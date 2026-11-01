// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact-byte verifier for PITB proof-search action calibration receipts.

use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const VERSION: &str = "pitb.proof-search-calibration-receipt.v0";
const BIN_COUNT: usize = 10;
const MIN_BIN_COUNT: u64 = 20;
const ACTION_KINDS: [&str; 8] = [
    "continue_current_target",
    "retrieve_context",
    "change_decomposition",
    "typed_repair",
    "tactic_probe",
    "spawn_snapshot_branch",
    "semantic_check",
    "restart_search",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: String,
    evidence_class: String,
    receipt_id: String,
    split: String,
    dataset_id: String,
    dataset_manifest_sha256: String,
    prediction_model_id: String,
    prediction_model_version: String,
    calibration_method: String,
    minimum_bin_count: u64,
    total_sample_count: u64,
    worst_action_ece_ppm: u64,
    mean_action_brier_score_ppm: u64,
    source_receipt_manifest_sha256: String,
    action_groups: Vec<ActionGroup>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionGroup {
    action_kind: String,
    sample_count: u64,
    ece_ppm: u64,
    brier_score_ppm: u64,
    bins: Vec<Bin>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bin {
    bin_id: u64,
    count: u64,
    min_predicted_ppm: u64,
    max_predicted_ppm: u64,
    sum_predicted_ppm: u64,
    sum_probability_squared_ppm2: u64,
    sum_probability_on_success_ppm: u64,
    success_count: u64,
    outcome_receipt_manifest_sha256: String,
}

#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    pub receipt_sha256: String,
    pub dataset_manifest_sha256: String,
    pub prediction_model_id: String,
    pub prediction_model_version: String,
    pub total_sample_count: u64,
    pub worst_action_ece_ppm: u64,
    pub mean_action_brier_score_ppm: u64,
    pub action_kinds: BTreeSet<String>,
    pub evidence_class: String,
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_sha(value: &str, label: &str) -> Result<()> {
    if !lower_hex(value, 64) {
        bail!("{label} must be a 64-character lowercase SHA-256 digest");
    }
    Ok(())
}

fn recompute_group(group: &ActionGroup) -> Result<(u64, u64)> {
    if !ACTION_KINDS.contains(&group.action_kind.as_str()) {
        bail!("unknown action kind in calibration receipt: {}", group.action_kind);
    }
    if group.bins.len() != BIN_COUNT {
        bail!("action {} must contain exactly {BIN_COUNT} calibration bins", group.action_kind);
    }

    let mut sample_count = 0_u64;
    let mut ece_numerator = 0_u128;
    let mut brier_numerator = 0_u128;
    for (index, bin) in group.bins.iter().enumerate() {
        if bin.bin_id != index as u64 {
            bail!("calibration bins must be ordered and contiguous from zero");
        }
        if bin.count < MIN_BIN_COUNT {
            bail!("calibration bin {} has fewer than {MIN_BIN_COUNT} observations", bin.bin_id);
        }
        if bin.success_count > bin.count
            || bin.min_predicted_ppm > bin.max_predicted_ppm
            || bin.max_predicted_ppm > 1_000_000
        {
            bail!("calibration bin {} has invalid count or probability bounds", bin.bin_id);
        }
        let lower = bin.bin_id * 100_000;
        let upper = if bin.bin_id == 9 { 1_000_000 } else { (bin.bin_id + 1) * 100_000 };
        if bin.min_predicted_ppm < lower || bin.max_predicted_ppm >= upper && bin.bin_id != 9
            || bin.bin_id == 9 && bin.max_predicted_ppm > upper
        {
            bail!("calibration bin {} contains predictions outside its registered interval", bin.bin_id);
        }
        let count = bin.count as u128;
        let sum_pred = bin.sum_predicted_ppm as u128;
        let min_sum = bin.min_predicted_ppm as u128 * count;
        let max_sum = bin.max_predicted_ppm as u128 * count;
        if sum_pred < min_sum || sum_pred > max_sum {
            bail!("calibration bin {} prediction sum conflicts with min/max bounds", bin.bin_id);
        }
        let max_square_sum = count * 1_000_000_000_000_u128;
        let min_square_sum = bin.min_predicted_ppm as u128
            * bin.min_predicted_ppm as u128
            * count;
        let max_square_bound = bin.max_predicted_ppm as u128
            * bin.max_predicted_ppm as u128
            * count;
        if (bin.sum_probability_squared_ppm2 as u128) < min_square_sum
            || (bin.sum_probability_squared_ppm2 as u128) > max_square_sum
            || (bin.sum_probability_squared_ppm2 as u128) > max_square_bound
        {
            bail!("calibration bin {} squared-probability sum conflicts with its min/max bounds", bin.bin_id);
        }
        let success_count = bin.success_count as u128;
        let probability_on_success = bin.sum_probability_on_success_ppm as u128;
        if probability_on_success < success_count * bin.min_predicted_ppm as u128
            || probability_on_success > success_count * bin.max_predicted_ppm as u128
            || probability_on_success > success_count * 1_000_000_u128
            || probability_on_success > sum_pred
        {
            bail!("calibration bin {} success-probability sum is impossible", bin.bin_id);
        }
        let observed_success_ppm = success_count * 1_000_000_u128;
        ece_numerator += sum_pred.abs_diff(observed_success_ppm);

        let squared = bin.sum_probability_squared_ppm2 as i128;
        let cross = 2_i128 * 1_000_000_i128 * probability_on_success as i128;
        let outcome_square = success_count as i128 * 1_000_000_000_000_i128;
        let brier = squared - cross + outcome_square;
        if brier < 0 || brier as u128 > max_square_sum {
            bail!("calibration bin {} has an impossible Brier sufficient statistic", bin.bin_id);
        }
        brier_numerator += brier as u128;
        sample_count = sample_count
            .checked_add(bin.count)
            .context("calibration sample count overflow")?;
        check_sha(
            &bin.outcome_receipt_manifest_sha256,
            "bin outcome receipt manifest",
        )?;
    }

    if sample_count != group.sample_count {
        bail!("action {} sample_count does not equal its bins", group.action_kind);
    }
    let ece = ((ece_numerator + (sample_count as u128 / 2)) / sample_count as u128) as u64;
    let brier_denominator = sample_count as u128 * 1_000_000_u128;
    let brier = ((brier_numerator + (brier_denominator / 2)) / brier_denominator) as u64;
    if ece != group.ece_ppm || brier != group.brier_score_ppm {
        bail!(
            "action {} claims ECE/Brier {}/{} but recomputation gives {}/{}",
            group.action_kind,
            group.ece_ppm,
            group.brier_score_ppm,
            ece,
            brier
        );
    }
    Ok((ece, brier))
}

pub fn verify_bytes(bytes: &[u8], expected_sha256: Option<&str>, allow_synthetic: bool) -> Result<CalibrationSummary> {
    let receipt_sha256 = hex::encode(Sha256::digest(bytes));
    if let Some(expected) = expected_sha256 {
        check_sha(expected, "expected calibration receipt SHA-256")?;
        if expected != receipt_sha256 {
            bail!("calibration receipt bytes do not match the decision's SHA-256 link");
        }
    }
    let receipt: Receipt = serde_json::from_slice(bytes).context("decode strict calibration receipt")?;
    if receipt.schema_version != VERSION
        || receipt.split != "validation"
        || receipt.calibration_method != "equal_width_10_bin_per_action"
        || receipt.minimum_bin_count != MIN_BIN_COUNT
        || receipt.receipt_id.trim().is_empty()
        || receipt.dataset_id.trim().is_empty()
        || receipt.prediction_model_id.trim().is_empty()
        || receipt.prediction_model_version.trim().is_empty()
    {
        bail!("calibration receipt metadata differs from the v0 contract");
    }
    if !matches!(receipt.evidence_class.as_str(), "synthetic_contract_fixture" | "measured_validation") {
        bail!("unknown calibration evidence class");
    }
    if receipt.evidence_class == "synthetic_contract_fixture" && !allow_synthetic {
        bail!("synthetic calibration receipt is not eligible for production selection");
    }
    check_sha(&receipt.dataset_manifest_sha256, "calibration dataset manifest")?;
    check_sha(&receipt.source_receipt_manifest_sha256, "source outcome receipt manifest")?;
    if receipt.action_groups.len() != ACTION_KINDS.len() {
        bail!("calibration receipt must cover all eight registered action kinds");
    }

    let mut action_kinds = BTreeSet::new();
    let mut total_sample_count = 0_u64;
    let mut worst_action_ece_ppm = 0_u64;
    let mut total_action_brier = 0_u128;
    for group in &receipt.action_groups {
        if !action_kinds.insert(group.action_kind.clone()) {
            bail!("duplicate calibration group for action kind {}", group.action_kind);
        }
        let (ece, brier) = recompute_group(group)?;
        total_sample_count = total_sample_count
            .checked_add(group.sample_count)
            .context("total calibration samples overflow")?;
        worst_action_ece_ppm = worst_action_ece_ppm.max(ece);
        total_action_brier += brier as u128;
    }
    if action_kinds.len() != ACTION_KINDS.len()
        || ACTION_KINDS.iter().any(|kind| !action_kinds.contains(*kind))
    {
        bail!("calibration receipt does not contain exactly the eight required action kinds");
    }
    let mean_action_brier_score_ppm =
        ((total_action_brier + (ACTION_KINDS.len() as u128 / 2)) / ACTION_KINDS.len() as u128) as u64;
    if total_sample_count != receipt.total_sample_count
        || worst_action_ece_ppm != receipt.worst_action_ece_ppm
        || mean_action_brier_score_ppm != receipt.mean_action_brier_score_ppm
    {
        bail!("calibration receipt summary metrics do not match independently recomputed action groups");
    }

    Ok(CalibrationSummary {
        receipt_sha256,
        dataset_manifest_sha256: receipt.dataset_manifest_sha256,
        prediction_model_id: receipt.prediction_model_id,
        prediction_model_version: receipt.prediction_model_version,
        total_sample_count,
        worst_action_ece_ppm,
        mean_action_brier_score_ppm,
        action_kinds,
        evidence_class: receipt.evidence_class,
    })
}

pub fn verify_file(path: &Path, expected_sha256: Option<&str>, allow_synthetic: bool) -> Result<CalibrationSummary> {
    let bytes = fs::read(path).with_context(|| format!("read calibration receipt {}", path.display()))?;
    verify_bytes(&bytes, expected_sha256, allow_synthetic)
}
