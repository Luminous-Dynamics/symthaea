// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-freeze exact-universe measurement for the Matbench Benchmark Zero ladder.
//!
//! Truth enters only after the comparison has been frozen. Full official truth is
//! first reduced to the exact retained candidate universe with a replayable
//! provenance receipt. Measurement then evaluates the already-frozen runs
//! without invoking any screening or model-fitting API.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use symthaea_energy_benchmark_zero::{
    BandgapTarget, BandgapTruthSet, BenchmarkError, DataSliceId, DatasetProvenance,
};
use symthaea_energy_benchmark_zero_exact::{
    ExactUniverseBenchmarkReceipt, ExactUniverseError, candidate_universe_sha256,
    evaluate_exact_universe,
};
use symthaea_matbench_gap::{
    MATBENCH_EXPT_GAP_DATASET_ID, MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE,
    MATBENCH_EXPT_GAP_SHA256, MATBENCH_EXPT_GAP_URL, MatbenchGapDataset,
};
use symthaea_matbench_gap_comparison_freeze::{
    ComparisonFreezeError, ComparisonFreezeReceipt, EndpointSpec, endpoint_contract,
};
use symthaea_matbench_gap_exposure_plan::{ExposedTrainingExclusionPlan, PlanError};
use thiserror::Error;

pub const RESTRICTED_TRUTH_SCHEMA: &str =
    "symthaea.matbench-gap.restricted-benchmark-truth.v0";
pub const MEASUREMENT_SCHEMA: &str = "symthaea.matbench-gap.frozen-comparison-measurement.v0";
pub const RESTRICTED_TRUTH_CAPABILITY: &str =
    "POST-FREEZE TRUTH RESTRICTION RECEIPT ONLY -- not a screening result, holdout-cleanliness certificate, or scientific conclusion.";
pub const MEASUREMENT_CAPABILITY: &str =
    "POST-FREEZE EXACT-UNIVERSE MEASUREMENT ONLY -- not a scalar winner claim, material certification, or promotion authority.";

const RESTRICTED_TRUTH_CONTENT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.restricted-truth-content.v0\0";
const RESTRICTED_TRUTH_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.restricted-truth-receipt.v0\0";
const MEASUREMENT_SUBJECT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.frozen-comparison-measurement.v0\0";
const MEASUREMENT_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.frozen-comparison-measurement-receipt.v0\0";

/// Exact post-freeze truth subset derived from the pinned official dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestrictedTruthReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub source_plan_sha256: String,
    pub comparison_subject_sha256: String,
    pub parent_compressed_sha256: String,
    pub parent_row_order_sha256: String,
    pub parent_truth_provenance: DatasetProvenance,
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    /// SHA-256 over sorted `(candidate_id, f64::to_bits(gap))` records.
    pub restricted_truth_sha256: String,
    pub truth: BandgapTruthSet,
}

impl RestrictedTruthReceipt {
    pub fn validate(&self) -> Result<(), FrozenMeasurementError> {
        if self.schema != RESTRICTED_TRUTH_SCHEMA
            || self.capability_classification != RESTRICTED_TRUTH_CAPABILITY
        {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "schema or capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan digest", self.source_plan_sha256.as_str()),
            (
                "comparison subject digest",
                self.comparison_subject_sha256.as_str(),
            ),
            (
                "parent compressed SHA-256",
                self.parent_compressed_sha256.as_str(),
            ),
            (
                "parent row-order SHA-256",
                self.parent_row_order_sha256.as_str(),
            ),
            (
                "candidate universe SHA-256",
                self.candidate_universe_sha256.as_str(),
            ),
            (
                "restricted truth SHA-256",
                self.restricted_truth_sha256.as_str(),
            ),
        ] {
            validate_sha256(value, name)?;
        }
        if self.parent_compressed_sha256 != MATBENCH_EXPT_GAP_SHA256 {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "parent compressed artifact is not the pinned official Matbench artifact".into(),
            ));
        }
        self.parent_truth_provenance.validate()?;
        self.truth.validate()?;
        if self.parent_truth_provenance.slice.dataset_id != MATBENCH_EXPT_GAP_DATASET_ID
            || self.parent_truth_provenance.source_uri != MATBENCH_EXPT_GAP_URL
            || self.parent_truth_provenance.content_digest
                != format!("sha256:{MATBENCH_EXPT_GAP_SHA256}")
            || self.parent_truth_provenance.license != MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE
        {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "parent truth provenance differs from the pinned official Matbench identity"
                    .into(),
            ));
        }
        if self.candidate_count == 0
            || self.truth.experimental_gap_ev.len() != self.candidate_count
        {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "restricted truth count must equal non-zero candidate count".into(),
            ));
        }

        let observed_candidate_universe = candidate_universe_sha256(
            self.truth.experimental_gap_ev.keys().map(String::as_str),
        )?;
        if observed_candidate_universe != self.candidate_universe_sha256 {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "restricted truth candidate set does not match its universe digest".into(),
            ));
        }
        let observed_truth_sha256 = restricted_truth_content_sha256(&self.truth.experimental_gap_ev)?;
        if observed_truth_sha256 != self.restricted_truth_sha256 {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "restricted truth values do not match their content digest".into(),
            ));
        }

        let expected_split = restricted_split_id(&self.candidate_universe_sha256);
        if self.truth.provenance.slice.dataset_id != self.parent_truth_provenance.slice.dataset_id
            || self.truth.provenance.slice.split_id != expected_split
            || self.truth.provenance.source_uri != self.parent_truth_provenance.source_uri
            || self.truth.provenance.license != self.parent_truth_provenance.license
            || self.truth.provenance.content_digest
                != format!("sha256:{}", self.restricted_truth_sha256)
        {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "restricted truth provenance does not bind parent source, candidate universe and restricted content"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, FrozenMeasurementError> {
        self.validate()?;
        domain_separated_sha256(RESTRICTED_TRUTH_RECEIPT_DOMAIN, self)
    }
}

/// Restrict official parsed truth to the exact candidate universe that was
/// frozen before truth was supplied.
pub fn restrict_official_truth_after_freeze(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
    dataset: &MatbenchGapDataset,
) -> Result<RestrictedTruthReceipt, FrozenMeasurementError> {
    plan.validate()?;
    freeze.validate()?;
    dataset.truth.validate()?;

    if freeze.source_plan_sha256 != plan.sha256()?
        || freeze.source_composition_order_sha256 != plan.source_composition_order_sha256
        || freeze.source_partition_sha256 != plan.partition_sha256
        || freeze.symthaea_training_snapshot_sha256 != plan.symthaea_training_snapshot_sha256
        || freeze.retained_universe_sha256 != plan.retained_universe_sha256
        || freeze.candidate_count != plan.retained_composition_count
    {
        return Err(FrozenMeasurementError::FreezePlanMismatch);
    }

    if plan.source_dataset_id != MATBENCH_EXPT_GAP_DATASET_ID
        || plan.source_compressed_sha256 != MATBENCH_EXPT_GAP_SHA256
        || plan.source_content_digest != format!("sha256:{MATBENCH_EXPT_GAP_SHA256}")
        || dataset.compressed_sha256 != plan.source_compressed_sha256
        || dataset.row_order_sha256 != plan.source_row_order_sha256
        || dataset.truth.provenance.slice.dataset_id != plan.source_dataset_id
        || dataset.truth.provenance.content_digest != plan.source_content_digest
        || dataset.truth.provenance.source_uri != MATBENCH_EXPT_GAP_URL
        || dataset.truth.provenance.license != MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE
    {
        return Err(FrozenMeasurementError::ParentDatasetMismatch);
    }

    if dataset.ordered_records.len() != plan.source_row_count
        || plan.rows.len() != plan.source_row_count
        || dataset.truth.experimental_gap_ev.len() != plan.source_row_count
    {
        return Err(FrozenMeasurementError::ParentDatasetMismatch);
    }

    let mut plan_ids = BTreeSet::new();
    for (index, (record, planned)) in dataset
        .ordered_records
        .iter()
        .zip(&plan.rows)
        .enumerate()
    {
        let truth_gap = dataset
            .truth
            .experimental_gap_ev
            .get(&record.candidate_id)
            .ok_or(FrozenMeasurementError::ParentDatasetMismatch)?;
        if planned.source_row_index != index
            || record.candidate_id != planned.candidate_id
            || record.composition != planned.composition
            || truth_gap.to_bits() != record.experimental_gap_ev.to_bits()
        {
            return Err(FrozenMeasurementError::ParentDatasetMismatch);
        }
        plan_ids.insert(planned.candidate_id.as_str());
    }
    let truth_ids: BTreeSet<&str> = dataset
        .truth
        .experimental_gap_ev
        .keys()
        .map(String::as_str)
        .collect();
    if plan_ids != truth_ids || plan_ids.len() != plan.source_row_count {
        return Err(FrozenMeasurementError::ParentDatasetMismatch);
    }

    let retained_ids: BTreeSet<&str> = freeze
        .learned
        .run
        .ranked
        .iter()
        .map(|record| record.candidate_id.as_str())
        .collect();
    if retained_ids.len() != freeze.candidate_count {
        return Err(FrozenMeasurementError::FreezePlanMismatch);
    }
    let observed_universe = candidate_universe_sha256(retained_ids.iter().copied())?;
    if observed_universe != freeze.candidate_universe_sha256 {
        return Err(FrozenMeasurementError::FreezePlanMismatch);
    }

    let mut restricted_values = BTreeMap::new();
    for candidate_id in retained_ids {
        let value = dataset
            .truth
            .experimental_gap_ev
            .get(candidate_id)
            .copied()
            .ok_or_else(|| FrozenMeasurementError::MissingFrozenTruth(candidate_id.to_owned()))?;
        restricted_values.insert(candidate_id.to_owned(), value);
    }
    let restricted_truth_sha256 = restricted_truth_content_sha256(&restricted_values)?;
    let restricted_truth = BandgapTruthSet {
        provenance: DatasetProvenance {
            slice: DataSliceId::new(
                dataset.truth.provenance.slice.dataset_id.clone(),
                restricted_split_id(&freeze.candidate_universe_sha256),
            )?,
            source_uri: dataset.truth.provenance.source_uri.clone(),
            content_digest: format!("sha256:{restricted_truth_sha256}"),
            license: dataset.truth.provenance.license.clone(),
        },
        experimental_gap_ev: restricted_values,
    };

    let receipt = RestrictedTruthReceipt {
        schema: RESTRICTED_TRUTH_SCHEMA.into(),
        capability_classification: RESTRICTED_TRUTH_CAPABILITY.into(),
        source_plan_sha256: freeze.source_plan_sha256.clone(),
        comparison_subject_sha256: freeze.comparison_subject_sha256.clone(),
        parent_compressed_sha256: dataset.compressed_sha256.clone(),
        parent_row_order_sha256: dataset.row_order_sha256.clone(),
        parent_truth_provenance: dataset.truth.provenance.clone(),
        candidate_universe_sha256: freeze.candidate_universe_sha256.clone(),
        candidate_count: freeze.candidate_count,
        restricted_truth_sha256,
        truth: restricted_truth,
    };
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_restricted_truth_receipt(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
    dataset: &MatbenchGapDataset,
    expected: &RestrictedTruthReceipt,
) -> Result<(), FrozenMeasurementError> {
    expected.validate()?;
    let observed = restrict_official_truth_after_freeze(plan, freeze, dataset)?;
    if &observed != expected {
        return Err(FrozenMeasurementError::RestrictedTruthReplayMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyRole {
    DeterministicRandomNull,
    LegacyTargetDistance,
    CompositionOnlyLearned,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEndpointValues {
    pub role: PolicyRole,
    pub target_regret_ev: f64,
    pub top_k_hits: usize,
    pub mean_abs_prediction_error_ev: f64,
}

/// One post-freeze comparison measurement. No scalar winner is stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenComparisonMeasurementReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub source_plan_sha256: String,
    pub comparison_freeze_sha256: String,
    pub comparison_subject_sha256: String,
    pub restricted_truth_receipt_sha256: String,
    pub restricted_truth_sha256: String,
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    pub target: BandgapTarget,
    pub top_k: usize,
    pub endpoints: Vec<EndpointSpec>,
    pub random_control: ExactUniverseBenchmarkReceipt,
    pub legacy_target_distance: ExactUniverseBenchmarkReceipt,
    pub learned: ExactUniverseBenchmarkReceipt,
    pub endpoint_values: Vec<PolicyEndpointValues>,
    pub measurement_subject_sha256: String,
}

impl FrozenComparisonMeasurementReceipt {
    pub fn validate(&self) -> Result<(), FrozenMeasurementError> {
        if self.schema != MEASUREMENT_SCHEMA
            || self.capability_classification != MEASUREMENT_CAPABILITY
        {
            return Err(FrozenMeasurementError::InvalidMeasurement(
                "schema or capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan digest", self.source_plan_sha256.as_str()),
            (
                "comparison freeze digest",
                self.comparison_freeze_sha256.as_str(),
            ),
            (
                "comparison subject digest",
                self.comparison_subject_sha256.as_str(),
            ),
            (
                "restricted truth receipt digest",
                self.restricted_truth_receipt_sha256.as_str(),
            ),
            (
                "restricted truth digest",
                self.restricted_truth_sha256.as_str(),
            ),
            (
                "candidate universe digest",
                self.candidate_universe_sha256.as_str(),
            ),
            (
                "measurement subject digest",
                self.measurement_subject_sha256.as_str(),
            ),
        ] {
            validate_sha256(value, name)?;
        }
        self.target.validate()?;
        if self.candidate_count == 0 || self.top_k == 0 || self.top_k > self.candidate_count {
            return Err(FrozenMeasurementError::InvalidMeasurement(
                "top_k must be in 1..=candidate_count".into(),
            ));
        }
        if self.endpoints != endpoint_contract() {
            return Err(FrozenMeasurementError::InvalidMeasurement(
                "measurement endpoint contract differs from preregistered V0 vector".into(),
            ));
        }

        self.random_control.validate()?;
        self.legacy_target_distance.validate()?;
        self.learned.validate()?;
        for measurement in [
            &self.random_control,
            &self.legacy_target_distance,
            &self.learned,
        ] {
            if measurement.candidate_universe_sha256 != self.candidate_universe_sha256
                || measurement.candidate_count != self.candidate_count
                || measurement.benchmark.metrics.k != self.top_k
                || measurement.benchmark.target != self.target
            {
                return Err(FrozenMeasurementError::InvalidMeasurement(
                    "policy measurement differs from frozen universe/target/top-k".into(),
                ));
            }
        }

        let expected_values = endpoint_values_from_receipts(
            &self.random_control,
            &self.legacy_target_distance,
            &self.learned,
        );
        if self.endpoint_values != expected_values {
            return Err(FrozenMeasurementError::InvalidMeasurement(
                "endpoint summary does not match exact Benchmark Zero receipts".into(),
            ));
        }
        for values in &self.endpoint_values {
            if !values.target_regret_ev.is_finite()
                || values.target_regret_ev < 0.0
                || !values.mean_abs_prediction_error_ev.is_finite()
                || values.mean_abs_prediction_error_ev < 0.0
            {
                return Err(FrozenMeasurementError::InvalidMeasurement(
                    "endpoint values must be finite and non-negative".into(),
                ));
            }
        }

        let observed_subject = measurement_subject_sha256(self)?;
        if observed_subject != self.measurement_subject_sha256 {
            return Err(FrozenMeasurementError::InvalidMeasurement(
                "measurement subject digest does not match frozen inputs/results".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, FrozenMeasurementError> {
        self.validate()?;
        domain_separated_sha256(MEASUREMENT_RECEIPT_DOMAIN, self)
    }
}

/// Measure the three already-frozen runs against an already-restricted truth
/// set. This function performs no model fitting, screening, reranking, or truth
/// restriction.
pub fn measure_frozen_comparison(
    freeze: &ComparisonFreezeReceipt,
    restricted: &RestrictedTruthReceipt,
) -> Result<FrozenComparisonMeasurementReceipt, FrozenMeasurementError> {
    freeze.validate()?;
    restricted.validate()?;
    if restricted.source_plan_sha256 != freeze.source_plan_sha256
        || restricted.comparison_subject_sha256 != freeze.comparison_subject_sha256
        || restricted.candidate_universe_sha256 != freeze.candidate_universe_sha256
        || restricted.candidate_count != freeze.candidate_count
    {
        return Err(FrozenMeasurementError::RestrictedTruthFreezeMismatch);
    }

    let random_control = evaluate_exact_universe(
        &freeze.random_control.run,
        &restricted.truth,
        freeze.top_k,
    )?;
    let legacy_target_distance = evaluate_exact_universe(
        &freeze.legacy_target_distance.run,
        &restricted.truth,
        freeze.top_k,
    )?;
    let learned = evaluate_exact_universe(&freeze.learned.run, &restricted.truth, freeze.top_k)?;

    for measurement in [&random_control, &legacy_target_distance, &learned] {
        if measurement.candidate_universe_sha256 != freeze.candidate_universe_sha256 {
            return Err(FrozenMeasurementError::RestrictedTruthFreezeMismatch);
        }
    }

    let endpoint_values =
        endpoint_values_from_receipts(&random_control, &legacy_target_distance, &learned);
    let mut receipt = FrozenComparisonMeasurementReceipt {
        schema: MEASUREMENT_SCHEMA.into(),
        capability_classification: MEASUREMENT_CAPABILITY.into(),
        source_plan_sha256: freeze.source_plan_sha256.clone(),
        comparison_freeze_sha256: freeze.sha256()?,
        comparison_subject_sha256: freeze.comparison_subject_sha256.clone(),
        restricted_truth_receipt_sha256: restricted.sha256()?,
        restricted_truth_sha256: restricted.restricted_truth_sha256.clone(),
        candidate_universe_sha256: freeze.candidate_universe_sha256.clone(),
        candidate_count: freeze.candidate_count,
        target: freeze.target,
        top_k: freeze.top_k,
        endpoints: freeze.endpoints.clone(),
        random_control,
        legacy_target_distance,
        learned,
        endpoint_values,
        measurement_subject_sha256: String::new(),
    };
    receipt.measurement_subject_sha256 = measurement_subject_sha256(&receipt)?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_frozen_comparison_measurement(
    freeze: &ComparisonFreezeReceipt,
    restricted: &RestrictedTruthReceipt,
    expected: &FrozenComparisonMeasurementReceipt,
) -> Result<(), FrozenMeasurementError> {
    expected.validate()?;
    let observed = measure_frozen_comparison(freeze, restricted)?;
    if &observed != expected {
        return Err(FrozenMeasurementError::MeasurementReplayMismatch);
    }
    Ok(())
}

fn endpoint_values_from_receipts(
    random: &ExactUniverseBenchmarkReceipt,
    legacy: &ExactUniverseBenchmarkReceipt,
    learned: &ExactUniverseBenchmarkReceipt,
) -> Vec<PolicyEndpointValues> {
    [
        (PolicyRole::DeterministicRandomNull, random),
        (PolicyRole::LegacyTargetDistance, legacy),
        (PolicyRole::CompositionOnlyLearned, learned),
    ]
    .into_iter()
    .map(|(role, receipt)| PolicyEndpointValues {
        role,
        target_regret_ev: receipt.benchmark.metrics.target_regret_ev,
        top_k_hits: receipt.benchmark.metrics.top_k_hits,
        mean_abs_prediction_error_ev: receipt.benchmark.metrics.mean_abs_prediction_error_ev,
    })
    .collect()
}

#[derive(Serialize)]
struct TruthBitRecord<'a> {
    candidate_id: &'a str,
    gap_bits: u64,
}

fn restricted_truth_content_sha256(
    values: &BTreeMap<String, f64>,
) -> Result<String, FrozenMeasurementError> {
    if values.is_empty() {
        return Err(FrozenMeasurementError::InvalidRestrictedTruth(
            "restricted truth cannot be empty".into(),
        ));
    }
    let mut records = Vec::with_capacity(values.len());
    for (candidate_id, gap) in values {
        if candidate_id.trim().is_empty() || !gap.is_finite() || *gap < 0.0 {
            return Err(FrozenMeasurementError::InvalidRestrictedTruth(
                "restricted truth requires non-empty candidate ids and finite non-negative gaps"
                    .into(),
            ));
        }
        records.push(TruthBitRecord {
            candidate_id,
            gap_bits: gap.to_bits(),
        });
    }
    domain_separated_sha256(RESTRICTED_TRUTH_CONTENT_DOMAIN, &records)
}

fn restricted_split_id(candidate_universe_sha256: &str) -> String {
    format!("retained-universe-sha256:{candidate_universe_sha256}")
}

#[derive(Serialize)]
struct MeasurementSubject<'a> {
    source_plan_sha256: &'a str,
    comparison_freeze_sha256: &'a str,
    comparison_subject_sha256: &'a str,
    restricted_truth_receipt_sha256: &'a str,
    restricted_truth_sha256: &'a str,
    candidate_universe_sha256: &'a str,
    candidate_count: usize,
    target: BandgapTarget,
    top_k: usize,
    endpoints: &'a [EndpointSpec],
    random_receipt_sha256: String,
    legacy_receipt_sha256: String,
    learned_receipt_sha256: String,
}

fn measurement_subject_sha256(
    receipt: &FrozenComparisonMeasurementReceipt,
) -> Result<String, FrozenMeasurementError> {
    let subject = MeasurementSubject {
        source_plan_sha256: &receipt.source_plan_sha256,
        comparison_freeze_sha256: &receipt.comparison_freeze_sha256,
        comparison_subject_sha256: &receipt.comparison_subject_sha256,
        restricted_truth_receipt_sha256: &receipt.restricted_truth_receipt_sha256,
        restricted_truth_sha256: &receipt.restricted_truth_sha256,
        candidate_universe_sha256: &receipt.candidate_universe_sha256,
        candidate_count: receipt.candidate_count,
        target: receipt.target,
        top_k: receipt.top_k,
        endpoints: &receipt.endpoints,
        random_receipt_sha256: receipt.random_control.sha256()?,
        legacy_receipt_sha256: receipt.legacy_target_distance.sha256()?,
        learned_receipt_sha256: receipt.learned.sha256()?,
    };
    domain_separated_sha256(MEASUREMENT_SUBJECT_DOMAIN, &subject)
}

fn validate_sha256(value: &str, name: &str) -> Result<(), FrozenMeasurementError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FrozenMeasurementError::InvalidMeasurement(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, FrozenMeasurementError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum FrozenMeasurementError {
    #[error("exposure plan rejected: {0}")]
    Plan(#[from] PlanError),
    #[error("comparison freeze rejected: {0}")]
    Freeze(#[from] ComparisonFreezeError),
    #[error("Benchmark Zero contract rejected: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("exact-universe measurement rejected: {0}")]
    ExactUniverse(#[from] ExactUniverseError),
    #[error("comparison freeze does not bind the supplied exposure plan")]
    FreezePlanMismatch,
    #[error("parsed parent dataset does not exactly match the frozen exposure-plan source")]
    ParentDatasetMismatch,
    #[error("official truth is missing frozen candidate {0:?}")]
    MissingFrozenTruth(String),
    #[error("restricted truth receipt replay differs from supplied receipt")]
    RestrictedTruthReplayMismatch,
    #[error("restricted truth does not bind the frozen comparison")]
    RestrictedTruthFreezeMismatch,
    #[error("invalid restricted truth receipt: {0}")]
    InvalidRestrictedTruth(String),
    #[error("invalid frozen comparison measurement: {0}")]
    InvalidMeasurement(String),
    #[error("frozen comparison measurement replay differs from supplied receipt")]
    MeasurementReplayMismatch,
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_truth_digest_is_map_order_independent() {
        let first = BTreeMap::from([("A".to_owned(), 1.25), ("B".to_owned(), 2.5)]);
        let mut second = BTreeMap::new();
        second.insert("B".to_owned(), 2.5);
        second.insert("A".to_owned(), 1.25);
        assert_eq!(
            restricted_truth_content_sha256(&first).unwrap(),
            restricted_truth_content_sha256(&second).unwrap()
        );
    }

    #[test]
    fn restricted_truth_digest_binds_exact_float_bits() {
        let first = BTreeMap::from([("A".to_owned(), 1.25)]);
        let second = BTreeMap::from([(
            "A".to_owned(),
            f64::from_bits(1.25f64.to_bits() + 1),
        )]);
        assert_ne!(
            restricted_truth_content_sha256(&first).unwrap(),
            restricted_truth_content_sha256(&second).unwrap()
        );
    }

    #[test]
    fn retained_split_id_is_candidate_universe_specific() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        assert_ne!(restricted_split_id(&a), restricted_split_id(&b));
        assert!(restricted_split_id(&a).contains(&a));
    }
}
