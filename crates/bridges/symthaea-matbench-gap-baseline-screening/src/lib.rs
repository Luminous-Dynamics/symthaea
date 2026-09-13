// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Truth-blind baseline screening over the frozen Matbench composition universe.
//!
//! This crate consumes only the truth-free partition from
//! `symthaea-matbench-gap-exposure-plan`. It never accepts `BandgapTruthSet` and
//! never reads benchmark experimental gap values.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use symthaea_bandgap::bandgap_baseline::electronegativity_bandgap;
use symthaea_energy_benchmark_zero::{
    BandgapTarget, BenchmarkError, ScreeningMethodProvenance, ScreeningRecord, ScreeningRun,
};
use symthaea_matbench_gap_exposure_plan::{
    ExposedTrainingExclusionPlan, ExposureDisposition, PlanError, PlannedComposition,
};
use thiserror::Error;

pub const RECEIPT_SCHEMA: &str = "symthaea.matbench-gap.baseline-screening-receipt.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "TRUTH-BLIND BASELINE SCREENING RECEIPT ONLY -- not benchmark truth, not a clean-holdout certificate, and not a scientific result.";
pub const METHOD_VERSION: &str = "v0";
pub const TARGET_DISTANCE_METHOD_ID: &str =
    "symthaea.matbench-gap.legacy-electronegativity-target-distance";
pub const RANDOM_ORDER_METHOD_ID_PREFIX: &str =
    "symthaea.matbench-gap.legacy-electronegativity-random-order";
pub const PRIOR_KNOWLEDGE_DISCLOSURE: &str =
    "The electronegativity baseline is a legacy empirical/physics-inspired heuristic whose source comments describe fitted/chosen semiconductor parameters. An empty machine-readable training_slices set means no exact registered DataSliceId is declared here; it does not mean the method had no historical empirical tuning or prior knowledge.";

const SCREENING_SUBJECT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.baseline-screening-subject.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.baseline-screening-receipt.v0\0";
const RANDOM_ORDER_KEY_DOMAIN: &[u8] = b"symthaea.matbench-gap.random-order-key.v0\0";

/// Truth-blind ordering policy applied to the same composition-only baseline
/// predictions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum BaselineScreeningPolicy {
    /// Rank by absolute distance from baseline prediction to the preregistered
    /// target-window midpoint.
    TargetDistance,
    /// Null ordering control. Prediction values remain identical to the target
    /// policy; only ordering changes.
    DeterministicRandomOrder { seed: u64 },
}

impl BaselineScreeningPolicy {
    fn method_id(&self) -> String {
        match self {
            Self::TargetDistance => TARGET_DISTANCE_METHOD_ID.to_owned(),
            Self::DeterministicRandomOrder { seed } => {
                format!("{RANDOM_ORDER_METHOD_ID_PREFIX}.seed-{seed:016x}")
            }
        }
    }
}

/// Replayable truth-blind screening evidence.
///
/// `source_plan_sha256` is retained for exact provenance but is deliberately
/// excluded from `screening_subject_sha256`: the full plan binds exact official
/// artifact identities that may change when truth values change, whereas the
/// screening subject is defined only by truth-free universe/policy inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineScreeningReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub source_plan_sha256: String,
    pub source_composition_order_sha256: String,
    pub source_partition_sha256: String,
    pub symthaea_training_snapshot_sha256: String,
    pub retained_universe_sha256: String,
    pub policy: BaselineScreeningPolicy,
    pub target: BandgapTarget,
    pub screening_subject_sha256: String,
    pub ranking_digest: String,
    pub ranked_candidate_count: usize,
    pub run: ScreeningRun,
    pub prior_knowledge_disclosure: String,
}

impl BaselineScreeningReceipt {
    pub fn validate(&self) -> Result<(), ScreeningError> {
        if self.schema != RECEIPT_SCHEMA
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(ScreeningError::InvalidReceipt(
                "receipt schema/capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan SHA-256", self.source_plan_sha256.as_str()),
            (
                "source composition-order SHA-256",
                self.source_composition_order_sha256.as_str(),
            ),
            ("source partition SHA-256", self.source_partition_sha256.as_str()),
            (
                "Symthaea training snapshot SHA-256",
                self.symthaea_training_snapshot_sha256.as_str(),
            ),
            (
                "retained-universe SHA-256",
                self.retained_universe_sha256.as_str(),
            ),
            (
                "screening-subject SHA-256",
                self.screening_subject_sha256.as_str(),
            ),
            ("ranking digest", self.ranking_digest.as_str()),
        ] {
            validate_sha256(value, name)?;
        }
        if self.prior_knowledge_disclosure != PRIOR_KNOWLEDGE_DISCLOSURE {
            return Err(ScreeningError::InvalidReceipt(
                "prior-knowledge disclosure was altered".into(),
            ));
        }
        self.target.validate()?;
        self.run.validate()?;
        if self.run.target != self.target {
            return Err(ScreeningError::InvalidReceipt(
                "screening run target differs from receipt target".into(),
            ));
        }
        let expected_method_id = self.policy.method_id();
        if self.run.method.method_id != expected_method_id
            || self.run.method.version != METHOD_VERSION
            || !self.run.method.training_slices.is_empty()
        {
            return Err(ScreeningError::InvalidReceipt(
                "screening method provenance differs from the fixed baseline policy".into(),
            ));
        }
        if self.ranked_candidate_count == 0
            || self.ranked_candidate_count != self.run.ranked.len()
        {
            return Err(ScreeningError::InvalidReceipt(
                "ranked candidate count must equal the non-empty screening run".into(),
            ));
        }
        let expected_ranking = self.run.ranking_digest()?;
        if self.ranking_digest != expected_ranking {
            return Err(ScreeningError::InvalidReceipt(
                "ranking digest does not match the ordered screening run".into(),
            ));
        }
        let expected_subject = screening_subject_digest(
            &self.source_composition_order_sha256,
            &self.source_partition_sha256,
            &self.symthaea_training_snapshot_sha256,
            &self.retained_universe_sha256,
            &self.policy,
            self.target,
        )?;
        if self.screening_subject_sha256 != expected_subject {
            return Err(ScreeningError::InvalidReceipt(
                "screening-subject digest does not match truth-free screening inputs".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ScreeningError> {
        self.validate()?;
        domain_separated_sha256(RECEIPT_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug)]
struct ScoredCandidate {
    source_row_index: usize,
    candidate_id: String,
    predicted_gap_ev: f64,
    target_error_ev: f64,
    random_order_key: [u8; 32],
}

/// Screen the exact retained universe using one truth-blind baseline policy.
pub fn screen_baseline(
    plan: &ExposedTrainingExclusionPlan,
    target: BandgapTarget,
    policy: BaselineScreeningPolicy,
) -> Result<BaselineScreeningReceipt, ScreeningError> {
    plan.validate_against_current_training_snapshot()?;
    target.validate()?;

    let source_plan_sha256 = plan.sha256()?;
    let ranked = rank_retained_rows(plan.retained_rows(), target, &policy)?;
    if ranked.is_empty() {
        return Err(ScreeningError::EmptyRetainedUniverse);
    }

    let method = ScreeningMethodProvenance {
        method_id: policy.method_id(),
        version: METHOD_VERSION.to_owned(),
        training_slices: BTreeSet::new(),
    };
    let run = ScreeningRun {
        method,
        target,
        ranked,
    };
    run.validate()?;
    let ranking_digest = run.ranking_digest()?;
    let screening_subject_sha256 = screening_subject_digest(
        &plan.source_composition_order_sha256,
        &plan.partition_sha256,
        &plan.symthaea_training_snapshot_sha256,
        &plan.retained_universe_sha256,
        &policy,
        target,
    )?;

    let receipt = BaselineScreeningReceipt {
        schema: RECEIPT_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source_plan_sha256,
        source_composition_order_sha256: plan.source_composition_order_sha256.clone(),
        source_partition_sha256: plan.partition_sha256.clone(),
        symthaea_training_snapshot_sha256: plan.symthaea_training_snapshot_sha256.clone(),
        retained_universe_sha256: plan.retained_universe_sha256.clone(),
        policy,
        target,
        screening_subject_sha256,
        ranking_digest,
        ranked_candidate_count: run.ranked.len(),
        run,
        prior_knowledge_disclosure: PRIOR_KNOWLEDGE_DISCLOSURE.into(),
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Recompute the screening result from the frozen truth-free plan and require
/// exact receipt equality.
pub fn verify_baseline_screening_receipt(
    plan: &ExposedTrainingExclusionPlan,
    receipt: &BaselineScreeningReceipt,
) -> Result<(), ScreeningError> {
    receipt.validate()?;
    plan.validate_against_current_training_snapshot()?;
    if receipt.source_plan_sha256 != plan.sha256()?
        || receipt.source_composition_order_sha256 != plan.source_composition_order_sha256
        || receipt.source_partition_sha256 != plan.partition_sha256
        || receipt.symthaea_training_snapshot_sha256 != plan.symthaea_training_snapshot_sha256
        || receipt.retained_universe_sha256 != plan.retained_universe_sha256
    {
        return Err(ScreeningError::PlanIdentityMismatch);
    }

    let observed = screen_baseline(plan, receipt.target, receipt.policy.clone())?;
    if &observed != receipt {
        return Err(ScreeningError::ReplayMismatch);
    }
    Ok(())
}

fn rank_retained_rows<'a>(
    rows: impl Iterator<Item = &'a PlannedComposition>,
    target: BandgapTarget,
    policy: &BaselineScreeningPolicy,
) -> Result<Vec<ScreeningRecord>, ScreeningError> {
    let mut scored = Vec::new();
    for row in rows {
        if !matches!(&row.disposition, ExposureDisposition::Retained) {
            continue;
        }
        let composition = composition_from_fingerprint(row)?;
        let predicted_gap_ev = electronegativity_bandgap(&composition);
        if !predicted_gap_ev.is_finite() || predicted_gap_ev < 0.0 {
            return Err(ScreeningError::InvalidPrediction {
                candidate_id: row.candidate_id.clone(),
                value: predicted_gap_ev,
            });
        }
        scored.push(ScoredCandidate {
            source_row_index: row.source_row_index,
            candidate_id: row.candidate_id.clone(),
            predicted_gap_ev,
            target_error_ev: target.midpoint_error(predicted_gap_ev),
            random_order_key: deterministic_random_key(policy, &row.candidate_id),
        });
    }

    match policy {
        BaselineScreeningPolicy::TargetDistance => scored.sort_by(|left, right| {
            left.target_error_ev
                .total_cmp(&right.target_error_ev)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
                .then_with(|| left.source_row_index.cmp(&right.source_row_index))
        }),
        BaselineScreeningPolicy::DeterministicRandomOrder { .. } => {
            scored.sort_by(|left, right| {
                left.random_order_key
                    .cmp(&right.random_order_key)
                    .then_with(|| left.candidate_id.cmp(&right.candidate_id))
                    .then_with(|| left.source_row_index.cmp(&right.source_row_index))
            });
        }
    }

    Ok(scored
        .into_iter()
        .map(|candidate| ScreeningRecord {
            candidate_id: candidate.candidate_id,
            predicted_gap_ev: candidate.predicted_gap_ev,
            uncertainty_ev: None,
        })
        .collect())
}

fn composition_from_fingerprint(
    row: &PlannedComposition,
) -> Result<Vec<(u8, f64)>, ScreeningError> {
    if row.composition.0.is_empty() {
        return Err(ScreeningError::InvalidComposition(
            row.candidate_id.clone(),
            "composition fingerprint is empty".into(),
        ));
    }
    let total: u128 = row
        .composition
        .0
        .iter()
        .map(|(_, fraction)| u128::from(*fraction))
        .sum();
    if total == 0 {
        return Err(ScreeningError::InvalidComposition(
            row.candidate_id.clone(),
            "composition fingerprint has zero total".into(),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut composition = Vec::with_capacity(row.composition.0.len());
    for &(atomic_number, fraction) in &row.composition.0 {
        if atomic_number == 0 || fraction == 0 || !seen.insert(atomic_number) {
            return Err(ScreeningError::InvalidComposition(
                row.candidate_id.clone(),
                "composition fingerprint requires positive unique element entries".into(),
            ));
        }
        composition.push((atomic_number, fraction as f64 / total as f64));
    }
    Ok(composition)
}

fn deterministic_random_key(policy: &BaselineScreeningPolicy, candidate_id: &str) -> [u8; 32] {
    let seed = match policy {
        BaselineScreeningPolicy::TargetDistance => 0,
        BaselineScreeningPolicy::DeterministicRandomOrder { seed } => *seed,
    };
    let mut hasher = Sha256::new();
    hasher.update(RANDOM_ORDER_KEY_DOMAIN);
    hasher.update(seed.to_le_bytes());
    hasher.update(candidate_id.as_bytes());
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    key
}

#[derive(Serialize)]
struct ScreeningSubject<'a> {
    source_composition_order_sha256: &'a str,
    source_partition_sha256: &'a str,
    symthaea_training_snapshot_sha256: &'a str,
    retained_universe_sha256: &'a str,
    policy: &'a BaselineScreeningPolicy,
    target: BandgapTarget,
    method_id: String,
    method_version: &'static str,
}

fn screening_subject_digest(
    source_composition_order_sha256: &str,
    source_partition_sha256: &str,
    symthaea_training_snapshot_sha256: &str,
    retained_universe_sha256: &str,
    policy: &BaselineScreeningPolicy,
    target: BandgapTarget,
) -> Result<String, ScreeningError> {
    let subject = ScreeningSubject {
        source_composition_order_sha256,
        source_partition_sha256,
        symthaea_training_snapshot_sha256,
        retained_universe_sha256,
        policy,
        target,
        method_id: policy.method_id(),
        method_version: METHOD_VERSION,
    };
    domain_separated_sha256(SCREENING_SUBJECT_DIGEST_DOMAIN, &subject)
}

fn validate_sha256(value: &str, name: &str) -> Result<(), ScreeningError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ScreeningError::InvalidReceipt(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, ScreeningError> {
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
pub enum ScreeningError {
    #[error("exposure plan rejected: {0}")]
    Plan(#[from] PlanError),
    #[error("Benchmark Zero contract rejected: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("retained screening universe is empty")]
    EmptyRetainedUniverse,
    #[error("invalid composition for {0:?}: {1}")]
    InvalidComposition(String, String),
    #[error("baseline prediction for {candidate_id:?} is invalid: {value}")]
    InvalidPrediction { candidate_id: String, value: f64 },
    #[error("screening receipt does not bind the supplied exposure plan")]
    PlanIdentityMismatch,
    #[error("screening receipt replay differs from the supplied receipt")]
    ReplayMismatch,
    #[error("invalid baseline screening receipt: {0}")]
    InvalidReceipt(String),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_matbench_gap::CompositionFingerprint;

    fn retained(index: usize, id: &str, composition: Vec<(u8, u64)>) -> PlannedComposition {
        PlannedComposition {
            source_row_index: index,
            candidate_id: id.into(),
            composition: CompositionFingerprint(composition),
            disposition: ExposureDisposition::Retained,
        }
    }

    fn excluded(index: usize, id: &str, composition: Vec<(u8, u64)>) -> PlannedComposition {
        PlannedComposition {
            source_row_index: index,
            candidate_id: id.into(),
            composition: CompositionFingerprint(composition),
            disposition: ExposureDisposition::ExcludedExposedTraining {
                symthaea_training_labels: vec!["fixture".into()],
            },
        }
    }

    #[test]
    fn fingerprint_conversion_normalizes_quantized_fractions() {
        let row = retained(0, "fixture", vec![(31, 333_333_333), (33, 666_666_667)]);
        let composition = composition_from_fingerprint(&row).unwrap();
        let sum: f64 = composition.iter().map(|(_, fraction)| fraction).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(composition[0].0, 31);
        assert_eq!(composition[1].0, 33);
    }

    #[test]
    fn target_distance_uses_only_retained_compositions() {
        let rows = vec![
            retained(0, "si", vec![(14, 1_000_000_000)]),
            excluded(1, "diamond", vec![(6, 1_000_000_000)]),
            retained(2, "ge", vec![(32, 1_000_000_000)]),
        ];
        let target = BandgapTarget::new(1.0, 1.2).unwrap();
        let ranked = rank_retained_rows(
            rows.iter(),
            target,
            &BaselineScreeningPolicy::TargetDistance,
        )
        .unwrap();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].candidate_id, "si");
        assert!(ranked.iter().all(|record| record.candidate_id != "diamond"));
    }

    #[test]
    fn both_policies_preserve_identical_prediction_surface() {
        let rows = vec![
            retained(0, "si", vec![(14, 1_000_000_000)]),
            retained(1, "ge", vec![(32, 1_000_000_000)]),
            retained(2, "diamond", vec![(6, 1_000_000_000)]),
        ];
        let target = BandgapTarget::new(1.0, 2.0).unwrap();
        let target_ranked = rank_retained_rows(
            rows.iter(),
            target,
            &BaselineScreeningPolicy::TargetDistance,
        )
        .unwrap();
        let random_ranked = rank_retained_rows(
            rows.iter(),
            target,
            &BaselineScreeningPolicy::DeterministicRandomOrder { seed: 7 },
        )
        .unwrap();

        let target_predictions: BTreeMap<_, _> = target_ranked
            .iter()
            .map(|record| (record.candidate_id.clone(), record.predicted_gap_ev.to_bits()))
            .collect();
        let random_predictions: BTreeMap<_, _> = random_ranked
            .iter()
            .map(|record| (record.candidate_id.clone(), record.predicted_gap_ev.to_bits()))
            .collect();
        assert_eq!(target_predictions, random_predictions);
    }

    #[test]
    fn random_order_is_deterministic_and_seed_is_method_identity() {
        let rows = vec![
            retained(0, "a", vec![(14, 1_000_000_000)]),
            retained(1, "b", vec![(32, 1_000_000_000)]),
            retained(2, "c", vec![(6, 1_000_000_000)]),
            retained(3, "d", vec![(31, 500_000_000), (33, 500_000_000)]),
        ];
        let target = BandgapTarget::new(1.0, 2.0).unwrap();
        let policy = BaselineScreeningPolicy::DeterministicRandomOrder { seed: 42 };
        let first = rank_retained_rows(rows.iter(), target, &policy).unwrap();
        let second = rank_retained_rows(rows.iter(), target, &policy).unwrap();
        assert_eq!(
            first.iter().map(|r| &r.candidate_id).collect::<Vec<_>>(),
            second.iter().map(|r| &r.candidate_id).collect::<Vec<_>>()
        );
        assert!(policy.method_id().ends_with("seed-000000000000002a"));
        assert_ne!(
            policy.method_id(),
            BaselineScreeningPolicy::DeterministicRandomOrder { seed: 43 }.method_id()
        );
    }

    #[test]
    fn screening_subject_does_not_include_truth_sensitive_source_plan_digest() {
        let policy = BaselineScreeningPolicy::TargetDistance;
        let target = BandgapTarget::new(1.0, 2.0).unwrap();
        let first = screening_subject_digest(
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &policy,
            target,
        )
        .unwrap();
        let second = screening_subject_digest(
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &policy,
            target,
        )
        .unwrap();
        assert_eq!(first, second);
    }
}
