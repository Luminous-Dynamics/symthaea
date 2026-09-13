// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Truth-blind learned screening over the frozen Matbench composition universe.
//!
//! This adapter joins #2653's exact exposed-training snapshot with #2674's
//! frozen composition-only model recipe, then ranks only retained compositions.
//! It has no `BandgapTruthSet` input.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use symthaea_bandgap::composition_only::{
    CompositionOnlyBandgapPredictor, CompositionOnlyError, CompositionOnlyModelRecipe, MODEL_ID,
    model_recipe,
};
use symthaea_energy_benchmark_zero::{
    BandgapTarget, BenchmarkError, DataSliceId, ScreeningMethodProvenance, ScreeningRecord,
    ScreeningRun,
};
use symthaea_matbench_gap_exposure_plan::{
    ExposedTrainingExclusionPlan, ExposureDisposition, PlanError, PlannedComposition,
};
use thiserror::Error;

pub const RECEIPT_SCHEMA: &str = "symthaea.matbench-gap.learned-screening-receipt.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "TRUTH-BLIND COMPOSITION-ONLY LEARNED SCREENING RECEIPT -- not benchmark truth, not a clean-holdout certificate, and not a scientific result.";
pub const TRAINING_DATASET_ID: &str = "symthaea_bandgap_curated";
pub const PRIOR_KNOWLEDGE_DISCLOSURE: &str =
    "The machine-readable training slice identifies the exact curated Symthaea band-gap table used to fit the composition-only Random Forest. It does not exhaust historical/manual prior knowledge: the residual electronegativity baseline is a legacy empirical/physics-inspired heuristic whose source comments describe fitted/chosen semiconductor parameters, and exact-composition exclusion does not rule out near-neighbor, publication, pretrained-model, or other leakage.";
pub const UNCERTAINTY_DISCLOSURE: &str =
    "uncertainty_ev is Random-Forest inter-tree standard deviation only. It is not a calibrated confidence interval, total prediction error, within-composition polymorph ambiguity, or experimental uncertainty.";

const MODEL_IDENTITY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.composition-only-model-identity.v0\0";
const SCREENING_SUBJECT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.learned-screening-subject.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.learned-screening-receipt.v0\0";

/// Replayable evidence for one truth-blind learned screening run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedScreeningReceipt {
    pub schema: String,
    pub capability_classification: String,
    /// Exact source plan identity retained for provenance. This is deliberately
    /// not part of the truth-free screening subject digest.
    pub source_plan_sha256: String,
    pub source_composition_order_sha256: String,
    pub source_partition_sha256: String,
    pub symthaea_training_snapshot_sha256: String,
    pub retained_universe_sha256: String,
    pub model_recipe: CompositionOnlyModelRecipe,
    pub model_identity_sha256: String,
    pub target: BandgapTarget,
    pub screening_subject_sha256: String,
    /// Benchmark Zero's domain-separated BLAKE3 identity of the ordered run.
    pub ranking_digest: String,
    pub ranked_candidate_count: usize,
    pub run: ScreeningRun,
    pub prior_knowledge_disclosure: String,
    pub uncertainty_disclosure: String,
}

impl LearnedScreeningReceipt {
    pub fn validate(&self) -> Result<(), LearnedScreeningError> {
        if self.schema != RECEIPT_SCHEMA
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(LearnedScreeningError::InvalidReceipt(
                "receipt schema/capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan digest", self.source_plan_sha256.as_str()),
            (
                "source composition-order digest",
                self.source_composition_order_sha256.as_str(),
            ),
            ("source partition digest", self.source_partition_sha256.as_str()),
            (
                "training snapshot digest",
                self.symthaea_training_snapshot_sha256.as_str(),
            ),
            ("retained universe digest", self.retained_universe_sha256.as_str()),
            ("model identity digest", self.model_identity_sha256.as_str()),
            (
                "screening subject digest",
                self.screening_subject_sha256.as_str(),
            ),
            ("ranking digest", self.ranking_digest.as_str()),
        ] {
            validate_256_bit_hex_digest(value, name)?;
        }
        if self.prior_knowledge_disclosure != PRIOR_KNOWLEDGE_DISCLOSURE
            || self.uncertainty_disclosure != UNCERTAINTY_DISCLOSURE
        {
            return Err(LearnedScreeningError::InvalidReceipt(
                "fixed epistemic disclosure was altered".into(),
            ));
        }
        self.target.validate()?;
        self.run.validate()?;
        if self.run.target != self.target {
            return Err(LearnedScreeningError::InvalidReceipt(
                "screening run target differs from receipt target".into(),
            ));
        }

        let canonical_recipe = model_recipe();
        if self.model_recipe != canonical_recipe {
            return Err(LearnedScreeningError::InvalidReceipt(
                "serialized model recipe differs from the canonical composition-only recipe"
                    .into(),
            ));
        }
        let expected_model_identity = model_identity_sha256(
            &self.symthaea_training_snapshot_sha256,
            &self.model_recipe,
        )?;
        if self.model_identity_sha256 != expected_model_identity {
            return Err(LearnedScreeningError::InvalidReceipt(
                "model identity does not bind the recorded training snapshot and recipe".into(),
            ));
        }

        let expected_method = screening_method_provenance(
            &self.symthaea_training_snapshot_sha256,
            &self.model_recipe,
            &self.model_identity_sha256,
        )?;
        if self.run.method != expected_method {
            return Err(LearnedScreeningError::InvalidReceipt(
                "ScreeningMethodProvenance does not bind the learned model identity/training slice"
                    .into(),
            ));
        }
        if self.ranked_candidate_count == 0
            || self.ranked_candidate_count != self.run.ranked.len()
        {
            return Err(LearnedScreeningError::InvalidReceipt(
                "ranked candidate count must equal the non-empty screening run".into(),
            ));
        }
        let expected_ranking = self.run.ranking_digest()?;
        if self.ranking_digest != expected_ranking {
            return Err(LearnedScreeningError::InvalidReceipt(
                "ranking digest does not match the ordered ScreeningRun".into(),
            ));
        }
        let expected_subject = screening_subject_sha256(
            &self.source_composition_order_sha256,
            &self.source_partition_sha256,
            &self.retained_universe_sha256,
            &self.model_identity_sha256,
            self.target,
        )?;
        if self.screening_subject_sha256 != expected_subject {
            return Err(LearnedScreeningError::InvalidReceipt(
                "screening subject does not bind the truth-free universe/model/target".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, LearnedScreeningError> {
        self.validate()?;
        domain_separated_sha256(RECEIPT_DIGEST_DOMAIN, self)
    }
}

/// Rank the exact retained universe with the exact composition-only learned
/// model fitted from the current exposed Symthaea training snapshot.
pub fn screen_learned(
    plan: &ExposedTrainingExclusionPlan,
    target: BandgapTarget,
) -> Result<LearnedScreeningReceipt, LearnedScreeningError> {
    plan.validate_against_current_training_snapshot()?;
    target.validate()?;

    let recipe = model_recipe();
    let model_identity = model_identity_sha256(&plan.symthaea_training_snapshot_sha256, &recipe)?;
    let method = screening_method_provenance(
        &plan.symthaea_training_snapshot_sha256,
        &recipe,
        &model_identity,
    )?;
    let predictor = CompositionOnlyBandgapPredictor::new();
    let ranked = rank_retained_rows(plan.retained_rows(), target, &predictor)?;
    if ranked.is_empty() {
        return Err(LearnedScreeningError::EmptyRetainedUniverse);
    }

    let run = ScreeningRun {
        method,
        target,
        ranked,
    };
    run.validate()?;
    let ranking_digest = run.ranking_digest()?;
    let screening_subject_sha256 = screening_subject_sha256(
        &plan.source_composition_order_sha256,
        &plan.partition_sha256,
        &plan.retained_universe_sha256,
        &model_identity,
        target,
    )?;

    let receipt = LearnedScreeningReceipt {
        schema: RECEIPT_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source_plan_sha256: plan.sha256()?,
        source_composition_order_sha256: plan.source_composition_order_sha256.clone(),
        source_partition_sha256: plan.partition_sha256.clone(),
        symthaea_training_snapshot_sha256: plan.symthaea_training_snapshot_sha256.clone(),
        retained_universe_sha256: plan.retained_universe_sha256.clone(),
        model_recipe: recipe,
        model_identity_sha256: model_identity,
        target,
        screening_subject_sha256,
        ranking_digest,
        ranked_candidate_count: run.ranked.len(),
        run,
        prior_knowledge_disclosure: PRIOR_KNOWLEDGE_DISCLOSURE.into(),
        uncertainty_disclosure: UNCERTAINTY_DISCLOSURE.into(),
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Replay the exact fit + ranking from the frozen plan and require exact receipt
/// equality.
pub fn verify_learned_screening_receipt(
    plan: &ExposedTrainingExclusionPlan,
    receipt: &LearnedScreeningReceipt,
) -> Result<(), LearnedScreeningError> {
    receipt.validate()?;
    plan.validate_against_current_training_snapshot()?;
    if receipt.source_plan_sha256 != plan.sha256()?
        || receipt.source_composition_order_sha256 != plan.source_composition_order_sha256
        || receipt.source_partition_sha256 != plan.partition_sha256
        || receipt.symthaea_training_snapshot_sha256 != plan.symthaea_training_snapshot_sha256
        || receipt.retained_universe_sha256 != plan.retained_universe_sha256
    {
        return Err(LearnedScreeningError::PlanIdentityMismatch);
    }

    let observed = screen_learned(plan, receipt.target)?;
    if &observed != receipt {
        return Err(LearnedScreeningError::ReplayMismatch);
    }
    Ok(())
}

fn rank_retained_rows<'a>(
    rows: impl Iterator<Item = &'a PlannedComposition>,
    target: BandgapTarget,
    predictor: &CompositionOnlyBandgapPredictor,
) -> Result<Vec<ScreeningRecord>, LearnedScreeningError> {
    struct Scored {
        source_row_index: usize,
        candidate_id: String,
        predicted_gap_ev: f64,
        uncertainty_ev: f64,
        target_error_ev: f64,
    }

    let mut scored = Vec::new();
    for row in rows {
        if !matches!(&row.disposition, ExposureDisposition::Retained) {
            continue;
        }
        let composition = fingerprint_as_amounts(row)?;
        let prediction = predictor.predict(&composition)?;
        if !prediction.bandgap.is_finite()
            || prediction.bandgap < 0.0
            || !prediction.uncertainty.is_finite()
            || prediction.uncertainty < 0.0
        {
            return Err(LearnedScreeningError::InvalidPrediction {
                candidate_id: row.candidate_id.clone(),
            });
        }
        scored.push(Scored {
            source_row_index: row.source_row_index,
            candidate_id: row.candidate_id.clone(),
            predicted_gap_ev: prediction.bandgap,
            uncertainty_ev: prediction.uncertainty,
            target_error_ev: target.midpoint_error(prediction.bandgap),
        });
    }

    scored.sort_by(|left, right| {
        left.target_error_ev
            .total_cmp(&right.target_error_ev)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.source_row_index.cmp(&right.source_row_index))
    });

    Ok(scored
        .into_iter()
        .map(|candidate| ScreeningRecord {
            candidate_id: candidate.candidate_id,
            predicted_gap_ev: candidate.predicted_gap_ev,
            uncertainty_ev: Some(candidate.uncertainty_ev),
        })
        .collect())
}

fn fingerprint_as_amounts(
    row: &PlannedComposition,
) -> Result<Vec<(u8, f64)>, LearnedScreeningError> {
    if row.composition.0.is_empty() {
        return Err(LearnedScreeningError::InvalidComposition(
            row.candidate_id.clone(),
        ));
    }
    let mut previous = None;
    let mut composition = Vec::with_capacity(row.composition.0.len());
    for &(atomic_number, amount) in &row.composition.0 {
        if atomic_number == 0
            || amount == 0
            || previous.is_some_and(|prior| prior >= atomic_number)
        {
            return Err(LearnedScreeningError::InvalidComposition(
                row.candidate_id.clone(),
            ));
        }
        previous = Some(atomic_number);
        composition.push((atomic_number, amount as f64));
    }
    Ok(composition)
}

#[derive(Serialize)]
struct ModelIdentityInput<'a> {
    training_snapshot_sha256: &'a str,
    recipe: &'a CompositionOnlyModelRecipe,
}

pub fn model_identity_sha256(
    training_snapshot_sha256: &str,
    recipe: &CompositionOnlyModelRecipe,
) -> Result<String, LearnedScreeningError> {
    validate_256_bit_hex_digest(training_snapshot_sha256, "training snapshot digest")?;
    domain_separated_sha256(
        MODEL_IDENTITY_DIGEST_DOMAIN,
        &ModelIdentityInput {
            training_snapshot_sha256,
            recipe,
        },
    )
}

fn screening_method_provenance(
    training_snapshot_sha256: &str,
    recipe: &CompositionOnlyModelRecipe,
    model_identity_sha256: &str,
) -> Result<ScreeningMethodProvenance, LearnedScreeningError> {
    validate_256_bit_hex_digest(training_snapshot_sha256, "training snapshot digest")?;
    validate_256_bit_hex_digest(model_identity_sha256, "model identity digest")?;
    let training_slice = DataSliceId::new(
        TRAINING_DATASET_ID,
        format!("sha256:{training_snapshot_sha256}"),
    )?;
    let mut training_slices = BTreeSet::new();
    training_slices.insert(training_slice);
    Ok(ScreeningMethodProvenance {
        method_id: format!("{MODEL_ID}@sha256:{model_identity_sha256}"),
        version: recipe.model_version.clone(),
        training_slices,
    })
}

#[derive(Serialize)]
struct ScreeningSubject<'a> {
    source_composition_order_sha256: &'a str,
    source_partition_sha256: &'a str,
    retained_universe_sha256: &'a str,
    model_identity_sha256: &'a str,
    target: BandgapTarget,
}

fn screening_subject_sha256(
    source_composition_order_sha256: &str,
    source_partition_sha256: &str,
    retained_universe_sha256: &str,
    model_identity_sha256: &str,
    target: BandgapTarget,
) -> Result<String, LearnedScreeningError> {
    target.validate()?;
    for (name, value) in [
        ("composition-order digest", source_composition_order_sha256),
        ("partition digest", source_partition_sha256),
        ("retained-universe digest", retained_universe_sha256),
        ("model identity digest", model_identity_sha256),
    ] {
        validate_256_bit_hex_digest(value, name)?;
    }
    domain_separated_sha256(
        SCREENING_SUBJECT_DIGEST_DOMAIN,
        &ScreeningSubject {
            source_composition_order_sha256,
            source_partition_sha256,
            retained_universe_sha256,
            model_identity_sha256,
            target,
        },
    )
}

fn validate_256_bit_hex_digest(
    value: &str,
    name: &str,
) -> Result<(), LearnedScreeningError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LearnedScreeningError::InvalidReceipt(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, LearnedScreeningError> {
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
pub enum LearnedScreeningError {
    #[error("exposure plan rejected: {0}")]
    Plan(#[from] PlanError),
    #[error("composition-only model rejected input: {0}")]
    Model(#[from] CompositionOnlyError),
    #[error("Benchmark Zero contract rejected: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("retained screening universe is empty")]
    EmptyRetainedUniverse,
    #[error("invalid retained composition for {0:?}")]
    InvalidComposition(String),
    #[error("composition-only model emitted invalid prediction/uncertainty for {candidate_id:?}")]
    InvalidPrediction { candidate_id: String },
    #[error("learned screening receipt does not bind the supplied exposure plan")]
    PlanIdentityMismatch,
    #[error("learned screening receipt replay differs from recomputation")]
    ReplayMismatch,
    #[error("invalid learned screening receipt: {0}")]
    InvalidReceipt(String),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn model_identity_binds_training_snapshot_and_recipe() {
        let recipe = model_recipe();
        let first = model_identity_sha256(&"a".repeat(64), &recipe).unwrap();
        let second = model_identity_sha256(&"b".repeat(64), &recipe).unwrap();
        assert_ne!(first, second);

        let mut changed = recipe.clone();
        changed.rng_seed += 1;
        let third = model_identity_sha256(&"a".repeat(64), &changed).unwrap();
        assert_ne!(first, third);
    }

    #[test]
    fn method_provenance_carries_exact_training_slice_and_model_identity() {
        let snapshot = "a".repeat(64);
        let recipe = model_recipe();
        let identity = model_identity_sha256(&snapshot, &recipe).unwrap();
        let method = screening_method_provenance(&snapshot, &recipe, &identity).unwrap();
        assert_eq!(method.training_slices.len(), 1);
        let only = method.training_slices.iter().next().unwrap();
        assert_eq!(only.dataset_id, TRAINING_DATASET_ID);
        assert_eq!(only.split_id, format!("sha256:{snapshot}"));
        assert!(method.method_id.ends_with(&identity));
    }

    #[test]
    fn learned_ranking_uses_only_retained_rows_and_has_uncertainty() {
        let predictor = CompositionOnlyBandgapPredictor::new();
        let rows = vec![
            retained(0, "si", vec![(14, 1_000_000_000)]),
            excluded(1, "diamond", vec![(6, 1_000_000_000)]),
            retained(2, "gaas", vec![(31, 500_000_000), (33, 500_000_000)]),
        ];
        let ranked = rank_retained_rows(
            rows.iter(),
            BandgapTarget::new(1.0, 2.0).unwrap(),
            &predictor,
        )
        .unwrap();
        assert_eq!(ranked.len(), 2);
        assert!(ranked.iter().all(|record| record.candidate_id != "diamond"));
        assert!(ranked.iter().all(|record| record.uncertainty_ev.is_some()));
    }

    #[test]
    fn learned_ranking_is_bitwise_deterministic() {
        let first_model = CompositionOnlyBandgapPredictor::new();
        let second_model = CompositionOnlyBandgapPredictor::new();
        let rows = vec![
            retained(0, "si", vec![(14, 1_000_000_000)]),
            retained(1, "gaas", vec![(31, 500_000_000), (33, 500_000_000)]),
            retained(2, "zno", vec![(30, 500_000_000), (8, 500_000_000)]),
        ];
        let target = BandgapTarget::new(1.0, 2.0).unwrap();
        let first = rank_retained_rows(rows.iter(), target, &first_model).unwrap();
        let second = rank_retained_rows(rows.iter(), target, &second_model).unwrap();
        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.candidate_id, right.candidate_id);
            assert_eq!(left.predicted_gap_ev.to_bits(), right.predicted_gap_ev.to_bits());
            assert_eq!(
                left.uncertainty_ev.unwrap().to_bits(),
                right.uncertainty_ev.unwrap().to_bits()
            );
        }
    }

    #[test]
    fn screening_subject_changes_with_model_identity_but_not_full_plan_identity() {
        let target = BandgapTarget::new(1.0, 2.0).unwrap();
        let first = screening_subject_sha256(
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            target,
        )
        .unwrap();
        let second = screening_subject_sha256(
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"e".repeat(64),
            target,
        )
        .unwrap();
        assert_ne!(first, second);
    }
}
