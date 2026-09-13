// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic random-order null ensemble for the Matbench Benchmark Zero ladder.
//!
//! The ensemble is frozen before truth using the random seed already committed
//! by the V0 comparison freeze. Post-truth measurement consumes only compact
//! top-k commitments; it never regenerates or reranks null policies.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use symthaea_energy_benchmark_zero::{BandgapTarget, BenchmarkError, ScreeningRun};
use symthaea_energy_benchmark_zero_exact::{candidate_universe_sha256, ExactUniverseError};
use symthaea_matbench_gap_baseline_screening::{
    screen_baseline, BaselineScreeningPolicy, ScreeningError,
};
use symthaea_matbench_gap_comparison_freeze::{
    ComparisonFreezeError, ComparisonFreezeReceipt,
};
use symthaea_matbench_gap_exposure_plan::{ExposedTrainingExclusionPlan, PlanError};
use symthaea_matbench_gap_frozen_measurement::{
    verify_frozen_comparison_measurement, FrozenComparisonMeasurementReceipt,
    FrozenMeasurementError, PolicyEndpointValues, PolicyRole, RestrictedTruthReceipt,
};
use thiserror::Error;

pub const NULL_REPLICATES: usize = 256;
pub const FREEZE_SCHEMA: &str = "symthaea.matbench-gap.null-ensemble-freeze.v1";
pub const MEASUREMENT_SCHEMA: &str = "symthaea.matbench-gap.null-ensemble-measurement.v1";
pub const FREEZE_CAPABILITY: &str =
    "PRE-TRUTH RANDOM-ORDER NULL ENSEMBLE FREEZE ONLY -- not benchmark truth, statistical significance, or promotion authority.";
pub const MEASUREMENT_CAPABILITY: &str =
    "POST-FREEZE NULL-REFERENCE MEASUREMENT ONLY -- empirical finite-ensemble tail fractions, not a generic significance certificate.";
pub const SEED_DERIVATION: &str =
    "sha256-domain(master_seed_le_u64 || replicate_index_le_u64), first-8-bytes little-endian u64";
pub const MAE_DISCLOSURE: &str =
    "Mean absolute prediction error is not permutation-referenced here because every random-order null replicate preserves the exact legacy baseline candidate-to-prediction surface. The null ensemble tests ordering-dependent selection outcomes only.";

const SEED_DERIVATION_DOMAIN: &[u8] = b"symthaea.matbench-gap.null-seed.v1\0";
const TOP_K_SELECTION_DOMAIN: &[u8] = b"symthaea.matbench-gap.null-top-k-selection.v1\0";
const ENSEMBLE_SUBJECT_DOMAIN: &[u8] = b"symthaea.matbench-gap.null-ensemble-subject.v1\0";
const ENSEMBLE_RECEIPT_DOMAIN: &[u8] = b"symthaea.matbench-gap.null-ensemble-receipt.v1\0";
const MEASUREMENT_SUBJECT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.null-ensemble-measurement.v1\0";
const MEASUREMENT_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.null-ensemble-measurement-receipt.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NullReplicateCommitment {
    pub replicate_index: usize,
    pub seed: u64,
    pub screening_subject_sha256: String,
    /// Benchmark Zero BLAKE3 digest of the complete ordered ranking.
    pub ranking_digest: String,
    pub top_k_candidate_ids: Vec<String>,
    pub top_k_selection_sha256: String,
}

impl NullReplicateCommitment {
    fn validate(&self, master_seed: u64, top_k: usize) -> Result<(), NullEnsembleError> {
        if self.replicate_index >= NULL_REPLICATES {
            return Err(NullEnsembleError::InvalidFreeze(
                "replicate index exceeds fixed ensemble size".into(),
            ));
        }
        if self.seed != derive_seed(master_seed, self.replicate_index) {
            return Err(NullEnsembleError::InvalidFreeze(
                "replicate seed differs from deterministic derivation".into(),
            ));
        }
        validate_256_bit_hex_digest(
            &self.screening_subject_sha256,
            "screening subject digest",
            ErrorSurface::Freeze,
        )?;
        validate_256_bit_hex_digest(&self.ranking_digest, "ranking digest", ErrorSurface::Freeze)?;
        validate_256_bit_hex_digest(
            &self.top_k_selection_sha256,
            "top-k selection digest",
            ErrorSurface::Freeze,
        )?;
        if top_k == 0 || self.top_k_candidate_ids.len() != top_k {
            return Err(NullEnsembleError::InvalidFreeze(
                "replicate shortlist length differs from frozen top_k".into(),
            ));
        }
        let unique: BTreeSet<&str> = self
            .top_k_candidate_ids
            .iter()
            .map(String::as_str)
            .collect();
        if unique.len() != top_k || unique.iter().any(|id| id.trim().is_empty()) {
            return Err(NullEnsembleError::InvalidFreeze(
                "replicate shortlist requires unique non-empty candidate ids".into(),
            ));
        }
        if self.top_k_selection_sha256 != top_k_selection_sha256(&self.top_k_candidate_ids)? {
            return Err(NullEnsembleError::InvalidFreeze(
                "replicate top-k selection digest does not match candidate order".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NullEnsembleFreezeReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub source_plan_sha256: String,
    pub comparison_freeze_sha256: String,
    pub comparison_subject_sha256: String,
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    pub target: BandgapTarget,
    pub top_k: usize,
    /// Exactly `ComparisonFreezeReceipt.random_seed`; not a new choice.
    pub master_seed: u64,
    pub seed_derivation: String,
    pub replicate_count: usize,
    pub legacy_prediction_surface_sha256: String,
    pub replicates: Vec<NullReplicateCommitment>,
    pub ensemble_subject_sha256: String,
}

impl NullEnsembleFreezeReceipt {
    pub fn validate(&self) -> Result<(), NullEnsembleError> {
        if self.schema != FREEZE_SCHEMA || self.capability_classification != FREEZE_CAPABILITY {
            return Err(NullEnsembleError::InvalidFreeze(
                "schema or capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("source plan digest", self.source_plan_sha256.as_str()),
            ("comparison freeze digest", self.comparison_freeze_sha256.as_str()),
            ("comparison subject digest", self.comparison_subject_sha256.as_str()),
            ("candidate universe digest", self.candidate_universe_sha256.as_str()),
            (
                "legacy prediction-surface digest",
                self.legacy_prediction_surface_sha256.as_str(),
            ),
            ("ensemble subject digest", self.ensemble_subject_sha256.as_str()),
        ] {
            validate_256_bit_hex_digest(value, name, ErrorSurface::Freeze)?;
        }
        self.target.validate()?;
        if self.candidate_count == 0 || self.top_k == 0 || self.top_k > self.candidate_count {
            return Err(NullEnsembleError::InvalidFreeze(
                "top_k must be in 1..=candidate_count".into(),
            ));
        }
        if self.seed_derivation != SEED_DERIVATION
            || self.replicate_count != NULL_REPLICATES
            || self.replicates.len() != NULL_REPLICATES
        {
            return Err(NullEnsembleError::InvalidFreeze(
                "null ensemble derivation/count differs from fixed V1 contract".into(),
            ));
        }

        let mut seeds = BTreeSet::new();
        let mut screening_subjects = BTreeSet::new();
        for (expected_index, replicate) in self.replicates.iter().enumerate() {
            if replicate.replicate_index != expected_index {
                return Err(NullEnsembleError::InvalidFreeze(
                    "replicates must be stored in exact canonical index order".into(),
                ));
            }
            replicate.validate(self.master_seed, self.top_k)?;
            if !seeds.insert(replicate.seed) {
                return Err(NullEnsembleError::SeedCollision(replicate.seed));
            }
            if !screening_subjects.insert(replicate.screening_subject_sha256.as_str()) {
                return Err(NullEnsembleError::InvalidFreeze(
                    "two null replicates share the same screening-subject digest".into(),
                ));
            }
        }
        if seeds.len() != NULL_REPLICATES {
            return Err(NullEnsembleError::InvalidFreeze(
                "null ensemble does not contain the fixed number of unique seeds".into(),
            ));
        }
        if ensemble_subject_sha256(self)? != self.ensemble_subject_sha256 {
            return Err(NullEnsembleError::InvalidFreeze(
                "ensemble subject digest does not match frozen commitments".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, NullEnsembleError> {
        self.validate()?;
        domain_separated_sha256(ENSEMBLE_RECEIPT_DOMAIN, self)
    }
}

pub fn freeze_null_ensemble(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
) -> Result<NullEnsembleFreezeReceipt, NullEnsembleError> {
    plan.validate_against_current_training_snapshot()?;
    freeze.validate()?;
    if freeze.source_plan_sha256 != plan.sha256()?
        || freeze.source_composition_order_sha256 != plan.source_composition_order_sha256
        || freeze.source_partition_sha256 != plan.partition_sha256
        || freeze.symthaea_training_snapshot_sha256 != plan.symthaea_training_snapshot_sha256
        || freeze.retained_universe_sha256 != plan.retained_universe_sha256
        || freeze.candidate_count != plan.retained_composition_count
    {
        return Err(NullEnsembleError::FreezePlanMismatch);
    }
    if candidate_universe_sha256(plan.retained_candidate_ids())?
        != freeze.candidate_universe_sha256
    {
        return Err(NullEnsembleError::FreezePlanMismatch);
    }

    let legacy_surface = prediction_surface(&freeze.legacy_target_distance.run)?;
    if prediction_surface(&freeze.random_control.run)? != legacy_surface {
        return Err(NullEnsembleError::LegacyPredictionSurfaceMismatch);
    }

    let mut commitments = Vec::with_capacity(NULL_REPLICATES);
    let mut seen_seeds = BTreeSet::new();
    for replicate_index in 0..NULL_REPLICATES {
        let seed = derive_seed(freeze.random_seed, replicate_index);
        if !seen_seeds.insert(seed) {
            return Err(NullEnsembleError::SeedCollision(seed));
        }
        let receipt = screen_baseline(
            plan,
            freeze.target,
            BaselineScreeningPolicy::DeterministicRandomOrder { seed },
        )?;
        if receipt.ranked_candidate_count != freeze.candidate_count
            || candidate_universe_sha256(
                receipt.run.ranked.iter().map(|record| record.candidate_id.as_str()),
            )? != freeze.candidate_universe_sha256
            || prediction_surface(&receipt.run)? != legacy_surface
        {
            return Err(NullEnsembleError::NullReplicateMismatch { replicate_index });
        }
        let top_k_candidate_ids: Vec<String> = receipt.run.ranked[..freeze.top_k]
            .iter()
            .map(|record| record.candidate_id.clone())
            .collect();
        commitments.push(NullReplicateCommitment {
            replicate_index,
            seed,
            screening_subject_sha256: receipt.screening_subject_sha256,
            ranking_digest: receipt.ranking_digest,
            top_k_selection_sha256: top_k_selection_sha256(&top_k_candidate_ids)?,
            top_k_candidate_ids,
        });
    }

    let mut receipt = NullEnsembleFreezeReceipt {
        schema: FREEZE_SCHEMA.into(),
        capability_classification: FREEZE_CAPABILITY.into(),
        source_plan_sha256: freeze.source_plan_sha256.clone(),
        comparison_freeze_sha256: freeze.sha256()?,
        comparison_subject_sha256: freeze.comparison_subject_sha256.clone(),
        candidate_universe_sha256: freeze.candidate_universe_sha256.clone(),
        candidate_count: freeze.candidate_count,
        target: freeze.target,
        top_k: freeze.top_k,
        master_seed: freeze.random_seed,
        seed_derivation: SEED_DERIVATION.into(),
        replicate_count: NULL_REPLICATES,
        legacy_prediction_surface_sha256: freeze.legacy_prediction_surface_sha256.clone(),
        replicates: commitments,
        ensemble_subject_sha256: String::new(),
    };
    receipt.ensemble_subject_sha256 = ensemble_subject_sha256(&receipt)?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_null_ensemble_freeze(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
    expected: &NullEnsembleFreezeReceipt,
) -> Result<(), NullEnsembleError> {
    expected.validate()?;
    let observed = freeze_null_ensemble(plan, freeze)?;
    if &observed != expected {
        return Err(NullEnsembleError::FreezeReplayMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NullReplicateMeasurement {
    pub replicate_index: usize,
    pub seed: u64,
    pub top_k_hits: usize,
    pub target_regret_ev: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddOneTailFraction {
    pub extreme_plus_one: usize,
    pub denominator: usize,
    pub value: f64,
}

impl AddOneTailFraction {
    fn new(extreme_count: usize) -> Self {
        let extreme_plus_one = extreme_count + 1;
        let denominator = NULL_REPLICATES + 1;
        Self {
            extreme_plus_one,
            denominator,
            value: extreme_plus_one as f64 / denominator as f64,
        }
    }

    fn validate(&self) -> Result<(), NullEnsembleError> {
        if self.denominator != NULL_REPLICATES + 1
            || self.extreme_plus_one == 0
            || self.extreme_plus_one > self.denominator
        {
            return Err(NullEnsembleError::InvalidMeasurement(
                "add-one tail fraction numerator/denominator is invalid".into(),
            ));
        }
        let expected = self.extreme_plus_one as f64 / self.denominator as f64;
        if self.value.to_bits() != expected.to_bits() {
            return Err(NullEnsembleError::InvalidMeasurement(
                "add-one tail fraction value differs from exact numerator/denominator".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyNullReference {
    pub role: PolicyRole,
    pub observed_top_k_hits: usize,
    /// Upper tail: null hits >= observed hits.
    pub null_hits_at_least_observed: usize,
    pub top_k_hits_add_one_tail_fraction: AddOneTailFraction,
    pub observed_target_regret_ev: f64,
    /// Lower tail: null regret <= observed regret.
    pub null_regret_at_most_observed: usize,
    pub target_regret_add_one_tail_fraction: AddOneTailFraction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NullEnsembleMeasurementReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub ensemble_freeze_sha256: String,
    pub ensemble_subject_sha256: String,
    pub comparison_freeze_sha256: String,
    pub comparison_measurement_sha256: String,
    pub restricted_truth_receipt_sha256: String,
    pub restricted_truth_sha256: String,
    pub candidate_universe_sha256: String,
    pub candidate_count: usize,
    pub target: BandgapTarget,
    pub top_k: usize,
    pub replicate_count: usize,
    pub replicate_metrics: Vec<NullReplicateMeasurement>,
    pub policy_references: Vec<PolicyNullReference>,
    pub mae_disclosure: String,
    pub measurement_subject_sha256: String,
}

impl NullEnsembleMeasurementReceipt {
    pub fn validate(&self) -> Result<(), NullEnsembleError> {
        if self.schema != MEASUREMENT_SCHEMA
            || self.capability_classification != MEASUREMENT_CAPABILITY
        {
            return Err(NullEnsembleError::InvalidMeasurement(
                "schema or capability classification was altered".into(),
            ));
        }
        for (name, value) in [
            ("ensemble freeze digest", self.ensemble_freeze_sha256.as_str()),
            ("ensemble subject digest", self.ensemble_subject_sha256.as_str()),
            ("comparison freeze digest", self.comparison_freeze_sha256.as_str()),
            (
                "comparison measurement digest",
                self.comparison_measurement_sha256.as_str(),
            ),
            (
                "restricted truth receipt digest",
                self.restricted_truth_receipt_sha256.as_str(),
            ),
            ("restricted truth digest", self.restricted_truth_sha256.as_str()),
            ("candidate universe digest", self.candidate_universe_sha256.as_str()),
            ("measurement subject digest", self.measurement_subject_sha256.as_str()),
        ] {
            validate_256_bit_hex_digest(value, name, ErrorSurface::Measurement)?;
        }
        self.target.validate()?;
        if self.candidate_count == 0 || self.top_k == 0 || self.top_k > self.candidate_count {
            return Err(NullEnsembleError::InvalidMeasurement(
                "top_k must be in 1..=candidate_count".into(),
            ));
        }
        if self.replicate_count != NULL_REPLICATES
            || self.replicate_metrics.len() != NULL_REPLICATES
        {
            return Err(NullEnsembleError::InvalidMeasurement(
                "measurement does not contain the fixed null replicate count".into(),
            ));
        }
        let mut metric_seeds = BTreeSet::new();
        for (index, metrics) in self.replicate_metrics.iter().enumerate() {
            if metrics.replicate_index != index
                || metrics.top_k_hits > self.top_k
                || !metrics.target_regret_ev.is_finite()
                || metrics.target_regret_ev < 0.0
                || !metric_seeds.insert(metrics.seed)
            {
                return Err(NullEnsembleError::InvalidMeasurement(
                    "null replicate metrics are malformed, duplicated, or out of canonical order"
                        .into(),
                ));
            }
        }
        if self.policy_references.len() != 2
            || self.policy_references[0].role != PolicyRole::LegacyTargetDistance
            || self.policy_references[1].role != PolicyRole::CompositionOnlyLearned
        {
            return Err(NullEnsembleError::InvalidMeasurement(
                "policy null references must contain legacy then learned exactly once".into(),
            ));
        }
        for reference in &self.policy_references {
            if reference.observed_top_k_hits > self.top_k
                || !reference.observed_target_regret_ev.is_finite()
                || reference.observed_target_regret_ev < 0.0
            {
                return Err(NullEnsembleError::InvalidMeasurement(
                    "policy null-reference values are invalid".into(),
                ));
            }
            let expected_hits_extreme = self
                .replicate_metrics
                .iter()
                .filter(|null| null.top_k_hits >= reference.observed_top_k_hits)
                .count();
            let expected_regret_extreme = self
                .replicate_metrics
                .iter()
                .filter(|null| null.target_regret_ev <= reference.observed_target_regret_ev)
                .count();
            if reference.null_hits_at_least_observed != expected_hits_extreme
                || reference.null_regret_at_most_observed != expected_regret_extreme
            {
                return Err(NullEnsembleError::InvalidMeasurement(
                    "policy null-reference extreme counts differ from stored replicate metrics"
                        .into(),
                ));
            }
            reference.top_k_hits_add_one_tail_fraction.validate()?;
            reference.target_regret_add_one_tail_fraction.validate()?;
            if reference.top_k_hits_add_one_tail_fraction.extreme_plus_one
                != expected_hits_extreme + 1
                || reference.target_regret_add_one_tail_fraction.extreme_plus_one
                    != expected_regret_extreme + 1
            {
                return Err(NullEnsembleError::InvalidMeasurement(
                    "tail fractions do not bind their recomputed extreme counts".into(),
                ));
            }
        }
        if self.mae_disclosure != MAE_DISCLOSURE {
            return Err(NullEnsembleError::InvalidMeasurement(
                "MAE null-reference disclosure was altered".into(),
            ));
        }
        if null_measurement_subject_sha256(self)? != self.measurement_subject_sha256 {
            return Err(NullEnsembleError::InvalidMeasurement(
                "null measurement subject digest does not match receipt".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, NullEnsembleError> {
        self.validate()?;
        domain_separated_sha256(MEASUREMENT_RECEIPT_DOMAIN, self)
    }
}

/// Measure only pre-frozen null top-k selections against restricted truth.
/// The observed three-policy comparison is independently replayed from the
/// supplied frozen rankings and restricted truth before its endpoints are used.
pub fn measure_null_ensemble(
    freeze: &ComparisonFreezeReceipt,
    ensemble: &NullEnsembleFreezeReceipt,
    restricted: &RestrictedTruthReceipt,
    comparison_measurement: &FrozenComparisonMeasurementReceipt,
) -> Result<NullEnsembleMeasurementReceipt, NullEnsembleError> {
    freeze.validate()?;
    ensemble.validate()?;
    restricted.validate()?;
    comparison_measurement.validate()?;
    verify_frozen_comparison_measurement(freeze, restricted, comparison_measurement)?;

    let freeze_sha256 = freeze.sha256()?;
    if ensemble.comparison_freeze_sha256 != freeze_sha256
        || ensemble.comparison_subject_sha256 != freeze.comparison_subject_sha256
        || ensemble.master_seed != freeze.random_seed
        || ensemble.target != freeze.target
        || ensemble.top_k != freeze.top_k
        || ensemble.candidate_universe_sha256 != freeze.candidate_universe_sha256
        || ensemble.candidate_count != freeze.candidate_count
        || ensemble.legacy_prediction_surface_sha256 != freeze.legacy_prediction_surface_sha256
        || ensemble.comparison_subject_sha256 != restricted.comparison_subject_sha256
        || ensemble.comparison_subject_sha256 != comparison_measurement.comparison_subject_sha256
        || ensemble.source_plan_sha256 != restricted.source_plan_sha256
        || ensemble.source_plan_sha256 != comparison_measurement.source_plan_sha256
        || ensemble.candidate_universe_sha256 != restricted.candidate_universe_sha256
        || ensemble.candidate_universe_sha256 != comparison_measurement.candidate_universe_sha256
        || ensemble.candidate_count != restricted.candidate_count
        || ensemble.candidate_count != comparison_measurement.candidate_count
        || comparison_measurement.comparison_freeze_sha256 != freeze_sha256
        || comparison_measurement.restricted_truth_receipt_sha256 != restricted.sha256()?
        || comparison_measurement.restricted_truth_sha256 != restricted.restricted_truth_sha256
    {
        return Err(NullEnsembleError::MeasurementCrossWire);
    }

    let global_best_target_error_ev = restricted
        .truth
        .experimental_gap_ev
        .values()
        .map(|&gap| ensemble.target.midpoint_error(gap))
        .fold(f64::INFINITY, f64::min);
    if !global_best_target_error_ev.is_finite() {
        return Err(NullEnsembleError::InvalidMeasurement(
            "restricted truth produced no finite global target error".into(),
        ));
    }

    let mut replicate_metrics = Vec::with_capacity(NULL_REPLICATES);
    for replicate in &ensemble.replicates {
        let mut hits = 0usize;
        let mut best_selected = f64::INFINITY;
        for candidate_id in &replicate.top_k_candidate_ids {
            let observed = *restricted
                .truth
                .experimental_gap_ev
                .get(candidate_id)
                .ok_or_else(|| NullEnsembleError::MissingRestrictedTruth(candidate_id.clone()))?;
            if ensemble.target.qualifies(observed) {
                hits += 1;
            }
            best_selected = best_selected.min(ensemble.target.midpoint_error(observed));
        }
        if !best_selected.is_finite() {
            return Err(NullEnsembleError::InvalidMeasurement(
                "null shortlist produced no finite selected target error".into(),
            ));
        }
        replicate_metrics.push(NullReplicateMeasurement {
            replicate_index: replicate.replicate_index,
            seed: replicate.seed,
            top_k_hits: hits,
            target_regret_ev: (best_selected - global_best_target_error_ev).max(0.0),
        });
    }

    let legacy = find_observed_endpoint(
        &comparison_measurement.endpoint_values,
        PolicyRole::LegacyTargetDistance,
    )?;
    let learned = find_observed_endpoint(
        &comparison_measurement.endpoint_values,
        PolicyRole::CompositionOnlyLearned,
    )?;
    let policy_references = vec![
        policy_null_reference(legacy, &replicate_metrics),
        policy_null_reference(learned, &replicate_metrics),
    ];

    let mut receipt = NullEnsembleMeasurementReceipt {
        schema: MEASUREMENT_SCHEMA.into(),
        capability_classification: MEASUREMENT_CAPABILITY.into(),
        ensemble_freeze_sha256: ensemble.sha256()?,
        ensemble_subject_sha256: ensemble.ensemble_subject_sha256.clone(),
        comparison_freeze_sha256: freeze_sha256,
        comparison_measurement_sha256: comparison_measurement.sha256()?,
        restricted_truth_receipt_sha256: restricted.sha256()?,
        restricted_truth_sha256: restricted.restricted_truth_sha256.clone(),
        candidate_universe_sha256: ensemble.candidate_universe_sha256.clone(),
        candidate_count: ensemble.candidate_count,
        target: ensemble.target,
        top_k: ensemble.top_k,
        replicate_count: NULL_REPLICATES,
        replicate_metrics,
        policy_references,
        mae_disclosure: MAE_DISCLOSURE.into(),
        measurement_subject_sha256: String::new(),
    };
    receipt.measurement_subject_sha256 = null_measurement_subject_sha256(&receipt)?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn verify_null_ensemble_measurement(
    freeze: &ComparisonFreezeReceipt,
    ensemble: &NullEnsembleFreezeReceipt,
    restricted: &RestrictedTruthReceipt,
    comparison_measurement: &FrozenComparisonMeasurementReceipt,
    expected: &NullEnsembleMeasurementReceipt,
) -> Result<(), NullEnsembleError> {
    expected.validate()?;
    let observed = measure_null_ensemble(freeze, ensemble, restricted, comparison_measurement)?;
    if &observed != expected {
        return Err(NullEnsembleError::MeasurementReplayMismatch);
    }
    Ok(())
}

fn find_observed_endpoint(
    values: &[PolicyEndpointValues],
    role: PolicyRole,
) -> Result<&PolicyEndpointValues, NullEnsembleError> {
    let mut matches = values.iter().filter(|value| value.role == role);
    let first = matches
        .next()
        .ok_or(NullEnsembleError::MissingObservedPolicy(role))?;
    if matches.next().is_some() {
        return Err(NullEnsembleError::DuplicateObservedPolicy(role));
    }
    Ok(first)
}

fn policy_null_reference(
    observed: &PolicyEndpointValues,
    nulls: &[NullReplicateMeasurement],
) -> PolicyNullReference {
    let hits_extreme = nulls
        .iter()
        .filter(|null| null.top_k_hits >= observed.top_k_hits)
        .count();
    let regret_extreme = nulls
        .iter()
        .filter(|null| null.target_regret_ev <= observed.target_regret_ev)
        .count();
    PolicyNullReference {
        role: observed.role,
        observed_top_k_hits: observed.top_k_hits,
        null_hits_at_least_observed: hits_extreme,
        top_k_hits_add_one_tail_fraction: AddOneTailFraction::new(hits_extreme),
        observed_target_regret_ev: observed.target_regret_ev,
        null_regret_at_most_observed: regret_extreme,
        target_regret_add_one_tail_fraction: AddOneTailFraction::new(regret_extreme),
    }
}

fn derive_seed(master_seed: u64, replicate_index: usize) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(SEED_DERIVATION_DOMAIN);
    hasher.update(master_seed.to_le_bytes());
    hasher.update((replicate_index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn prediction_surface(run: &ScreeningRun) -> Result<BTreeMap<String, u64>, NullEnsembleError> {
    run.validate()?;
    Ok(run
        .ranked
        .iter()
        .map(|record| (record.candidate_id.clone(), record.predicted_gap_ev.to_bits()))
        .collect())
}

fn top_k_selection_sha256(ids: &[String]) -> Result<String, NullEnsembleError> {
    domain_separated_sha256(TOP_K_SELECTION_DOMAIN, ids)
}

#[derive(Serialize)]
struct EnsembleSubject<'a> {
    source_plan_sha256: &'a str,
    comparison_freeze_sha256: &'a str,
    comparison_subject_sha256: &'a str,
    candidate_universe_sha256: &'a str,
    candidate_count: usize,
    target: BandgapTarget,
    top_k: usize,
    master_seed: u64,
    seed_derivation: &'a str,
    replicate_count: usize,
    legacy_prediction_surface_sha256: &'a str,
    replicates: &'a [NullReplicateCommitment],
}

fn ensemble_subject_sha256(
    receipt: &NullEnsembleFreezeReceipt,
) -> Result<String, NullEnsembleError> {
    domain_separated_sha256(
        ENSEMBLE_SUBJECT_DOMAIN,
        &EnsembleSubject {
            source_plan_sha256: &receipt.source_plan_sha256,
            comparison_freeze_sha256: &receipt.comparison_freeze_sha256,
            comparison_subject_sha256: &receipt.comparison_subject_sha256,
            candidate_universe_sha256: &receipt.candidate_universe_sha256,
            candidate_count: receipt.candidate_count,
            target: receipt.target,
            top_k: receipt.top_k,
            master_seed: receipt.master_seed,
            seed_derivation: &receipt.seed_derivation,
            replicate_count: receipt.replicate_count,
            legacy_prediction_surface_sha256: &receipt.legacy_prediction_surface_sha256,
            replicates: &receipt.replicates,
        },
    )
}

#[derive(Serialize)]
struct NullMeasurementSubject<'a> {
    ensemble_freeze_sha256: &'a str,
    ensemble_subject_sha256: &'a str,
    comparison_freeze_sha256: &'a str,
    comparison_measurement_sha256: &'a str,
    restricted_truth_receipt_sha256: &'a str,
    restricted_truth_sha256: &'a str,
    candidate_universe_sha256: &'a str,
    candidate_count: usize,
    target: BandgapTarget,
    top_k: usize,
    replicate_count: usize,
    replicate_metrics: &'a [NullReplicateMeasurement],
    policy_references: &'a [PolicyNullReference],
    mae_disclosure: &'a str,
}

fn null_measurement_subject_sha256(
    receipt: &NullEnsembleMeasurementReceipt,
) -> Result<String, NullEnsembleError> {
    domain_separated_sha256(
        MEASUREMENT_SUBJECT_DOMAIN,
        &NullMeasurementSubject {
            ensemble_freeze_sha256: &receipt.ensemble_freeze_sha256,
            ensemble_subject_sha256: &receipt.ensemble_subject_sha256,
            comparison_freeze_sha256: &receipt.comparison_freeze_sha256,
            comparison_measurement_sha256: &receipt.comparison_measurement_sha256,
            restricted_truth_receipt_sha256: &receipt.restricted_truth_receipt_sha256,
            restricted_truth_sha256: &receipt.restricted_truth_sha256,
            candidate_universe_sha256: &receipt.candidate_universe_sha256,
            candidate_count: receipt.candidate_count,
            target: receipt.target,
            top_k: receipt.top_k,
            replicate_count: receipt.replicate_count,
            replicate_metrics: &receipt.replicate_metrics,
            policy_references: &receipt.policy_references,
            mae_disclosure: &receipt.mae_disclosure,
        },
    )
}

#[derive(Clone, Copy)]
enum ErrorSurface {
    Freeze,
    Measurement,
}

fn validate_256_bit_hex_digest(
    value: &str,
    name: &str,
    surface: ErrorSurface,
) -> Result<(), NullEnsembleError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        let message = format!("{name} must be exactly 64 hexadecimal characters");
        return Err(match surface {
            ErrorSurface::Freeze => NullEnsembleError::InvalidFreeze(message),
            ErrorSurface::Measurement => NullEnsembleError::InvalidMeasurement(message),
        });
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, NullEnsembleError> {
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
pub enum NullEnsembleError {
    #[error("exposure plan rejected: {0}")]
    Plan(#[from] PlanError),
    #[error("comparison freeze rejected: {0}")]
    Freeze(#[from] ComparisonFreezeError),
    #[error("baseline screening rejected: {0}")]
    Screening(#[from] ScreeningError),
    #[error("Benchmark Zero rejected input: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("exact-universe contract rejected: {0}")]
    ExactUniverse(#[from] ExactUniverseError),
    #[error("frozen measurement contract rejected: {0}")]
    FrozenMeasurement(#[from] FrozenMeasurementError),
    #[error("null ensemble does not bind the supplied plan/comparison freeze")]
    FreezePlanMismatch,
    #[error("legacy prediction surfaces differ")]
    LegacyPredictionSurfaceMismatch,
    #[error("derived null seed collision at seed {0}")]
    SeedCollision(u64),
    #[error("null replicate {replicate_index} differs from frozen universe/prediction surface")]
    NullReplicateMismatch { replicate_index: usize },
    #[error("null ensemble freeze replay differs from supplied receipt")]
    FreezeReplayMismatch,
    #[error("null measurement inputs are cross-wired")]
    MeasurementCrossWire,
    #[error("restricted truth is missing null shortlist candidate {0:?}")]
    MissingRestrictedTruth(String),
    #[error("observed endpoint summary is missing policy {0:?}")]
    MissingObservedPolicy(PolicyRole),
    #[error("observed endpoint summary duplicates policy {0:?}")]
    DuplicateObservedPolicy(PolicyRole),
    #[error("null ensemble measurement replay differs from supplied receipt")]
    MeasurementReplayMismatch,
    #[error("invalid null ensemble freeze: {0}")]
    InvalidFreeze(String),
    #[error("invalid null ensemble measurement: {0}")]
    InvalidMeasurement(String),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_derivation_is_deterministic_and_unique_for_fixed_ensemble() {
        let first: Vec<u64> = (0..NULL_REPLICATES)
            .map(|index| derive_seed(42, index))
            .collect();
        let second: Vec<u64> = (0..NULL_REPLICATES)
            .map(|index| derive_seed(42, index))
            .collect();
        assert_eq!(first, second);
        assert_eq!(
            first.iter().copied().collect::<BTreeSet<_>>().len(),
            NULL_REPLICATES
        );
    }

    #[test]
    fn changing_master_seed_changes_derived_sequence() {
        let first: Vec<u64> = (0..8).map(|index| derive_seed(42, index)).collect();
        let second: Vec<u64> = (0..8).map(|index| derive_seed(43, index)).collect();
        assert_ne!(first, second);
    }

    #[test]
    fn top_k_selection_digest_is_order_sensitive() {
        let first = vec!["A".to_owned(), "B".to_owned()];
        let second = vec!["B".to_owned(), "A".to_owned()];
        assert_ne!(
            top_k_selection_sha256(&first).unwrap(),
            top_k_selection_sha256(&second).unwrap()
        );
    }

    #[test]
    fn add_one_tail_fraction_has_fixed_denominator() {
        let zero_extreme = AddOneTailFraction::new(0);
        zero_extreme.validate().unwrap();
        assert_eq!(zero_extreme.extreme_plus_one, 1);
        assert_eq!(zero_extreme.denominator, NULL_REPLICATES + 1);

        let all_extreme = AddOneTailFraction::new(NULL_REPLICATES);
        all_extreme.validate().unwrap();
        assert_eq!(all_extreme.value.to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn better_direction_tail_counts_are_explicit() {
        let nulls = vec![
            NullReplicateMeasurement {
                replicate_index: 0,
                seed: 1,
                top_k_hits: 3,
                target_regret_ev: 0.4,
            },
            NullReplicateMeasurement {
                replicate_index: 1,
                seed: 2,
                top_k_hits: 5,
                target_regret_ev: 0.2,
            },
            NullReplicateMeasurement {
                replicate_index: 2,
                seed: 3,
                top_k_hits: 7,
                target_regret_ev: 0.1,
            },
        ];
        let observed = PolicyEndpointValues {
            role: PolicyRole::CompositionOnlyLearned,
            target_regret_ev: 0.2,
            top_k_hits: 5,
            mean_abs_prediction_error_ev: 0.5,
        };
        let reference = policy_null_reference(&observed, &nulls);
        assert_eq!(reference.null_hits_at_least_observed, 2);
        assert_eq!(reference.null_regret_at_most_observed, 2);
    }
}
