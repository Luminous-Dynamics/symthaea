// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Matched never-written shadow-distractor control for validity-memory capacity.
//!
//! The score-moment oracle measures a deterministic non-target candidate drawn
//! from the real value vocabulary. That is useful, but a real candidate may have
//! been written elsewhere in the archive, mixing generic finite-dimensional
//! crosstalk with structured semantic-codeword reuse.
//!
//! A second confound matters too: #2813 changes which vocabulary distractor is
//! scored from query to query. A single fixed shadow role would therefore have a
//! different codeword-selection schedule and would not be a clean variance
//! control.
//!
//! This module reconstructs the frozen synthetic archive independently and builds
//! a **matched never-written shadow vocabulary** with the same size as the real
//! candidate vocabulary. Every shadow role is:
//!
//! - generated from a domain-separated deterministic seed namespace;
//! - mechanically checked to differ from every real candidate and every earlier
//!   shadow codeword;
//! - never passed to `write_span`;
//! - fixed before any query score is observed.
//!
//! For each query, the shadow role is selected using the exact deterministic
//! non-target index schedule used by #2813. Thus real-vocabulary and shadow score
//! streams have the same codeword-switching schedule while differing in one key
//! property: shadow codewords never occur in archive writes.
//!
//! No favorable threshold is defined here.

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::TemporalAxis;
use crate::validity_capacity::{ValidityCapacityCase, ValidityCapacityPlan};
use crate::validity_capacity_score_moments::ScoreMomentSummary;
use crate::validity_capacity_theory::{
    ValidityCapacityNullModel, ValidityCapacityTheoryError,
};
use crate::validity_interval_memory::{ValidityIntervalMemory, ValidityMemoryError};
use std::collections::HashSet;
use std::fmt;

const SHADOW_SEED_DOMAIN: u64 = 0x5348_4144_4F57_5632; // "SHADOWV2"
const SHADOW_INDEX_STRIDE: u64 = 0xE703_7ED1_A0B4_28DB;
const SHADOW_RETRY_STRIDE: u64 = 0xA076_1D64_78BD_642F;
const MAX_SHADOW_COLLISION_RETRIES: u64 = 1_024;

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityShadowObservation {
    pub case: ValidityCapacityCase,
    pub seed: u64,
    /// Seeds actually used to construct the never-written shadow vocabulary, in
    /// candidate-index order.
    pub shadow_role_seeds: Vec<u64>,
    /// Total exact-codeword collision retries across the shadow vocabulary.
    pub total_shadow_collision_retries: u64,
    pub total_queries: u64,
    pub shadow_scores: ScoreMomentSummary,
    /// Maximum absolute cosine between any shadow role and any real candidate.
    /// Diagnostic only; no post-result threshold is defined.
    pub max_abs_shadow_candidate_similarity: f64,
    /// Maximum absolute cosine among distinct shadow roles.
    pub max_abs_shadow_shadow_similarity: f64,
    pub null_distractor_noise_variance: f64,
    pub shadow_mean_bias_from_zero: f64,
    pub shadow_variance_ratio_to_null: f64,
    pub shadow_mean_squared_residual: f64,
    pub shadow_mse_ratio_to_null: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityShadowResult {
    pub observations: Vec<ValidityCapacityShadowObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacityShadowError {
    EmptyCases,
    EmptySeeds,
    DuplicateSeed(u64),
    InvalidCase(String),
    Memory(String),
    Theory(String),
    SizeOverflow,
    ShadowRoleCollisionExhausted { index: usize },
}

impl fmt::Display for ValidityCapacityShadowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCases => write!(f, "shadow probe requires at least one capacity case"),
            Self::EmptySeeds => write!(f, "shadow probe requires replicate seeds"),
            Self::DuplicateSeed(seed) => write!(f, "shadow probe contains duplicate seed {seed}"),
            Self::InvalidCase(error) => write!(f, "shadow probe case error: {error}"),
            Self::Memory(error) => write!(f, "shadow probe memory error: {error}"),
            Self::Theory(error) => write!(f, "shadow probe theory error: {error}"),
            Self::SizeOverflow => write!(f, "shadow probe derived size overflowed u64"),
            Self::ShadowRoleCollisionExhausted { index } => write!(
                f,
                "shadow probe exhausted deterministic collision retries for shadow index {index}"
            ),
        }
    }
}

impl std::error::Error for ValidityCapacityShadowError {}

impl From<ValidityMemoryError> for ValidityCapacityShadowError {
    fn from(value: ValidityMemoryError) -> Self {
        Self::Memory(value.to_string())
    }
}

impl From<ValidityCapacityTheoryError> for ValidityCapacityShadowError {
    fn from(value: ValidityCapacityTheoryError) -> Self {
        Self::Theory(value.to_string())
    }
}

pub fn measure_validity_capacity_shadow_distractors(
    plan: &ValidityCapacityPlan,
) -> Result<ValidityCapacityShadowResult, ValidityCapacityShadowError> {
    validate_plan(plan)?;

    let mut observations = Vec::with_capacity(plan.cases.len() * plan.replicate_seeds.len());
    for &case in &plan.cases {
        for &seed in &plan.replicate_seeds {
            observations.push(measure_case(case, seed)?);
        }
    }
    Ok(ValidityCapacityShadowResult { observations })
}

fn measure_case(
    case: ValidityCapacityCase,
    seed: u64,
) -> Result<ValidityCapacityShadowObservation, ValidityCapacityShadowError> {
    let axis = TemporalAxis::new(case.dim, seed ^ 0x5449_4D45)
        .map_err(|error| ValidityCapacityShadowError::Memory(error.to_string()))?;
    let keys = (0..case.key_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(1_000 + index as u64)))
        .collect::<Vec<_>>();
    let values = (0..case.candidate_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(100_000 + index as u64)))
        .collect::<Vec<_>>();
    let (shadow_values, shadow_role_seeds, total_shadow_collision_retries) =
        construct_shadow_vocabulary(case.dim, seed, case.candidate_count, &values)?;

    let mut memory = ValidityIntervalMemory::new(case.dim)?;
    for (key_index, key) in keys.iter().enumerate() {
        let mut start = 0u64;
        let mut span_index = 0u64;
        while start < case.horizon {
            let end = (start + case.span_length).min(case.horizon);
            let value_index = assigned_value(seed, key_index, span_index, case.candidate_count);
            memory.write_span(&axis, key, &values[value_index], start, end)?;
            start = end;
            span_index = span_index
                .checked_add(1)
                .ok_or(ValidityCapacityShadowError::SizeOverflow)?;
        }
    }

    let mut moments = RunningMoments::default();
    let mut total_queries = 0u64;
    for (key_index, key) in keys.iter().enumerate() {
        for checkpoint in 0..case.horizon {
            let span_index = checkpoint / case.span_length;
            let expected = assigned_value(seed, key_index, span_index, case.candidate_count);
            let probe_index = probe_distractor_index(
                seed,
                key_index,
                checkpoint,
                expected,
                case.candidate_count,
            );
            let score = memory.score_candidate(&axis, key, &shadow_values[probe_index], checkpoint)?;
            moments.push(score);
            total_queries = total_queries
                .checked_add(1)
                .ok_or(ValidityCapacityShadowError::SizeOverflow)?;
        }
    }

    let expected_queries = u64::try_from(case.key_count)
        .ok()
        .and_then(|keys| keys.checked_mul(case.horizon))
        .ok_or(ValidityCapacityShadowError::SizeOverflow)?;
    if total_queries != expected_queries {
        return Err(ValidityCapacityShadowError::InvalidCase(format!(
            "query count mismatch: observed {total_queries}, expected {expected_queries}"
        )));
    }

    let shadow_scores = moments.finish();
    let null = ValidityCapacityNullModel::new(
        case.dim,
        case.key_count,
        case.candidate_count,
        case.horizon,
    )?;
    let shadow_mean_squared_residual =
        shadow_scores.variance + shadow_scores.mean * shadow_scores.mean;

    Ok(ValidityCapacityShadowObservation {
        case,
        seed,
        shadow_role_seeds,
        total_shadow_collision_retries,
        total_queries,
        shadow_scores,
        max_abs_shadow_candidate_similarity: max_cross_similarity(&shadow_values, &values),
        max_abs_shadow_shadow_similarity: max_pairwise_similarity(&shadow_values),
        null_distractor_noise_variance: null.distractor_noise_variance,
        shadow_mean_bias_from_zero: shadow_scores.mean,
        shadow_variance_ratio_to_null: ratio_or_infinity(
            shadow_scores.variance,
            null.distractor_noise_variance,
        ),
        shadow_mean_squared_residual,
        shadow_mse_ratio_to_null: ratio_or_infinity(
            shadow_mean_squared_residual,
            null.distractor_noise_variance,
        ),
    })
}

fn construct_shadow_vocabulary(
    dim: usize,
    seed: u64,
    count: usize,
    real_candidates: &[UnitaryRole],
) -> Result<(Vec<UnitaryRole>, Vec<u64>, u64), ValidityCapacityShadowError> {
    let mut roles = Vec::with_capacity(count);
    let mut seeds = Vec::with_capacity(count);
    let mut total_retries = 0u64;

    for index in 0..count {
        let mut accepted = None;
        for retry in 0..=MAX_SHADOW_COLLISION_RETRIES {
            let role_seed = shadow_seed(seed, index, retry);
            let role = UnitaryRole::new(dim, role_seed);
            let collides_real = real_candidates.iter().any(|candidate| candidate == &role);
            let collides_shadow = roles.iter().any(|candidate| candidate == &role);
            if !collides_real && !collides_shadow {
                accepted = Some((role, role_seed, retry));
                break;
            }
        }

        let (role, role_seed, retries) = accepted.ok_or(
            ValidityCapacityShadowError::ShadowRoleCollisionExhausted { index },
        )?;
        total_retries = total_retries
            .checked_add(retries)
            .ok_or(ValidityCapacityShadowError::SizeOverflow)?;
        roles.push(role);
        seeds.push(role_seed);
    }

    Ok((roles, seeds, total_retries))
}

#[inline]
fn shadow_seed(seed: u64, index: usize, retry: u64) -> u64 {
    splitmix64(
        seed ^ SHADOW_SEED_DOMAIN
            ^ (index as u64).wrapping_mul(SHADOW_INDEX_STRIDE)
            ^ retry.wrapping_mul(SHADOW_RETRY_STRIDE),
    )
}

/// Exact query-to-probe index schedule from #2813. Repeating the small pure
/// function here is deliberate: the shadow control must match the vocabulary
/// selection schedule without depending on the score oracle's private helper.
fn probe_distractor_index(
    seed: u64,
    key_index: usize,
    checkpoint: u64,
    expected: usize,
    candidate_count: usize,
) -> usize {
    debug_assert!(candidate_count >= 2);
    let mixed = splitmix64(
        seed ^ (key_index as u64).wrapping_mul(0xA24B_AED4_963E_E407)
            ^ checkpoint.wrapping_mul(0x9FB2_1C65_1E98_DF25),
    );
    let jump = 1 + (mixed as usize % (candidate_count - 1));
    (expected + jump) % candidate_count
}

fn max_cross_similarity(left: &[UnitaryRole], right: &[UnitaryRole]) -> f64 {
    left.iter()
        .flat_map(|left| right.iter().map(move |right| bipolar_cosine(left, right).abs()))
        .fold(0.0_f64, f64::max)
}

fn max_pairwise_similarity(roles: &[UnitaryRole]) -> f64 {
    let mut maximum = 0.0_f64;
    for left in 0..roles.len() {
        for right in (left + 1)..roles.len() {
            maximum = maximum.max(bipolar_cosine(&roles[left], &roles[right]).abs());
        }
    }
    maximum
}

fn bipolar_cosine(left: &UnitaryRole, right: &UnitaryRole) -> f64 {
    debug_assert_eq!(left.dim(), right.dim());
    left.as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(left, right)| (*left as f64) * (*right as f64))
        .sum::<f64>()
        / left.dim() as f64
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn finish(self) -> ScoreMomentSummary {
        ScoreMomentSummary {
            count: self.count,
            mean: self.mean,
            variance: if self.count == 0 {
                0.0
            } else {
                self.m2 / self.count as f64
            },
        }
    }
}

fn assigned_value(seed: u64, key_index: usize, span_index: u64, candidate_count: usize) -> usize {
    let mut value = seed
        ^ (key_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ span_index.wrapping_mul(0xD1B5_4A32_D192_ED03);
    value = splitmix64(value);
    (value as usize) % candidate_count
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn ratio_or_infinity(observed: f64, predicted: f64) -> f64 {
    if predicted == 0.0 {
        if observed == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        observed / predicted
    }
}

fn validate_plan(plan: &ValidityCapacityPlan) -> Result<(), ValidityCapacityShadowError> {
    if plan.cases.is_empty() {
        return Err(ValidityCapacityShadowError::EmptyCases);
    }
    if plan.replicate_seeds.is_empty() {
        return Err(ValidityCapacityShadowError::EmptySeeds);
    }

    let mut seeds = HashSet::new();
    for &seed in &plan.replicate_seeds {
        if !seeds.insert(seed) {
            return Err(ValidityCapacityShadowError::DuplicateSeed(seed));
        }
    }
    for &case in &plan.cases {
        validate_case(case)?;
    }
    Ok(())
}

fn validate_case(case: ValidityCapacityCase) -> Result<(), ValidityCapacityShadowError> {
    if case.dim == 0 {
        return Err(ValidityCapacityShadowError::InvalidCase(
            "dimension must be non-zero".to_string(),
        ));
    }
    if case.key_count == 0 {
        return Err(ValidityCapacityShadowError::InvalidCase(
            "key count must be non-zero".to_string(),
        ));
    }
    if case.candidate_count < 2 {
        return Err(ValidityCapacityShadowError::InvalidCase(
            "candidate count must be at least two".to_string(),
        ));
    }
    if case.horizon == 0 {
        return Err(ValidityCapacityShadowError::InvalidCase(
            "horizon must be non-zero".to_string(),
        ));
    }
    if case.span_length == 0 {
        return Err(ValidityCapacityShadowError::InvalidCase(
            "span length must be non-zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_values(case: ValidityCapacityCase, seed: u64) -> Vec<UnitaryRole> {
        (0..case.candidate_count)
            .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(100_000 + index as u64)))
            .collect()
    }

    #[test]
    fn shadow_vocabulary_is_disjoint_from_real_and_itself() {
        for case in ValidityCapacityPlan::smoke().cases {
            for seed in [30_001_u64, 30_002] {
                let values = candidate_values(case, seed);
                let (shadow, seeds, _) =
                    construct_shadow_vocabulary(case.dim, seed, case.candidate_count, &values)
                        .unwrap();
                assert_eq!(shadow.len(), case.candidate_count);
                assert_eq!(seeds.len(), case.candidate_count);
                for (index, role) in shadow.iter().enumerate() {
                    assert!(values.iter().all(|candidate| candidate != role));
                    assert!(shadow[..index].iter().all(|candidate| candidate != role));
                }
            }
        }
    }

    #[test]
    fn exact_collision_forces_deterministic_retry() {
        let dim = 256;
        let seed = 91_337_u64;
        let first_seed = shadow_seed(seed, 0, 0);
        let colliding_candidate = UnitaryRole::new(dim, first_seed);
        let (shadow, seeds, retries) =
            construct_shadow_vocabulary(dim, seed, 1, &[colliding_candidate.clone()]).unwrap();
        assert_eq!(retries, 1);
        assert_eq!(shadow.len(), 1);
        assert_eq!(seeds.len(), 1);
        assert_ne!(seeds[0], first_seed);
        assert_ne!(shadow[0], colliding_candidate);
    }

    #[test]
    fn matched_probe_schedule_has_frozen_vectors() {
        assert_eq!(probe_distractor_index(30_001, 0, 0, 1, 4), 0);
        assert_eq!(probe_distractor_index(30_001, 1, 7, 2, 4), 1);
        assert_eq!(probe_distractor_index(31_001, 3, 127, 5, 8), 6);
        assert_eq!(probe_distractor_index(31_005, 7, 255, 0, 8), 1);
    }

    #[test]
    fn smoke_probe_is_deterministic_complete_and_finite() {
        let plan = ValidityCapacityPlan::smoke();
        let first = measure_validity_capacity_shadow_distractors(&plan).unwrap();
        let second = measure_validity_capacity_shadow_distractors(&plan).unwrap();
        assert_eq!(first.observations.len(), plan.cases.len() * plan.replicate_seeds.len());
        assert_eq!(first.observations.len(), second.observations.len());

        for (left, right) in first.observations.iter().zip(&second.observations) {
            assert_eq!(left.case, right.case);
            assert_eq!(left.seed, right.seed);
            assert_eq!(left.shadow_role_seeds, right.shadow_role_seeds);
            assert_eq!(
                left.total_shadow_collision_retries,
                right.total_shadow_collision_retries
            );
            assert_eq!(left.shadow_role_seeds.len(), left.case.candidate_count);
            assert_eq!(left.total_queries, right.total_queries);
            assert_eq!(left.shadow_scores.count, left.total_queries);
            assert_eq!(left.shadow_scores.mean.to_bits(), right.shadow_scores.mean.to_bits());
            assert_eq!(
                left.shadow_scores.variance.to_bits(),
                right.shadow_scores.variance.to_bits()
            );
            assert!(left.shadow_scores.mean.is_finite());
            assert!(left.shadow_scores.variance.is_finite());
            assert!(left.max_abs_shadow_candidate_similarity.is_finite());
            assert!((0.0..=1.0).contains(&left.max_abs_shadow_candidate_similarity));
            assert!(left.max_abs_shadow_shadow_similarity.is_finite());
            assert!((0.0..=1.0).contains(&left.max_abs_shadow_shadow_similarity));
            assert!(left.shadow_variance_ratio_to_null.is_finite());
            assert!(left.shadow_mean_squared_residual.is_finite());
            assert!(left.shadow_mse_ratio_to_null.is_finite());
        }
    }

    #[test]
    fn null_prediction_depends_on_case_not_replicate_seed() {
        let plan = ValidityCapacityPlan::smoke();
        let result = measure_validity_capacity_shadow_distractors(&plan).unwrap();
        for case in plan.cases {
            let matching = result
                .observations
                .iter()
                .filter(|observation| observation.case == case)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), plan.replicate_seeds.len());
            let predicted = matching[0].null_distractor_noise_variance;
            assert!(
                matching
                    .iter()
                    .all(|observation| observation.null_distractor_noise_variance == predicted)
            );
        }
    }

    #[test]
    fn duplicate_seed_is_rejected_before_measurement() {
        let mut plan = ValidityCapacityPlan::smoke();
        plan.replicate_seeds.push(plan.replicate_seeds[0]);
        assert!(matches!(
            measure_validity_capacity_shadow_distractors(&plan),
            Err(ValidityCapacityShadowError::DuplicateSeed(_))
        ));
    }
}
