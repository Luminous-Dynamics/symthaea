// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic capacity sweep for checkpoint-validity associative memory.
//!
//! This is an exploratory measurement protocol, not a success-guaranteeing test.
//! It varies dimension, relation-key count, candidate-value count, checkpoint
//! horizon, and validity-span length independently. The key load coordinate is
//! `key_count * horizon / dim`: the number of represented key/checkpoint facts per
//! temporal dimension.
//!
//! The span-length axis changes both archive segmentation and the expected rate at
//! which independently resampled span values differ. It is therefore reported with
//! the *realized* semantic-change count rather than being interpreted as either a
//! pure write-density axis or a guaranteed one-change-per-boundary mutation axis.

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::TemporalAxis;
use crate::validity_interval_memory::{ValidityIntervalMemory, ValidityMemoryError};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidityCapacityAxis {
    Smoke,
    Dimension,
    KeyCount,
    CandidateCount,
    Horizon,
    SpanLength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValidityCapacityCase {
    pub axis: ValidityCapacityAxis,
    pub dim: usize,
    pub key_count: usize,
    pub candidate_count: usize,
    pub horizon: u64,
    pub span_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityCapacityPlan {
    pub cases: Vec<ValidityCapacityCase>,
    pub replicate_seeds: Vec<u64>,
}

impl ValidityCapacityPlan {
    pub fn smoke() -> Self {
        Self {
            cases: vec![
                ValidityCapacityCase {
                    axis: ValidityCapacityAxis::Smoke,
                    dim: 256,
                    key_count: 4,
                    candidate_count: 4,
                    horizon: 16,
                    span_length: 4,
                },
                ValidityCapacityCase {
                    axis: ValidityCapacityAxis::Smoke,
                    dim: 512,
                    key_count: 8,
                    candidate_count: 4,
                    horizon: 32,
                    span_length: 8,
                },
            ],
            replicate_seeds: vec![30_001, 30_002],
        }
    }

    /// Exploratory preregistration. Any changed axis values or seeds define a new
    /// experiment version rather than a revised `research_v0`.
    pub fn research_v0() -> Self {
        let mut cases = Vec::new();

        for dim in [512, 1_024, 2_048, 4_096, 8_192] {
            cases.push(ValidityCapacityCase {
                axis: ValidityCapacityAxis::Dimension,
                dim,
                key_count: 8,
                candidate_count: 8,
                horizon: 128,
                span_length: 8,
            });
        }
        for key_count in [2, 4, 8, 16, 32] {
            cases.push(ValidityCapacityCase {
                axis: ValidityCapacityAxis::KeyCount,
                dim: 4_096,
                key_count,
                candidate_count: 8,
                horizon: 128,
                span_length: 8,
            });
        }
        for candidate_count in [2, 4, 8, 16, 32] {
            cases.push(ValidityCapacityCase {
                axis: ValidityCapacityAxis::CandidateCount,
                dim: 4_096,
                key_count: 8,
                candidate_count,
                horizon: 128,
                span_length: 8,
            });
        }
        for horizon in [32, 64, 128, 256, 512] {
            cases.push(ValidityCapacityCase {
                axis: ValidityCapacityAxis::Horizon,
                dim: 4_096,
                key_count: 8,
                candidate_count: 8,
                horizon,
                span_length: 8,
            });
        }
        for span_length in [1, 2, 4, 8, 16, 32, 64] {
            cases.push(ValidityCapacityCase {
                axis: ValidityCapacityAxis::SpanLength,
                dim: 4_096,
                key_count: 8,
                candidate_count: 8,
                horizon: 256,
                span_length,
            });
        }

        Self {
            cases,
            replicate_seeds: (31_001..=31_005).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityObservation {
    pub case: ValidityCapacityCase,
    pub seed: u64,
    pub correct: u64,
    pub total_queries: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
    pub smallest_margin: f64,
    pub spans_written: u64,
    /// Number of adjacent span boundaries whose assigned semantic value changed.
    pub realized_semantic_changes: u64,
    /// Number of candidate values actually used by at least one written span.
    pub used_candidate_values: usize,
    pub represented_key_checkpoint_facts: u64,
    pub facts_per_dimension: f64,
    pub candidate_score_evaluations: u64,
    /// Maximum absolute pairwise cosine within the key role codebook.
    pub max_abs_key_similarity: f64,
    /// Maximum absolute pairwise cosine within the candidate value codebook.
    pub max_abs_candidate_similarity: f64,
    /// Maximum absolute cosine between any key role and any candidate role.
    pub max_abs_key_candidate_similarity: f64,
    /// Complex history accumulator only (`real` + `imag` f64 payloads).
    pub history_payload_bytes: u64,
    /// Temporal-axis frequency payload only.
    pub temporal_axis_payload_bytes: u64,
    /// Bipolar key/value codebook payload only.
    pub codebook_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacitySweepResult {
    pub observations: Vec<ValidityCapacityObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacityError {
    EmptyCases,
    EmptySeeds,
    DuplicateSeed(u64),
    ZeroDimension,
    ZeroKeyCount,
    TooFewCandidates,
    ZeroHorizon,
    ZeroSpanLength,
    SizeOverflow,
    Memory(String),
}

impl fmt::Display for ValidityCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCases => write!(f, "validity capacity plan requires at least one case"),
            Self::EmptySeeds => write!(f, "validity capacity plan requires replicate seeds"),
            Self::DuplicateSeed(seed) => write!(f, "validity capacity plan contains duplicate seed {seed}"),
            Self::ZeroDimension => write!(f, "validity capacity dimension must be non-zero"),
            Self::ZeroKeyCount => write!(f, "validity capacity key count must be non-zero"),
            Self::TooFewCandidates => write!(f, "validity capacity requires at least two candidate values"),
            Self::ZeroHorizon => write!(f, "validity capacity horizon must be non-zero"),
            Self::ZeroSpanLength => write!(f, "validity capacity span length must be non-zero"),
            Self::SizeOverflow => write!(f, "validity capacity derived size overflowed u64"),
            Self::Memory(error) => write!(f, "validity capacity memory error: {error}"),
        }
    }
}

impl std::error::Error for ValidityCapacityError {}

impl From<ValidityMemoryError> for ValidityCapacityError {
    fn from(value: ValidityMemoryError) -> Self {
        Self::Memory(value.to_string())
    }
}

pub fn run_validity_capacity_sweep(
    plan: &ValidityCapacityPlan,
) -> Result<ValidityCapacitySweepResult, ValidityCapacityError> {
    validate_plan(plan)?;
    let mut observations = Vec::with_capacity(plan.cases.len() * plan.replicate_seeds.len());
    for &case in &plan.cases {
        for &seed in &plan.replicate_seeds {
            observations.push(run_case(case, seed)?);
        }
    }
    Ok(ValidityCapacitySweepResult { observations })
}

fn run_case(
    case: ValidityCapacityCase,
    seed: u64,
) -> Result<ValidityCapacityObservation, ValidityCapacityError> {
    validate_case(case)?;
    let axis = TemporalAxis::new(case.dim, seed ^ 0x5449_4D45)
        .map_err(|error| ValidityCapacityError::Memory(error.to_string()))?;
    let keys = (0..case.key_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(1_000 + index as u64)))
        .collect::<Vec<_>>();
    let values = (0..case.candidate_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(100_000 + index as u64)))
        .collect::<Vec<_>>();
    let mut memory = ValidityIntervalMemory::new(case.dim)?;

    let mut expected_spans = 0u64;
    let mut realized_semantic_changes = 0u64;
    let mut used_candidate_values = HashSet::new();
    for (key_index, key) in keys.iter().enumerate() {
        let mut start = 0u64;
        let mut span_index = 0u64;
        let mut previous_value = None::<usize>;
        while start < case.horizon {
            let end = (start + case.span_length).min(case.horizon);
            let value_index = assigned_value(seed, key_index, span_index, case.candidate_count);
            if previous_value.is_some_and(|previous| previous != value_index) {
                realized_semantic_changes = realized_semantic_changes
                    .checked_add(1)
                    .ok_or(ValidityCapacityError::SizeOverflow)?;
            }
            previous_value = Some(value_index);
            used_candidate_values.insert(value_index);
            memory.write_span(&axis, key, &values[value_index], start, end)?;
            expected_spans = expected_spans
                .checked_add(1)
                .ok_or(ValidityCapacityError::SizeOverflow)?;
            start = end;
            span_index += 1;
        }
    }

    let mut correct = 0u64;
    let mut total_queries = 0u64;
    let mut margin_sum = 0.0f64;
    let mut smallest_margin = f64::INFINITY;
    for (key_index, key) in keys.iter().enumerate() {
        for checkpoint in 0..case.horizon {
            let span_index = checkpoint / case.span_length;
            let expected = assigned_value(seed, key_index, span_index, case.candidate_count);
            let cleanup = memory.cleanup(&axis, key, &values, checkpoint)?;
            correct += (cleanup.best_index == expected) as u64;
            total_queries += 1;
            margin_sum += cleanup.margin;
            smallest_margin = smallest_margin.min(cleanup.margin);
        }
    }

    let represented_facts = u64::try_from(case.key_count)
        .ok()
        .and_then(|keys| keys.checked_mul(case.horizon))
        .ok_or(ValidityCapacityError::SizeOverflow)?;
    let candidate_count_u64 = u64::try_from(case.candidate_count)
        .map_err(|_| ValidityCapacityError::SizeOverflow)?;
    let candidate_score_evaluations = total_queries
        .checked_mul(candidate_count_u64)
        .ok_or(ValidityCapacityError::SizeOverflow)?;
    let dim_u64 = u64::try_from(case.dim).map_err(|_| ValidityCapacityError::SizeOverflow)?;
    let key_count_u64 = u64::try_from(case.key_count)
        .map_err(|_| ValidityCapacityError::SizeOverflow)?;
    let history_payload_bytes = dim_u64
        .checked_mul(16)
        .ok_or(ValidityCapacityError::SizeOverflow)?;
    let temporal_axis_payload_bytes = dim_u64
        .checked_mul(8)
        .ok_or(ValidityCapacityError::SizeOverflow)?;
    let codebook_payload_bytes = key_count_u64
        .checked_add(candidate_count_u64)
        .and_then(|roles| roles.checked_mul(dim_u64))
        .and_then(|components| components.checked_mul(4))
        .ok_or(ValidityCapacityError::SizeOverflow)?;

    debug_assert_eq!(memory.spans_written() as u64, expected_spans);
    Ok(ValidityCapacityObservation {
        case,
        seed,
        correct,
        total_queries,
        accuracy: correct as f64 / total_queries as f64,
        mean_margin: margin_sum / total_queries as f64,
        smallest_margin,
        spans_written: expected_spans,
        realized_semantic_changes,
        used_candidate_values: used_candidate_values.len(),
        represented_key_checkpoint_facts: represented_facts,
        facts_per_dimension: represented_facts as f64 / case.dim as f64,
        candidate_score_evaluations,
        max_abs_key_similarity: max_abs_pairwise_similarity(&keys),
        max_abs_candidate_similarity: max_abs_pairwise_similarity(&values),
        max_abs_key_candidate_similarity: max_abs_cross_similarity(&keys, &values),
        history_payload_bytes,
        temporal_axis_payload_bytes,
        codebook_payload_bytes,
    })
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

fn role_similarity(left: &UnitaryRole, right: &UnitaryRole) -> f64 {
    debug_assert_eq!(left.dim(), right.dim());
    left.as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(&left, &right)| (left * right) as f64)
        .sum::<f64>()
        / left.dim() as f64
}

fn max_abs_pairwise_similarity(roles: &[UnitaryRole]) -> f64 {
    let mut maximum = 0.0_f64;
    for left in 0..roles.len() {
        for right in (left + 1)..roles.len() {
            maximum = maximum.max(role_similarity(&roles[left], &roles[right]).abs());
        }
    }
    maximum
}

fn max_abs_cross_similarity(left: &[UnitaryRole], right: &[UnitaryRole]) -> f64 {
    let mut maximum = 0.0_f64;
    for left_role in left {
        for right_role in right {
            maximum = maximum.max(role_similarity(left_role, right_role).abs());
        }
    }
    maximum
}

fn validate_plan(plan: &ValidityCapacityPlan) -> Result<(), ValidityCapacityError> {
    if plan.cases.is_empty() {
        return Err(ValidityCapacityError::EmptyCases);
    }
    if plan.replicate_seeds.is_empty() {
        return Err(ValidityCapacityError::EmptySeeds);
    }
    let mut seeds = HashSet::new();
    for &seed in &plan.replicate_seeds {
        if !seeds.insert(seed) {
            return Err(ValidityCapacityError::DuplicateSeed(seed));
        }
    }
    for &case in &plan.cases {
        validate_case(case)?;
    }
    Ok(())
}

fn validate_case(case: ValidityCapacityCase) -> Result<(), ValidityCapacityError> {
    if case.dim == 0 {
        return Err(ValidityCapacityError::ZeroDimension);
    }
    if case.key_count == 0 {
        return Err(ValidityCapacityError::ZeroKeyCount);
    }
    if case.candidate_count < 2 {
        return Err(ValidityCapacityError::TooFewCandidates);
    }
    if case.horizon == 0 {
        return Err(ValidityCapacityError::ZeroHorizon);
    }
    if case.span_length == 0 {
        return Err(ValidityCapacityError::ZeroSpanLength);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_sweep_is_deterministic_without_accuracy_assumption() {
        let plan = ValidityCapacityPlan::smoke();
        let first = run_validity_capacity_sweep(&plan).unwrap();
        let second = run_validity_capacity_sweep(&plan).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.observations.len(), plan.cases.len() * plan.replicate_seeds.len());
        for observation in &first.observations {
            assert!(observation.accuracy.is_finite());
            assert!(observation.mean_margin.is_finite());
            assert!(observation.smallest_margin.is_finite());
            assert_eq!(
                observation.represented_key_checkpoint_facts,
                observation.case.key_count as u64 * observation.case.horizon
            );
            assert!(observation.used_candidate_values <= observation.case.candidate_count);
            let maximum_boundaries = observation
                .spans_written
                .saturating_sub(observation.case.key_count as u64);
            assert!(observation.realized_semantic_changes <= maximum_boundaries);
            for similarity in [
                observation.max_abs_key_similarity,
                observation.max_abs_candidate_similarity,
                observation.max_abs_key_candidate_similarity,
            ] {
                assert!(similarity.is_finite());
                assert!((0.0..=1.0).contains(&similarity));
            }
        }
    }

    #[test]
    fn research_v0_freezes_five_axes_and_five_replicates() {
        let plan = ValidityCapacityPlan::research_v0();
        assert_eq!(plan.cases.len(), 27);
        assert_eq!(plan.replicate_seeds, vec![31_001, 31_002, 31_003, 31_004, 31_005]);
        assert_eq!(
            plan.cases
                .iter()
                .filter(|case| case.axis == ValidityCapacityAxis::Dimension)
                .count(),
            5
        );
        assert_eq!(
            plan.cases
                .iter()
                .filter(|case| case.axis == ValidityCapacityAxis::KeyCount)
                .count(),
            5
        );
        assert_eq!(
            plan.cases
                .iter()
                .filter(|case| case.axis == ValidityCapacityAxis::CandidateCount)
                .count(),
            5
        );
        assert_eq!(
            plan.cases
                .iter()
                .filter(|case| case.axis == ValidityCapacityAxis::Horizon)
                .count(),
            5
        );
        assert_eq!(
            plan.cases
                .iter()
                .filter(|case| case.axis == ValidityCapacityAxis::SpanLength)
                .count(),
            7
        );
    }

    #[test]
    fn duplicate_replicate_seed_is_rejected() {
        let mut plan = ValidityCapacityPlan::smoke();
        plan.replicate_seeds.push(plan.replicate_seeds[0]);
        assert!(matches!(
            run_validity_capacity_sweep(&plan),
            Err(ValidityCapacityError::DuplicateSeed(_))
        ));
    }
}
