// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Orthogonal controls for causal-time validity-memory capacity.
//!
//! The exploratory capacity sweep varies span length, but that changes both write
//! segmentation and the expected number of semantic changes. This module freezes
//! two cleaner controls before those results are interpreted:
//!
//! 1. **semantic-density control** — every checkpoint is written as its own span,
//!    so write count is fixed while semantic run length changes; adjacent semantic
//!    runs are forced to use different candidate values;
//! 2. **segmentation control** — semantic run length and the entire checkpoint-level
//!    semantic history are fixed, while each semantic run is split into different
//!    numbers of equivalent writes.
//!
//! Both controls keep `D`, `K`, `H`, and `C` fixed. Consequently the idealized
//! [`crate::validity_capacity_theory::ValidityCapacityNullModel`] makes the same
//! score-variance prediction for every case. Systematic case effects therefore
//! expose structure omitted by the null model rather than changing its load term.

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::TemporalAxis;
use crate::validity_capacity_theory::{
    ValidityCapacityNullModel, ValidityCapacityTheoryError,
};
use crate::validity_interval_memory::{ValidityIntervalMemory, ValidityMemoryError};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidityCapacityControlAxis {
    Smoke,
    /// Fixed one-checkpoint writes; semantic run length changes.
    SemanticRunLength,
    /// Fixed semantic history; equivalent runs are split into smaller writes.
    WriteSegmentation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValidityCapacityControlCase {
    pub axis: ValidityCapacityControlAxis,
    pub dim: usize,
    pub key_count: usize,
    pub candidate_count: usize,
    pub horizon: u64,
    /// Number of checkpoints for which a semantic value remains unchanged.
    pub semantic_run_length: u64,
    /// Number of checkpoints covered by each archive write.
    pub write_segment_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityCapacityControlPlan {
    pub cases: Vec<ValidityCapacityControlCase>,
    pub replicate_seeds: Vec<u64>,
}

impl ValidityCapacityControlPlan {
    pub fn smoke() -> Self {
        Self {
            cases: vec![
                ValidityCapacityControlCase {
                    axis: ValidityCapacityControlAxis::SemanticRunLength,
                    dim: 256,
                    key_count: 2,
                    candidate_count: 4,
                    horizon: 16,
                    semantic_run_length: 4,
                    write_segment_length: 1,
                },
                ValidityCapacityControlCase {
                    axis: ValidityCapacityControlAxis::WriteSegmentation,
                    dim: 256,
                    key_count: 2,
                    candidate_count: 4,
                    horizon: 16,
                    semantic_run_length: 4,
                    write_segment_length: 2,
                },
            ],
            replicate_seeds: vec![33_001, 33_002],
        }
    }

    /// Pre-result control protocol on seeds disjoint from #2768's `research_v0`.
    ///
    /// All cases fix `D=4096`, `K=8`, `C=8`, and `H=256`. Therefore every case
    /// has `rho = 0.5` and exactly the same idealized null-model variance.
    pub fn research_v0() -> Self {
        let mut cases = Vec::new();

        // Semantic-density control: every checkpoint is written separately, so
        // spans_written = K * H in every case. Only how often the semantic value
        // changes varies.
        for semantic_run_length in [1, 2, 4, 8, 16, 32, 64] {
            cases.push(ValidityCapacityControlCase {
                axis: ValidityCapacityControlAxis::SemanticRunLength,
                dim: 4096,
                key_count: 8,
                candidate_count: 8,
                horizon: 256,
                semantic_run_length,
                write_segment_length: 1,
            });
        }

        // Segmentation control: semantic history is held fixed at 32-checkpoint
        // forced-change runs. Only the number of writes used to represent each
        // run changes. Every segment length divides 32 exactly.
        for write_segment_length in [1, 2, 4, 8, 16, 32] {
            cases.push(ValidityCapacityControlCase {
                axis: ValidityCapacityControlAxis::WriteSegmentation,
                dim: 4096,
                key_count: 8,
                candidate_count: 8,
                horizon: 256,
                semantic_run_length: 32,
                write_segment_length,
            });
        }

        Self {
            cases,
            replicate_seeds: (32_001..=32_008).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityControlObservation {
    pub case: ValidityCapacityControlCase,
    pub seed: u64,
    pub correct: u64,
    pub total_queries: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
    pub smallest_margin: f64,
    pub spans_written: u64,
    /// Guaranteed semantic changes across all keys. Adjacent semantic runs are
    /// constructed to differ, so this value is not an estimate.
    pub semantic_changes: u64,
    pub represented_key_checkpoint_facts: u64,
    pub facts_per_dimension: f64,
    pub candidate_score_evaluations: u64,
    /// Null prediction shared by all cases with the same D/K/C/H coordinates.
    pub null_target_noise_variance: f64,
    pub null_distractor_noise_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityControlResult {
    pub observations: Vec<ValidityCapacityControlObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacityControlError {
    EmptyCases,
    EmptySeeds,
    DuplicateSeed(u64),
    ZeroDimension,
    ZeroKeyCount,
    TooFewCandidates,
    ZeroHorizon,
    ZeroSemanticRunLength,
    ZeroWriteSegmentLength,
    SemanticRunDoesNotDivideHorizon,
    WriteSegmentDoesNotDivideSemanticRun,
    SemanticDensityRequiresUnitWrites,
    SizeOverflow,
    Memory(String),
    Theory(String),
}

impl fmt::Display for ValidityCapacityControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCases => write!(f, "validity control plan requires at least one case"),
            Self::EmptySeeds => write!(f, "validity control plan requires replicate seeds"),
            Self::DuplicateSeed(seed) => write!(f, "validity control plan contains duplicate seed {seed}"),
            Self::ZeroDimension => write!(f, "validity control dimension must be non-zero"),
            Self::ZeroKeyCount => write!(f, "validity control key count must be non-zero"),
            Self::TooFewCandidates => write!(f, "validity control requires at least two candidates"),
            Self::ZeroHorizon => write!(f, "validity control horizon must be non-zero"),
            Self::ZeroSemanticRunLength => write!(f, "semantic run length must be non-zero"),
            Self::ZeroWriteSegmentLength => write!(f, "write segment length must be non-zero"),
            Self::SemanticRunDoesNotDivideHorizon => write!(f, "semantic run length must divide horizon exactly"),
            Self::WriteSegmentDoesNotDivideSemanticRun => write!(f, "write segment length must divide semantic run length exactly"),
            Self::SemanticDensityRequiresUnitWrites => write!(f, "semantic-density control requires one-checkpoint writes"),
            Self::SizeOverflow => write!(f, "validity control derived size overflowed u64"),
            Self::Memory(error) => write!(f, "validity control memory error: {error}"),
            Self::Theory(error) => write!(f, "validity control theory error: {error}"),
        }
    }
}

impl std::error::Error for ValidityCapacityControlError {}

impl From<ValidityMemoryError> for ValidityCapacityControlError {
    fn from(value: ValidityMemoryError) -> Self {
        Self::Memory(value.to_string())
    }
}

impl From<ValidityCapacityTheoryError> for ValidityCapacityControlError {
    fn from(value: ValidityCapacityTheoryError) -> Self {
        Self::Theory(value.to_string())
    }
}

pub fn run_validity_capacity_controls(
    plan: &ValidityCapacityControlPlan,
) -> Result<ValidityCapacityControlResult, ValidityCapacityControlError> {
    validate_plan(plan)?;
    let mut observations = Vec::with_capacity(plan.cases.len() * plan.replicate_seeds.len());
    for &case in &plan.cases {
        for &seed in &plan.replicate_seeds {
            observations.push(run_case(case, seed)?);
        }
    }
    Ok(ValidityCapacityControlResult { observations })
}

fn run_case(
    case: ValidityCapacityControlCase,
    seed: u64,
) -> Result<ValidityCapacityControlObservation, ValidityCapacityControlError> {
    validate_case(case)?;
    let axis = TemporalAxis::new(case.dim, seed ^ 0x4354_524C)
        .map_err(|error| ValidityCapacityControlError::Memory(error.to_string()))?;
    let keys = (0..case.key_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(1_000 + index as u64)))
        .collect::<Vec<_>>();
    let values = (0..case.candidate_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(100_000 + index as u64)))
        .collect::<Vec<_>>();
    let mut memory = ValidityIntervalMemory::new(case.dim)?;

    let semantic_runs = case.horizon / case.semantic_run_length;
    let segments_per_semantic_run = case.semantic_run_length / case.write_segment_length;
    let mut semantic_history = Vec::with_capacity(case.key_count);
    let mut spans_written = 0u64;

    for (key_index, key) in keys.iter().enumerate() {
        let values_for_runs = forced_semantic_values(
            seed,
            key_index,
            semantic_runs,
            case.candidate_count,
        );
        for (semantic_run, &value_index) in values_for_runs.iter().enumerate() {
            let semantic_start = semantic_run as u64 * case.semantic_run_length;
            for segment in 0..segments_per_semantic_run {
                let start = semantic_start + segment * case.write_segment_length;
                let end = start + case.write_segment_length;
                memory.write_span(&axis, key, &values[value_index], start, end)?;
                spans_written = spans_written
                    .checked_add(1)
                    .ok_or(ValidityCapacityControlError::SizeOverflow)?;
            }
        }
        semantic_history.push(values_for_runs);
    }

    let mut correct = 0u64;
    let mut total_queries = 0u64;
    let mut margin_sum = 0.0f64;
    let mut smallest_margin = f64::INFINITY;
    for (key_index, key) in keys.iter().enumerate() {
        for checkpoint in 0..case.horizon {
            let semantic_run = (checkpoint / case.semantic_run_length) as usize;
            let expected = semantic_history[key_index][semantic_run];
            let cleanup = memory.cleanup(&axis, key, &values, checkpoint)?;
            correct += (cleanup.best_index == expected) as u64;
            total_queries = total_queries
                .checked_add(1)
                .ok_or(ValidityCapacityControlError::SizeOverflow)?;
            margin_sum += cleanup.margin;
            smallest_margin = smallest_margin.min(cleanup.margin);
        }
    }

    let key_count_u64 = u64::try_from(case.key_count)
        .map_err(|_| ValidityCapacityControlError::SizeOverflow)?;
    let candidate_count_u64 = u64::try_from(case.candidate_count)
        .map_err(|_| ValidityCapacityControlError::SizeOverflow)?;
    let represented_facts = key_count_u64
        .checked_mul(case.horizon)
        .ok_or(ValidityCapacityControlError::SizeOverflow)?;
    let semantic_changes = key_count_u64
        .checked_mul(semantic_runs.saturating_sub(1))
        .ok_or(ValidityCapacityControlError::SizeOverflow)?;
    let candidate_score_evaluations = total_queries
        .checked_mul(candidate_count_u64)
        .ok_or(ValidityCapacityControlError::SizeOverflow)?;
    let null = ValidityCapacityNullModel::new(
        case.dim,
        case.key_count,
        case.candidate_count,
        case.horizon,
    )?;

    debug_assert_eq!(memory.spans_written() as u64, spans_written);
    Ok(ValidityCapacityControlObservation {
        case,
        seed,
        correct,
        total_queries,
        accuracy: correct as f64 / total_queries as f64,
        mean_margin: margin_sum / total_queries as f64,
        smallest_margin,
        spans_written,
        semantic_changes,
        represented_key_checkpoint_facts: represented_facts,
        facts_per_dimension: represented_facts as f64 / case.dim as f64,
        candidate_score_evaluations,
        null_target_noise_variance: null.target_noise_variance,
        null_distractor_noise_variance: null.distractor_noise_variance,
    })
}

/// Deterministically generate semantic-run values while guaranteeing that every
/// adjacent semantic run changes candidate identity.
fn forced_semantic_values(
    seed: u64,
    key_index: usize,
    run_count: u64,
    candidate_count: usize,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(run_count as usize);
    let mut state = splitmix64(seed ^ (key_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut current = (state as usize) % candidate_count;
    result.push(current);

    for run_index in 1..run_count {
        state = splitmix64(state ^ run_index.wrapping_mul(0xD1B5_4A32_D192_ED03));
        let jump = 1 + (state as usize % (candidate_count - 1));
        current = (current + jump) % candidate_count;
        result.push(current);
    }
    result
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn validate_plan(plan: &ValidityCapacityControlPlan) -> Result<(), ValidityCapacityControlError> {
    if plan.cases.is_empty() {
        return Err(ValidityCapacityControlError::EmptyCases);
    }
    if plan.replicate_seeds.is_empty() {
        return Err(ValidityCapacityControlError::EmptySeeds);
    }
    let mut seeds = HashSet::new();
    for &seed in &plan.replicate_seeds {
        if !seeds.insert(seed) {
            return Err(ValidityCapacityControlError::DuplicateSeed(seed));
        }
    }
    for &case in &plan.cases {
        validate_case(case)?;
    }
    Ok(())
}

fn validate_case(case: ValidityCapacityControlCase) -> Result<(), ValidityCapacityControlError> {
    if case.dim == 0 {
        return Err(ValidityCapacityControlError::ZeroDimension);
    }
    if case.key_count == 0 {
        return Err(ValidityCapacityControlError::ZeroKeyCount);
    }
    if case.candidate_count < 2 {
        return Err(ValidityCapacityControlError::TooFewCandidates);
    }
    if case.horizon == 0 {
        return Err(ValidityCapacityControlError::ZeroHorizon);
    }
    if case.semantic_run_length == 0 {
        return Err(ValidityCapacityControlError::ZeroSemanticRunLength);
    }
    if case.write_segment_length == 0 {
        return Err(ValidityCapacityControlError::ZeroWriteSegmentLength);
    }
    if !case.horizon.is_multiple_of(case.semantic_run_length) {
        return Err(ValidityCapacityControlError::SemanticRunDoesNotDivideHorizon);
    }
    if !case.semantic_run_length.is_multiple_of(case.write_segment_length) {
        return Err(ValidityCapacityControlError::WriteSegmentDoesNotDivideSemanticRun);
    }
    if case.axis == ValidityCapacityControlAxis::SemanticRunLength
        && case.write_segment_length != 1
    {
        return Err(ValidityCapacityControlError::SemanticDensityRequiresUnitWrites);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_semantic_schedule_never_repeats_adjacent_value() {
        let values = forced_semantic_values(71, 3, 128, 8);
        assert_eq!(values.len(), 128);
        assert!(values.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn smoke_controls_are_deterministic_without_success_assumption() {
        let plan = ValidityCapacityControlPlan::smoke();
        let first = run_validity_capacity_controls(&plan).unwrap();
        let second = run_validity_capacity_controls(&plan).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.observations.len(), 4);
        for observation in first.observations {
            assert!(observation.accuracy.is_finite());
            assert!(observation.mean_margin.is_finite());
            assert!(observation.smallest_margin.is_finite());
            assert_eq!(
                observation.represented_key_checkpoint_facts,
                observation.case.key_count as u64 * observation.case.horizon
            );
        }
    }

    #[test]
    fn research_v0_holds_null_coordinates_constant() {
        let plan = ValidityCapacityControlPlan::research_v0();
        assert_eq!(plan.cases.len(), 13);
        assert_eq!(plan.replicate_seeds, (32_001..=32_008).collect::<Vec<_>>());

        let first = plan.cases[0];
        for case in &plan.cases {
            assert_eq!(case.dim, first.dim);
            assert_eq!(case.key_count, first.key_count);
            assert_eq!(case.candidate_count, first.candidate_count);
            assert_eq!(case.horizon, first.horizon);
            let predicted = ValidityCapacityNullModel::new(
                case.dim,
                case.key_count,
                case.candidate_count,
                case.horizon,
            )
            .unwrap();
            let reference = ValidityCapacityNullModel::new(
                first.dim,
                first.key_count,
                first.candidate_count,
                first.horizon,
            )
            .unwrap();
            assert_eq!(predicted, reference);
        }
    }

    #[test]
    fn semantic_density_axis_fixes_write_count_and_forces_change_count() {
        let plan = ValidityCapacityControlPlan::research_v0();
        let cases = plan
            .cases
            .into_iter()
            .filter(|case| case.axis == ValidityCapacityControlAxis::SemanticRunLength)
            .collect::<Vec<_>>();
        assert_eq!(cases.len(), 7);
        for case in cases {
            assert_eq!(case.write_segment_length, 1);
            let writes = case.key_count as u64 * case.horizon;
            let changes = case.key_count as u64
                * (case.horizon / case.semantic_run_length - 1);
            assert_eq!(writes, 2048);
            assert!(changes <= writes);
        }
    }

    #[test]
    fn segmentation_axis_fixes_semantic_history_but_varies_writes() {
        let plan = ValidityCapacityControlPlan::research_v0();
        let cases = plan
            .cases
            .into_iter()
            .filter(|case| case.axis == ValidityCapacityControlAxis::WriteSegmentation)
            .collect::<Vec<_>>();
        assert_eq!(cases.len(), 6);
        let semantic_changes = cases[0].key_count as u64
            * (cases[0].horizon / cases[0].semantic_run_length - 1);
        assert_eq!(semantic_changes, 56);
        let write_counts = cases
            .iter()
            .map(|case| case.key_count as u64 * case.horizon / case.write_segment_length)
            .collect::<Vec<_>>();
        assert_eq!(write_counts, vec![2048, 1024, 512, 256, 128, 64]);
    }
}
