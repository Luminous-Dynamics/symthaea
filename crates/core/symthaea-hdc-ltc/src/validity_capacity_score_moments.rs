// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent score-moment oracle for the preregistered validity-capacity sweep.
//!
//! The primary #2768 runner intentionally records compact capacity metrics. The
//! analytic null model added later predicts something more specific: the mean and
//! variance of target and distractor candidate scores. This module reconstructs
//! the same deterministic synthetic histories independently, verifies parity with
//! the primary runner in public tests, and records those additional observables.
//!
//! Only two extra direct scores are evaluated per query:
//!
//! - the correct target candidate;
//! - one deterministic seed-hashed distractor that is guaranteed to differ from
//!   the target.
//!
//! The existing cleanup result supplies the maximum distractor score: if the
//! target wins it is `second_score`; otherwise the winning score itself is the
//! maximum distractor. Therefore the signed correctness margin
//!
//! `target_score - max_distractor_score`
//!
//! is available without rescoring the full candidate set.

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::TemporalAxis;
use crate::validity_capacity::{
    ValidityCapacityCase, ValidityCapacityError, ValidityCapacityPlan,
};
use crate::validity_capacity_theory::{
    ValidityCapacityNullModel, ValidityCapacityTheoryError,
};
use crate::validity_interval_memory::{ValidityIntervalMemory, ValidityMemoryError};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreMomentSummary {
    pub count: u64,
    pub mean: f64,
    /// Population variance `sum((x - mean)^2) / count` over the observed scores.
    pub variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityScoreMomentObservation {
    pub case: ValidityCapacityCase,
    pub seed: u64,
    /// Primary-runner parity fields.
    pub correct: u64,
    pub total_queries: u64,
    pub accuracy: f64,
    pub mean_winner_margin: f64,
    pub smallest_winner_margin: f64,
    pub spans_written: u64,
    pub represented_key_checkpoint_facts: u64,
    pub facts_per_dimension: f64,
    /// Signed margin of the true target against the strongest distractor.
    pub mean_true_margin: f64,
    pub smallest_true_margin: f64,
    /// Empirical target score moments across every key/checkpoint query.
    pub target_scores: ScoreMomentSummary,
    /// Empirical moments of one seed-hashed non-target candidate per query.
    pub probe_distractor_scores: ScoreMomentSummary,
    /// Pre-result idealized predictions for direct comparison.
    pub null_target_noise_variance: f64,
    pub null_distractor_noise_variance: f64,
    /// Descriptive ratios; values above/below one indicate more/less empirical
    /// score variance than the idealized independent-interference model predicts.
    pub target_variance_ratio_to_null: f64,
    pub distractor_variance_ratio_to_null: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityScoreMomentResult {
    pub observations: Vec<ValidityCapacityScoreMomentObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacityScoreMomentError {
    Capacity(String),
    Memory(String),
    Theory(String),
    SizeOverflow,
}

impl fmt::Display for ValidityCapacityScoreMomentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity(error) => write!(f, "capacity score oracle case error: {error}"),
            Self::Memory(error) => write!(f, "capacity score oracle memory error: {error}"),
            Self::Theory(error) => write!(f, "capacity score oracle theory error: {error}"),
            Self::SizeOverflow => write!(f, "capacity score oracle derived size overflowed u64"),
        }
    }
}

impl std::error::Error for ValidityCapacityScoreMomentError {}

impl From<ValidityMemoryError> for ValidityCapacityScoreMomentError {
    fn from(value: ValidityMemoryError) -> Self {
        Self::Memory(value.to_string())
    }
}

impl From<ValidityCapacityTheoryError> for ValidityCapacityScoreMomentError {
    fn from(value: ValidityCapacityTheoryError) -> Self {
        Self::Theory(value.to_string())
    }
}

impl From<ValidityCapacityError> for ValidityCapacityScoreMomentError {
    fn from(value: ValidityCapacityError) -> Self {
        Self::Capacity(value.to_string())
    }
}

pub fn measure_validity_capacity_score_moments(
    plan: &ValidityCapacityPlan,
) -> Result<ValidityCapacityScoreMomentResult, ValidityCapacityScoreMomentError> {
    // Reuse the primary runner's public validation surface without depending on
    // its private synthetic state. Running it would duplicate the full workload,
    // so validation is mirrored below at the case level instead.
    if plan.cases.is_empty() {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity plan requires at least one case".to_string(),
        ));
    }
    if plan.replicate_seeds.is_empty() {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity plan requires replicate seeds".to_string(),
        ));
    }

    let mut observations = Vec::with_capacity(plan.cases.len() * plan.replicate_seeds.len());
    for &case in &plan.cases {
        validate_case(case)?;
        for &seed in &plan.replicate_seeds {
            observations.push(measure_case(case, seed)?);
        }
    }
    Ok(ValidityCapacityScoreMomentResult { observations })
}

fn measure_case(
    case: ValidityCapacityCase,
    seed: u64,
) -> Result<ValidityCapacityScoreMomentObservation, ValidityCapacityScoreMomentError> {
    let axis = TemporalAxis::new(case.dim, seed ^ 0x5449_4D45)
        .map_err(|error| ValidityCapacityScoreMomentError::Memory(error.to_string()))?;
    let keys = (0..case.key_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(1_000 + index as u64)))
        .collect::<Vec<_>>();
    let values = (0..case.candidate_count)
        .map(|index| UnitaryRole::new(case.dim, seed.wrapping_add(100_000 + index as u64)))
        .collect::<Vec<_>>();
    let mut memory = ValidityIntervalMemory::new(case.dim)?;

    let mut spans_written = 0u64;
    for (key_index, key) in keys.iter().enumerate() {
        let mut start = 0u64;
        let mut span_index = 0u64;
        while start < case.horizon {
            let end = (start + case.span_length).min(case.horizon);
            let value_index = assigned_value(seed, key_index, span_index, case.candidate_count);
            memory.write_span(&axis, key, &values[value_index], start, end)?;
            spans_written = spans_written
                .checked_add(1)
                .ok_or(ValidityCapacityScoreMomentError::SizeOverflow)?;
            start = end;
            span_index += 1;
        }
    }

    let mut correct = 0u64;
    let mut total_queries = 0u64;
    let mut winner_margin_sum = 0.0f64;
    let mut smallest_winner_margin = f64::INFINITY;
    let mut true_margin_sum = 0.0f64;
    let mut smallest_true_margin = f64::INFINITY;
    let mut target_moments = RunningMoments::default();
    let mut distractor_moments = RunningMoments::default();

    for (key_index, key) in keys.iter().enumerate() {
        for checkpoint in 0..case.horizon {
            let span_index = checkpoint / case.span_length;
            let expected = assigned_value(seed, key_index, span_index, case.candidate_count);
            let cleanup = memory.cleanup(&axis, key, &values, checkpoint)?;
            let target_score = memory.score_candidate(&axis, key, &values[expected], checkpoint)?;
            let probe_index = probe_distractor_index(
                seed,
                key_index,
                checkpoint,
                expected,
                case.candidate_count,
            );
            let distractor_score =
                memory.score_candidate(&axis, key, &values[probe_index], checkpoint)?;

            let max_distractor_score = if cleanup.best_index == expected {
                cleanup.second_score
            } else {
                cleanup.best_score
            };
            let true_margin = target_score - max_distractor_score;

            correct += (cleanup.best_index == expected) as u64;
            total_queries = total_queries
                .checked_add(1)
                .ok_or(ValidityCapacityScoreMomentError::SizeOverflow)?;
            winner_margin_sum += cleanup.margin;
            smallest_winner_margin = smallest_winner_margin.min(cleanup.margin);
            true_margin_sum += true_margin;
            smallest_true_margin = smallest_true_margin.min(true_margin);
            target_moments.push(target_score);
            distractor_moments.push(distractor_score);
        }
    }

    let represented_facts = u64::try_from(case.key_count)
        .ok()
        .and_then(|keys| keys.checked_mul(case.horizon))
        .ok_or(ValidityCapacityScoreMomentError::SizeOverflow)?;
    let null = ValidityCapacityNullModel::new(
        case.dim,
        case.key_count,
        case.candidate_count,
        case.horizon,
    )?;
    let target_scores = target_moments.finish();
    let probe_distractor_scores = distractor_moments.finish();

    Ok(ValidityCapacityScoreMomentObservation {
        case,
        seed,
        correct,
        total_queries,
        accuracy: correct as f64 / total_queries as f64,
        mean_winner_margin: winner_margin_sum / total_queries as f64,
        smallest_winner_margin,
        spans_written,
        represented_key_checkpoint_facts: represented_facts,
        facts_per_dimension: represented_facts as f64 / case.dim as f64,
        mean_true_margin: true_margin_sum / total_queries as f64,
        smallest_true_margin,
        target_scores,
        probe_distractor_scores,
        null_target_noise_variance: null.target_noise_variance,
        null_distractor_noise_variance: null.distractor_noise_variance,
        target_variance_ratio_to_null: ratio_or_infinity(
            target_scores.variance,
            null.target_noise_variance,
        ),
        distractor_variance_ratio_to_null: ratio_or_infinity(
            probe_distractor_scores.variance,
            null.distractor_noise_variance,
        ),
    })
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

fn validate_case(case: ValidityCapacityCase) -> Result<(), ValidityCapacityScoreMomentError> {
    if case.dim == 0 {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity dimension must be non-zero".to_string(),
        ));
    }
    if case.key_count == 0 {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity key count must be non-zero".to_string(),
        ));
    }
    if case.candidate_count < 2 {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity requires at least two candidate values".to_string(),
        ));
    }
    if case.horizon == 0 {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity horizon must be non-zero".to_string(),
        ));
    }
    if case.span_length == 0 {
        return Err(ValidityCapacityScoreMomentError::Capacity(
            "validity capacity span length must be non-zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validity_capacity::run_validity_capacity_sweep;

    #[test]
    fn independent_oracle_matches_primary_smoke_runner() {
        let plan = ValidityCapacityPlan::smoke();
        let primary = run_validity_capacity_sweep(&plan).unwrap();
        let diagnostic = measure_validity_capacity_score_moments(&plan).unwrap();
        assert_eq!(primary.observations.len(), diagnostic.observations.len());

        for (primary, diagnostic) in primary.observations.iter().zip(&diagnostic.observations) {
            assert_eq!(primary.case, diagnostic.case);
            assert_eq!(primary.seed, diagnostic.seed);
            assert_eq!(primary.correct, diagnostic.correct);
            assert_eq!(primary.total_queries, diagnostic.total_queries);
            assert_eq!(primary.accuracy.to_bits(), diagnostic.accuracy.to_bits());
            assert_eq!(primary.mean_margin.to_bits(), diagnostic.mean_winner_margin.to_bits());
            assert_eq!(
                primary.smallest_margin.to_bits(),
                diagnostic.smallest_winner_margin.to_bits()
            );
            assert_eq!(primary.spans_written, diagnostic.spans_written);
            assert_eq!(
                primary.represented_key_checkpoint_facts,
                diagnostic.represented_key_checkpoint_facts
            );
        }
    }

    #[test]
    fn signed_true_margin_agrees_with_correctness_sign() {
        let result = measure_validity_capacity_score_moments(&ValidityCapacityPlan::smoke()).unwrap();
        for observation in result.observations {
            assert!(observation.mean_true_margin.is_finite());
            assert!(observation.smallest_true_margin.is_finite());
            assert!(observation.target_scores.mean.is_finite());
            assert!(observation.target_scores.variance.is_finite());
            assert!(observation.probe_distractor_scores.mean.is_finite());
            assert!(observation.probe_distractor_scores.variance.is_finite());
            assert_eq!(observation.target_scores.count, observation.total_queries);
            assert_eq!(observation.probe_distractor_scores.count, observation.total_queries);

            if observation.correct == observation.total_queries {
                assert!(observation.smallest_true_margin >= 0.0);
            }
            if observation.smallest_true_margin < 0.0 {
                assert!(observation.correct < observation.total_queries);
            }
        }
    }

    #[test]
    fn null_predictions_depend_only_on_case_not_replicate_seed() {
        let plan = ValidityCapacityPlan::smoke();
        let result = measure_validity_capacity_score_moments(&plan).unwrap();
        for case in plan.cases {
            let matching = result
                .observations
                .iter()
                .filter(|observation| observation.case == case)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), plan.replicate_seeds.len());
            let target = matching[0].null_target_noise_variance;
            let distractor = matching[0].null_distractor_noise_variance;
            assert!(matching.iter().all(|observation| {
                observation.null_target_noise_variance == target
                    && observation.null_distractor_noise_variance == distractor
            }));
        }
    }
}
