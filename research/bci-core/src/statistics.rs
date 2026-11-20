// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Participant-clustered paired bootstrap intervals for BCI model comparisons.
//!
//! Each row must be one held-out participant's aggregate score, calculated from
//! the same preregistered evaluation cells for both methods. Do not pass trial- or
//! window-level rows: overlapping windows are not independent participants.
//! This estimates uncertainty for an equal-participant-weighted mean. It is a
//! complement to (not a replacement for) a benchmark's official cell-weighted
//! leaderboard metric, and does not establish clinical benefit or safety.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::benchmark::{
    evaluate_track02_reference_compatible_cell_mean, CellBalancedAccuracyError,
    EvaluationCell, LabeledPrediction,
};

/// Scores use integer parts per million: 1_000_000 represents 1.0 / 100%.
pub const SCORE_SCALE_PPM: u32 = 1_000_000;
const MIN_REPLICATES: u32 = 100;
const MAX_REPLICATES: u32 = 1_000_000;

/// Paired score summaries for one independent held-out participant.
///
/// Aggregate each method over the same frozen cells before constructing this
/// value. Participant identifiers should be pseudonymous and purpose-limited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedParticipantScore {
    pub participant_id: String,
    pub baseline_score_ppm: u32,
    pub candidate_score_ppm: u32,
}

/// All randomness and interval choices are explicit and replayable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    /// Must be in [100, 1_000_000]. 10,000 is a useful starting value.
    pub replicates: u32,
    /// Confidence level in permille, supported range [500, 999].
    pub confidence_level_permille: u16,
    /// Zero is valid; the internal generator maps it to a non-degenerate stream.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            replicates: 10_000,
            confidence_level_permille: 950,
            seed: 0xBC1_2026_5EED,
        }
    }
}

/// Percentile interval for the paired candidate-minus-baseline score difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedBootstrapReport {
    /// Stable identifier for this exact resampling and quantile convention.
    pub method_id: &'static str,
    pub participant_count: u32,
    pub replicates: u32,
    pub confidence_level_permille: u16,
    pub seed: u64,
    pub baseline_mean_score_ppm: u32,
    pub candidate_mean_score_ppm: u32,
    /// Arithmetic mean paired difference, candidate minus baseline.
    pub mean_paired_delta_ppm: i64,
    pub ci_lower_delta_ppm: i64,
    pub ci_upper_delta_ppm: i64,
    /// Count of bootstrap replicates with a positive mean delta. This is a
    /// descriptive resampling fraction, not a Bayesian posterior probability.
    pub positive_delta_replicates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedBootstrapError {
    EmptyScores,
    TooFewParticipants { actual: usize },
    TooFewCells { actual: usize },
    EmptyParticipantId { index: usize },
    DuplicateParticipantId(String),
    ScoreOutOfRange { index: usize, candidate: bool, score_ppm: u32 },
    InvalidReplicateCount(u32),
    InvalidConfidenceLevel(u16),
    CounterOverflow,
}

impl fmt::Display for PairedBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PairedBootstrapError {}

/// Compute a deterministic paired percentile bootstrap by resampling participants.
///
/// Every replicate draws N participants with replacement from the N supplied
/// participants and averages their paired score differences. The same resampled
/// participant is used for both candidate and baseline, preserving the pairing.
/// The returned interval describes between-participant uncertainty in the supplied
/// cohort; it is not a guarantee about arbitrary devices, populations, or tasks.
pub fn paired_participant_bootstrap(
    scores: &[PairedParticipantScore],
    config: BootstrapConfig,
) -> Result<PairedBootstrapReport, PairedBootstrapError> {
    if scores.is_empty() {
        return Err(PairedBootstrapError::EmptyScores);
    }
    if scores.len() < 2 {
        return Err(PairedBootstrapError::TooFewParticipants { actual: scores.len() });
    }
    if !(MIN_REPLICATES..=MAX_REPLICATES).contains(&config.replicates) {
        return Err(PairedBootstrapError::InvalidReplicateCount(config.replicates));
    }
    if !(500..=999).contains(&config.confidence_level_permille) {
        return Err(PairedBootstrapError::InvalidConfidenceLevel(
            config.confidence_level_permille,
        ));
    }

    let mut participant_ids = BTreeSet::new();
    for (index, score) in scores.iter().enumerate() {
        if score.participant_id.trim().is_empty() {
            return Err(PairedBootstrapError::EmptyParticipantId { index });
        }
        if !participant_ids.insert(score.participant_id.as_str()) {
            return Err(PairedBootstrapError::DuplicateParticipantId(
                score.participant_id.clone(),
            ));
        }
        if score.baseline_score_ppm > SCORE_SCALE_PPM {
            return Err(PairedBootstrapError::ScoreOutOfRange {
                index,
                candidate: false,
                score_ppm: score.baseline_score_ppm,
            });
        }
        if score.candidate_score_ppm > SCORE_SCALE_PPM {
            return Err(PairedBootstrapError::ScoreOutOfRange {
                index,
                candidate: true,
                score_ppm: score.candidate_score_ppm,
            });
        }
    }

    let participant_count_u32 =
        u32::try_from(scores.len()).map_err(|_| PairedBootstrapError::CounterOverflow)?;
    let participant_count_u64 =
        u64::try_from(scores.len()).map_err(|_| PairedBootstrapError::CounterOverflow)?;
    let participant_count_i128 =
        i128::try_from(scores.len()).map_err(|_| PairedBootstrapError::CounterOverflow)?;

    let mut baseline_total = 0_u128;
    let mut candidate_total = 0_u128;
    let mut delta_total = 0_i128;
    for score in scores {
        baseline_total = baseline_total
            .checked_add(u128::from(score.baseline_score_ppm))
            .ok_or(PairedBootstrapError::CounterOverflow)?;
        candidate_total = candidate_total
            .checked_add(u128::from(score.candidate_score_ppm))
            .ok_or(PairedBootstrapError::CounterOverflow)?;
        let delta = i128::from(score.candidate_score_ppm)
            - i128::from(score.baseline_score_ppm);
        delta_total = delta_total
            .checked_add(delta)
            .ok_or(PairedBootstrapError::CounterOverflow)?;
    }

    let baseline_mean_score_ppm = u32::try_from(baseline_total / u128::from(participant_count_u64))
        .map_err(|_| PairedBootstrapError::CounterOverflow)?;
    let candidate_mean_score_ppm =
        u32::try_from(candidate_total / u128::from(participant_count_u64))
            .map_err(|_| PairedBootstrapError::CounterOverflow)?;
    let mean_paired_delta_ppm =
        i64::try_from(delta_total / participant_count_i128)
            .map_err(|_| PairedBootstrapError::CounterOverflow)?;

    let mut random_state = config.seed;
    let mut bootstrap_deltas = Vec::with_capacity(config.replicates as usize);
    let mut positive_delta_replicates = 0_u32;
    for _ in 0..config.replicates {
        let mut replicate_total = 0_i128;
        for _ in 0..participant_count_u64 {
            let index = sample_below(&mut random_state, participant_count_u64) as usize;
            let row = &scores[index];
            let delta =
                i128::from(row.candidate_score_ppm) - i128::from(row.baseline_score_ppm);
            replicate_total = replicate_total
                .checked_add(delta)
                .ok_or(PairedBootstrapError::CounterOverflow)?;
        }
        let replicate_mean = i64::try_from(replicate_total / participant_count_i128)
            .map_err(|_| PairedBootstrapError::CounterOverflow)?;
        if replicate_mean > 0 {
            positive_delta_replicates = positive_delta_replicates
                .checked_add(1)
                .ok_or(PairedBootstrapError::CounterOverflow)?;
        }
        bootstrap_deltas.push(replicate_mean);
    }

    bootstrap_deltas.sort_unstable();
    let last_index = (bootstrap_deltas.len() - 1) as u128;
    let tail_permille = u128::from(1000 - config.confidence_level_permille);
    // Percentile positions are (1-confidence)/2 and 1-(1-confidence)/2.
    // Integer indexing and fixed-point scores keep output deterministic.
    let lower_index = (tail_permille * last_index / 2000) as usize;
    let upper_index = ((2000 - tail_permille) * last_index / 2000) as usize;

    Ok(PairedBootstrapReport {
        method_id: "paired-participant-bootstrap-percentile-order-statistic-v1",
        participant_count: participant_count_u32,
        replicates: config.replicates,
        confidence_level_permille: config.confidence_level_permille,
        seed: config.seed,
        baseline_mean_score_ppm,
        candidate_mean_score_ppm,
        mean_paired_delta_ppm,
        ci_lower_delta_ppm: bootstrap_deltas[lower_index],
        ci_upper_delta_ppm: bootstrap_deltas[upper_index],
        positive_delta_replicates,
    })
}

/// SplitMix64 provides a small deterministic PRNG without external dependencies.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Uniform index in [0, upper), using rejection sampling to avoid modulo bias.
fn sample_below(state: &mut u64, upper: u64) -> u64 {
    let threshold = upper.wrapping_neg() % upper;
    loop {
        let value = next_u64(state);
        if value >= threshold {
            return value % upper;
        }
    }
}

/// One prediction tied to a stable evaluation-window identity.
///
/// Window IDs must be pseudonymous, unique within a scored run, and identical
/// between candidate and baseline for the same labelled observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedLabeledPrediction {
    pub window_id: String,
    pub cell: EvaluationCell,
    pub target_class: u16,
    pub predicted_class: u16,
}

/// Error returned by the paired participant-cluster bootstrap of the official-style
/// cell-mean metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedCellBootstrapError {
    EmptyPredictions,
    EmptyWindowId { index: usize },
    DuplicateWindowId(String),
    PairedWindowInventoryMismatch,
    TooFewParticipants { actual: usize },
    BaselineMetric(CellBalancedAccuracyError),
    CandidateMetric(CellBalancedAccuracyError),
    InvalidReplicateCount(u32),
    InvalidConfidenceLevel(u16),
    CounterOverflow,
}

impl fmt::Display for PairedCellBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PairedCellBootstrapError {}

/// Paired confidence interval for the official-style equal-cell-weighted score.
///
/// The same participants are resampled for baseline and candidate. For each draw,
/// every selected participant contributes all of their declared session/context
/// cells, including repeated copies when sampled more than once, and the score is
/// recomputed as the mean over the sampled cells. This preserves the competition's
/// equal-cell aggregation even when participants contribute differing cell counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellMeanPairedBootstrapReport {
    pub method_id: &'static str,
    pub participant_count: u32,
    pub cell_count: u32,
    pub replicates: u32,
    pub confidence_level_permille: u16,
    pub seed: u64,
    pub baseline_score_ppm: u32,
    pub candidate_score_ppm: u32,
    /// Candidate official-style score minus baseline official-style score.
    pub observed_delta_ppm: i64,
    /// Arithmetic mean of paired bootstrap score differences.
    pub mean_bootstrap_delta_ppm: i64,
    pub ci_lower_delta_ppm: i64,
    pub ci_upper_delta_ppm: i64,
    /// Descriptive fraction of resamples whose score difference was positive,
    /// not a posterior probability or p-value.
    pub positive_delta_replicates: u32,
}

/// Compare two models using a participant-cluster bootstrap of the Track 02-style
/// cell-mean balanced-accuracy score. Both prediction arrays must identify the same labelled evaluation windows. Stable window IDs,
/// cell identity, and target class are checked exactly; prediction order does not affect the score.
///
/// The official-style score is recomputed from cell-level metrics on every draw.
/// One draw samples N participants with replacement from N unique participants;
/// each selected participant contributes all their expected cells. The seed,
/// method, confidence level, and replication count are returned for replay.
fn score_paired_cells(
    baseline_predictions: &[PairedLabeledPrediction],
    candidate_predictions: &[PairedLabeledPrediction],
    class_ids: &[u16],
    expected_cells: &[EvaluationCell],
) -> Result<(u32, u32, Vec<(EvaluationCell, u32, u32)>), PairedCellBootstrapError> {
    if baseline_predictions.is_empty() || candidate_predictions.is_empty() {
        return Err(PairedCellBootstrapError::EmptyPredictions);
    }

    fn window_inventory(
        predictions: &[PairedLabeledPrediction],
    ) -> Result<BTreeMap<String, (EvaluationCell, u16)>, PairedCellBootstrapError> {
        let mut inventory = BTreeMap::new();
        for (index, row) in predictions.iter().enumerate() {
            if row.window_id.trim().is_empty() {
                return Err(PairedCellBootstrapError::EmptyWindowId { index });
            }
            if inventory.insert(
                row.window_id.clone(),
                (row.cell.clone(), row.target_class),
            ).is_some() {
                return Err(PairedCellBootstrapError::DuplicateWindowId(row.window_id.clone()));
            }
        }
        Ok(inventory)
    }

    if window_inventory(baseline_predictions)? != window_inventory(candidate_predictions)? {
        return Err(PairedCellBootstrapError::PairedWindowInventoryMismatch);
    }

    let baseline_rows: Vec<LabeledPrediction> = baseline_predictions.iter().map(|row| {
        LabeledPrediction {
            cell: row.cell.clone(),
            target_class: row.target_class,
            predicted_class: row.predicted_class,
        }
    }).collect();
    let candidate_rows: Vec<LabeledPrediction> = candidate_predictions.iter().map(|row| {
        LabeledPrediction {
            cell: row.cell.clone(),
            target_class: row.target_class,
            predicted_class: row.predicted_class,
        }
    }).collect();

    let baseline = evaluate_track02_reference_compatible_cell_mean(
        &baseline_rows,
        class_ids,
        expected_cells,
    )
    .map_err(PairedCellBootstrapError::BaselineMetric)?;
    let candidate = evaluate_track02_reference_compatible_cell_mean(
        &candidate_rows,
        class_ids,
        expected_cells,
    )
    .map_err(PairedCellBootstrapError::CandidateMetric)?;

    if baseline.cells.len() != candidate.cells.len() {
        return Err(PairedCellBootstrapError::PairedWindowInventoryMismatch);
    }

    let mut paired_cells = Vec::with_capacity(baseline.cells.len());
    for (baseline_cell, candidate_cell) in baseline.cells.iter().zip(candidate.cells.iter()) {
        if baseline_cell.cell != candidate_cell.cell {
            return Err(PairedCellBootstrapError::PairedWindowInventoryMismatch);
        }
        paired_cells.push((
            baseline_cell.cell.clone(),
            baseline_cell.balanced_accuracy_ppm,
            candidate_cell.balanced_accuracy_ppm,
        ));
    }

    Ok((
        baseline.cell_mean_balanced_accuracy_ppm,
        candidate.cell_mean_balanced_accuracy_ppm,
        paired_cells,
    ))
}

fn validate_bootstrap_config(config: BootstrapConfig) -> Result<(), PairedCellBootstrapError> {
    if !(MIN_REPLICATES..=MAX_REPLICATES).contains(&config.replicates) {
        return Err(PairedCellBootstrapError::InvalidReplicateCount(config.replicates));
    }
    if !(500..=999).contains(&config.confidence_level_permille) {
        return Err(PairedCellBootstrapError::InvalidConfidenceLevel(
            config.confidence_level_permille,
        ));
    }
    Ok(())
}

fn finish_cell_bootstrap(
    baseline_score_ppm: u32,
    candidate_score_ppm: u32,
    participant_count: u32,
    cell_count: u32,
    config: BootstrapConfig,
    mut deltas: Vec<i64>,
    positive_delta_replicates: u32,
    method_id: &'static str,
) -> Result<CellMeanPairedBootstrapReport, PairedCellBootstrapError> {
    let delta_total = deltas.iter().try_fold(0_i128, |total, delta| {
        total.checked_add(i128::from(*delta))
            .ok_or(PairedCellBootstrapError::CounterOverflow)
    })?;
    let mean_bootstrap_delta_ppm =
        i64::try_from(delta_total / i128::from(config.replicates))
            .map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
    deltas.sort_unstable();
    let last_index = (deltas.len() - 1) as u128;
    let tail_permille = u128::from(1000 - config.confidence_level_permille);
    let lower_index = (tail_permille * last_index / 2000) as usize;
    let upper_index = ((2000 - tail_permille) * last_index / 2000) as usize;
    Ok(CellMeanPairedBootstrapReport {
        method_id,
        participant_count,
        cell_count,
        replicates: config.replicates,
        confidence_level_permille: config.confidence_level_permille,
        seed: config.seed,
        baseline_score_ppm,
        candidate_score_ppm,
        observed_delta_ppm: i64::from(candidate_score_ppm) - i64::from(baseline_score_ppm),
        mean_bootstrap_delta_ppm,
        ci_lower_delta_ppm: deltas[lower_index],
        ci_upper_delta_ppm: deltas[upper_index],
        positive_delta_replicates,
    })
}

/// Participant-cluster sensitivity analysis for the Track 02-style cell-mean score.
///
/// A draw samples N participants with replacement from N participants and retains
/// every sampled participant's complete set of session/context cells. This reflects
/// participant dependence and is a sensitivity analysis, not the competition site's
/// published default post-evaluation bootstrap, which samples independent cells.
pub fn paired_cell_mean_participant_bootstrap(
    baseline_predictions: &[PairedLabeledPrediction],
    candidate_predictions: &[PairedLabeledPrediction],
    class_ids: &[u16],
    expected_cells: &[EvaluationCell],
    config: BootstrapConfig,
) -> Result<CellMeanPairedBootstrapReport, PairedCellBootstrapError> {
    validate_bootstrap_config(config)?;
    let (baseline_score, candidate_score, paired_cells) = score_paired_cells(
        baseline_predictions, candidate_predictions, class_ids, expected_cells,
    )?;
    let mut participant_cells: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for (cell, baseline, candidate) in &paired_cells {
        participant_cells
            .entry(cell.participant_id.clone())
            .or_default()
            .push((*baseline, *candidate));
    }
    let participants: Vec<&str> = participant_cells.keys().map(String::as_str).collect();
    if participants.len() < 2 {
        return Err(PairedCellBootstrapError::TooFewParticipants {
            actual: participants.len(),
        });
    }
    let participant_count =
        u32::try_from(participants.len()).map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
    let cell_count =
        u32::try_from(paired_cells.len()).map_err(|_| PairedCellBootstrapError::CounterOverflow)?;

    let mut random_state = config.seed;
    let mut deltas = Vec::with_capacity(config.replicates as usize);
    let mut positive_delta_replicates = 0_u32;
    let n = u64::from(participant_count);
    for _ in 0..config.replicates {
        let mut baseline_total = 0_u128;
        let mut candidate_total = 0_u128;
        let mut sampled_cell_count = 0_u128;
        for _ in 0..n {
            let participant_index = sample_below(&mut random_state, n) as usize;
            let cells = participant_cells
                .get(participants[participant_index])
                .ok_or(PairedCellBootstrapError::CounterOverflow)?;
            for &(baseline, candidate) in cells {
                baseline_total = baseline_total.checked_add(u128::from(baseline))
                    .ok_or(PairedCellBootstrapError::CounterOverflow)?;
                candidate_total = candidate_total.checked_add(u128::from(candidate))
                    .ok_or(PairedCellBootstrapError::CounterOverflow)?;
                sampled_cell_count = sampled_cell_count.checked_add(1)
                    .ok_or(PairedCellBootstrapError::CounterOverflow)?;
            }
        }
        if sampled_cell_count == 0 {
            return Err(PairedCellBootstrapError::CounterOverflow);
        }
        let baseline_draw = i64::try_from(baseline_total / sampled_cell_count)
            .map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
        let candidate_draw = i64::try_from(candidate_total / sampled_cell_count)
            .map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
        let delta = candidate_draw - baseline_draw;
        if delta > 0 {
            positive_delta_replicates = positive_delta_replicates.checked_add(1)
                .ok_or(PairedCellBootstrapError::CounterOverflow)?;
        }
        deltas.push(delta);
    }
    finish_cell_bootstrap(
        baseline_score, candidate_score, participant_count, cell_count, config, deltas,
        positive_delta_replicates,
        "paired-participant-cluster-cell-mean-percentile-order-statistic-v1",
    )
}

/// Challenge-method sensitivity/bootstrap: resample participant × session × context
/// cells directly, using the same resampled cells for baseline and candidate.
///
/// The 2026 competition website describes these cell scores as the independent
/// units for its planned post-evaluation intervals and rank stability analysis.
/// This helper returns a paired score-difference interval under that convention.
pub fn paired_cell_mean_unit_bootstrap(
    baseline_predictions: &[PairedLabeledPrediction],
    candidate_predictions: &[PairedLabeledPrediction],
    class_ids: &[u16],
    expected_cells: &[EvaluationCell],
    config: BootstrapConfig,
) -> Result<CellMeanPairedBootstrapReport, PairedCellBootstrapError> {
    validate_bootstrap_config(config)?;
    let (baseline_score, candidate_score, paired_cells) = score_paired_cells(
        baseline_predictions, candidate_predictions, class_ids, expected_cells,
    )?;
    if paired_cells.len() < 2 {
        return Err(PairedCellBootstrapError::TooFewCells {
            actual: paired_cells.len(),
        });
    }
    let mut participants = BTreeSet::new();
    for (cell, _, _) in &paired_cells {
        participants.insert(cell.participant_id.as_str());
    }
    let participant_count =
        u32::try_from(participants.len()).map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
    let cell_count =
        u32::try_from(paired_cells.len()).map_err(|_| PairedCellBootstrapError::CounterOverflow)?;

    let mut random_state = config.seed;
    let mut deltas = Vec::with_capacity(config.replicates as usize);
    let mut positive_delta_replicates = 0_u32;
    let n = u64::from(cell_count);
    for _ in 0..config.replicates {
        let mut baseline_total = 0_u128;
        let mut candidate_total = 0_u128;
        for _ in 0..n {
            let index = sample_below(&mut random_state, n) as usize;
            baseline_total = baseline_total
                .checked_add(u128::from(paired_cells[index].1))
                .ok_or(PairedCellBootstrapError::CounterOverflow)?;
            candidate_total = candidate_total
                .checked_add(u128::from(paired_cells[index].2))
                .ok_or(PairedCellBootstrapError::CounterOverflow)?;
        }
        let baseline_draw = i64::try_from(baseline_total / u128::from(n))
            .map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
        let candidate_draw = i64::try_from(candidate_total / u128::from(n))
            .map_err(|_| PairedCellBootstrapError::CounterOverflow)?;
        let delta = candidate_draw - baseline_draw;
        if delta > 0 {
            positive_delta_replicates = positive_delta_replicates.checked_add(1)
                .ok_or(PairedCellBootstrapError::CounterOverflow)?;
        }
        deltas.push(delta);
    }
    finish_cell_bootstrap(
        baseline_score, candidate_score, participant_count, cell_count, config, deltas,
        positive_delta_replicates,
        "paired-cell-unit-cell-mean-percentile-order-statistic-v1",
    )
}

/// Maximum number of model × unit × bootstrap evaluations per invocation.
/// This bounds accidental quadratic/explosive runs in a dependency-free core.
const MAX_LEADERBOARD_BOOTSTRAP_OPERATIONS: u128 = 100_000_000;
const MAX_LEADERBOARD_BOOTSTRAP_VALUES: u128 = 10_000_000;
const MAX_LEADERBOARD_PAIRWISE_COMPARISONS: u128 = 100_000;

/// One model's score on one frozen, independent evaluation unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderboardUnitScore {
    pub model_id: String,
    pub unit_id: String,
    pub score_ppm: u32,
}

/// Configuration for paired full-field inference. Alpha is expressed in permille:
/// 50 means a family-wise 5% Holm threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaderboardInferenceConfig {
    pub bootstrap: BootstrapConfig,
    pub familywise_alpha_permille: u16,
}

impl Default for LeaderboardInferenceConfig {
    fn default() -> Self {
        Self {
            bootstrap: BootstrapConfig::default(),
            familywise_alpha_permille: 50,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaderboardInferenceError {
    EmptyScores,
    EmptyIdentity { index: usize, field: &'static str },
    EmptyExpectedModelInventory,
    EmptyExpectedUnitInventory,
    DuplicateExpectedModelId(String),
    DuplicateExpectedUnitId(String),
    UnexpectedModelId(String),
    UnexpectedUnitId(String),
    DuplicateModelUnit { model_id: String, unit_id: String },
    IncompleteScoreGrid { expected: usize, actual: usize },
    TooFewModels { actual: usize },
    TooFewUnits { actual: usize },
    ScoreOutOfRange { index: usize, score_ppm: u32 },
    InvalidAlpha(u16),
    InvalidReplicateCount(u32),
    InvalidConfidenceLevel(u16),
    InvalidPValue { index: usize, value_ppm: u32 },
    EmptyPValues,
    ComputationBudgetExceeded,
    CounterOverflow,
}

impl fmt::Display for LeaderboardInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LeaderboardInferenceError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRankStability {
    pub model_id: String,
    pub point_score_ppm: u32,
    pub mean_bootstrap_score_ppm: u32,
    /// Mean rank × 1000, rounded down. Tied scores share competition rank.
    pub mean_rank_milli: u32,
    /// Normalized distribution for ranks 1..=model_count; values sum to exactly 1000.
    /// Largest-remainder apportionment removes fixed-point rounding drift.
    pub rank_frequency_permille: Vec<u16>,
    pub top1_frequency_permille: u16,
    pub top3_frequency_permille: u16,
    pub top5_frequency_permille: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairwiseModelInference {
    /// IDs are sorted lexicographically; delta is model_a minus model_b.
    pub model_a_id: String,
    pub model_b_id: String,
    pub point_delta_ppm: i64,
    pub mean_bootstrap_delta_ppm: i64,
    pub ci_lower_delta_ppm: i64,
    pub ci_upper_delta_ppm: i64,
    pub p_boot_ppm: u32,
    pub p_holm_ppm: u32,
    pub interval_includes_zero: bool,
    pub significant_after_holm: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderboardInferenceReport {
    pub method_id: &'static str,
    pub independent_unit_count: u32,
    pub model_count: u32,
    pub replicates: u32,
    pub confidence_level_permille: u16,
    pub familywise_alpha_permille: u16,
    pub seed: u64,
    pub models: Vec<ModelRankStability>,
    pub pairwise: Vec<PairwiseModelInference>,
}

/// Holm step-down adjustment for p-values scaled to [0, 1_000_000].
///
/// Returns adjusted p-values in original input order. The input values should
/// already be conservative fixed-point estimates of the raw p-values.
pub fn holm_adjust_p_values(
    p_values_ppm: &[u32],
) -> Result<Vec<u32>, LeaderboardInferenceError> {
    if p_values_ppm.is_empty() {
        return Err(LeaderboardInferenceError::EmptyPValues);
    }
    for (index, value) in p_values_ppm.iter().copied().enumerate() {
        if value > SCORE_SCALE_PPM {
            return Err(LeaderboardInferenceError::InvalidPValue {
                index,
                value_ppm: value,
            });
        }
    }
    let mut order: Vec<usize> = (0..p_values_ppm.len()).collect();
    order.sort_by_key(|&index| (p_values_ppm[index], index));
    let mut adjusted = vec![0_u32; p_values_ppm.len()];
    let mut running_max = 0_u32;
    let count = p_values_ppm.len();
    for (rank, &index) in order.iter().enumerate() {
        let remaining = u128::try_from(count - rank)
            .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        let product = u128::from(p_values_ppm[index])
            .checked_mul(remaining)
            .ok_or(LeaderboardInferenceError::CounterOverflow)?;
        let capped = product.min(u128::from(SCORE_SCALE_PPM));
        let candidate = u32::try_from(capped)
            .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        running_max = running_max.max(candidate);
        adjusted[index] = running_max;
    }
    Ok(adjusted)
}

fn percentile_interval(
    deltas: &mut [i64],
    confidence_level_permille: u16,
) -> Result<(i64, i64), LeaderboardInferenceError> {
    if deltas.is_empty() {
        return Err(LeaderboardInferenceError::EmptyScores);
    }
    deltas.sort_unstable();
    let last_index = u128::try_from(deltas.len() - 1)
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    let tail_permille = u128::from(1000 - confidence_level_permille);
    let lower = (tail_permille * last_index / 2000) as usize;
    let upper = ((2000 - tail_permille) * last_index / 2000) as usize;
    Ok((deltas[lower], deltas[upper]))
}

fn conservative_two_sided_bootstrap_p_ppm(
    deltas: &[i64],
) -> Result<u32, LeaderboardInferenceError> {
    if deltas.is_empty() {
        return Err(LeaderboardInferenceError::EmptyScores);
    }
    let mut nonpositive = 0_u128;
    let mut nonnegative = 0_u128;
    for delta in deltas {
        if *delta <= 0 {
            nonpositive += 1;
        }
        if *delta >= 0 {
            nonnegative += 1;
        }
    }
    // Match the challenge page's doubled smaller tail. Integer ceil avoids
    // understating a p-value through quantization. Zero deltas count in both tails.
    let tail = nonpositive.min(nonnegative);
    let numerator = tail
        .checked_mul(2)
        .and_then(|v| v.checked_mul(u128::from(SCORE_SCALE_PPM)))
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    let denominator = u128::try_from(deltas.len())
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    let rounded_up = numerator
        .checked_add(denominator - 1)
        .ok_or(LeaderboardInferenceError::CounterOverflow)?
        / denominator;
    u32::try_from(rounded_up.min(u128::from(SCORE_SCALE_PPM)))
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)
}

/// Compare a complete field of models on paired independent-unit scores.
///
/// The caller must supply model and independent-unit inventories from the frozen
/// benchmark manifest. The function rejects scores for unknown inventory IDs and
/// any absent model × unit pair, including a unit omitted from every model's rows.
/// A bootstrap draw resamples units once and applies the same draw to all models,
/// enabling paired differences and full-field rank stability. Higher scores are
/// assumed better (appropriate for Track 02 balanced accuracy). Ties share
/// competition rank rather than being broken arbitrarily by model name.
///
/// This is a post-evaluation analysis: unit scores must be frozen before calling
/// this function. It does not select a model, validate data provenance, reproduce
/// the organizers' float64 implementation bit-for-bit, or account for tuning against
/// the same evaluation set.
pub fn analyze_leaderboard(
    unit_scores: &[LeaderboardUnitScore],
    expected_model_ids: &[&str],
    expected_unit_ids: &[&str],
    config: LeaderboardInferenceConfig,
) -> Result<LeaderboardInferenceReport, LeaderboardInferenceError> {
    if unit_scores.is_empty() {
        return Err(LeaderboardInferenceError::EmptyScores);
    }
    if config.familywise_alpha_permille == 0 || config.familywise_alpha_permille > 500 {
        return Err(LeaderboardInferenceError::InvalidAlpha(
            config.familywise_alpha_permille,
        ));
    }
    if !(MIN_REPLICATES..=MAX_REPLICATES).contains(&config.bootstrap.replicates) {
        return Err(LeaderboardInferenceError::InvalidReplicateCount(
            config.bootstrap.replicates,
        ));
    }
    if !(500..=999).contains(&config.bootstrap.confidence_level_permille) {
        return Err(LeaderboardInferenceError::InvalidConfidenceLevel(
            config.bootstrap.confidence_level_permille,
        ));
    }

    if expected_model_ids.is_empty() {
        return Err(LeaderboardInferenceError::EmptyExpectedModelInventory);
    }
    if expected_unit_ids.is_empty() {
        return Err(LeaderboardInferenceError::EmptyExpectedUnitInventory);
    }
    let mut expected_models = BTreeSet::<String>::new();
    for (index, &model_id) in expected_model_ids.iter().enumerate() {
        if model_id.trim().is_empty() {
            return Err(LeaderboardInferenceError::EmptyIdentity {
                index,
                field: "expected_model_id",
            });
        }
        if !expected_models.insert(model_id.to_owned()) {
            return Err(LeaderboardInferenceError::DuplicateExpectedModelId(
                model_id.to_owned(),
            ));
        }
    }
    let mut expected_units = BTreeSet::<String>::new();
    for (index, &unit_id) in expected_unit_ids.iter().enumerate() {
        if unit_id.trim().is_empty() {
            return Err(LeaderboardInferenceError::EmptyIdentity {
                index,
                field: "expected_unit_id",
            });
        }
        if !expected_units.insert(unit_id.to_owned()) {
            return Err(LeaderboardInferenceError::DuplicateExpectedUnitId(
                unit_id.to_owned(),
            ));
        }
    }
    if expected_models.len() < 2 {
        return Err(LeaderboardInferenceError::TooFewModels {
            actual: expected_models.len(),
        });
    }
    if expected_units.len() < 2 {
        return Err(LeaderboardInferenceError::TooFewUnits {
            actual: expected_units.len(),
        });
    }

    let mut seen = BTreeMap::<(String, String), u32>::new();
    for (index, row) in unit_scores.iter().enumerate() {
        if row.model_id.trim().is_empty() {
            return Err(LeaderboardInferenceError::EmptyIdentity {
                index,
                field: "model_id",
            });
        }
        if row.unit_id.trim().is_empty() {
            return Err(LeaderboardInferenceError::EmptyIdentity {
                index,
                field: "unit_id",
            });
        }
        if !expected_models.contains(&row.model_id) {
            return Err(LeaderboardInferenceError::UnexpectedModelId(
                row.model_id.clone(),
            ));
        }
        if !expected_units.contains(&row.unit_id) {
            return Err(LeaderboardInferenceError::UnexpectedUnitId(
                row.unit_id.clone(),
            ));
        }
        if row.score_ppm > SCORE_SCALE_PPM {
            return Err(LeaderboardInferenceError::ScoreOutOfRange {
                index,
                score_ppm: row.score_ppm,
            });
        }
        let key = (row.model_id.clone(), row.unit_id.clone());
        if seen.insert(key, row.score_ppm).is_some() {
            return Err(LeaderboardInferenceError::DuplicateModelUnit {
                model_id: row.model_id.clone(),
                unit_id: row.unit_id.clone(),
            });
        }
    }

    let model_ids: Vec<String> = expected_models.iter().cloned().collect();
    let unit_ids: Vec<String> = expected_units.iter().cloned().collect();
    let expected = model_ids.len().checked_mul(unit_ids.len())
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    if seen.len() != expected {
        return Err(LeaderboardInferenceError::IncompleteScoreGrid {
            expected,
            actual: seen.len(),
        });
    }
    let model_count = model_ids.len();
    let unit_count = unit_ids.len();
    let model_count_u128 = u128::try_from(model_count)
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    let unit_count_u128 = u128::try_from(unit_count)
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    let replicate_count_u128 = u128::from(config.bootstrap.replicates);
    let point_operations = model_count_u128
        .checked_mul(unit_count_u128)
        .and_then(|v| v.checked_mul(replicate_count_u128))
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    let pairwise_count = model_count_u128
        .checked_mul(model_count_u128 - 1)
        .ok_or(LeaderboardInferenceError::CounterOverflow)? / 2;
    let pairwise_operations = pairwise_count
        .checked_mul(replicate_count_u128)
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    let rank_operations = model_count_u128
        .checked_mul(model_count_u128)
        .and_then(|v| v.checked_mul(replicate_count_u128))
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    let operations = point_operations
        .checked_add(pairwise_operations)
        .and_then(|v| v.checked_add(rank_operations))
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    let stored_values = model_count_u128
        .checked_mul(replicate_count_u128)
        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
    if operations > MAX_LEADERBOARD_BOOTSTRAP_OPERATIONS
        || stored_values > MAX_LEADERBOARD_BOOTSTRAP_VALUES
        || pairwise_count > MAX_LEADERBOARD_PAIRWISE_COMPARISONS
    {
        return Err(LeaderboardInferenceError::ComputationBudgetExceeded);
    }

    let model_count = model_ids.len();
    let unit_count = unit_ids.len();
    let model_index: BTreeMap<&str, usize> = model_ids.iter().enumerate()
        .map(|(index, id)| (id.as_str(), index)).collect();
    let unit_index: BTreeMap<&str, usize> = unit_ids.iter().enumerate()
        .map(|(index, id)| (id.as_str(), index)).collect();
    let mut matrix = vec![vec![0_u32; unit_count]; model_count];
    for ((model_id, unit_id), score) in &seen {
        let mi = *model_index.get(model_id.as_str())
            .ok_or(LeaderboardInferenceError::CounterOverflow)?;
        let ui = *unit_index.get(unit_id.as_str())
            .ok_or(LeaderboardInferenceError::CounterOverflow)?;
        matrix[mi][ui] = *score;
    }

    let unit_count_u128 = u128::try_from(unit_count)
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    let mut point_scores = Vec::with_capacity(model_count);
    for values in &matrix {
        let total = values.iter().try_fold(0_u128, |sum, score| {
            sum.checked_add(u128::from(*score))
                .ok_or(LeaderboardInferenceError::CounterOverflow)
        })?;
        point_scores.push(u32::try_from(total / unit_count_u128)
            .map_err(|_| LeaderboardInferenceError::CounterOverflow)?);
    }

    let replicates = config.bootstrap.replicates as usize;
    let mut draws = vec![vec![0_u32; replicates]; model_count];
    let mut random_state = config.bootstrap.seed;
    let unit_count_u64 = u64::try_from(unit_count)
        .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
    for replicate in 0..replicates {
        let mut sums = vec![0_u128; model_count];
        for _ in 0..unit_count_u64 {
            let selected = sample_below(&mut random_state, unit_count_u64) as usize;
            for model in 0..model_count {
                sums[model] = sums[model]
                    .checked_add(u128::from(matrix[model][selected]))
                    .ok_or(LeaderboardInferenceError::CounterOverflow)?;
            }
        }
        for model in 0..model_count {
            draws[model][replicate] = u32::try_from(sums[model] / unit_count_u128)
                .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        }
    }

    let mut rank_sums = vec![0_u128; model_count];
    let mut rank_counts = vec![vec![0_u64; model_count]; model_count];
    let mut top_counts = vec![[0_u64; 3]; model_count];
    for replicate in 0..replicates {
        for model in 0..model_count {
            let rank = 1_u32 + (0..model_count)
                .filter(|&other| draws[other][replicate] > draws[model][replicate])
                .count() as u32;
            rank_sums[model] = rank_sums[model]
                .checked_add(u128::from(rank))
                .ok_or(LeaderboardInferenceError::CounterOverflow)?;
            let rank_slot = usize::try_from(rank - 1)
                .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
            rank_counts[model][rank_slot] = rank_counts[model][rank_slot]
                .checked_add(1)
                .ok_or(LeaderboardInferenceError::CounterOverflow)?;
            for (slot, cutoff) in [1_u32, 3, 5].iter().copied().enumerate() {
                if rank <= cutoff {
                    top_counts[model][slot] = top_counts[model][slot]
                        .checked_add(1)
                        .ok_or(LeaderboardInferenceError::CounterOverflow)?;
                }
            }
        }
    }

    let mut model_reports = Vec::with_capacity(model_count);
    for model in 0..model_count {
        let mean_total = draws[model].iter().try_fold(0_u128, |sum, score| {
            sum.checked_add(u128::from(*score))
                .ok_or(LeaderboardInferenceError::CounterOverflow)
        })?;
        let mean_bootstrap_score_ppm = u32::try_from(
            mean_total / u128::from(config.bootstrap.replicates),
        ).map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        let mean_rank_milli = u32::try_from(
            rank_sums[model].checked_mul(1000)
                .ok_or(LeaderboardInferenceError::CounterOverflow)?
                / u128::from(config.bootstrap.replicates),
        ).map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        let mut frequencies = [0_u16; 3];
        for slot in 0..3 {
            frequencies[slot] = u16::try_from(
                (u128::from(top_counts[model][slot]) * 1000)
                    / u128::from(config.bootstrap.replicates),
            ).map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
        }
        let mut rank_frequency_permille = vec![0_u16; model_count];
        let mut remainders = Vec::with_capacity(model_count);
        let mut allocated = 0_u32;
        for rank_index in 0..model_count {
            let numerator = u128::from(rank_counts[model][rank_index]) * 1000;
            let base = u32::try_from(numerator / u128::from(config.bootstrap.replicates))
                .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
            let remainder = numerator % u128::from(config.bootstrap.replicates);
            rank_frequency_permille[rank_index] = u16::try_from(base)
                .map_err(|_| LeaderboardInferenceError::CounterOverflow)?;
            allocated = allocated.checked_add(base)
                .ok_or(LeaderboardInferenceError::CounterOverflow)?;
            remainders.push((rank_index, remainder));
        }
        remainders.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        let remaining = 1000_u32.checked_sub(allocated)
            .ok_or(LeaderboardInferenceError::CounterOverflow)?;
        for (rank_index, _) in remainders.iter().take(remaining as usize) {
            rank_frequency_permille[*rank_index] = rank_frequency_permille[*rank_index]
                .checked_add(1)
                .ok_or(LeaderboardInferenceError::CounterOverflow)?;
        }
        model_reports.push(ModelRankStability {
            model_id: model_ids[model].clone(),
            point_score_ppm: point_scores[model],
            mean_bootstrap_score_ppm,
            mean_rank_milli,
            rank_frequency_permille,
            top1_frequency_permille: frequencies[0],
            top3_frequency_permille: frequencies[1],
            top5_frequency_permille: frequencies[2],
        });
    }

    let mut pairs = Vec::<PairwiseModelInference>::new();
    let mut raw_p_values = Vec::<u32>::new();
    for model_a in 0..model_count {
        for model_b in (model_a + 1)..model_count {
            let point_delta_ppm = i64::from(point_scores[model_a])
                - i64::from(point_scores[model_b]);
            let mut deltas = Vec::with_capacity(replicates);
            for replicate in 0..replicates {
                deltas.push(i64::from(draws[model_a][replicate])
                    - i64::from(draws[model_b][replicate]));
            }
            let p_boot_ppm = conservative_two_sided_bootstrap_p_ppm(&deltas)?;
            let (ci_lower_delta_ppm, ci_upper_delta_ppm) = percentile_interval(
                &mut deltas,
                config.bootstrap.confidence_level_permille,
            )?;
            raw_p_values.push(p_boot_ppm);
            pairs.push(PairwiseModelInference {
                model_a_id: model_ids[model_a].clone(),
                model_b_id: model_ids[model_b].clone(),
                point_delta_ppm,
                mean_bootstrap_delta_ppm: i64::try_from(
                    draws[model_a].iter().zip(draws[model_b].iter())
                        .try_fold(0_i128, |sum, (a, b)| {
                            sum.checked_add(i128::from(*a) - i128::from(*b))
                                .ok_or(LeaderboardInferenceError::CounterOverflow)
                        })? / i128::from(config.bootstrap.replicates),
                ).map_err(|_| LeaderboardInferenceError::CounterOverflow)?,
                ci_lower_delta_ppm,
                ci_upper_delta_ppm,
                p_boot_ppm,
                p_holm_ppm: 0,
                interval_includes_zero: ci_lower_delta_ppm <= 0 && ci_upper_delta_ppm >= 0,
                significant_after_holm: false,
            });
        }
    }

    let adjusted = holm_adjust_p_values(&raw_p_values)?;
    let alpha_ppm = u32::from(config.familywise_alpha_permille) * 1000;
    for (pair, p_holm_ppm) in pairs.iter_mut().zip(adjusted.into_iter()) {
        pair.p_holm_ppm = p_holm_ppm;
        pair.significant_after_holm = p_holm_ppm <= alpha_ppm;
    }

    Ok(LeaderboardInferenceReport {
        method_id: "paired-unit-bootstrap-holm-shared-ranks-v1",
        independent_unit_count: u32::try_from(unit_count)
            .map_err(|_| LeaderboardInferenceError::CounterOverflow)?,
        model_count: u32::try_from(model_count)
            .map_err(|_| LeaderboardInferenceError::CounterOverflow)?,
        replicates: config.bootstrap.replicates,
        confidence_level_permille: config.bootstrap.confidence_level_permille,
        familywise_alpha_permille: config.familywise_alpha_permille,
        seed: config.bootstrap.seed,
        models: model_reports,
        pairwise: pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BootstrapConfig {
        BootstrapConfig {
            replicates: 1_000,
            confidence_level_permille: 950,
            seed: 42,
        }
    }

    fn leader_config() -> LeaderboardInferenceConfig {
        LeaderboardInferenceConfig {
            bootstrap: BootstrapConfig {
                replicates: 1_000,
                confidence_level_permille: 950,
                seed: 2026,
            },
            familywise_alpha_permille: 50,
        }
    }

    fn leader_score(model: &str, unit: &str, ppm: u32) -> LeaderboardUnitScore {
        LeaderboardUnitScore {
            model_id: model.into(),
            unit_id: unit.into(),
            score_ppm: ppm,
        }
    }

    fn row(id: &str, baseline: u32, candidate: u32) -> PairedParticipantScore {
        PairedParticipantScore {
            participant_id: id.to_owned(),
            baseline_score_ppm: baseline,
            candidate_score_ppm: candidate,
        }
    }

    fn cell(participant: &str, session: &str, context: &str) -> EvaluationCell {
        EvaluationCell {
            participant_id: participant.into(),
            session_id: session.into(),
            context_id: context.into(),
        }
    }

    fn prediction(
        window_id: &str,
        cell: &EvaluationCell,
        target: u16,
        predicted: u16,
    ) -> PairedLabeledPrediction {
        PairedLabeledPrediction {
            window_id: window_id.into(),
            cell: cell.clone(),
            target_class: target,
            predicted_class: predicted,
        }
    }

    #[test]
    fn cluster_bootstrap_recomputes_official_cell_weighting_and_replays() {
        let a1 = cell("p-a", "s-1", "graz");
        let a2 = cell("p-a", "s-2", "graz");
        let b1 = cell("p-b", "s-1", "graz");
        let expected = vec![a1.clone(), a2.clone(), b1.clone()];
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for (index, c) in [&a1, &a2, &b1].iter().enumerate() {
            baseline.push(prediction(&format!("w{index}-0"), c, 0, 0));
            baseline.push(prediction(&format!("w{index}-1"), c, 1, 0));
        }
        for (index, c) in [&a1, &a2].iter().enumerate() {
            candidate.push(prediction(&format!("w{index}-0"), c, 0, 0));
            candidate.push(prediction(&format!("w{index}-1"), c, 1, 1));
        }
        candidate.push(prediction("w2-0", &b1, 0, 1));
        candidate.push(prediction("w2-1", &b1, 1, 0));

        let config = BootstrapConfig {
            replicates: 1_000,
            confidence_level_permille: 950,
            seed: 77,
        };
        let report = paired_cell_mean_participant_bootstrap(
            &baseline, &candidate, &[0, 1], &expected, config,
        ).unwrap();
        // Per-cell scores are baseline .5, candidate [1, 1, 0].
        // Deltas are averaged over 3 cells, not over the 2 participants.
        assert_eq!(report.participant_count, 2);
        assert_eq!(report.cell_count, 3);
        assert_eq!(report.baseline_score_ppm, 500_000);
        assert_eq!(report.candidate_score_ppm, 666_666);
        assert_eq!(report.observed_delta_ppm, 166_666);
        assert_eq!(report.seed, 77);
        assert_eq!(
            report.method_id,
            "paired-participant-cluster-cell-mean-percentile-order-statistic-v1"
        );
        assert_eq!(
            report,
            paired_cell_mean_participant_bootstrap(
                &baseline, &candidate, &[0, 1], &expected, config,
            ).unwrap()
        );
    }

    #[test]
    fn cell_unit_bootstrap_matches_paired_primary_metric_and_exact_interval() {
        let a = cell("p-a", "s-1", "graz");
        let b = cell("p-b", "s-1", "graz");
        let baseline = vec![
            prediction("a-0", &a, 0, 0), prediction("a-1", &a, 1, 0),
            prediction("b-0", &b, 0, 0), prediction("b-1", &b, 1, 0),
        ];
        let candidate = vec![
            prediction("a-0", &a, 0, 0), prediction("a-1", &a, 1, 1),
            prediction("b-0", &b, 0, 0), prediction("b-1", &b, 1, 1),
        ];
        let report = paired_cell_mean_unit_bootstrap(
            &baseline, &candidate, &[0, 1], &[a, b], config(),
        ).unwrap();
        assert_eq!(report.method_id, "paired-cell-unit-cell-mean-percentile-order-statistic-v1");
        assert_eq!(report.participant_count, 2);
        assert_eq!(report.cell_count, 2);
        assert_eq!(report.baseline_score_ppm, 500_000);
        assert_eq!(report.candidate_score_ppm, 1_000_000);
        assert_eq!(report.observed_delta_ppm, 500_000);
        assert_eq!(report.ci_lower_delta_ppm, 500_000);
        assert_eq!(report.ci_upper_delta_ppm, 500_000);
    }

    #[test]
    fn cluster_bootstrap_rejects_changed_ground_truth_inventory() {
        let a = cell("p-a", "s-1", "graz");
        let b = cell("p-b", "s-1", "graz");
        let baseline = vec![
            prediction("w-a", &a, 0, 0),
            prediction("w-b", &b, 0, 0),
        ];
        let candidate = vec![
            prediction("w-a", &a, 1, 1),
            prediction("w-b", &b, 0, 0),
        ];
        assert_eq!(
            paired_cell_mean_participant_bootstrap(
                &baseline, &candidate, &[0, 1], &[a, b], config(),
            ),
            Err(PairedCellBootstrapError::PairedWindowInventoryMismatch)
        );
    }

    #[test]
    fn paired_cell_bootstrap_rejects_duplicate_and_blank_window_ids() {
        let a = cell("p-a", "s-1", "graz");
        let b = cell("p-b", "s-1", "graz");
        let baseline_duplicate = vec![
            prediction("same-id", &a, 0, 0),
            prediction("same-id", &b, 0, 0),
        ];
        assert_eq!(
            paired_cell_mean_unit_bootstrap(
                &baseline_duplicate, &baseline_duplicate, &[0, 1], &[a.clone(), b.clone()], config(),
            ),
            Err(PairedCellBootstrapError::DuplicateWindowId("same-id".into()))
        );
        let baseline_blank = vec![
            prediction("  ", &a, 0, 0),
            prediction("good-id", &b, 0, 0),
        ];
        assert_eq!(
            paired_cell_mean_participant_bootstrap(
                &baseline_blank, &baseline_blank, &[0, 1], &[a, b], config(),
            ),
            Err(PairedCellBootstrapError::EmptyWindowId { index: 0 })
        );
    }

    #[test]
    fn cluster_bootstrap_requires_two_independent_participants() {
        let a1 = cell("p-a", "s-1", "graz");
        let a2 = cell("p-a", "s-2", "graz");
        let baseline = vec![
            prediction("w1-0", &a1, 0, 0), prediction("w1-1", &a1, 1, 1),
            prediction("w2-0", &a2, 0, 0), prediction("w2-1", &a2, 1, 1),
        ];
        assert_eq!(
            paired_cell_mean_participant_bootstrap(
                &baseline, &baseline, &[0, 1], &[a1, a2], config(),
            ),
            Err(PairedCellBootstrapError::TooFewParticipants { actual: 1 })
        );
    }

    #[test]
    fn uniform_positive_paired_delta_has_exact_positive_interval() {
        let scores = [
            row("p-01", 500_000, 600_000),
            row("p-02", 650_000, 750_000),
            row("p-03", 700_000, 800_000),
        ];
        let report = paired_participant_bootstrap(&scores, config()).unwrap();
        assert_eq!(report.participant_count, 3);
        assert_eq!(report.method_id, "paired-participant-bootstrap-percentile-order-statistic-v1");
        assert_eq!(report.seed, 42);
        assert_eq!(report.mean_paired_delta_ppm, 100_000);
        assert_eq!(report.ci_lower_delta_ppm, 100_000);
        assert_eq!(report.ci_upper_delta_ppm, 100_000);
        assert_eq!(report.positive_delta_replicates, 1_000);
    }

    #[test]
    fn identical_methods_produce_zero_interval() {
        let scores = [
            row("p-01", 500_000, 500_000),
            row("p-02", 800_000, 800_000),
            row("p-03", 250_000, 250_000),
        ];
        let report = paired_participant_bootstrap(&scores, config()).unwrap();
        assert_eq!(report.mean_paired_delta_ppm, 0);
        assert_eq!(report.ci_lower_delta_ppm, 0);
        assert_eq!(report.ci_upper_delta_ppm, 0);
        assert_eq!(report.positive_delta_replicates, 0);
    }

    #[test]
    fn same_seed_reproduces_interval_exactly() {
        let scores = [
            row("p-01", 500_000, 610_000),
            row("p-02", 550_000, 540_000),
            row("p-03", 650_000, 710_000),
            row("p-04", 600_000, 640_000),
            row("p-05", 400_000, 455_000),
        ];
        assert_eq!(
            paired_participant_bootstrap(&scores, config()).unwrap(),
            paired_participant_bootstrap(&scores, config()).unwrap()
        );
    }

    #[test]
    fn uniformly_worse_candidate_has_negative_interval() {
        let scores = [
            row("p-01", 700_000, 600_000),
            row("p-02", 750_000, 650_000),
            row("p-03", 800_000, 700_000),
        ];
        let report = paired_participant_bootstrap(&scores, config()).unwrap();
        assert_eq!(report.ci_lower_delta_ppm, -100_000);
        assert_eq!(report.ci_upper_delta_ppm, -100_000);
    }

    #[test]
    fn rejects_single_participant_duplicate_ids_and_invalid_scores() {
        assert_eq!(
            paired_participant_bootstrap(&[row("p-01", 1, 1)], config()),
            Err(PairedBootstrapError::TooFewParticipants { actual: 1 })
        );
        assert_eq!(
            paired_participant_bootstrap(&[row("p-01", 1, 1), row("p-01", 2, 2)], config()),
            Err(PairedBootstrapError::DuplicateParticipantId("p-01".into()))
        );
        assert_eq!(
            paired_participant_bootstrap(&[row("p-01", 1_000_001, 1), row("p-02", 1, 1)], config()),
            Err(PairedBootstrapError::ScoreOutOfRange {
                index: 0,
                candidate: false,
                score_ppm: 1_000_001
            })
        );
    }

    #[test]
    fn rejects_invalid_bootstrap_configuration() {
        let scores = [row("p-01", 1, 2), row("p-02", 1, 2)];
        let bad_replicates = BootstrapConfig { replicates: 99, ..config() };
        assert_eq!(
            paired_participant_bootstrap(&scores, bad_replicates),
            Err(PairedBootstrapError::InvalidReplicateCount(99))
        );
        let bad_confidence = BootstrapConfig { confidence_level_permille: 1000, ..config() };
        assert_eq!(
            paired_participant_bootstrap(&scores, bad_confidence),
            Err(PairedBootstrapError::InvalidConfidenceLevel(1000))
        );
    }

    #[test]
    fn holm_adjustment_matches_step_down_fixture_and_preserves_input_order() {
        assert_eq!(
            holm_adjust_p_values(&[30_000, 10_000, 20_000]).unwrap(),
            vec![40_000, 30_000, 40_000]
        );
        assert_eq!(
            holm_adjust_p_values(&[1_000_000, 0, 500_000]).unwrap(),
            vec![1_000_000, 0, 1_000_000]
        );
        assert_eq!(holm_adjust_p_values(&[]), Err(LeaderboardInferenceError::EmptyPValues));
        assert_eq!(
            holm_adjust_p_values(&[1_000_001]),
            Err(LeaderboardInferenceError::InvalidPValue { index: 0, value_ppm: 1_000_001 })
        );
    }

    #[test]
    fn leaderboard_inference_detects_clear_winner_and_holm_significance() {
        let mut rows = Vec::new();
        for (unit, _) in ["u1", "u2", "u3", "u4"].iter().enumerate() {
            let id = format!("u{}", unit + 1);
            rows.push(leader_score("alpha", &id, 900_000));
            rows.push(leader_score("beta", &id, 700_000));
            rows.push(leader_score("gamma", &id, 500_000));
        }
        let report = analyze_leaderboard(&rows, &["alpha", "beta", "gamma"], &["u1", "u2", "u3", "u4"], leader_config()).unwrap();
        assert_eq!(report.model_count, 3);
        assert_eq!(report.independent_unit_count, 4);
        assert_eq!(report.models[0].model_id, "alpha");
        assert_eq!(report.models[0].point_score_ppm, 900_000);
        assert_eq!(report.models[0].top1_frequency_permille, 1_000);
        assert_eq!(report.models[1].mean_rank_milli, 2_000);
        assert_eq!(report.models[2].mean_rank_milli, 3_000);
        assert!(report.models.iter().all(|model| model.rank_frequency_permille.iter().map(|v| u32::from(*v)).sum::<u32>() == 1_000));
        assert_eq!(report.pairwise.len(), 3);
        assert!(report.pairwise.iter().all(|p| p.significant_after_holm));
        assert!(report.pairwise.iter().all(|p| !p.interval_includes_zero));
    }

    #[test]
    fn leaderboard_ties_share_ranks_and_are_not_declared_significant() {
        let rows = vec![
            leader_score("alpha", "u1", 500_000),
            leader_score("alpha", "u2", 700_000),
            leader_score("beta", "u1", 500_000),
            leader_score("beta", "u2", 700_000),
            leader_score("gamma", "u1", 200_000),
            leader_score("gamma", "u2", 200_000),
        ];
        let report = analyze_leaderboard(&rows, &["alpha", "beta", "gamma"], &["u1", "u2"], leader_config()).unwrap();
        let alpha = report.models.iter().find(|m| m.model_id == "alpha").unwrap();
        let beta = report.models.iter().find(|m| m.model_id == "beta").unwrap();
        assert_eq!(alpha.point_score_ppm, beta.point_score_ppm);
        assert_eq!(alpha.top1_frequency_permille, 1_000);
        assert_eq!(beta.top1_frequency_permille, 1_000);
        let tie = report.pairwise.iter().find(|p| p.model_a_id == "alpha" && p.model_b_id == "beta").unwrap();
        assert_eq!(tie.p_boot_ppm, 1_000_000);
        assert_eq!(tie.p_holm_ppm, 1_000_000);
        assert!(!tie.significant_after_holm);
        assert!(tie.interval_includes_zero);
    }

    #[test]
    fn leaderboard_results_ignore_input_row_order_and_replay_seed() {
        let rows = vec![
            leader_score("alpha", "u1", 800_000),
            leader_score("alpha", "u2", 400_000),
            leader_score("alpha", "u3", 600_000),
            leader_score("beta", "u1", 500_000),
            leader_score("beta", "u2", 650_000),
            leader_score("beta", "u3", 450_000),
        ];
        let expected = analyze_leaderboard(&rows, &["alpha", "beta"], &["u1", "u2", "u3"], leader_config()).unwrap();
        let mut reversed = rows.clone();
        reversed.reverse();
        assert_eq!(expected, analyze_leaderboard(&reversed, &["alpha", "beta"], &["u1", "u2", "u3"], leader_config()).unwrap());
    }

    #[test]
    fn leaderboard_rejects_incomplete_grids_duplicates_invalid_scores_and_singletons() {
        assert_eq!(analyze_leaderboard(&[], &["alpha", "beta"], &["u1", "u2"], leader_config()), Err(LeaderboardInferenceError::EmptyScores));
        assert_eq!(
            analyze_leaderboard(
                &[leader_score("alpha", "u1", 500_000), leader_score("alpha", "u2", 500_000)],
                &["alpha"],
                &["u1", "u2"],
                leader_config(),
            ),
            Err(LeaderboardInferenceError::TooFewModels { actual: 1 })
        );
        let incomplete = vec![
            leader_score("alpha", "u1", 500_000),
            leader_score("alpha", "u2", 500_000),
            leader_score("beta", "u1", 600_000),
        ];
        assert_eq!(
            analyze_leaderboard(&incomplete, &["alpha", "beta"], &["u1", "u2"], leader_config()),
            Err(LeaderboardInferenceError::IncompleteScoreGrid { expected: 4, actual: 3 })
        );
        let duplicate = vec![
            leader_score("alpha", "u1", 500_000),
            leader_score("alpha", "u1", 500_000),
            leader_score("beta", "u1", 600_000),
            leader_score("beta", "u2", 600_000),
        ];
        assert!(matches!(
            analyze_leaderboard(&duplicate, &["alpha", "beta"], &["u1", "u2"], leader_config()),
            Err(LeaderboardInferenceError::DuplicateModelUnit { .. })
        ));
        let invalid = vec![
            leader_score("alpha", "u1", 1_000_001),
            leader_score("alpha", "u2", 500_000),
            leader_score("beta", "u1", 600_000),
            leader_score("beta", "u2", 600_000),
        ];
        assert!(matches!(
            analyze_leaderboard(&invalid, &["alpha", "beta"], &["u1", "u2"], leader_config()),
            Err(LeaderboardInferenceError::ScoreOutOfRange { .. })
        ));
    }

    #[test]
    fn leaderboard_uses_frozen_inventory_to_reject_a_wholly_omitted_unit() {
        // Both models omit u2, so an inventory inferred from rows alone would
        // silently accept this smaller grid. The frozen manifest must catch it.
        let rows = vec![
            leader_score("alpha", "u1", 500_000),
            leader_score("beta", "u1", 600_000),
        ];
        assert_eq!(
            analyze_leaderboard(&rows, &["alpha", "beta"], &["u1", "u2"], leader_config()),
            Err(LeaderboardInferenceError::IncompleteScoreGrid {
                expected: 4,
                actual: 2,
            })
        );
    }

    #[test]
    fn leaderboard_rejects_excessive_work_and_invalid_alpha() {
        let rows = vec![
            leader_score("alpha", "u1", 500_000),
            leader_score("alpha", "u2", 500_000),
            leader_score("beta", "u1", 600_000),
            leader_score("beta", "u2", 600_000),
        ];
        let bad_alpha = LeaderboardInferenceConfig { familywise_alpha_permille: 0, ..leader_config() };
        assert_eq!(
            analyze_leaderboard(&rows, &["alpha", "beta"], &["u1", "u2"], bad_alpha),
            Err(LeaderboardInferenceError::InvalidAlpha(0))
        );
        let mut budget_rows = Vec::new();
        for model in 0..11 {
            for unit in 0..10 {
                budget_rows.push(leader_score(
                    &format!("model-{model:02}"),
                    &format!("unit-{unit:02}"),
                    500_000 + model * 1000,
                ));
            }
        }
        let too_much = LeaderboardInferenceConfig {
            bootstrap: BootstrapConfig { replicates: 1_000_000, ..leader_config().bootstrap },
            ..leader_config()
        };
        let model_names: Vec<String> = (0..11).map(|model| format!("model-{model:02}")).collect();
        let unit_names: Vec<String> = (0..10).map(|unit| format!("unit-{unit:02}")).collect();
        let model_refs: Vec<&str> = model_names.iter().map(String::as_str).collect();
        let unit_refs: Vec<&str> = unit_names.iter().map(String::as_str).collect();
        assert_eq!(
            analyze_leaderboard(&budget_rows, &model_refs, &unit_refs, too_much),
            Err(LeaderboardInferenceError::ComputationBudgetExceeded)
        );
    }

}
