//! Finite-sample split-conformal calibration over complete action-horizon episodes.
//!
//! One independent episode contributes one maximum nonconformity score over
//! every frozen action and every tick in the declared horizon. This preserves
//! within-episode dependence and provides simultaneous marginal coverage over
//! the frozen rectangle under exchangeability of complete episodes.
//!
//! The caller supplies predictions, observed outcomes and provenance digests.
//! This crate does not prove their truth, exchangeability or physical validity.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

/// A fixed-point interval for expected task loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossInterval {
    /// Lower bound in caller-declared task-loss units.
    pub lower_units: u64,
    /// Upper bound in caller-declared task-loss units.
    pub upper_units: u64,
}

impl LossInterval {
    /// Construct an interval. Ordering is checked when the calibration grid is evaluated.
    pub const fn new(lower_units: u64, upper_units: u64) -> Self {
        Self {
            lower_units,
            upper_units,
        }
    }
}

/// One predicted interval and its independently observed realized loss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedLossBound {
    /// Monotonic tick index within the episode.
    pub tick: u64,
    /// Candidate action identity from the frozen action domain.
    pub action_id: String,
    /// Predicted interval for the action's expected task loss.
    pub predicted: LossInterval,
    /// Independently adjudicated expected-loss reference interval in the same fixed-point units.
    ///
    /// This must bound the target expected loss (for example, from an exact finite
    /// oracle or a separately qualified reference estimator), not merely one noisy
    /// realized outcome from the episode.
    pub reference_expected_loss: LossInterval,
}

/// One complete held-out calibration episode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationEpisode {
    /// Unique independent episode/mission-run identity.
    pub episode_id: String,
    /// Frozen scope digest this episode was collected under.
    pub scope_digest: String,
    /// Prediction/outcome cells for the full action × tick rectangle.
    pub observations: Vec<ObservedLossBound>,
}

/// Frozen calibration scope and finite-sample target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationScope {
    /// Digest committing task, loss, model, action domain, horizon, stratum and split semantics.
    pub scope_digest: String,
    /// Expected-loss predictor identity.
    pub loss_model_digest: String,
    /// Versioned task-loss unit identity.
    pub loss_scale_id: String,
    /// Exact frozen candidate action-domain manifest identity.
    pub action_domain_digest: String,
    /// Exact candidate action identities in the frozen domain.
    pub candidate_action_ids: Vec<String>,
    /// First tick covered by the simultaneous calibration scope.
    pub first_tick: u64,
    /// Number of contiguous ticks covered by the scope.
    pub horizon_ticks: u64,
    /// Target miscoverage in parts per million; must lie in [1, 999_999].
    pub alpha_ppm: u32,
    /// Digest of the frozen calibration split / dataset manifest.
    pub calibration_split_digest: String,
    /// Evidence digest supporting the claimed episode-level exchangeability scope.
    pub exchangeability_evidence_digest: String,
    /// Declared stratum identity, such as morphology/task/ambient/hardware profile.
    pub stratum_id: String,
}

/// Finite-sample order-statistic metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformalRank {
    /// Number of independent calibration episodes.
    pub sample_count: usize,
    /// Target miscoverage in parts per million.
    pub alpha_ppm: u32,
    /// One-indexed split-conformal rank.
    pub one_indexed_rank: usize,
}

/// Receipt emitted by a successful simultaneous episode-level calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimultaneousCalibrationReceipt {
    /// Frozen calibration scope identity.
    pub scope_digest: String,
    /// Predictor identity.
    pub loss_model_digest: String,
    /// Task-loss scale identity.
    pub loss_scale_id: String,
    /// Frozen candidate action-domain digest.
    pub action_domain_digest: String,
    /// Exact candidate action IDs covered.
    pub candidate_action_ids: Vec<String>,
    /// First covered control tick.
    pub first_tick: u64,
    /// Number of contiguous ticks covered.
    pub horizon_ticks: u64,
    /// Declared exchangeability stratum.
    pub stratum_id: String,
    /// Calibration split manifest digest.
    pub calibration_split_digest: String,
    /// Evidence digest supplied for the episode-level exchangeability claim.
    pub exchangeability_evidence_digest: String,
    /// Number of independent complete calibration episodes (not rows/ticks/actions).
    pub calibration_episode_count: usize,
    /// Target miscoverage in parts per million.
    pub alpha_ppm: u32,
    /// Nominal marginal coverage in parts per million under exchangeability.
    pub coverage_ppm: u32,
    /// Exact order-statistic metadata.
    pub rank: ConformalRank,
    /// Maximum score at the selected conformal rank, in task-loss units.
    pub expansion_units: u64,
}

/// Conservatively expanded loss interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandedLossInterval {
    /// Lower bound after expansion and zero-flooring.
    pub lower_units: u64,
    /// Upper bound after expansion.
    pub upper_units: u64,
    /// True when upper addition reached the representable u64 maximum.
    pub upper_saturated: bool,
}

/// Error or non-calibrated input disposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    /// A required identity/digest field is empty.
    EmptyScopeField {
        /// Name of the missing field.
        field: &'static str,
    },
    /// Target miscoverage is outside the open interval (0, 1).
    InvalidAlpha {
        /// Supplied alpha in parts per million.
        alpha_ppm: u32,
    },
    /// Horizon length must be positive.
    EmptyHorizon,
    /// Candidate action domain must contain at least one action.
    EmptyActionDomain,
    /// Candidate action identity is empty.
    EmptyActionId,
    /// Candidate action identity occurs more than once.
    DuplicateCandidateAction(String),
    /// The final tick index cannot be represented.
    TickRangeOverflow,
    /// No calibration episodes were provided.
    EmptyCalibrationSet,
    /// One calibration episode is missing the required frozen scope identity.
    EpisodeScopeMismatch {
        /// Episode identity.
        episode_id: String,
    },
    /// Episode identity is empty.
    EmptyEpisodeId,
    /// One independent episode identity occurs more than once.
    DuplicateEpisodeId(String),
    /// Episode does not contain exactly the frozen number of action × tick cells.
    IncompleteGrid {
        /// Episode identity.
        episode_id: String,
        /// Expected number of action × tick cells.
        expected_cells: usize,
        /// Supplied number of cells.
        observed_cells: usize,
    },
    /// An observation names an action outside the frozen candidate domain.
    UnexpectedAction {
        /// Episode identity.
        episode_id: String,
        /// Unknown action identity.
        action_id: String,
    },
    /// An observation tick is outside the frozen contiguous horizon.
    UnexpectedTick {
        /// Episode identity.
        episode_id: String,
        /// Supplied tick index.
        tick: u64,
    },
    /// An action × tick cell occurs more than once.
    DuplicateCell {
        /// Episode identity.
        episode_id: String,
        /// Duplicated tick.
        tick: u64,
        /// Duplicated action.
        action_id: String,
    },
    /// A predicted interval is malformed.
    InvalidLossInterval {
        /// Episode identity.
        episode_id: String,
        /// Action identity.
        action_id: String,
        /// Tick identity.
        tick: u64,
        /// Lower endpoint.
        lower_units: u64,
        /// Upper endpoint.
        upper_units: u64,
    },
    /// The action × tick cardinality overflows the platform index type.
    GridCardinalityOverflow,
    /// The exact conformal rank exceeds the available calibration episode count.
    InsufficientCalibrationEpisodes {
        /// Required one-indexed rank.
        required_rank: usize,
        /// Number of independent calibration episodes.
        sample_count: usize,
    },
    /// A requested interval expansion uses a different scope.
    ReceiptScopeMismatch,
    /// A requested action is outside the calibrated action domain.
    ReceiptActionMismatch(String),
    /// A requested tick is outside the calibrated horizon.
    ReceiptTickMismatch(u64),
}

/// Calculate the exact one-indexed split-conformal rank using integer arithmetic.
///
/// Computes ceil((n + 1) * (1 - alpha)). Returns an error when the requested
/// nominal coverage has no finite order statistic for the supplied sample size.
pub fn conformal_rank(
    sample_count: usize,
    alpha_ppm: u32,
) -> Result<ConformalRank, CalibrationError> {
    if sample_count == 0 {
        return Err(CalibrationError::EmptyCalibrationSet);
    }
    if !(1..1_000_000).contains(&alpha_ppm) {
        return Err(CalibrationError::InvalidAlpha { alpha_ppm });
    }

    let numerator = (sample_count as u128 + 1)
        .checked_mul((1_000_000 - alpha_ppm) as u128)
        .ok_or(CalibrationError::GridCardinalityOverflow)?;
    let denominator = 1_000_000_u128;
    let rank = numerator
        .checked_add(denominator - 1)
        .ok_or(CalibrationError::GridCardinalityOverflow)?
        / denominator;

    if rank > sample_count as u128 {
        return Err(CalibrationError::InsufficientCalibrationEpisodes {
            required_rank: usize::try_from(rank)
                .map_err(|_| CalibrationError::GridCardinalityOverflow)?,
            sample_count,
        });
    }

    Ok(ConformalRank {
        sample_count,
        alpha_ppm,
        one_indexed_rank: usize::try_from(rank)
            .map_err(|_| CalibrationError::GridCardinalityOverflow)?,
    })
}

fn validate_scope(scope: &CalibrationScope) -> Result<BTreeSet<&str>, CalibrationError> {
    for (field, value) in [
        ("scope_digest", scope.scope_digest.as_str()),
        ("loss_model_digest", scope.loss_model_digest.as_str()),
        ("loss_scale_id", scope.loss_scale_id.as_str()),
        ("action_domain_digest", scope.action_domain_digest.as_str()),
        (
            "calibration_split_digest",
            scope.calibration_split_digest.as_str(),
        ),
        (
            "exchangeability_evidence_digest",
            scope.exchangeability_evidence_digest.as_str(),
        ),
        ("stratum_id", scope.stratum_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(CalibrationError::EmptyScopeField { field });
        }
    }
    if scope.horizon_ticks == 0 {
        return Err(CalibrationError::EmptyHorizon);
    }
    if scope.candidate_action_ids.is_empty() {
        return Err(CalibrationError::EmptyActionDomain);
    }
    if !(1..1_000_000).contains(&scope.alpha_ppm) {
        return Err(CalibrationError::InvalidAlpha {
            alpha_ppm: scope.alpha_ppm,
        });
    }
    scope
        .first_tick
        .checked_add(scope.horizon_ticks)
        .ok_or(CalibrationError::TickRangeOverflow)?;

    let mut action_ids = BTreeSet::new();
    for action_id in &scope.candidate_action_ids {
        if action_id.trim().is_empty() {
            return Err(CalibrationError::EmptyActionId);
        }
        if !action_ids.insert(action_id.as_str()) {
            return Err(CalibrationError::DuplicateCandidateAction(
                action_id.clone(),
            ));
        }
    }
    Ok(action_ids)
}

fn episode_nonconformity(
    scope: &CalibrationScope,
    action_ids: &BTreeSet<&str>,
    episode: &CalibrationEpisode,
) -> Result<u64, CalibrationError> {
    if episode.episode_id.trim().is_empty() {
        return Err(CalibrationError::EmptyEpisodeId);
    }
    if episode.scope_digest != scope.scope_digest {
        return Err(CalibrationError::EpisodeScopeMismatch {
            episode_id: episode.episode_id.clone(),
        });
    }

    let expected_cells_u64 = scope
        .horizon_ticks
        .checked_mul(action_ids.len() as u64)
        .ok_or(CalibrationError::GridCardinalityOverflow)?;
    let expected_cells = usize::try_from(expected_cells_u64)
        .map_err(|_| CalibrationError::GridCardinalityOverflow)?;
    if episode.observations.len() != expected_cells {
        return Err(CalibrationError::IncompleteGrid {
            episode_id: episode.episode_id.clone(),
            expected_cells,
            observed_cells: episode.observations.len(),
        });
    }

    let end_tick = scope
        .first_tick
        .checked_add(scope.horizon_ticks)
        .ok_or(CalibrationError::TickRangeOverflow)?;
    let mut cells = BTreeSet::new();
    let mut maximum_score = 0_u64;

    for observation in &episode.observations {
        if !action_ids.contains(observation.action_id.as_str()) {
            return Err(CalibrationError::UnexpectedAction {
                episode_id: episode.episode_id.clone(),
                action_id: observation.action_id.clone(),
            });
        }
        if observation.tick < scope.first_tick || observation.tick >= end_tick {
            return Err(CalibrationError::UnexpectedTick {
                episode_id: episode.episode_id.clone(),
                tick: observation.tick,
            });
        }
        if !cells.insert((observation.tick, observation.action_id.as_str())) {
            return Err(CalibrationError::DuplicateCell {
                episode_id: episode.episode_id.clone(),
                tick: observation.tick,
                action_id: observation.action_id.clone(),
            });
        }
        if observation.predicted.lower_units > observation.predicted.upper_units {
            return Err(CalibrationError::InvalidLossInterval {
                episode_id: episode.episode_id.clone(),
                action_id: observation.action_id.clone(),
                tick: observation.tick,
                lower_units: observation.predicted.lower_units,
                upper_units: observation.predicted.upper_units,
            });
        }
        if observation.reference_expected_loss.lower_units
            > observation.reference_expected_loss.upper_units
        {
            return Err(CalibrationError::InvalidLossInterval {
                episode_id: episode.episode_id.clone(),
                action_id: observation.action_id.clone(),
                tick: observation.tick,
                lower_units: observation.reference_expected_loss.lower_units,
                upper_units: observation.reference_expected_loss.upper_units,
            });
        }

        // The score expands the prediction until it contains the full
        // independently adjudicated reference interval, not a noisy scalar
        // realized outcome that may differ from the expected loss target.
        let residual_below = observation
            .predicted
            .lower_units
            .saturating_sub(observation.reference_expected_loss.lower_units);
        let residual_above = observation
            .reference_expected_loss
            .upper_units
            .saturating_sub(observation.predicted.upper_units);
        maximum_score = maximum_score.max(residual_below.max(residual_above));
    }

    if cells.len() != expected_cells {
        return Err(CalibrationError::IncompleteGrid {
            episode_id: episode.episode_id.clone(),
            expected_cells,
            observed_cells: cells.len(),
        });
    }
    Ok(maximum_score)
}

/// Fit one simultaneous split-conformal expansion from complete independent episodes.
///
/// Each episode contributes exactly one maximum residual score over every
/// candidate action and every tick in the frozen horizon. The finite-sample
/// coverage statement is conditional on exchangeability of complete episodes.
pub fn calibrate_simultaneous_episode_envelope(
    scope: &CalibrationScope,
    episodes: &[CalibrationEpisode],
) -> Result<SimultaneousCalibrationReceipt, CalibrationError> {
    let action_ids = validate_scope(scope)?;
    if episodes.is_empty() {
        return Err(CalibrationError::EmptyCalibrationSet);
    }

    let expected_cells = scope
        .horizon_ticks
        .checked_mul(action_ids.len() as u64)
        .ok_or(CalibrationError::GridCardinalityOverflow)?;
    usize::try_from(expected_cells).map_err(|_| CalibrationError::GridCardinalityOverflow)?;

    let mut episode_ids = BTreeSet::new();
    let mut scores = Vec::with_capacity(episodes.len());
    for episode in episodes {
        if !episode_ids.insert(episode.episode_id.as_str()) {
            return Err(CalibrationError::DuplicateEpisodeId(
                episode.episode_id.clone(),
            ));
        }
        scores.push(episode_nonconformity(scope, &action_ids, episode)?);
    }

    let rank = conformal_rank(episodes.len(), scope.alpha_ppm)?;
    scores.sort_unstable();
    let expansion_units = scores[rank.one_indexed_rank - 1];

    Ok(SimultaneousCalibrationReceipt {
        scope_digest: scope.scope_digest.clone(),
        loss_model_digest: scope.loss_model_digest.clone(),
        loss_scale_id: scope.loss_scale_id.clone(),
        action_domain_digest: scope.action_domain_digest.clone(),
        candidate_action_ids: scope.candidate_action_ids.clone(),
        first_tick: scope.first_tick,
        horizon_ticks: scope.horizon_ticks,
        stratum_id: scope.stratum_id.clone(),
        calibration_split_digest: scope.calibration_split_digest.clone(),
        exchangeability_evidence_digest: scope.exchangeability_evidence_digest.clone(),
        calibration_episode_count: episodes.len(),
        alpha_ppm: scope.alpha_ppm,
        coverage_ppm: 1_000_000 - scope.alpha_ppm,
        rank,
        expansion_units,
    })
}

/// Expand a predicted interval only within the exact calibrated scope.
///
/// Lower bounds are floored at zero. Upper bounds saturate at u64::MAX rather
/// than wrapping; the caller receives an explicit saturation flag.
pub fn expand_interval_for(
    receipt: &SimultaneousCalibrationReceipt,
    scope_digest: &str,
    action_id: &str,
    tick: u64,
    interval: LossInterval,
) -> Result<ExpandedLossInterval, CalibrationError> {
    if scope_digest != receipt.scope_digest {
        return Err(CalibrationError::ReceiptScopeMismatch);
    }
    if !receipt
        .candidate_action_ids
        .iter()
        .any(|candidate| candidate == action_id)
    {
        return Err(CalibrationError::ReceiptActionMismatch(
            action_id.to_owned(),
        ));
    }
    let end_tick = receipt
        .first_tick
        .checked_add(receipt.horizon_ticks)
        .ok_or(CalibrationError::TickRangeOverflow)?;
    if tick < receipt.first_tick || tick >= end_tick {
        return Err(CalibrationError::ReceiptTickMismatch(tick));
    }
    if interval.lower_units > interval.upper_units {
        return Err(CalibrationError::InvalidLossInterval {
            episode_id: "future".to_owned(),
            action_id: action_id.to_owned(),
            tick,
            lower_units: interval.lower_units,
            upper_units: interval.upper_units,
        });
    }

    let unbounded_upper = interval.upper_units.checked_add(receipt.expansion_units);
    Ok(ExpandedLossInterval {
        lower_units: interval.lower_units.saturating_sub(receipt.expansion_units),
        upper_units: unbounded_upper.unwrap_or(u64::MAX),
        upper_saturated: unbounded_upper.is_none(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(alpha_ppm: u32, horizon_ticks: u64, actions: &[&str]) -> CalibrationScope {
        CalibrationScope {
            scope_digest: "scope-v1".to_owned(),
            loss_model_digest: "loss-model-v1".to_owned(),
            loss_scale_id: "task-loss-v1".to_owned(),
            action_domain_digest: "action-domain-v1".to_owned(),
            candidate_action_ids: actions.iter().map(|x| (*x).to_owned()).collect(),
            first_tick: 10,
            horizon_ticks,
            alpha_ppm,
            calibration_split_digest: "calibration-split-v1".to_owned(),
            exchangeability_evidence_digest: "exchangeability-evidence-v1".to_owned(),
            stratum_id: "robot-task-stratum-v1".to_owned(),
        }
    }

    fn episode(
        id: &str,
        scope_digest: &str,
        ticks: u64,
        actions: &[&str],
        base_loss: u64,
        upper_error: u64,
    ) -> CalibrationEpisode {
        let mut observations = Vec::new();
        for tick in 10..10 + ticks {
            for action in actions {
                let lower = base_loss;
                let upper = base_loss + 2;
                observations.push(ObservedLossBound {
                    tick,
                    action_id: (*action).to_owned(),
                    predicted: LossInterval::new(lower, upper),
                    reference_expected_loss: LossInterval::new(
                        lower,
                        upper.saturating_add(upper_error),
                    ),
                });
            }
        }
        CalibrationEpisode {
            episode_id: id.to_owned(),
            scope_digest: scope_digest.to_owned(),
            observations,
        }
    }

    fn nineteen_episodes(errors: &[u64]) -> Vec<CalibrationEpisode> {
        (0..19)
            .map(|index| {
                episode(
                    &format!("mission-{index:02}"),
                    "scope-v1",
                    2,
                    &["a0", "a1"],
                    10,
                    errors.get(index).copied().unwrap_or(0),
                )
            })
            .collect()
    }

    #[test]
    fn exact_split_conformal_rank_uses_integer_ceiling() {
        assert_eq!(
            conformal_rank(19, 50_000).unwrap(),
            ConformalRank {
                sample_count: 19,
                alpha_ppm: 50_000,
                one_indexed_rank: 19,
            }
        );
        assert_eq!(conformal_rank(9, 100_000).unwrap().one_indexed_rank, 9);
        assert_eq!(conformal_rank(99, 10_000).unwrap().one_indexed_rank, 99);
    }

    #[test]
    fn insufficient_episode_count_has_no_finite_quantile() {
        assert_eq!(
            conformal_rank(18, 50_000),
            Err(CalibrationError::InsufficientCalibrationEpisodes {
                required_rank: 19,
                sample_count: 18,
            })
        );
    }

    #[test]
    fn one_episode_is_one_sample_even_with_many_ticks_and_actions() {
        let one = episode(
            "one-long-mission",
            "scope-v1",
            100,
            &["a0", "a1", "a2", "a3"],
            10,
            0,
        );
        assert_eq!(
            calibrate_simultaneous_episode_envelope(
                &scope(50_000, 100, &["a0", "a1", "a2", "a3"]),
                &[one]
            ),
            Err(CalibrationError::InsufficientCalibrationEpisodes {
                required_rank: 1,
                sample_count: 1,
            })
        );
    }

    #[test]
    fn zero_residuals_produce_zero_expansion_only_with_sufficient_complete_episodes() {
        let episodes = nineteen_episodes(&[]);
        let receipt =
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &episodes)
                .unwrap();
        assert_eq!(receipt.expansion_units, 0);
        assert_eq!(receipt.coverage_ppm, 950_000);
        assert_eq!(receipt.calibration_episode_count, 19);
    }

    #[test]
    fn one_large_action_tick_error_expands_the_entire_simultaneous_scope() {
        let mut episodes = nineteen_episodes(&[]);
        // One cell in one independent episode is an outlier. The episode
        // maximum makes the expansion simultaneous over every action/tick.
        episodes[18].observations[3]
            .reference_expected_loss
            .upper_units += 7;
        let receipt =
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &episodes)
                .unwrap();
        assert_eq!(receipt.expansion_units, 7);

        for tick in [10, 11] {
            for action in ["a0", "a1"] {
                let expanded = expand_interval_for(
                    &receipt,
                    "scope-v1",
                    action,
                    tick,
                    LossInterval::new(20, 24),
                )
                .unwrap();
                assert_eq!(expanded.lower_units, 13);
                assert_eq!(expanded.upper_units, 31);
                assert!(!expanded.upper_saturated);
            }
        }
    }

    #[test]
    fn lower_bound_expansion_cannot_be_negative() {
        let mut errors = vec![0; 19];
        errors[18] = 5;
        let receipt = calibrate_simultaneous_episode_envelope(
            &scope(50_000, 2, &["a0", "a1"]),
            &nineteen_episodes(&errors),
        )
        .unwrap();
        let expanded =
            expand_interval_for(&receipt, "scope-v1", "a0", 10, LossInterval::new(2, 4)).unwrap();
        assert_eq!(expanded.lower_units, 0);
    }

    #[test]
    fn upper_bound_overflow_saturates_and_is_reported() {
        let mut errors = vec![0; 19];
        errors[18] = 10;
        let receipt = calibrate_simultaneous_episode_envelope(
            &scope(50_000, 2, &["a0", "a1"]),
            &nineteen_episodes(&errors),
        )
        .unwrap();
        let expanded = expand_interval_for(
            &receipt,
            "scope-v1",
            "a0",
            10,
            LossInterval::new(u64::MAX - 2, u64::MAX - 1),
        )
        .unwrap();
        assert_eq!(expanded.upper_units, u64::MAX);
        assert!(expanded.upper_saturated);
    }

    #[test]
    fn missing_action_tick_cell_rejects_the_episode() {
        let mut e = episode("m1", "scope-v1", 2, &["a0", "a1"], 10, 0);
        e.observations.pop();
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::IncompleteGrid { .. })
        ));
    }

    #[test]
    fn duplicate_episode_ids_are_rejected() {
        let e1 = episode("same", "scope-v1", 2, &["a0", "a1"], 10, 0);
        let e2 = e1.clone();
        assert_eq!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e1, e2]),
            Err(CalibrationError::DuplicateEpisodeId("same".to_owned()))
        );
    }

    #[test]
    fn duplicate_action_tick_cell_is_rejected() {
        let mut e = episode("m1", "scope-v1", 2, &["a0", "a1"], 10, 0);
        e.observations[1] = e.observations[0].clone();
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::DuplicateCell { .. })
        ));
    }

    #[test]
    fn wrong_scope_digest_rejects_episode() {
        let e = episode("m1", "other-scope", 2, &["a0", "a1"], 10, 0);
        assert_eq!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::EpisodeScopeMismatch {
                episode_id: "m1".to_owned()
            })
        );
    }

    #[test]
    fn unexpected_candidate_action_is_rejected() {
        let e = episode("m1", "scope-v1", 2, &["a0", "wrong"], 10, 0);
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::UnexpectedAction { .. })
        ));
    }

    #[test]
    fn unexpected_tick_is_rejected() {
        let mut e = episode("m1", "scope-v1", 2, &["a0", "a1"], 10, 0);
        e.observations[0].tick = 99;
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::UnexpectedTick { .. })
        ));
    }

    #[test]
    fn malformed_loss_interval_is_rejected() {
        let mut e = episode("m1", "scope-v1", 2, &["a0", "a1"], 10, 0);
        e.observations[0].predicted = LossInterval::new(5, 2);
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::InvalidLossInterval { .. })
        ));
    }

    #[test]
    fn invalid_reference_expected_loss_interval_is_rejected() {
        let mut e = episode("m1", "scope-v1", 2, &["a0", "a1"], 10, 0);
        e.observations[0].reference_expected_loss = LossInterval::new(9, 4);
        assert!(matches!(
            calibrate_simultaneous_episode_envelope(&scope(50_000, 2, &["a0", "a1"]), &[e]),
            Err(CalibrationError::InvalidLossInterval { .. })
        ));
    }

    #[test]
    fn calibration_split_digest_is_mandatory() {
        let mut s = scope(50_000, 2, &["a0", "a1"]);
        s.calibration_split_digest.clear();
        assert_eq!(
            calibrate_simultaneous_episode_envelope(&s, &nineteen_episodes(&[])),
            Err(CalibrationError::EmptyScopeField {
                field: "calibration_split_digest"
            })
        );
    }

    #[test]
    fn exchangeability_evidence_digest_is_mandatory() {
        let mut s = scope(50_000, 2, &["a0", "a1"]);
        s.exchangeability_evidence_digest.clear();
        assert_eq!(
            calibrate_simultaneous_episode_envelope(&s, &nineteen_episodes(&[])),
            Err(CalibrationError::EmptyScopeField {
                field: "exchangeability_evidence_digest"
            })
        );
    }

    #[test]
    fn action_domain_manifest_must_be_unique() {
        let s = scope(50_000, 2, &["a0", "a0"]);
        assert_eq!(
            calibrate_simultaneous_episode_envelope(&s, &nineteen_episodes(&[])),
            Err(CalibrationError::DuplicateCandidateAction("a0".to_owned()))
        );
    }

    #[test]
    fn receipt_rejects_scope_action_and_tick_outside_calibration() {
        let receipt = calibrate_simultaneous_episode_envelope(
            &scope(50_000, 2, &["a0", "a1"]),
            &nineteen_episodes(&[]),
        )
        .unwrap();
        assert_eq!(
            expand_interval_for(&receipt, "other-scope", "a0", 10, LossInterval::new(1, 2)),
            Err(CalibrationError::ReceiptScopeMismatch)
        );
        assert_eq!(
            expand_interval_for(&receipt, "scope-v1", "a9", 10, LossInterval::new(1, 2)),
            Err(CalibrationError::ReceiptActionMismatch("a9".to_owned()))
        );
        assert_eq!(
            expand_interval_for(&receipt, "scope-v1", "a0", 12, LossInterval::new(1, 2)),
            Err(CalibrationError::ReceiptTickMismatch(12))
        );
    }

    #[test]
    fn same_episode_with_many_action_rows_still_contributes_one_score() {
        let e = episode("long", "scope-v1", 100, &["a0", "a1"], 10, 0);
        assert_eq!(e.observations.len(), 200);
        assert_eq!(
            conformal_rank(1, 50_000),
            Err(CalibrationError::InsufficientCalibrationEpisodes {
                required_rank: 2,
                sample_count: 1
            })
        );
    }
}
