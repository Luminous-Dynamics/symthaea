//! Selection-aware uncertainty metrics for CMAQ-018.
//!
//! These are metric and interval-construction primitives, not a production
//! predictor and not a claim that any model is calibrated. In particular,
//! simultaneous coverage must be measured over the complete action set;
//! pointwise intervals do not automatically remain calibrated after selecting
//! the most attractive action.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictiveInterval {
    pub lower: f64,
    pub upper: f64,
    pub nominal_coverage: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalObservation {
    pub interval: PredictiveInterval,
    pub realized: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalMetrics {
    pub sample_count: usize,
    pub nominal_coverage: f64,
    pub empirical_coverage: f64,
    pub mean_width: f64,
    pub mean_interval_score: f64,
    pub absolute_coverage_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimultaneousMetrics {
    pub episode_count: usize,
    pub intervals_per_episode: usize,
    pub familywise_nominal_coverage: f64,
    pub empirical_simultaneous_coverage: f64,
    pub mean_max_interval_width: f64,
    pub absolute_coverage_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointDrawCoupling {
    SharedPosteriorAndWorldNoise,
    PairedExogenousDisturbance,
    DeterministicSharedLatent,
}

/// All action outcomes predicted from one exact observed history and one
/// shared joint-draw bank. The sample matrix is [action][joint_draw]; column
/// j must denote the same posterior/latent/disturbance draw for every action.
#[derive(Clone, Debug, PartialEq)]
pub struct JointActionSetSamples {
    pub history_sha256: String,
    pub horizon_steps: usize,
    pub action_manifest_sha256: String,
    pub outcome_channel: String,
    pub action_ids: Vec<u64>,
    pub joint_draw_ids: Vec<u64>,
    pub joint_draw_bank_sha256: String,
    pub coupling: JointDrawCoupling,
    pub outcome_samples: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionPredictionInterval {
    pub action_id: u64,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionEffectInterval {
    pub left_action_id: u64,
    pub right_action_id: u64,
    pub paired_draw_count: usize,
    pub interval: PredictiveInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalError {
    InvalidNominalCoverage,
    NonFiniteLowerBound,
    NonFiniteUpperBound,
    ReversedBounds,
    NonFiniteIntervalWidth,
    NonFiniteRealization,
    NonFiniteDerivedMetric,
    MixedNominalCoverage,
    EmptySamples,
    InsufficientSamples,
    UnequalPairedSampleCount,
    EmptyEpisodeSet,
    EmptyActionSet,
    UnequalActionSetCardinality,
    InsufficientActions,
    InvalidSelectedIndex,
    InvalidHorizon,
    InvalidHistoryDigest,
    InvalidActionManifestDigest,
    EmptyOutcomeChannel,
    InvalidJointDrawDigest,
    InsufficientJointDraws,
    DuplicateActionId,
    DuplicateJointDrawId,
    ActionSampleCardinalityMismatch,
    SameActionRequested,
}

/// Construct a central empirical interval from predictive samples.
///
/// Quantiles use linear interpolation (the common type-7 sample quantile).
/// This construction alone is not a finite-sample calibration guarantee;
/// calibration must be assessed on held-out data or added by a separately
/// specified procedure such as split conformal prediction.
pub fn central_interval(
    samples: &[f64],
    nominal_coverage: f64,
) -> Result<PredictiveInterval, IntervalError> {
    validate_nominal_coverage(nominal_coverage)?;
    if samples.is_empty() {
        return Err(IntervalError::EmptySamples);
    }
    if samples.len() < 2 {
        return Err(IntervalError::InsufficientSamples);
    }

    let mut sorted = samples.to_vec();
    if sorted.iter().any(|sample| !sample.is_finite()) {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    sorted.sort_by(f64::total_cmp);

    let tail_probability = (1.0 - nominal_coverage) / 2.0;
    let lower = linear_quantile(&sorted, tail_probability);
    let upper = linear_quantile(&sorted, 1.0 - tail_probability);
    let interval = PredictiveInterval {
        lower,
        upper,
        nominal_coverage,
    };
    validate_interval(interval)?;
    Ok(interval)
}

/// Construct an action-effect interval from paired joint predictive samples.
///
/// left_samples[i] and right_samples[i] must belong to the same observed
/// history and the same joint posterior/latent draw. Pairing preserves
/// common-mode error cancellation. Subtracting independently constructed
/// marginal interval endpoints is not an equivalent operation.
pub fn paired_effect_interval(
    left_samples: &[f64],
    right_samples: &[f64],
    nominal_coverage: f64,
) -> Result<PredictiveInterval, IntervalError> {
    if left_samples.len() != right_samples.len() {
        return Err(IntervalError::UnequalPairedSampleCount);
    }
    if left_samples.is_empty() {
        return Err(IntervalError::EmptySamples);
    }
    if left_samples.len() < 2 {
        return Err(IntervalError::InsufficientSamples);
    }

    let mut paired_differences = Vec::with_capacity(left_samples.len());
    for (left, right) in left_samples.iter().zip(right_samples) {
        if !left.is_finite() || !right.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
        let difference = left - right;
        if !difference.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
        paired_differences.push(difference);
    }
    central_interval(&paired_differences, nominal_coverage)
}


/// Validate the shape and provenance bindings of one same-history action bank.
///
/// This checks that an adapter supplies a shared draw axis. It cannot prove
/// that the producer honestly generated the values using the declared
/// coupling; the source/recorder and draw-bank bytes must be independently
/// retained and verified by the adapter's evidence layer.
pub fn validate_joint_action_set(
    bank: &JointActionSetSamples,
) -> Result<(), IntervalError> {
    if bank.horizon_steps == 0 {
        return Err(IntervalError::InvalidHorizon);
    }
    if !is_lower_hex_sha256(&bank.history_sha256) {
        return Err(IntervalError::InvalidHistoryDigest);
    }
    if !is_lower_hex_sha256(&bank.action_manifest_sha256) {
        return Err(IntervalError::InvalidActionManifestDigest);
    }
    if bank.outcome_channel.trim().is_empty() {
        return Err(IntervalError::EmptyOutcomeChannel);
    }
    if !is_lower_hex_sha256(&bank.joint_draw_bank_sha256) {
        return Err(IntervalError::InvalidJointDrawDigest);
    }
    if bank.action_ids.len() < 2 || bank.outcome_samples.len() != bank.action_ids.len() {
        return Err(IntervalError::UnequalActionSetCardinality);
    }
    if bank.joint_draw_ids.len() < 2 {
        return Err(IntervalError::InsufficientJointDraws);
    }

    let mut action_ids = bank.action_ids.clone();
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateActionId);
    }

    let mut draw_ids = bank.joint_draw_ids.clone();
    draw_ids.sort_unstable();
    if draw_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntervalError::DuplicateJointDrawId);
    }

    for samples in &bank.outcome_samples {
        if samples.len() != bank.joint_draw_ids.len() {
            return Err(IntervalError::ActionSampleCardinalityMismatch);
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }
    }
    Ok(())
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Produce a predictive interval for each action in the original action-id
/// order. Every interval is derived from the same declared joint draw bank.
pub fn joint_action_prediction_intervals(
    bank: &JointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<ActionPredictionInterval>, IntervalError> {
    validate_joint_action_set(bank)?;
    validate_nominal_coverage(nominal_coverage)?;
    bank.action_ids
        .iter()
        .copied()
        .zip(&bank.outcome_samples)
        .map(|(action_id, samples)| {
            Ok(ActionPredictionInterval {
                action_id,
                interval: central_interval(samples, nominal_coverage)?,
            })
        })
        .collect()
}

/// Construct the paired effect distribution for two action identities from
/// the shared draw axis. Action order is semantic by ID, not row position.
pub fn joint_action_effect_interval(
    bank: &JointActionSetSamples,
    left_action_id: u64,
    right_action_id: u64,
    nominal_coverage: f64,
) -> Result<ActionEffectInterval, IntervalError> {
    validate_joint_action_set(bank)?;
    if left_action_id == right_action_id {
        return Err(IntervalError::SameActionRequested);
    }
    let left_index = bank.action_ids.iter().position(|id| *id == left_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let right_index = bank.action_ids.iter().position(|id| *id == right_action_id)
        .ok_or(IntervalError::InvalidSelectedIndex)?;
    let interval = paired_effect_interval(
        &bank.outcome_samples[left_index],
        &bank.outcome_samples[right_index],
        nominal_coverage,
    )?;
    Ok(ActionEffectInterval {
        left_action_id,
        right_action_id,
        paired_draw_count: bank.joint_draw_ids.len(),
        interval,
    })
}

/// Calculate all unordered action-pair effect intervals, in lexicographic
/// action-id order, independent of the input action-row order.
pub fn all_joint_action_effect_intervals(
    bank: &JointActionSetSamples,
    nominal_coverage: f64,
) -> Result<Vec<ActionEffectInterval>, IntervalError> {
    validate_joint_action_set(bank)?;
    let mut ids = bank.action_ids.clone();
    ids.sort_unstable();
    let mut effects = Vec::with_capacity(ids.len() * (ids.len() - 1) / 2);
    for left in 0..ids.len() {
        for right in (left + 1)..ids.len() {
            effects.push(joint_action_effect_interval(
                bank, ids[left], ids[right], nominal_coverage
            )?);
        }
    }
    Ok(effects)
}

/// Measure marginal interval coverage, sharpness and the proper interval score.
///
/// All observations must use the same nominal coverage. Empty input returns
/// None so a missing evaluation is not silently reported as zero error.
pub fn evaluate_intervals(
    observations: &[IntervalObservation],
) -> Result<Option<IntervalMetrics>, IntervalError> {
    let Some(first) = observations.first() else {
        return Ok(None);
    };
    let nominal_coverage = first.interval.nominal_coverage;
    let mut covered_count = 0usize;
    let mut total_width = 0.0;
    let mut total_interval_score = 0.0;

    for observation in observations {
        validate_observation(*observation)?;
        if observation.interval.nominal_coverage.to_bits() != nominal_coverage.to_bits() {
            return Err(IntervalError::MixedNominalCoverage);
        }

        let interval = observation.interval;
        let width = interval.upper - interval.lower;
        let alpha = 1.0 - nominal_coverage;
        let miss_penalty = if observation.realized < interval.lower {
            (2.0 / alpha) * (interval.lower - observation.realized)
        } else if observation.realized > interval.upper {
            (2.0 / alpha) * (observation.realized - interval.upper)
        } else {
            0.0
        };
        let interval_score = width + miss_penalty;
        if !interval_score.is_finite() {
            return Err(IntervalError::NonFiniteDerivedMetric);
        }

        if interval.lower <= observation.realized && observation.realized <= interval.upper {
            covered_count += 1;
        }
        total_width += width;
        total_interval_score += interval_score;
    }

    let count = observations.len() as f64;
    let empirical_coverage = covered_count as f64 / count;
    let mean_width = total_width / count;
    let mean_interval_score = total_interval_score / count;
    if !mean_width.is_finite() || !mean_interval_score.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(Some(IntervalMetrics {
        sample_count: observations.len(),
        nominal_coverage,
        empirical_coverage,
        mean_width,
        mean_interval_score,
        absolute_coverage_error: (empirical_coverage - nominal_coverage).abs(),
    }))
}

/// Measure whether every action interval in an episode covers its realized
/// outcome. The episode—not the individual action interval—is the sampling
/// unit for simultaneous action-set coverage.
pub fn evaluate_simultaneous_sets(
    episodes: &[Vec<IntervalObservation>],
    familywise_nominal_coverage: f64,
) -> Result<Option<SimultaneousMetrics>, IntervalError> {
    validate_nominal_coverage(familywise_nominal_coverage)?;
    let Some(first_episode) = episodes.first() else {
        return Ok(None);
    };
    if first_episode.is_empty() {
        return Err(IntervalError::EmptyActionSet);
    }

    let intervals_per_episode = first_episode.len();
    let mut covered_episode_count = 0usize;
    let mut total_max_width = 0.0;

    for episode in episodes {
        if episode.is_empty() {
            return Err(IntervalError::EmptyActionSet);
        }
        if episode.len() != intervals_per_episode {
            return Err(IntervalError::UnequalActionSetCardinality);
        }

        let mut all_actions_covered = true;
        let mut max_width = 0.0_f64;
        for observation in episode {
            validate_observation(*observation)?;
            let width = observation.interval.upper - observation.interval.lower;
            max_width = max_width.max(width);
            all_actions_covered &= observation.interval.lower <= observation.realized
                && observation.realized <= observation.interval.upper;
        }
        if all_actions_covered {
            covered_episode_count += 1;
        }
        total_max_width += max_width;
    }

    let episode_count = episodes.len();
    let empirical_coverage = covered_episode_count as f64 / episode_count as f64;
    let mean_max_interval_width = total_max_width / episode_count as f64;
    if !mean_max_interval_width.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }

    Ok(Some(SimultaneousMetrics {
        episode_count,
        intervals_per_episode,
        familywise_nominal_coverage,
        empirical_simultaneous_coverage: empirical_coverage,
        mean_max_interval_width,
        absolute_coverage_error: (empirical_coverage - familywise_nominal_coverage).abs(),
    }))
}

/// Return a conservative selected-action regret upper bound, conditional on
/// every action interval in the set containing its true outcome.
///
/// The bound is max(0, max competitor upper bound - selected lower bound).
/// Its empirical reliability depends on simultaneous—not merely marginal—
/// coverage of the complete action set.
pub fn selected_action_regret_upper_bound(
    intervals: &[PredictiveInterval],
    selected_index: usize,
) -> Result<f64, IntervalError> {
    if intervals.len() < 2 {
        return Err(IntervalError::InsufficientActions);
    }
    if selected_index >= intervals.len() {
        return Err(IntervalError::InvalidSelectedIndex);
    }
    for interval in intervals {
        validate_interval(*interval)?;
    }

    let selected_lower = intervals[selected_index].lower;
    let competitor_upper = intervals
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected_index)
        .map(|(_, interval)| interval.upper)
        .fold(f64::NEG_INFINITY, f64::max);
    let bound = (competitor_upper - selected_lower).max(0.0);
    if !bound.is_finite() {
        return Err(IntervalError::NonFiniteDerivedMetric);
    }
    Ok(bound)
}

/// True only when the selected interval's lower bound is at least every
/// competitor's upper bound. This is a certificate conditional on the
/// action-set intervals being jointly valid; it is not a safety proof.
pub fn is_interval_dominant(
    intervals: &[PredictiveInterval],
    selected_index: usize,
) -> Result<bool, IntervalError> {
    Ok(selected_action_regret_upper_bound(intervals, selected_index)? == 0.0)
}

fn validate_nominal_coverage(nominal_coverage: f64) -> Result<(), IntervalError> {
    if nominal_coverage.is_finite() && nominal_coverage > 0.0 && nominal_coverage < 1.0 {
        Ok(())
    } else {
        Err(IntervalError::InvalidNominalCoverage)
    }
}

fn validate_interval(interval: PredictiveInterval) -> Result<(), IntervalError> {
    validate_nominal_coverage(interval.nominal_coverage)?;
    if !interval.lower.is_finite() {
        return Err(IntervalError::NonFiniteLowerBound);
    }
    if !interval.upper.is_finite() {
        return Err(IntervalError::NonFiniteUpperBound);
    }
    if interval.lower > interval.upper {
        return Err(IntervalError::ReversedBounds);
    }
    if !(interval.upper - interval.lower).is_finite() {
        return Err(IntervalError::NonFiniteIntervalWidth);
    }
    Ok(())
}

fn validate_observation(observation: IntervalObservation) -> Result<(), IntervalError> {
    validate_interval(observation.interval)?;
    if observation.realized.is_finite() {
        Ok(())
    } else {
        Err(IntervalError::NonFiniteRealization)
    }
}

fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * probability;
    let lower_index = rank.floor() as usize;
    let upper_index = rank.ceil() as usize;
    let fraction = rank - lower_index as f64;
    sorted[lower_index] + (sorted[upper_index] - sorted[lower_index]) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;


    fn joint_bank_fixture() -> JointActionSetSamples {
        JointActionSetSamples {
            history_sha256: "a".repeat(64),
            horizon_steps: 5,
            action_manifest_sha256: "c".repeat(64),
            outcome_channel: "predicted_task_utility".into(),
            action_ids: vec![17, 3, 9],
            joint_draw_ids: vec![100, 101, 102, 103, 104],
            joint_draw_bank_sha256: "b".repeat(64),
            coupling: JointDrawCoupling::SharedPosteriorAndWorldNoise,
            outcome_samples: vec![
                vec![100.0, 110.0, 120.0, 130.0, 140.0],
                vec![99.0, 109.0, 119.0, 129.0, 139.0],
                vec![97.0, 107.0, 117.0, 127.0, 137.0],
            ],
        }
    }

    #[test]
    fn joint_action_bank_preserves_common_mode_effect_cancellation() {
        let bank = joint_bank_fixture();
        validate_joint_action_set(&bank).unwrap();
        let effect = joint_action_effect_interval(&bank, 17, 3, 0.8).unwrap();
        assert_eq!(effect.left_action_id, 17);
        assert_eq!(effect.right_action_id, 3);
        assert_eq!(effect.paired_draw_count, 5);
        assert!((effect.interval.lower - 1.0).abs() < 1e-12);
        assert!((effect.interval.upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn joint_action_effects_are_invariant_to_action_row_permutation() {
        let original = joint_bank_fixture();
        let original_effect = joint_action_effect_interval(&original, 17, 9, 0.8).unwrap();
        let permuted = JointActionSetSamples {
            action_ids: vec![9, 17, 3],
            outcome_samples: vec![
                original.outcome_samples[2].clone(),
                original.outcome_samples[0].clone(),
                original.outcome_samples[1].clone(),
            ],
            ..original.clone()
        };
        let permuted_effect = joint_action_effect_interval(&permuted, 17, 9, 0.8).unwrap();
        assert_eq!(original_effect, permuted_effect);
        let all = all_joint_action_effect_intervals(&permuted, 0.8).unwrap();
        assert_eq!(
            all.iter().map(|effect| (effect.left_action_id, effect.right_action_id)).collect::<Vec<_>>(),
            vec![(3, 9), (3, 17), (9, 17)]
        );
    }

    #[test]
    fn joint_action_bank_rejects_duplicate_ids_and_bad_cardinality() {
        let mut bank = joint_bank_fixture();
        bank.action_ids[1] = bank.action_ids[0];
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::DuplicateActionId));

        let mut bank = joint_bank_fixture();
        bank.joint_draw_ids[1] = bank.joint_draw_ids[0];
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::DuplicateJointDrawId));

        let mut bank = joint_bank_fixture();
        bank.outcome_samples[1].pop();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::ActionSampleCardinalityMismatch));
    }

    #[test]
    fn joint_action_bank_requires_action_semantics_and_outcome_channel_identity() {
        let mut bank = joint_bank_fixture();
        bank.action_manifest_sha256 = "bad".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidActionManifestDigest));

        let mut bank = joint_bank_fixture();
        bank.outcome_channel = "  ".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::EmptyOutcomeChannel));
    }

    #[test]
    fn joint_action_bank_rejects_missing_or_invalid_provenance_and_horizon() {
        let mut bank = joint_bank_fixture();
        bank.history_sha256 = "not-a-digest".into();
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidHistoryDigest));

        let mut bank = joint_bank_fixture();
        bank.joint_draw_bank_sha256 = "B".repeat(64);
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidJointDrawDigest));

        let mut bank = joint_bank_fixture();
        bank.horizon_steps = 0;
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::InvalidHorizon));
    }

    #[test]
    fn joint_action_bank_rejects_nonfinite_draws_and_same_action_effect() {
        let mut bank = joint_bank_fixture();
        bank.outcome_samples[0][0] = f64::NAN;
        assert_eq!(validate_joint_action_set(&bank), Err(IntervalError::NonFiniteDerivedMetric));

        let bank = joint_bank_fixture();
        assert_eq!(joint_action_effect_interval(&bank, 17, 17, 0.8), Err(IntervalError::SameActionRequested));
        assert_eq!(joint_action_effect_interval(&bank, 17, 999, 0.8), Err(IntervalError::InvalidSelectedIndex));
    }

    fn interval(lower: f64, upper: f64, nominal_coverage: f64) -> PredictiveInterval {
        PredictiveInterval {
            lower,
            upper,
            nominal_coverage,
        }
    }

    fn observation(
        lower: f64,
        upper: f64,
        nominal_coverage: f64,
        realized: f64,
    ) -> IntervalObservation {
        IntervalObservation {
            interval: interval(lower, upper, nominal_coverage),
            realized,
        }
    }

    #[test]
    fn interval_metrics_report_coverage_width_and_proper_score() {
        let observations = [
            observation(0.0, 2.0, 0.9, 0.0),
            observation(0.0, 2.0, 0.9, 1.0),
            observation(0.0, 2.0, 0.9, 2.0),
            observation(0.0, 2.0, 0.9, 3.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert_eq!(metrics.sample_count, 4);
        assert!((metrics.empirical_coverage - 0.75).abs() < 1e-12);
        assert!((metrics.mean_width - 2.0).abs() < 1e-12);
        assert!((metrics.mean_interval_score - 7.0).abs() < 1e-12);
        assert!((metrics.absolute_coverage_error - 0.15).abs() < 1e-12);
    }

    #[test]
    fn interval_score_penalizes_misses_on_both_sides() {
        let observations = [
            observation(0.0, 2.0, 0.8, -1.0),
            observation(0.0, 2.0, 0.8, 3.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert!((metrics.mean_interval_score - 12.0).abs() < 1e-12);
        assert_eq!(metrics.empirical_coverage, 0.0);
    }

    #[test]
    fn interval_boundaries_count_as_covered_and_empty_input_is_missing() {
        let observations = [
            observation(0.0, 1.0, 0.9, 0.0),
            observation(0.0, 1.0, 0.9, 1.0),
        ];
        let metrics = evaluate_intervals(&observations).unwrap().unwrap();
        assert_eq!(metrics.empirical_coverage, 1.0);
        assert_eq!(evaluate_intervals(&[]).unwrap(), None);
    }

    #[test]
    fn mixed_nominal_levels_are_rejected() {
        let observations = [
            observation(0.0, 1.0, 0.9, 0.5),
            observation(0.0, 1.0, 0.8, 0.5),
        ];
        assert_eq!(
            evaluate_intervals(&observations),
            Err(IntervalError::MixedNominalCoverage)
        );
    }

    #[test]
    fn invalid_intervals_and_nonfinite_realizations_fail_closed() {
        assert_eq!(
            evaluate_intervals(&[observation(2.0, 1.0, 0.9, 1.5)]),
            Err(IntervalError::ReversedBounds)
        );
        assert_eq!(
            evaluate_intervals(&[observation(0.0, 1.0, 1.0, 0.5)]),
            Err(IntervalError::InvalidNominalCoverage)
        );
        assert_eq!(
            evaluate_intervals(&[observation(0.0, 1.0, 0.9, f64::NAN)]),
            Err(IntervalError::NonFiniteRealization)
        );
    }

    #[test]
    fn paired_effect_interval_preserves_common_mode_cancellation() {
        let left = [100.0, 110.0, 120.0, 130.0, 140.0];
        let right = [99.0, 109.0, 119.0, 129.0, 139.0];
        let effect = paired_effect_interval(&left, &right, 0.8).unwrap();
        assert!((effect.lower - 1.0).abs() < 1e-12);
        assert!((effect.upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn paired_effect_interval_rejects_unpaired_or_insufficient_samples() {
        assert_eq!(
            paired_effect_interval(&[1.0, 2.0], &[0.0], 0.9),
            Err(IntervalError::UnequalPairedSampleCount)
        );
        assert_eq!(
            paired_effect_interval(&[1.0], &[0.0], 0.9),
            Err(IntervalError::InsufficientSamples)
        );
    }

    #[test]
    fn simultaneous_coverage_counts_whole_action_sets_not_individual_pairs() {
        let episodes = vec![
            vec![
                observation(0.0, 2.0, 0.9, 1.0),
                observation(0.0, 2.0, 0.9, 1.0),
            ],
            vec![
                observation(0.0, 2.0, 0.9, 1.0),
                observation(0.0, 2.0, 0.9, 3.0),
            ],
        ];
        let metrics = evaluate_simultaneous_sets(&episodes, 0.9)
            .unwrap()
            .unwrap();
        assert_eq!(metrics.episode_count, 2);
        assert_eq!(metrics.intervals_per_episode, 2);
        assert_eq!(metrics.empirical_simultaneous_coverage, 0.5);
        assert_eq!(metrics.mean_max_interval_width, 2.0);
        assert_eq!(metrics.absolute_coverage_error, 0.4);
    }

    #[test]
    fn simultaneous_coverage_rejects_empty_and_mismatched_action_banks() {
        assert_eq!(
            evaluate_simultaneous_sets(&[], 0.9).unwrap(),
            None
        );
        assert_eq!(
            evaluate_simultaneous_sets(&[vec![]], 0.9),
            Err(IntervalError::EmptyActionSet)
        );
        let mismatched = vec![
            vec![observation(0.0, 1.0, 0.9, 0.5)],
            vec![
                observation(0.0, 1.0, 0.9, 0.5),
                observation(0.0, 1.0, 0.9, 0.5),
            ],
        ];
        assert_eq!(
            evaluate_simultaneous_sets(&mismatched, 0.9),
            Err(IntervalError::UnequalActionSetCardinality)
        );
    }

    #[test]
    fn selected_action_regret_bound_and_dominance_use_competitor_intervals() {
        let intervals = [
            interval(3.0, 4.0, 0.9),
            interval(0.0, 2.0, 0.9),
            interval(1.0, 2.5, 0.9),
        ];
        assert_eq!(selected_action_regret_upper_bound(&intervals, 0).unwrap(), 0.0);
        assert!(is_interval_dominant(&intervals, 0).unwrap());

        let uncertain = [
            interval(1.5, 2.0, 0.9),
            interval(0.0, 2.5, 0.9),
        ];
        assert_eq!(selected_action_regret_upper_bound(&uncertain, 0).unwrap(), 1.0);
        assert!(!is_interval_dominant(&uncertain, 0).unwrap());
    }

    #[test]
    fn selection_certificate_rejects_invalid_action_selection() {
        assert_eq!(
            selected_action_regret_upper_bound(&[interval(0.0, 1.0, 0.9)], 0),
            Err(IntervalError::InsufficientActions)
        );
        assert_eq!(
            selected_action_regret_upper_bound(
                &[interval(0.0, 1.0, 0.9), interval(0.0, 1.0, 0.9)],
                2
            ),
            Err(IntervalError::InvalidSelectedIndex)
        );
    }
}
