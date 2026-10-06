// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only held-out prediction for relational dynamics.
//!
//! The qualification question is deliberately predictive rather than
//! ontological:
//!
//!   Does a relational feature set predict an independently observed future
//!   interaction outcome better than isolated-agent, synchrony-only, or
//!   common-driver baselines on data that occur strictly later in time?
//!
//! This module uses a fixed temporal holdout, an explicit label horizon, and a
//! boundary check that prevents training labels from extending into the test
//! feature interval. It never touches production partnership state, cognition,
//! relational_psi, trust, or response generation.
//!
//! The null layer has two deterministic families:
//!
//! - CircularShift: shift all relational channels together within each split,
//!   destroying partner-specific alignment while preserving each channel's
//!   marginal sequence structure.
//! - FeatureDecoupling: shift relational channels by distinct offsets within
//!   each split, preserving their individual temporal structure while breaking
//!   the coherent relational bundle.
//!
//! Null outputs are empirical calibration diagnostics, not p-values.

use super::relational_harmonics::EvidenceStatus;

/// A future outcome paired with features available strictly before that outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelationalPredictionSample {
    /// Time at which the predictor features are available.
    pub feature_time: f64,
    /// Time at which the independently observed target becomes available.
    pub outcome_time: f64,
    /// Scalar state of agent A available at feature_time.
    pub agent_a: f64,
    /// Scalar state of agent B available at feature_time.
    pub agent_b: f64,
    /// Current relational alignment feature.
    pub alignment: f64,
    /// Current A -> B predictive-coupling proxy.
    pub a_to_b: f64,
    /// Current B -> A predictive-coupling proxy.
    pub b_to_a: f64,
    /// Current turn-taking feature.
    pub turn_taking: f64,
    /// Explicitly observed shared context available at feature_time.
    pub common_driver: f64,
    /// Independently observed future interaction outcome.
    pub future_outcome: f64,
}

impl RelationalPredictionSample {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_time: f64,
        outcome_time: f64,
        agent_a: f64,
        agent_b: f64,
        alignment: f64,
        a_to_b: f64,
        b_to_a: f64,
        turn_taking: f64,
        common_driver: f64,
        future_outcome: f64,
    ) -> Result<Self, RelationalPredictionError> {
        if !feature_time.is_finite()
            || !outcome_time.is_finite()
            || !agent_a.is_finite()
            || !agent_b.is_finite()
            || !common_driver.is_finite()
            || !future_outcome.is_finite()
        {
            return Err(RelationalPredictionError::NonFiniteSample);
        }

        for (name, value) in [
            ("alignment", alignment),
            ("a_to_b", a_to_b),
            ("b_to_a", b_to_a),
            ("turn_taking", turn_taking),
        ] {
            if !value.is_finite() {
                return Err(RelationalPredictionError::NonFiniteValue(name));
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(RelationalPredictionError::OutOfRange(name, value));
            }
        }

        if outcome_time <= feature_time {
            return Err(RelationalPredictionError::OutcomeNotAfterFeatures);
        }

        Ok(Self {
            feature_time,
            outcome_time,
            agent_a,
            agent_b,
            alignment,
            a_to_b,
            b_to_a,
            turn_taking,
            common_driver,
            future_outcome,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationalPredictionError {
    NonFiniteSample,
    NonFiniteValue(&'static str),
    OutOfRange(&'static str, f64),
    OutcomeNotAfterFeatures,
    InsufficientSamples(usize),
    InvalidSplit,
    TemporalLeakage,
    InvalidRidgeLambda,
    InvalidSurrogateCount,
    ModelFitFailed,
}

impl std::fmt::Display for RelationalPredictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteSample => write!(f, "prediction sample contains a non-finite value"),
            Self::NonFiniteValue(name) => write!(f, "{name} must be finite"),
            Self::OutOfRange(name, value) => write!(f, "{name}={value} is outside [0, 1]"),
            Self::OutcomeNotAfterFeatures => {
                write!(f, "future outcome time must be strictly after feature time")
            }
            Self::InsufficientSamples(n) => write!(f, "insufficient samples: got {n}"),
            Self::InvalidSplit => write!(f, "prediction train/test split is invalid"),
            Self::TemporalLeakage => {
                write!(f, "training outcome horizon overlaps the held-out feature interval")
            }
            Self::InvalidRidgeLambda => write!(f, "ridge_lambda must be finite and non-negative"),
            Self::InvalidSurrogateCount => write!(f, "surrogate_count must be greater than zero"),
            Self::ModelFitFailed => write!(f, "deterministic linear model fit failed"),
        }
    }
}

impl std::error::Error for RelationalPredictionError {}

/// Predictor families used for the ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionFeatureSet {
    /// Predict the held-out target using the last observed training target.
    PersistenceBaseline,
    /// Only isolated agent state summaries.
    IsolatedAgents,
    /// Only the supplied common-context signal.
    CommonDriver,
    /// Only simultaneous relational alignment.
    SynchronyOnly,
    /// Isolated agents + common driver + synchrony, with no relational
    /// directionality or turn-taking channels.
    NonRelationalContext,
    /// Non-relational context augmented with directional and turn-taking
    /// relational channels.
    RelationalAugmented,
    /// Alignment plus directional and turn-taking relational channels.
    RelationalProfile,
}

/// A fixed temporal train/test configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldOutRelationalPredictionConfig {
    /// Number of earliest observations used for fitting.
    pub train_samples: usize,
    /// Number of subsequent observations evaluated out of sample.
    pub test_samples: usize,
    /// Number of observations skipped between train and test feature windows.
    pub gap_samples: usize,
    /// Fixed ridge coefficient. A small prespecified value stabilizes
    /// deterministic normal-equation fits without adaptive tuning.
    pub ridge_lambda: f64,
}

impl Default for HeldOutRelationalPredictionConfig {
    fn default() -> Self {
        Self {
            train_samples: 32,
            test_samples: 16,
            gap_samples: 2,
            ridge_lambda: 1e-8,
        }
    }
}

impl HeldOutRelationalPredictionConfig {
    fn validate(&self, total_samples: usize) -> Result<(), RelationalPredictionError> {
        if self.train_samples < 8
            || self.test_samples < 4
            || self.train_samples
                .saturating_add(self.gap_samples)
                .saturating_add(self.test_samples)
                > total_samples
        {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            return Err(RelationalPredictionError::InvalidRidgeLambda);
        }

        Ok(())
    }

    fn test_start(&self) -> usize {
        self.train_samples + self.gap_samples
    }
}

/// Out-of-sample accuracy for one predictor family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionScore {
    pub feature_set: PredictionFeatureSet,
    pub parameter_count: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
}

impl PredictionScore {
    pub fn is_finite(&self) -> bool {
        self.mean_absolute_error.is_finite() && self.mean_squared_error.is_finite()
    }
}

/// Held-out comparison across the required baselines and the relational model.
///
/// No score is interpreted as a consciousness, relationship, value, or
/// causality measure. The only claim this structure supports is predictive
/// comparison under the supplied split and target definition.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutRelationalPredictionSummary {
    pub train_samples: usize,
    pub test_samples: usize,
    pub gap_samples: usize,
    pub minimum_outcome_horizon: f64,
    pub maximum_outcome_horizon: f64,
    pub persistence_baseline: PredictionScore,
    pub isolated_agents: PredictionScore,
    pub common_driver: PredictionScore,
    pub synchrony_only: PredictionScore,
    pub non_relational_context: PredictionScore,
    pub relational_augmented: PredictionScore,
    pub relational_profile: PredictionScore,
    pub status: EvidenceStatus,
}

impl HeldOutRelationalPredictionSummary {
    pub fn compute(
        samples: &[RelationalPredictionSample],
        config: HeldOutRelationalPredictionConfig,
    ) -> Result<Self, RelationalPredictionError> {
        validate_samples(samples)?;
        config.validate(samples.len())?;
        validate_temporal_boundary(samples, &config)?;

        let persistence_baseline =
            fit_and_score(samples, &config, PredictionFeatureSet::PersistenceBaseline)?;
        let scores = [
            fit_and_score(samples, &config, PredictionFeatureSet::IsolatedAgents)?,
            fit_and_score(samples, &config, PredictionFeatureSet::CommonDriver)?,
            fit_and_score(samples, &config, PredictionFeatureSet::SynchronyOnly)?,
            fit_and_score(samples, &config, PredictionFeatureSet::NonRelationalContext)?,
            fit_and_score(samples, &config, PredictionFeatureSet::RelationalAugmented)?,
            fit_and_score(samples, &config, PredictionFeatureSet::RelationalProfile)?,
        ];

        let horizon_values = samples
            .iter()
            .map(|sample| sample.outcome_time - sample.feature_time)
            .collect::<Vec<_>>();

        Ok(Self {
            train_samples: config.train_samples,
            test_samples: config.test_samples,
            gap_samples: config.gap_samples,
            minimum_outcome_horizon: horizon_values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            maximum_outcome_horizon: horizon_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            persistence_baseline,
            isolated_agents: scores[0],
            common_driver: scores[1],
            synchrony_only: scores[2],
            non_relational_context: scores[3],
            relational_augmented: scores[4],
            relational_profile: scores[5],
            status: EvidenceStatus::Measured,
        })
    }

    pub fn score(&self, feature_set: PredictionFeatureSet) -> PredictionScore {
        match feature_set {
            PredictionFeatureSet::PersistenceBaseline => self.persistence_baseline,
            PredictionFeatureSet::IsolatedAgents => self.isolated_agents,
            PredictionFeatureSet::CommonDriver => self.common_driver,
            PredictionFeatureSet::SynchronyOnly => self.synchrony_only,
            PredictionFeatureSet::NonRelationalContext => self.non_relational_context,
            PredictionFeatureSet::RelationalAugmented => self.relational_augmented,
            PredictionFeatureSet::RelationalProfile => self.relational_profile,
        }
    }

    /// Positive values mean the selected relational model has lower MSE than
    /// the specified baseline; negative values mean the baseline is better.
    pub fn relational_mse_improvement_over(
        &self,
        baseline: PredictionFeatureSet,
    ) -> Option<f64> {
        let baseline_mse = self.score(baseline).mean_squared_error;
        if baseline_mse <= 1e-20 {
            return None;
        }

        Some((baseline_mse - self.relational_profile.mean_squared_error) / baseline_mse)
    }

    /// Positive values mean relational channels add predictive information
    /// beyond the nested non-relational context model.
    pub fn augmented_mse_improvement_over_non_relational(&self) -> Option<f64> {
        let baseline_mse = self.non_relational_context.mean_squared_error;
        if baseline_mse <= 1e-20 {
            return None;
        }

        Some(
            (baseline_mse - self.relational_augmented.mean_squared_error) / baseline_mse,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RollingOriginRelationalPredictionConfig {
    /// Earliest index at which the first training window starts.
    pub first_origin: usize,
    /// Fixed training-window size.
    pub train_samples: usize,
    /// Fixed test-window size.
    pub test_samples: usize,
    /// Fixed gap between training and test feature windows.
    pub gap_samples: usize,
    /// Number of forward origins to evaluate.
    pub origin_count: usize,
    /// Forward step between origins.
    pub step_samples: usize,
    /// Fixed ridge coefficient shared across every origin.
    pub ridge_lambda: f64,
}

impl Default for RollingOriginRelationalPredictionConfig {
    fn default() -> Self {
        Self {
            first_origin: 0,
            train_samples: 48,
            test_samples: 16,
            gap_samples: 4,
            origin_count: 4,
            step_samples: 16,
            ridge_lambda: 1e-8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingOriginRelationalPredictionSummary {
    pub first_origin: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub gap_samples: usize,
    pub origin_count: usize,
    pub step_samples: usize,
    pub mean_persistence_mse: f64,
    pub mean_isolated_agents_mse: f64,
    pub mean_common_driver_mse: f64,
    pub mean_synchrony_only_mse: f64,
    pub mean_non_relational_context_mse: f64,
    pub mean_relational_augmented_mse: f64,
    pub mean_relational_profile_mse: f64,
    pub segments: Vec<HeldOutRelationalPredictionSummary>,
    pub status: EvidenceStatus,
}

impl RollingOriginRelationalPredictionSummary {
    pub fn compute(
        samples: &[RelationalPredictionSample],
        config: RollingOriginRelationalPredictionConfig,
    ) -> Result<Self, RelationalPredictionError> {
        validate_rolling_config(samples.len(), &config)?;

        let mut segments = Vec::with_capacity(config.origin_count);
        let segment_total = config
            .train_samples
            .checked_add(config.gap_samples)
            .and_then(|value| value.checked_add(config.test_samples))
            .ok_or(RelationalPredictionError::InvalidSplit)?;

        for origin in 0..config.origin_count {
            let start = config
                .step_samples
                .checked_mul(origin)
                .and_then(|offset| config.first_origin.checked_add(offset))
                .ok_or(RelationalPredictionError::InvalidSplit)?;
            let end = start
                .checked_add(segment_total)
                .ok_or(RelationalPredictionError::InvalidSplit)?;
            let segment = samples
                .get(start..end)
                .ok_or(RelationalPredictionError::InvalidSplit)?;

            segments.push(HeldOutRelationalPredictionSummary::compute(
                segment,
                HeldOutRelationalPredictionConfig {
                    train_samples: config.train_samples,
                    test_samples: config.test_samples,
                    gap_samples: config.gap_samples,
                    ridge_lambda: config.ridge_lambda,
                },
            )?);
        }

        let mean = |select: fn(&HeldOutRelationalPredictionSummary) -> f64| {
            segments.iter().map(select).sum::<f64>() / segments.len() as f64
        };

        Ok(Self {
            first_origin: config.first_origin,
            train_samples: config.train_samples,
            test_samples: config.test_samples,
            gap_samples: config.gap_samples,
            origin_count: config.origin_count,
            step_samples: config.step_samples,
            mean_persistence_mse: mean(|s| s.persistence_baseline.mean_squared_error),
            mean_isolated_agents_mse: mean(|s| s.isolated_agents.mean_squared_error),
            mean_common_driver_mse: mean(|s| s.common_driver.mean_squared_error),
            mean_synchrony_only_mse: mean(|s| s.synchrony_only.mean_squared_error),
            mean_non_relational_context_mse: mean(|s| s.non_relational_context.mean_squared_error),
            mean_relational_augmented_mse: mean(|s| s.relational_augmented.mean_squared_error),
            mean_relational_profile_mse: mean(|s| s.relational_profile.mean_squared_error),
            segments,
            status: EvidenceStatus::Measured,
        })
    }

    /// Positive means the relationally augmented model improves mean MSE over
    /// the nested non-relational context model.
    pub fn mean_augmented_mse_improvement(&self) -> Option<f64> {
        if self.mean_non_relational_context_mse <= 1e-20 {
            return None;
        }

        Some(
            (self.mean_non_relational_context_mse - self.mean_relational_augmented_mse)
                / self.mean_non_relational_context_mse,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingOriginRelationalPredictionQualification {
    pub observed: RollingOriginRelationalPredictionSummary,
    pub circular_shift_nulls: Vec<PredictionNullSummary>,
    pub feature_decoupling_nulls: Vec<PredictionNullSummary>,
    pub incremental_relational_nulls: Vec<PredictionNullSummary>,
}

impl RollingOriginRelationalPredictionQualification {
    pub fn compute(
        samples: &[RelationalPredictionSample],
        config: RollingOriginRelationalPredictionConfig,
        surrogate_count: usize,
    ) -> Result<Self, RelationalPredictionError> {
        if surrogate_count == 0 {
            return Err(RelationalPredictionError::InvalidSurrogateCount);
        }

        let observed = RollingOriginRelationalPredictionSummary::compute(samples, config)?;
        let segment_total = config.train_samples
            + config.gap_samples
            + config.test_samples;

        let mut circular_shift_nulls = Vec::with_capacity(config.origin_count);
        let mut feature_decoupling_nulls = Vec::with_capacity(config.origin_count);
        let mut incremental_relational_nulls = Vec::with_capacity(config.origin_count);

        for origin in 0..config.origin_count {
            let start = config.first_origin + origin * config.step_samples;
            let end = start + segment_total;
            let segment = &samples[start..end];
            let held_out_config = HeldOutRelationalPredictionConfig {
                train_samples: config.train_samples,
                test_samples: config.test_samples,
                gap_samples: config.gap_samples,
                ridge_lambda: config.ridge_lambda,
            };

            circular_shift_nulls.push(PredictionNullSummary::compute_for_feature_set(
                segment,
                held_out_config,
                PredictionNullFamily::CircularShift,
                PredictionFeatureSet::RelationalAugmented,
                surrogate_count,
            )?);

            feature_decoupling_nulls.push(PredictionNullSummary::compute_for_feature_set(
                segment,
                held_out_config,
                PredictionNullFamily::FeatureDecoupling,
                PredictionFeatureSet::RelationalAugmented,
                surrogate_count,
            )?);

            incremental_relational_nulls.push(PredictionNullSummary::compute_for_feature_set(
                segment,
                held_out_config,
                PredictionNullFamily::IncrementalRelationalShift,
                PredictionFeatureSet::RelationalAugmented,
                surrogate_count,
            )?);
        }

        Ok(Self {
            observed,
            circular_shift_nulls,
            feature_decoupling_nulls,
            incremental_relational_nulls,
        })
    }
}

fn validate_rolling_config(
    total_samples: usize,
    config: &RollingOriginRelationalPredictionConfig,
) -> Result<(), RelationalPredictionError> {
    if config.train_samples < 8
        || config.test_samples < 4
        || config.origin_count == 0
        || config.step_samples == 0
        || !config.ridge_lambda.is_finite()
        || config.ridge_lambda < 0.0
    {
        return Err(RelationalPredictionError::InvalidSplit);
    }

    let segment_total = config
        .train_samples
        .checked_add(config.gap_samples)
        .and_then(|value| value.checked_add(config.test_samples))
        .ok_or(RelationalPredictionError::InvalidSplit)?;

    let final_start = config
        .step_samples
        .checked_mul(config.origin_count - 1)
        .and_then(|offset| config.first_origin.checked_add(offset))
        .ok_or(RelationalPredictionError::InvalidSplit)?;

    let final_end = final_start
        .checked_add(segment_total)
        .ok_or(RelationalPredictionError::InvalidSplit)?;

    if final_end > total_samples {
        return Err(RelationalPredictionError::InvalidSplit);
    }

    Ok(())
}

/// Deterministic empirical calibration family for held-out relational prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionNullFamily {
    /// Shift the relational bundle together inside train and test partitions.
    CircularShift,
    /// Shift each relational channel by a distinct offset inside each partition.
    FeatureDecoupling,
    /// Keep synchrony and all non-relational context fixed while shifting only
    /// the incremental directionality/turn-taking channels.
    IncrementalRelationalShift,
}

/// Empirical null calibration for held-out relational prediction.
///
/// The exceedance fraction is the fraction of surrogates whose relational model
/// MSE is no worse than the observed relational model MSE. It is deliberately
/// not named or exposed as a formal p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionNullSummary {
    pub family: PredictionNullFamily,
    pub feature_set: PredictionFeatureSet,
    pub requested_surrogate_count: usize,
    pub surrogate_count: usize,
    pub observed_relational_mse: f64,
    pub minimum_surrogate_mse: f64,
    pub exceedance_count: usize,
    pub exceedance_fraction: f64,
    pub status: EvidenceStatus,
}

impl PredictionNullSummary {
    pub fn compute(
        samples: &[RelationalPredictionSample],
        config: HeldOutRelationalPredictionConfig,
        family: PredictionNullFamily,
        surrogate_count: usize,
    ) -> Result<Self, RelationalPredictionError> {
        Self::compute_for_feature_set(
            samples,
            config,
            family,
            PredictionFeatureSet::RelationalProfile,
            surrogate_count,
        )
    }

    pub fn compute_for_feature_set(
        samples: &[RelationalPredictionSample],
        config: HeldOutRelationalPredictionConfig,
        family: PredictionNullFamily,
        feature_set: PredictionFeatureSet,
        surrogate_count: usize,
    ) -> Result<Self, RelationalPredictionError> {
        validate_samples(samples)?;
        config.validate(samples.len())?;
        validate_temporal_boundary(samples, &config)?;

        if surrogate_count == 0 {
            return Err(RelationalPredictionError::InvalidSurrogateCount);
        }

        let observed = fit_and_score(samples, &config, feature_set)?;

        let capacity = config
            .train_samples
            .min(config.test_samples)
            .saturating_sub(1);
        if capacity == 0 {
            return Err(RelationalPredictionError::InsufficientSamples(
                config.train_samples.min(config.test_samples),
            ));
        }
        let count = surrogate_count.min(capacity);

        let mut minimum_surrogate_mse = f64::INFINITY;
        let mut exceedance_count = 0usize;

        for index in 0..count {
            let shift = 1 + (index * capacity / count);
            let surrogate = make_surrogate(samples, &config, family, shift);
            let score = fit_and_score(&surrogate, &config, feature_set)?;

            minimum_surrogate_mse = minimum_surrogate_mse.min(score.mean_squared_error);
            if score.mean_squared_error <= observed.mean_squared_error + 1e-12 {
                exceedance_count += 1;
            }
        }

        Ok(Self {
            family,
            feature_set,
            requested_surrogate_count: surrogate_count,
            surrogate_count: count,
            observed_relational_mse: observed.mean_squared_error,
            minimum_surrogate_mse,
            exceedance_count,
            exceedance_fraction: exceedance_count as f64 / count as f64,
            status: EvidenceStatus::Proxy,
        })
    }
}

/// Full held-out qualification bundle: observed ablation plus multiple null families.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutRelationalPredictionQualification {
    pub observed: HeldOutRelationalPredictionSummary,
    pub circular_shift_null: PredictionNullSummary,
    pub feature_decoupling_null: PredictionNullSummary,
}

impl HeldOutRelationalPredictionQualification {
    pub fn compute(
        samples: &[RelationalPredictionSample],
        config: HeldOutRelationalPredictionConfig,
        surrogate_count: usize,
    ) -> Result<Self, RelationalPredictionError> {
        let observed = HeldOutRelationalPredictionSummary::compute(samples, config)?;
        let circular_shift_null = PredictionNullSummary::compute(
            samples,
            config,
            PredictionNullFamily::CircularShift,
            surrogate_count,
        )?;
        let feature_decoupling_null = PredictionNullSummary::compute(
            samples,
            config,
            PredictionNullFamily::FeatureDecoupling,
            surrogate_count,
        )?;

        Ok(Self {
            observed,
            circular_shift_null,
            feature_decoupling_null,
        })
    }
}

fn validate_samples(
    samples: &[RelationalPredictionSample],
) -> Result<(), RelationalPredictionError> {
    if samples.len() < 12 {
        return Err(RelationalPredictionError::InsufficientSamples(samples.len()));
    }

    for pair in samples.windows(2) {
        if pair[1].feature_time <= pair[0].feature_time
            || pair[1].outcome_time <= pair[0].outcome_time
        {
            return Err(RelationalPredictionError::InvalidSplit);
        }
    }

    Ok(())
}

fn validate_temporal_boundary(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
) -> Result<(), RelationalPredictionError> {
    let test_start = config.test_start();
    let train_last = config.train_samples - 1;

    if samples[train_last].outcome_time >= samples[test_start].feature_time {
        return Err(RelationalPredictionError::TemporalLeakage);
    }

    Ok(())
}

fn fit_and_score(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
    feature_set: PredictionFeatureSet,
) -> Result<PredictionScore, RelationalPredictionError> {
    if feature_set == PredictionFeatureSet::PersistenceBaseline {
        return score_persistence_baseline(samples, config);
    }

    let coefficients = fit_linear_model(
        &samples[..config.train_samples],
        feature_set,
        config.ridge_lambda,
    )?;

    let test_start = config.test_start();
    let test_end = test_start + config.test_samples;
    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;

    for sample in &samples[test_start..test_end] {
        let features = feature_vector(sample, feature_set);
        let prediction = predict(&coefficients, &features);
        if !prediction.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        let error = prediction - sample.future_outcome;
        absolute_error += error.abs();
        squared_error += error * error;
    }

    let n = config.test_samples as f64;
    Ok(PredictionScore {
        feature_set,
        parameter_count: coefficients.len(),
        train_samples: config.train_samples,
        test_samples: config.test_samples,
        mean_absolute_error: absolute_error / n,
        mean_squared_error: squared_error / n,
    })
}

fn score_persistence_baseline(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
) -> Result<PredictionScore, RelationalPredictionError> {
    let prediction = samples[config.train_samples - 1].future_outcome;
    if !prediction.is_finite() {
        return Err(RelationalPredictionError::ModelFitFailed);
    }

    let test_start = config.test_start();
    let test_end = test_start + config.test_samples;
    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;

    for sample in &samples[test_start..test_end] {
        let error = prediction - sample.future_outcome;
        absolute_error += error.abs();
        squared_error += error * error;
    }

    let n = config.test_samples as f64;
    Ok(PredictionScore {
        feature_set: PredictionFeatureSet::PersistenceBaseline,
        parameter_count: 0,
        train_samples: config.train_samples,
        test_samples: config.test_samples,
        mean_absolute_error: absolute_error / n,
        mean_squared_error: squared_error / n,
    })
}

fn feature_vector(
    sample: &RelationalPredictionSample,
    feature_set: PredictionFeatureSet,
) -> Vec<f64> {
    match feature_set {
        PredictionFeatureSet::IsolatedAgents => vec![sample.agent_a, sample.agent_b],
        PredictionFeatureSet::CommonDriver => vec![sample.common_driver],
        PredictionFeatureSet::SynchronyOnly => vec![sample.alignment],
        PredictionFeatureSet::NonRelationalContext => vec![
            sample.agent_a,
            sample.agent_b,
            sample.common_driver,
            sample.alignment,
        ],
        PredictionFeatureSet::RelationalAugmented => vec![
            sample.agent_a,
            sample.agent_b,
            sample.common_driver,
            sample.alignment,
            sample.a_to_b,
            sample.b_to_a,
            sample.turn_taking,
        ],
        PredictionFeatureSet::RelationalProfile => vec![
            sample.alignment,
            sample.a_to_b,
            sample.b_to_a,
            sample.turn_taking,
        ],
    }
}

fn fit_linear_model(
    samples: &[RelationalPredictionSample],
    feature_set: PredictionFeatureSet,
    ridge_lambda: f64,
) -> Result<Vec<f64>, RelationalPredictionError> {
    let feature_count = feature_vector(&samples[0], feature_set).len();
    let dimension = feature_count + 1;
    let mut normal = vec![vec![0.0_f64; dimension + 1]; dimension];

    for sample in samples {
        let features = feature_vector(sample, feature_set);
        let mut row = Vec::with_capacity(dimension);
        row.push(1.0);
        row.extend(features);

        for i in 0..dimension {
            for j in 0..dimension {
                normal[i][j] += row[i] * row[j];
            }
            normal[i][dimension] += row[i] * sample.future_outcome;
        }
    }

    // Preserve the intercept; regularize only predictor coefficients.
    for i in 1..dimension {
        normal[i][i] += ridge_lambda;
    }

    gaussian_elimination(&mut normal)
}

fn gaussian_elimination(
    matrix: &mut [Vec<f64>],
) -> Result<Vec<f64>, RelationalPredictionError> {
    let dimension = matrix.len();

    for column in 0..dimension {
        let pivot_row = (column..dimension)
            .max_by(|&a, &b| {
                matrix[a][column]
                    .abs()
                    .partial_cmp(&matrix[b][column].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(RelationalPredictionError::ModelFitFailed)?;

        if matrix[pivot_row][column].abs() <= 1e-14 {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        matrix.swap(column, pivot_row);

        for row in (column + 1)..dimension {
            let factor = matrix[row][column] / matrix[column][column];
            if !factor.is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }

            for j in column..=dimension {
                matrix[row][j] -= factor * matrix[column][j];
            }
        }
    }

    let mut solution = vec![0.0_f64; dimension];
    for row in (0..dimension).rev() {
        let mut rhs = matrix[row][dimension];
        for j in (row + 1)..dimension {
            rhs -= matrix[row][j] * solution[j];
        }

        let divisor = matrix[row][row];
        if divisor.abs() <= 1e-14 {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        solution[row] = rhs / divisor;
        if !solution[row].is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
    }

    Ok(solution)
}

fn predict(coefficients: &[f64], features: &[f64]) -> f64 {
    let mut prediction = coefficients[0];
    for (coefficient, feature) in coefficients.iter().skip(1).zip(features) {
        prediction += coefficient * feature;
    }
    prediction
}

fn make_surrogate(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
    family: PredictionNullFamily,
    shift: usize,
) -> Vec<RelationalPredictionSample> {
    let test_start = config.test_start();

    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let segment_start = if index < config.train_samples {
                0
            } else if index >= test_start {
                test_start
            } else {
                return *sample;
            };
            let segment_len = if index < config.train_samples {
                config.train_samples
            } else {
                config.test_samples
            };
            let local_index = index - segment_start;
            let source = |channel_offset: usize| {
                let extra = match family {
                    PredictionNullFamily::CircularShift => 0,
                    PredictionNullFamily::FeatureDecoupling => channel_offset,
                    PredictionNullFamily::IncrementalRelationalShift => channel_offset,
                };
                segment_start + (local_index + shift * (extra + 1)) % segment_len
            };

            let shifted_alignment =
                !matches!(family, PredictionNullFamily::IncrementalRelationalShift);
            let alignment_sample = if shifted_alignment {
                samples[source(0)]
            } else {
                *sample
            };
            let a_to_b_sample = samples[source(1)];
            let b_to_a_sample = samples[source(2)];
            let turn_taking_sample = samples[source(3)];

            RelationalPredictionSample {
                alignment: alignment_sample.alignment,
                a_to_b: a_to_b_sample.a_to_b,
                b_to_a: b_to_a_sample.b_to_a,
                turn_taking: turn_taking_sample.turn_taking,
                ..*sample
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_sequence(i: usize, frequency: f64, phase: f64) -> f64 {
        (i as f64 * frequency + phase).sin() * 0.45 + 0.5
    }

    fn build_samples(outcome_horizon: f64) -> Vec<RelationalPredictionSample> {
        (0..80)
            .map(|i| {
                let alignment = deterministic_sequence(i, 0.13, 0.0);
                let a_to_b = deterministic_sequence(i, 0.071, 0.6);
                let b_to_a = deterministic_sequence(i, 0.097, 1.3);
                let turn_taking = deterministic_sequence(i, 0.31, 0.4);
                let agent_a = deterministic_sequence(i, 0.173, 0.2);
                let agent_b = deterministic_sequence(i, 0.117, 1.1);
                let common_driver = deterministic_sequence(i, 0.043, 0.9);
                let outcome =
                    0.61 * alignment + 0.23 * a_to_b + 0.11 * b_to_a + 0.05 * turn_taking;

                RelationalPredictionSample::new(
                    i as f64,
                    i as f64 + outcome_horizon,
                    agent_a,
                    agent_b,
                    alignment,
                    a_to_b,
                    b_to_a,
                    turn_taking,
                    common_driver,
                    outcome,
                )
                .unwrap()
            })
            .collect()
    }

    fn config() -> HeldOutRelationalPredictionConfig {
        HeldOutRelationalPredictionConfig {
            train_samples: 36,
            test_samples: 28,
            gap_samples: 4,
            ridge_lambda: 1e-8,
        }
    }

    #[test]
    fn held_out_prediction_is_temporally_separated() {
        let summary =
            HeldOutRelationalPredictionSummary::compute(&build_samples(0.5), config()).unwrap();

        assert_eq!(summary.status, EvidenceStatus::Measured);
        assert_eq!(summary.train_samples, 36);
        assert_eq!(summary.test_samples, 28);
        assert!(summary.minimum_outcome_horizon > 0.0);
        assert!(summary.relational_profile.mean_squared_error.is_finite());
    }

    #[test]
    fn relational_profile_outpredicts_synchrony_only_on_known_target() {
        let summary =
            HeldOutRelationalPredictionSummary::compute(&build_samples(0.5), config()).unwrap();

        assert!(
            summary.relational_profile.mean_squared_error
                < summary.synchrony_only.mean_squared_error * 0.05
        );
        assert!(
            summary.relational_profile.mean_squared_error
                < summary.persistence_baseline.mean_squared_error
        );
        assert!(
            summary.relational_augmented.mean_squared_error
                < summary.non_relational_context.mean_squared_error
        );
        assert!(summary.augmented_mse_improvement_over_non_relational().unwrap() > 0.0);
        assert!(
            summary
                .relational_mse_improvement_over(PredictionFeatureSet::SynchronyOnly)
                .unwrap()
                > 0.95
        );
    }

    #[test]
    fn temporal_leakage_is_rejected_by_outcome_horizon() {
        let result =
            HeldOutRelationalPredictionSummary::compute(&build_samples(5.0), config());

        assert_eq!(result, Err(RelationalPredictionError::TemporalLeakage));
    }

    #[test]
    fn deterministic_null_calibration_has_multiple_families() {
        let samples = build_samples(0.5);
        let first =
            HeldOutRelationalPredictionQualification::compute(&samples, config(), 12).unwrap();
        let second =
            HeldOutRelationalPredictionQualification::compute(&samples, config(), 12).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first.circular_shift_null.family,
            PredictionNullFamily::CircularShift
        );
        assert_eq!(
            first.feature_decoupling_null.family,
            PredictionNullFamily::FeatureDecoupling
        );
        assert_eq!(first.circular_shift_null.status, EvidenceStatus::Proxy);
        assert_eq!(first.feature_decoupling_null.status, EvidenceStatus::Proxy);
        assert_eq!(first.circular_shift_null.surrogate_count, 12);
        assert_eq!(first.feature_decoupling_null.surrogate_count, 12);
        assert!(
            (0.0..=1.0).contains(&first.circular_shift_null.exceedance_fraction)
        );
        assert!(
            (0.0..=1.0).contains(&first.feature_decoupling_null.exceedance_fraction)
        );
    }

    #[test]
    fn zero_surrogate_request_is_distinct_validation_failure() {
        let samples = build_samples(0.5);

        assert_eq!(
            PredictionNullSummary::compute(
                &samples,
                config(),
                PredictionNullFamily::CircularShift,
                0,
            ),
            Err(RelationalPredictionError::InvalidSurrogateCount)
        );
    }

    #[test]
    fn outcome_must_be_strictly_future() {
        assert_eq!(
            RelationalPredictionSample::new(
                1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0
            ),
            Err(RelationalPredictionError::OutcomeNotAfterFeatures)
        );
    }

    #[test]
    fn rolling_origin_repeats_the_temporal_boundary() {
        let samples = build_samples(0.5);
        let summary = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                train_samples: 32,
                test_samples: 8,
                gap_samples: 2,
                origin_count: 4,
                step_samples: 8,
                ridge_lambda: 1e-8,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(summary.status, EvidenceStatus::Measured);
        assert_eq!(summary.origin_count, 4);
        assert_eq!(summary.segments.len(), 4);
        assert!(summary.mean_relational_profile_mse.is_finite());
        assert!(summary.mean_augmented_mse_improvement().unwrap() >= 0.0);
    }

    #[test]
    fn rolling_origin_nulls_target_the_nested_relational_model() {
        let samples = build_samples(0.5);
        let config = RollingOriginRelationalPredictionConfig {
            train_samples: 32,
            test_samples: 8,
            gap_samples: 2,
            origin_count: 4,
            step_samples: 8,
            ridge_lambda: 1e-8,
            ..Default::default()
        };

        let qualification =
            RollingOriginRelationalPredictionQualification::compute(&samples, config, 8).unwrap();

        assert_eq!(qualification.observed.origin_count, 4);
        assert_eq!(qualification.circular_shift_nulls.len(), 4);
        assert_eq!(qualification.feature_decoupling_nulls.len(), 4);
        assert_eq!(qualification.incremental_relational_nulls.len(), 4);

        for null in qualification
            .circular_shift_nulls
            .iter()
            .chain(qualification.feature_decoupling_nulls.iter())
            .chain(qualification.incremental_relational_nulls.iter())
        {
            assert_eq!(null.feature_set, PredictionFeatureSet::RelationalAugmented);
            assert_eq!(null.status, EvidenceStatus::Proxy);
            assert_eq!(null.surrogate_count, 8);
            assert!((0.0..=1.0).contains(&null.exceedance_fraction));
        }
    }

    #[test]
    fn incremental_null_preserves_synchrony_and_breaks_relational_channels() {
        let samples = build_samples(0.5);
        let config = config();

        let summary = PredictionNullSummary::compute_for_feature_set(
            &samples,
            config,
            PredictionNullFamily::IncrementalRelationalShift,
            PredictionFeatureSet::RelationalAugmented,
            8,
        )
        .unwrap();

        assert_eq!(
            summary.feature_set,
            PredictionFeatureSet::RelationalAugmented
        );
        assert_eq!(summary.family, PredictionNullFamily::IncrementalRelationalShift);
        assert_eq!(summary.status, EvidenceStatus::Proxy);
    }

#[test]
    fn rolling_origin_is_deterministic() {
        let samples = build_samples(0.5);
        let config = RollingOriginRelationalPredictionConfig {
            train_samples: 32,
            test_samples: 8,
            gap_samples: 2,
            origin_count: 4,
            step_samples: 8,
            ridge_lambda: 1e-8,
            ..Default::default()
        };

        let first = RollingOriginRelationalPredictionSummary::compute(&samples, config).unwrap();
        let second = RollingOriginRelationalPredictionSummary::compute(&samples, config).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn rolling_origin_rejects_arithmetic_overflow() {
        let samples = build_samples(0.5);
        let result = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                first_origin: usize::MAX,
                ..Default::default()
            },
        );

        assert_eq!(result, Err(RelationalPredictionError::InvalidSplit));
    }

#[test]
    fn rolling_origin_rejects_zero_step_or_zero_origins() {
        let samples = build_samples(0.5);

        let zero_step = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                step_samples: 0,
                ..Default::default()
            },
        );
        assert_eq!(zero_step, Err(RelationalPredictionError::InvalidSplit));

        let zero_origins = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                origin_count: 0,
                ..Default::default()
            },
        );
        assert_eq!(zero_origins, Err(RelationalPredictionError::InvalidSplit));
    }

    #[test]
    fn relational_score_is_not_promoted_to_authority() {
        let summary =
            HeldOutRelationalPredictionSummary::compute(&build_samples(0.5), config()).unwrap();

        assert_eq!(summary.status, EvidenceStatus::Measured);
        assert_eq!(
            summary.relational_profile.feature_set,
            PredictionFeatureSet::RelationalProfile
        );
    }
}
