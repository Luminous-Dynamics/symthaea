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
    InvalidEvidenceProvenance(&'static str),
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
            Self::InvalidEvidenceProvenance(name) => {
                write!(f, "evidence provenance field {name} is invalid")
            }
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

#[derive(Debug, Clone, PartialEq)]
struct FittedLinearModel {
    coefficients: Vec<f64>,
    means: Vec<f64>,
    scales: Vec<f64>,
}

impl PredictionFeatureSet {
    fn all() -> [Self; 7] {
        [
            Self::PersistenceBaseline,
            Self::IsolatedAgents,
            Self::CommonDriver,
            Self::SynchronyOnly,
            Self::NonRelationalContext,
            Self::RelationalAugmented,
            Self::RelationalProfile,
        ]
    }
}

impl PredictionScore {
    pub fn is_finite(&self) -> bool {
        self.mean_absolute_error.is_finite() && self.mean_squared_error.is_finite()
    }
}

/// Caller-attested provenance for an empirical qualification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalPredictionProvenance {
    pub protocol_id: String,
    pub source_data_sha256: String,
    pub software_commit_sha: String,
}

impl RelationalPredictionProvenance {
    pub fn new(
        protocol_id: impl Into<String>,
        source_data_sha256: impl Into<String>,
        software_commit_sha: impl Into<String>,
    ) -> Result<Self, RelationalPredictionError> {
        let provenance = Self {
            protocol_id: protocol_id.into(),
            source_data_sha256: source_data_sha256.into(),
            software_commit_sha: software_commit_sha.into(),
        };

        if provenance.protocol_id.trim().is_empty() {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "protocol_id",
            ));
        }
        if !is_hex_digest(&provenance.source_data_sha256, 64) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "source_data_sha256",
            ));
        }
        if !is_hex_digest(&provenance.software_commit_sha, 40) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "software_commit_sha",
            ));
        }

        Ok(provenance)
    }
}

/// Exact held-out prediction trace for one feature family.
///
/// This retains fitted preprocessing/model parameters and every held-out
/// prediction so MAE/MSE can be recomputed from frozen inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionEvidenceRecord {
    pub feature_set: PredictionFeatureSet,
    pub train_samples: usize,
    pub test_samples: usize,
    pub feature_times: Vec<f64>,
    pub outcome_times: Vec<f64>,
    pub observed_outcomes: Vec<f64>,
    pub predictions: Vec<f64>,
    pub fit_coefficients: Option<Vec<f64>>,
    pub feature_means: Vec<f64>,
    pub feature_scales: Vec<f64>,
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
}

impl PredictionEvidenceRecord {
    pub fn score(&self) -> PredictionScore {
        PredictionScore {
            feature_set: self.feature_set,
            parameter_count: self.fit_coefficients.as_ref().map_or(0, Vec::len),
            train_samples: self.train_samples,
            test_samples: self.test_samples,
            mean_absolute_error: self.mean_absolute_error,
            mean_squared_error: self.mean_squared_error,
        }
    }

    /// Recompute the stored loss metrics from the retained held-out trace.
    pub fn validate_trace(&self) -> Result<(), RelationalPredictionError> {
        if self.predictions.len() != self.test_samples
            || self.observed_outcomes.len() != self.test_samples
            || self.feature_times.len() != self.test_samples
            || self.outcome_times.len() != self.test_samples
            || self.predictions.len() != self.observed_outcomes.len()
        {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        if let Some(coefficients) = &self.fit_coefficients {
            if coefficients.len() != self.feature_means.len() + 1
                || coefficients.len() != self.feature_scales.len() + 1
            {
                return Err(RelationalPredictionError::InvalidSplit);
            }
        } else if !self.feature_means.is_empty() || !self.feature_scales.is_empty() {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        for pair in self.feature_times.windows(2) {
            if pair[1] <= pair[0] {
                return Err(RelationalPredictionError::InvalidSplit);
            }
        }
        for (feature_time, outcome_time) in
            self.feature_times.iter().zip(&self.outcome_times)
        {
            if *outcome_time <= *feature_time {
                return Err(RelationalPredictionError::OutcomeNotAfterFeatures);
            }
        }

        let mut absolute_error = 0.0;
        let mut squared_error = 0.0;
        for (prediction, outcome) in self.predictions.iter().zip(&self.observed_outcomes) {
            let error = *prediction - *outcome;
            absolute_error += error.abs();
            squared_error += error * error;
            if !absolute_error.is_finite() || !squared_error.is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }
        }

        let n = self.test_samples as f64;
        if n <= 0.0 {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        let mean_absolute_error = absolute_error / n;
        let mean_squared_error = squared_error / n;
        let tolerance = 1e-12;

        if (mean_absolute_error - self.mean_absolute_error).abs() > tolerance
            || (mean_squared_error - self.mean_squared_error).abs() > tolerance
        {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        Ok(())
    }
}

/// One complete held-out evidence packet.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutRelationalPredictionEvidence {
    pub provenance: RelationalPredictionProvenance,
    pub summary: HeldOutRelationalPredictionSummary,
    pub records: Vec<PredictionEvidenceRecord>,
}

/// Rolling-origin packet retaining a complete trace at each origin.
#[derive(Debug, Clone, PartialEq)]
pub struct RollingOriginRelationalPredictionEvidence {
    pub provenance: RelationalPredictionProvenance,
    pub config: RollingOriginRelationalPredictionConfig,
    pub observed: RollingOriginRelationalPredictionSummary,
    pub origins: Vec<HeldOutRelationalPredictionEvidence>,
}

/// Caller-attested provenance for an empirical qualification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalPredictionProvenance {
    pub protocol_id: String,
    pub source_data_sha256: String,
    pub software_commit_sha: String,
}

impl RelationalPredictionProvenance {
    pub fn new(
        protocol_id: impl Into<String>,
        source_data_sha256: impl Into<String>,
        software_commit_sha: impl Into<String>,
    ) -> Result<Self, RelationalPredictionError> {
        let provenance = Self {
            protocol_id: protocol_id.into(),
            source_data_sha256: source_data_sha256.into(),
            software_commit_sha: software_commit_sha.into(),
        };

        if provenance.protocol_id.trim().is_empty() {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "protocol_id",
            ));
        }
        if !is_hex_digest(&provenance.source_data_sha256, 64) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "source_data_sha256",
            ));
        }
        if !is_hex_digest(&provenance.software_commit_sha, 40) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "software_commit_sha",
            ));
        }

        Ok(provenance)
    }
}

/// Exact held-out prediction trace for one feature family.
///
/// This retains fitted preprocessing/model parameters and every held-out
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

impl HeldOutRelationalPredictionEvidence {
    pub fn validate(&self) -> Result<(), RelationalPredictionError> {
        Self::validate_provenance(&self.provenance)?;
        if self.records.len() != PredictionFeatureSet::all().len() {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        for record in &self.records {
            record.validate_trace()?;
            if record.score() != self.summary.score(record.feature_set) {
                return Err(RelationalPredictionError::InvalidSplit);
            }
        }

        Ok(())
    }

    pub fn to_json(&self) -> Result<String, RelationalPredictionError> {
        self.validate()?;

        let scores = PredictionFeatureSet::all()
            .into_iter()
            .map(|feature_set| {
                let score = self.summary.score(feature_set);
                serde_json::json!({
                    "feature_set": feature_set_name(feature_set),
                    "parameter_count": score.parameter_count,
                    "train_samples": score.train_samples,
                    "test_samples": score.test_samples,
                    "mean_absolute_error": score.mean_absolute_error,
                    "mean_squared_error": score.mean_squared_error
                })
            })
            .collect::<Vec<_>>();

        let records = self.records
            .iter()
            .map(prediction_evidence_record_json)
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "schema": "relational-prediction-evidence/v1",
            "provenance": {
                "protocol_id": self.provenance.protocol_id,
                "source_data_sha256": self.provenance.source_data_sha256,
                "software_commit_sha": self.provenance.software_commit_sha
            },
            "split": {
                "train_samples": self.summary.train_samples,
                "test_samples": self.summary.test_samples,
                "gap_samples": self.summary.gap_samples,
                "minimum_outcome_horizon": self.summary.minimum_outcome_horizon,
                "maximum_outcome_horizon": self.summary.maximum_outcome_horizon
            },
            "scores": scores,
            "records": records
        }).to_string())
    }

    fn validate_provenance(
        provenance: &RelationalPredictionProvenance,
    ) -> Result<(), RelationalPredictionError> {
        if provenance.protocol_id.trim().is_empty() {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "protocol_id",
            ));
        }
        if !is_hex_digest(&provenance.source_data_sha256, 64) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "source_data_sha256",
            ));
        }
        if !is_hex_digest(&provenance.software_commit_sha, 40) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "software_commit_sha",
            ));
        }

        Ok(())
    }
}

impl HeldOutRelationalPredictionSummary {
    pub fn compute_evidence(
        samples: &[RelationalPredictionSample],
        config: HeldOutRelationalPredictionConfig,
        provenance: RelationalPredictionProvenance,
    ) -> Result<HeldOutRelationalPredictionEvidence, RelationalPredictionError> {
        let summary = Self::compute(samples, config)?;
        let records = PredictionFeatureSet::all()
            .into_iter()
            .map(|feature_set| fit_prediction_record(samples, &config, feature_set))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(HeldOutRelationalPredictionEvidence {
            provenance,
            summary,
            records,
        })
    }

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
    /// Forward step between origins. Must be at least test_samples so
    /// held-out target windows do not overlap.
    pub step_samples: usize,
    /// Fixed forecast horizon in the same time units as the samples.
    pub forecast_horizon: f64,
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
            forecast_horizon: 1.0,
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
    pub forecast_horizon: f64,
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

impl RollingOriginRelationalPredictionEvidence {
    pub fn validate(&self) -> Result<(), RelationalPredictionError> {
        if self.origins.len() != self.config.origin_count {
            return Err(RelationalPredictionError::InvalidSplit);
        }
        for origin in &self.origins {
            origin.validate()?;
        }
        if self.origins.iter().any(|origin| origin.summary.status != EvidenceStatus::Measured) {
            return Err(RelationalPredictionError::InvalidSplit);
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, RelationalPredictionError> {
        self.validate()?;

        let origins = self.origins
            .iter()
            .map(|origin| {
                serde_json::from_str::<serde_json::Value>(
                    &origin.to_json().expect("validated evidence must serialize"),
                )
                .map_err(|_| RelationalPredictionError::ModelFitFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(serde_json::json!({
            "schema": "relational-prediction-rolling-evidence/v1",
            "provenance": {
                "protocol_id": self.provenance.protocol_id,
                "source_data_sha256": self.provenance.source_data_sha256,
                "software_commit_sha": self.provenance.software_commit_sha
            },
            "rolling_config": {
                "first_origin": self.config.first_origin,
                "train_samples": self.config.train_samples,
                "test_samples": self.config.test_samples,
                "gap_samples": self.config.gap_samples,
                "origin_count": self.config.origin_count,
                "step_samples": self.config.step_samples,
                "forecast_horizon": self.config.forecast_horizon,
                "ridge_lambda": self.config.ridge_lambda
            },
            "observed": {
                "mean_persistence_mse": self.observed.mean_persistence_mse,
                "mean_isolated_agents_mse": self.observed.mean_isolated_agents_mse,
                "mean_common_driver_mse": self.observed.mean_common_driver_mse,
                "mean_synchrony_only_mse": self.observed.mean_synchrony_only_mse,
                "mean_non_relational_context_mse": self.observed.mean_non_relational_context_mse,
                "mean_relational_augmented_mse": self.observed.mean_relational_augmented_mse,
                "mean_relational_profile_mse": self.observed.mean_relational_profile_mse
            },
            "origins": origins
        }).to_string())
    }
}

impl RollingOriginRelationalPredictionSummary {
    pub fn compute_evidence(
        samples: &[RelationalPredictionSample],
        config: RollingOriginRelationalPredictionConfig,
        provenance: RelationalPredictionProvenance,
    ) -> Result<RollingOriginRelationalPredictionEvidence, RelationalPredictionError> {
        let observed = Self::compute(samples, config)?;
        let segment_total = config
            .train_samples
            .checked_add(config.gap_samples)
            .and_then(|value| value.checked_add(config.test_samples))
            .ok_or(RelationalPredictionError::InvalidSplit)?;

        let mut origins = Vec::with_capacity(config.origin_count);
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

            origins.push(HeldOutRelationalPredictionSummary::compute_evidence(
                segment,
                HeldOutRelationalPredictionConfig {
                    train_samples: config.train_samples,
                    test_samples: config.test_samples,
                    gap_samples: config.gap_samples,
                    ridge_lambda: config.ridge_lambda,
                },
                provenance.clone(),
            )?);
        }

        Ok(RollingOriginRelationalPredictionEvidence {
            provenance,
            config,
            observed,
            origins,
        })
    }

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

            validate_forecast_horizon(segment, config.forecast_horizon)?;

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
            forecast_horizon: config.forecast_horizon,
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

    /// Relative MSE improvement for every rolling origin, retaining the full
    /// vector so heterogeneity cannot be hidden by the mean.
    pub fn augmented_mse_improvement_per_origin(&self) -> Vec<Option<f64>> {
        self.segments
            .iter()
            .map(|segment| segment.augmented_mse_improvement_over_non_relational())
            .collect()
    }

    /// Median per-origin relative MSE improvement over the nested baseline.
    pub fn median_augmented_mse_improvement(&self) -> Option<f64> {
        let mut values = self
            .augmented_mse_improvement_per_origin()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        if values.is_empty() {
            return None;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let middle = values.len() / 2;

        if values.len() % 2 == 1 {
            Some(values[middle])
        } else {
            Some((values[middle - 1] + values[middle]) / 2.0)
        }
    }

    /// Worst per-origin relative MSE improvement over the nested baseline.
    pub fn minimum_augmented_mse_improvement(&self) -> Option<f64> {
        self.augmented_mse_improvement_per_origin()
            .into_iter()
            .flatten()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of origins where the relational augmentation strictly improves
    /// held-out MSE over the nested non-relational context model.
    pub fn origins_beating_non_relational(&self) -> usize {
        self.segments
            .iter()
            .filter(|segment| {
                segment.relational_augmented.mean_squared_error
                    < segment.non_relational_context.mean_squared_error
            })
            .count()
    }

    /// Number of origins where the relational augmentation strictly improves
    /// held-out MSE over the persistence baseline.
    pub fn origins_beating_persistence(&self) -> usize {
        self.segments
            .iter()
            .filter(|segment| {
                segment.relational_augmented.mean_squared_error
                    < segment.persistence_baseline.mean_squared_error
            })
            .count()
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
        let segment_total = config
            .train_samples
            .checked_add(config.gap_samples)
            .and_then(|value| value.checked_add(config.test_samples))
            .ok_or(RelationalPredictionError::InvalidSplit)?;

        let mut circular_shift_nulls = Vec::with_capacity(config.origin_count);
        let mut feature_decoupling_nulls = Vec::with_capacity(config.origin_count);
        let mut incremental_relational_nulls = Vec::with_capacity(config.origin_count);

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

fn validate_forecast_horizon(
    samples: &[RelationalPredictionSample],
    expected_horizon: f64,
) -> Result<(), RelationalPredictionError> {
    if !expected_horizon.is_finite() || expected_horizon <= 0.0 {
        return Err(RelationalPredictionError::InvalidSplit);
    }

    let tolerance = 1e-9 * expected_horizon.abs().max(1.0);
    for sample in samples {
        let horizon = sample.outcome_time - sample.feature_time;
        if (horizon - expected_horizon).abs() > tolerance {
            return Err(RelationalPredictionError::InvalidSplit);
        }
    }

    Ok(())
}

fn validate_rolling_config(
    total_samples: usize,
    config: &RollingOriginRelationalPredictionConfig,
) -> Result<(), RelationalPredictionError> {
    if config.train_samples < 8
        || config.test_samples < 4
        || config.origin_count == 0
        || config.step_samples < config.test_samples
        || !config.forecast_horizon.is_finite()
        || config.forecast_horizon <= 0.0
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
    /// Shift relational channels by channel-specific non-zero offsets inside each partition.
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
            let shift = index
                .checked_mul(capacity)
                .and_then(|value| value.checked_div(count))
                .and_then(|value| value.checked_add(1))
                .ok_or(RelationalPredictionError::InvalidSplit)?;
            let surrogate = make_surrogate(samples, &config, family, shift)?;
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

fn feature_set_name(feature_set: PredictionFeatureSet) -> &'static str {
    match feature_set {
        PredictionFeatureSet::PersistenceBaseline => "PersistenceBaseline",
        PredictionFeatureSet::IsolatedAgents => "IsolatedAgents",
        PredictionFeatureSet::CommonDriver => "CommonDriver",
        PredictionFeatureSet::SynchronyOnly => "SynchronyOnly",
        PredictionFeatureSet::NonRelationalContext => "NonRelationalContext",
        PredictionFeatureSet::RelationalAugmented => "RelationalAugmented",
        PredictionFeatureSet::RelationalProfile => "RelationalProfile",
    }
}

fn prediction_evidence_record_json(
    record: &PredictionEvidenceRecord,
) -> serde_json::Value {
    serde_json::json!({
        "feature_set": feature_set_name(record.feature_set),
        "train_samples": record.train_samples,
        "test_samples": record.test_samples,
        "feature_times": record.feature_times,
        "outcome_times": record.outcome_times,
        "observed_outcomes": record.observed_outcomes,
        "predictions": record.predictions,
        "fit_coefficients": record.fit_coefficients,
        "feature_means": record.feature_means,
        "feature_scales": record.feature_scales,
        "mean_absolute_error": record.mean_absolute_error,
        "mean_squared_error": record.mean_squared_error
    })
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

fn fit_prediction_record(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
    feature_set: PredictionFeatureSet,
) -> Result<PredictionEvidenceRecord, RelationalPredictionError> {
    validate_samples(samples)?;
    config.validate(samples.len())?;
    validate_temporal_boundary(samples, config)?;

    let test_start = config.test_start();
    let test_end = test_start
        .checked_add(config.test_samples)
        .ok_or(RelationalPredictionError::InvalidSplit)?;

    let (model, baseline) = if feature_set == PredictionFeatureSet::PersistenceBaseline {
        (
            None,
            Some(samples[config.train_samples - 1].future_outcome),
        )
    } else {
        (
            Some(fit_linear_model(
                &samples[..config.train_samples],
                feature_set,
                config.ridge_lambda,
            )?),
            None,
        )
    };

    let fit_coefficients = model.as_ref().map(|model| model.coefficients.clone());
    let feature_means = model
        .as_ref()
        .map_or_else(Vec::new, |model| model.means.clone());
    let feature_scales = model
        .as_ref()
        .map_or_else(Vec::new, |model| model.scales.clone());

    let mut feature_times = Vec::with_capacity(config.test_samples);
    let mut outcome_times = Vec::with_capacity(config.test_samples);
    let mut observed_outcomes = Vec::with_capacity(config.test_samples);
    let mut predictions = Vec::with_capacity(config.test_samples);
    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;

    for sample in &samples[test_start..test_end] {
        let prediction = match (&model, baseline) {
            (Some(model), None) => predict(model, &feature_vector(sample, feature_set)),
            (None, Some(value)) => value,
            _ => return Err(RelationalPredictionError::ModelFitFailed),
        };

        if !prediction.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        let error = prediction - sample.future_outcome;
        absolute_error += error.abs();
        squared_error += error * error;
        if !absolute_error.is_finite() || !squared_error.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        feature_times.push(sample.feature_time);
        outcome_times.push(sample.outcome_time);
        observed_outcomes.push(sample.future_outcome);
        predictions.push(prediction);
    }

    let n = config.test_samples as f64;
    Ok(PredictionEvidenceRecord {
        feature_set,
        train_samples: config.train_samples,
        test_samples: config.test_samples,
        feature_times,
        outcome_times,
        observed_outcomes,
        predictions,
        fit_coefficients,
        feature_means,
        feature_scales,
        mean_absolute_error: absolute_error / n,
        mean_squared_error: squared_error / n,
    })
}

fn fit_and_score(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
    feature_set: PredictionFeatureSet,
) -> Result<PredictionScore, RelationalPredictionError> {
    if feature_set == PredictionFeatureSet::PersistenceBaseline {
        return score_persistence_baseline(samples, config);
    }

    let model = fit_linear_model(
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
        let prediction = predict(&model, &features);
        if !prediction.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }

        let error = prediction - sample.future_outcome;
        absolute_error += error.abs();
        squared_error += error * error;
        if !absolute_error.is_finite() || !squared_error.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
    }

    let n = config.test_samples as f64;
    Ok(PredictionScore {
        feature_set,
        parameter_count: model.coefficients.len(),
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
        if !absolute_error.is_finite() || !squared_error.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
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
) -> Result<FittedLinearModel, RelationalPredictionError> {
    if samples.is_empty() {
        return Err(RelationalPredictionError::InsufficientSamples(0));
    }

    let feature_count = feature_vector(&samples[0], feature_set).len();
    let sample_count = samples.len() as f64;
    let mut means = vec![0.0_f64; feature_count];

    // Preprocessing statistics are fit exclusively on the training window.
    for sample in samples {
        let features = feature_vector(sample, feature_set);
        for (mean, feature) in means.iter_mut().zip(features) {
            *mean += feature;
        }
    }

    for mean in &mut means {
        *mean /= sample_count;
        if !mean.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
    }

    let mut scales = vec![0.0_f64; feature_count];
    for sample in samples {
        let features = feature_vector(sample, feature_set);
        for ((scale, feature), mean) in scales.iter_mut().zip(features).zip(&means) {
            let centered = feature - *mean;
            *scale += centered * centered;
            if !scale.is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }
        }
    }

    for scale in &mut scales {
        *scale = (*scale / sample_count).sqrt();
        if !scale.is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
        // Constant training features are centered to zero; unit scale keeps
        // the transform total and avoids division by zero.
        if *scale <= 1e-12 {
            *scale = 1.0;
        }
    }

    let dimension = feature_count + 1;
    let mut normal = vec![vec![0.0_f64; dimension + 1]; dimension];

    for sample in samples {
        let features = feature_vector(sample, feature_set);
        let mut row = Vec::with_capacity(dimension);
        row.push(1.0);

        for ((feature, mean), scale) in features.iter().zip(&means).zip(&scales) {
            let standardized = (*feature - *mean) / *scale;
            if !standardized.is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }
            row.push(standardized);
        }

        for i in 0..dimension {
            for j in 0..dimension {
                normal[i][j] += row[i] * row[j];
                if !normal[i][j].is_finite() {
                    return Err(RelationalPredictionError::ModelFitFailed);
                }
            }
            normal[i][dimension] += row[i] * sample.future_outcome;
            if !normal[i][dimension].is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }
        }
    }

    // Ridge is applied in standardized predictor space, so lambda is not
    // implicitly changed by the raw units of a feature.
    for i in 1..dimension {
        normal[i][i] += ridge_lambda;
        if !normal[i][i].is_finite() {
            return Err(RelationalPredictionError::ModelFitFailed);
        }
    }

    Ok(FittedLinearModel {
        coefficients: gaussian_elimination(&mut normal)?,
        means,
        scales,
    })
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
                if !matrix[row][j].is_finite() {
                    return Err(RelationalPredictionError::ModelFitFailed);
                }
            }
        }
    }

    let mut solution = vec![0.0_f64; dimension];
    for row in (0..dimension).rev() {
        let mut rhs = matrix[row][dimension];
        for j in (row + 1)..dimension {
            rhs -= matrix[row][j] * solution[j];
            if !rhs.is_finite() {
                return Err(RelationalPredictionError::ModelFitFailed);
            }
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

fn predict(model: &FittedLinearModel, features: &[f64]) -> f64 {
    let mut prediction = model.coefficients[0];

    for (((coefficient, mean), scale), feature) in model
        .coefficients
        .iter()
        .skip(1)
        .zip(&model.means)
        .zip(&model.scales)
        .zip(features)
    {
        let standardized = (*feature - *mean) / *scale;
        if !standardized.is_finite() {
            return f64::NAN;
        }
        prediction += coefficient * standardized;
    }

    prediction
}

fn make_surrogate(
    samples: &[RelationalPredictionSample],
    config: &HeldOutRelationalPredictionConfig,
    family: PredictionNullFamily,
    shift: usize,
) -> Result<Vec<RelationalPredictionSample>, RelationalPredictionError> {
    if shift == 0 {
        return Err(RelationalPredictionError::InvalidSurrogateCount);
    }

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
            let source = |channel_offset: usize| -> Result<usize, RelationalPredictionError> {
                let extra = match family {
                    PredictionNullFamily::CircularShift => 0,
                    PredictionNullFamily::FeatureDecoupling => channel_offset,
                    PredictionNullFamily::IncrementalRelationalShift => channel_offset,
                };
                if segment_len < 2 {
                    return Err(RelationalPredictionError::InsufficientSamples(segment_len));
                }
                if matches!(family, PredictionNullFamily::FeatureDecoupling) && segment_len < 5 {
                    return Err(RelationalPredictionError::InsufficientSamples(segment_len));
                }
                // Use consecutive non-zero offsets rather than multiplication:
                // for FeatureDecoupling this guarantees distinct channel offsets
                // whenever at least four non-zero circular offsets exist.
                let normalized_offset = shift
                    .checked_sub(1)
                    .and_then(|base| base.checked_add(extra))
                    .ok_or(RelationalPredictionError::InvalidSplit)?;
                let offset = 1 + normalized_offset % (segment_len - 1);
                let source_index = local_index
                    .checked_add(offset)
                    .ok_or(RelationalPredictionError::InvalidSplit)?
                    % segment_len;
                segment_start
                    .checked_add(source_index)
                    .ok_or(RelationalPredictionError::InvalidSplit)
            };

            let shifted_alignment =
                !matches!(family, PredictionNullFamily::IncrementalRelationalShift);
            let alignment_sample = if shifted_alignment {
                samples[source(0)?]
            } else {
                *sample
            };
            let a_to_b_sample = samples[source(1)?];
            let b_to_a_sample = samples[source(2)?];
            let turn_taking_sample = samples[source(3)?];

            Ok(RelationalPredictionSample {
                alignment: alignment_sample.alignment,
                a_to_b: a_to_b_sample.a_to_b,
                b_to_a: b_to_a_sample.b_to_a,
                turn_taking: turn_taking_sample.turn_taking,
                ..*sample
            })
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
    fn evidence_provenance_requires_exact_hashes() {
        assert_eq!(
            RelationalPredictionProvenance::new(
                "",
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "0123456789abcdef0123456789abcdef01234567",
            ),
            Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "protocol_id"
            ))
        );

        assert_eq!(
            RelationalPredictionProvenance::new(
                "RH-006-v1",
                "not-a-sha",
                "0123456789abcdef0123456789abcdef01234567",
            ),
            Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "source_data_sha256"
            ))
        );

        let valid = RelationalPredictionProvenance::new(
            "RH-006-v1",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        assert_eq!(valid.protocol_id, "RH-006-v1");
    }

    #[test]
    fn evidence_records_reproduce_compact_scores() {
        let samples = build_samples(0.5);
        let provenance = RelationalPredictionProvenance::new(
            "RH-006-v1",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();

        let evidence = HeldOutRelationalPredictionSummary::compute_evidence(
            &samples,
            config(),
            provenance,
        )
        .unwrap();

        assert_eq!(evidence.records.len(), 7);
        for record in &evidence.records {
            assert_eq!(record.predictions.len(), record.test_samples);
            assert_eq!(record.observed_outcomes.len(), record.test_samples);
            assert_eq!(record.feature_times.len(), record.test_samples);
            assert_eq!(record.outcome_times.len(), record.test_samples);
            assert_eq!(record.score(), evidence.summary.score(record.feature_set));
            record.validate_trace().unwrap();
        }
    }

    #[test]
    fn evidence_packet_validates_and_serializes() {
        let samples = build_samples(0.5);
        let provenance = RelationalPredictionProvenance::new(
            "RH-006-v1",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();

        let evidence = HeldOutRelationalPredictionSummary::compute_evidence(
            &samples,
            config(),
            provenance,
        )
        .unwrap();

        evidence.validate().unwrap();
        let json = evidence.to_json().unwrap();
        assert!(json.contains("relational-prediction-evidence/v1"));
        assert!(json.contains("RelationalAugmented"));
        assert!(json.contains("predictions"));
    }

    #[test]
    fn evidence_trace_rejects_tampered_loss() {
        let samples = build_samples(0.5);
        let provenance = RelationalPredictionProvenance::new(
            "RH-006-v1",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();

        let mut evidence = HeldOutRelationalPredictionSummary::compute_evidence(
            &samples,
            config(),
            provenance,
        )
        .unwrap();
        evidence.records[0].mean_squared_error += 0.1;

        assert_eq!(
            evidence.records[0].validate_trace(),
            Err(RelationalPredictionError::InvalidSplit)
        );
    }

    #[test]
    fn train_only_standardization_is_deterministic_and_finite() {
        let samples = build_samples(0.5);
        let first = fit_linear_model(
            &samples[..36],
            PredictionFeatureSet::RelationalAugmented,
            1e-8,
        )
        .unwrap();
        let second = fit_linear_model(
            &samples[..36],
            PredictionFeatureSet::RelationalAugmented,
            1e-8,
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.means.len(), 7);
        assert_eq!(first.scales.len(), 7);
        assert!(first.means.iter().all(|value| value.is_finite()));
        assert!(first.scales.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(first.coefficients.iter().all(|value| value.is_finite()));
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
    fn rolling_origin_rejects_overlapping_test_windows_or_wrong_horizon() {
        let samples = build_samples(0.5);

        let overlapping = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                train_samples: 32,
                test_samples: 8,
                gap_samples: 2,
                origin_count: 4,
                step_samples: 4,
                forecast_horizon: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(overlapping, Err(RelationalPredictionError::InvalidSplit));

        let wrong_horizon = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                train_samples: 32,
                test_samples: 8,
                gap_samples: 2,
                origin_count: 4,
                step_samples: 8,
                forecast_horizon: 1.0,
                ..Default::default()
            },
        );
        assert_eq!(wrong_horizon, Err(RelationalPredictionError::InvalidSplit));
    }

    #[test]
    fn rolling_origin_exposes_per_origin_stability_without_a_pass_threshold() {
        let samples = build_samples(0.5);
        let summary = RollingOriginRelationalPredictionSummary::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                train_samples: 32,
                test_samples: 8,
                gap_samples: 2,
                origin_count: 4,
                step_samples: 8,
                forecast_horizon: 0.5,
                ridge_lambda: 1e-8,
                ..Default::default()
            },
        )
        .unwrap();

        let improvements = summary.augmented_mse_improvement_per_origin();
        assert_eq!(improvements.len(), 4);
        assert!(improvements.iter().all(Option::is_some));
        assert!(summary.median_augmented_mse_improvement().is_some());
        assert!(summary.minimum_augmented_mse_improvement().is_some());
        assert!(summary.origins_beating_non_relational() <= 4);
        assert!(summary.origins_beating_persistence() <= 4);
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
                forecast_horizon: 0.5,
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
            forecast_horizon: 0.5,
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
    fn make_surrogate_rejects_zero_shift() {
        let samples = build_samples(0.5);
        let config = HeldOutRelationalPredictionConfig {
            train_samples: 8,
            test_samples: 4,
            gap_samples: 2,
            ridge_lambda: 1e-8,
        };

        assert_eq!(
            make_surrogate(
                &samples,
                &config,
                PredictionNullFamily::CircularShift,
                0,
            ),
            Err(RelationalPredictionError::InvalidSurrogateCount)
        );
    }

    #[test]
    fn feature_decoupling_never_leaves_a_shifted_channel_identity_aligned() {
        let samples = build_samples(0.5);
        let config = HeldOutRelationalPredictionConfig {
            train_samples: 8,
            test_samples: 5,
            gap_samples: 2,
            ridge_lambda: 1e-8,
        };

        let surrogate = make_surrogate(
            &samples,
            &config,
            PredictionNullFamily::FeatureDecoupling,
            2,
        )
        .unwrap();

        assert_ne!(surrogate[0].a_to_b, samples[0].a_to_b);
        assert_ne!(surrogate[0].b_to_a, samples[0].b_to_a);
        assert_ne!(surrogate[0].turn_taking, samples[0].turn_taking);

        let short_config = HeldOutRelationalPredictionConfig {
            train_samples: 8,
            test_samples: 4,
            gap_samples: 2,
            ridge_lambda: 1e-8,
        };
        assert_eq!(
            make_surrogate(
                &samples,
                &short_config,
                PredictionNullFamily::FeatureDecoupling,
                2,
            ),
            Err(RelationalPredictionError::InsufficientSamples(4))
        );
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
    fn rolling_null_qualification_rejects_arithmetic_overflow() {
        let samples = build_samples(0.5);

        let result = RollingOriginRelationalPredictionQualification::compute(
            &samples,
            RollingOriginRelationalPredictionConfig {
                first_origin: usize::MAX,
                forecast_horizon: 0.5,
                ..Default::default()
            },
            4,
        );

        assert_eq!(result, Err(RelationalPredictionError::InvalidSplit));
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
