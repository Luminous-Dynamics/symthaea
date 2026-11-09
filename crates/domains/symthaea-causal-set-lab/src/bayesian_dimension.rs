// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Empirically calibrated finite-size Bayesian dimension inference.
//!
//! The likelihood is calibrated from whole fixed-cardinality Poisson
//! sprinklings into flat Minkowski Alexandrov intervals. This preserves the
//! dependence among pairwise causal-relation indicators induced by shared
//! events and transitivity. It deliberately does not use an iid-binomial
//! likelihood for the related-pair count.
//!
//! The first version uses the Myrheim-Meyer ordering fraction as a single
//! summary statistic and a Gaussian kernel density estimate for
//! p(ordering_fraction | spacetime_dimension, event_count). Calibration and
//! qualification seeds must be disjoint. Curved, non-Poisson and non-manifold
//! causal sets are outside this likelihood family and should produce an
//! explicit out-of-support result, not a forced dimension assignment.

use crate::causal_set_nd::{CausalSetNd, CausalSetNdError};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const MIN_BAYESIAN_DIMENSION: usize = 2;
pub const MAX_BAYESIAN_DIMENSION: usize = 4;
pub const MAX_DIMENSION_CALIBRATION_REPLICATES: usize = 4096;
pub const MAX_DIMENSION_CALIBRATION_EVENTS: usize = 2048;
pub const MAX_DIMENSION_CALIBRATION_PAIR_EVALUATIONS: u128 = 200_000_000;
pub const MIN_DIMENSION_CALIBRATION_REPLICATES: usize = 8;
/// Observations farther than this many KDE bandwidths from every calibration
/// sample in every candidate dimension are outside the calibrated support.
pub const MAX_CALIBRATION_SUPPORT_DISTANCE: f64 = 6.0;
const GAUSSIAN_LOG_NORMALIZER: f64 = 0.918_938_533_204_672_7;
const IQR_TO_SIGMA: f64 = 1.348_979_500_392_163_4;
const SILVERMAN_FACTOR: f64 = 1.06;
const MIN_BANDWIDTH_FLOOR: f64 = 1.0e-8;

/// Empirical ordering-fraction samples for one spacetime dimension at one N.
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionCalibrationSamples {
    pub spacetime_dimension: usize,
    pub ordering_fractions: Vec<f64>,
}

/// Frozen Monte-Carlo calibration for fixed-cardinality Minkowski sprinklings.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteSizeDimensionCalibration {
    pub event_count: usize,
    pub proper_time: f64,
    pub base_seed: u64,
    /// Seed derivation version; changing the formula requires a new version.
    pub seed_scheme: String,
    pub samples_per_dimension: usize,
    pub dimensions: Vec<DimensionCalibrationSamples>,
}

/// Prior probability for one candidate integer spacetime dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionPrior {
    pub spacetime_dimension: usize,
    pub probability: f64,
}

/// Per-dimension likelihood and posterior contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionPosteriorEntry {
    pub spacetime_dimension: usize,
    pub prior_probability: f64,
    pub log_likelihood: f64,
    pub posterior_probability: Option<f64>,
    pub calibration_mean: f64,
    pub calibration_standard_deviation: f64,
    pub kde_bandwidth: f64,
    /// Nearest calibration sample distance divided by the KDE bandwidth.
    pub nearest_sample_distance_bandwidths: f64,
    pub within_calibration_support: bool,
}

/// Why a Bayesian finite-size dimension request cannot return a posterior.
#[derive(Clone, Debug, PartialEq)]
pub enum DimensionInferenceStatus {
    CalibratedPosterior,
    OutsideCalibrationSupport,
}

/// Posterior over integer dimensions for one observed finite causal set.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteSizeDimensionPosterior {
    pub status: DimensionInferenceStatus,
    pub event_count: usize,
    pub observed_ordering_fraction: f64,
    pub calibration_base_seed: u64,
    pub calibration_replicates_per_dimension: usize,
    pub posterior_mean_dimension: Option<f64>,
    /// Posterior mass at spacetime dimensions <= 3. This is a model-conditional
    /// statistic, not evidence of dimensional reduction outside this family.
    pub posterior_probability_dimension_at_most_three: Option<f64>,
    /// Highest-posterior-mass discrete set reaching at least 95% cumulative
    /// posterior probability. Entries are returned in ascending dimension.
    pub credible_set_95: Vec<usize>,
    pub dimensions: Vec<DimensionPosteriorEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BayesianDimensionError {
    UnsupportedDimension { dimension: usize },
    InvalidEventCount { event_count: usize },
    TooManyCalibrationEvents { event_count: usize, maximum: usize },
    InvalidProperTime,
    InvalidReplicateCount { count: usize },
    TooManyPairEvaluations { requested: u128, maximum: u128 },
    EmptyDimensionSet,
    TooManyDimensions { count: usize },
    DuplicateDimension { dimension: usize },
    InvalidCalibrationSample { dimension: usize, index: usize },
    CalibrationDimensionMismatch { expected: usize, actual: usize },
    CalibrationSampleCount { expected: usize, actual: usize },
    CalibrationEventCountMismatch { expected: usize, actual: usize },
    InvalidObservedOrderingFraction,
    EmptyPrior,
    DuplicatePriorDimension { dimension: usize },
    PriorDimensionMissingCalibration { dimension: usize },
    InvalidPriorProbability { dimension: usize },
    ZeroPriorMass,
    InvalidCalibrationMetadata,
    NumericalOverflow,
    Sprinkling(CausalSetNdError),
}

impl fmt::Display for BayesianDimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDimension { dimension } => write!(
                f,
                "spacetime dimension {dimension} is outside {MIN_BAYESIAN_DIMENSION}..={MAX_BAYESIAN_DIMENSION}"
            ),
            Self::InvalidEventCount { event_count } => {
                write!(f, "event count {event_count} must be at least 8")
            }
            Self::TooManyCalibrationEvents { event_count, maximum } => {
                write!(f, "event count {event_count} exceeds calibration cap {maximum}")
            }
            Self::InvalidProperTime => write!(f, "proper time must be finite and positive"),
            Self::InvalidReplicateCount { count } => write!(
                f,
                "calibration replicate count {count} must be in {MIN_DIMENSION_CALIBRATION_REPLICATES}..={MAX_DIMENSION_CALIBRATION_REPLICATES}"
            ),
            Self::TooManyPairEvaluations { requested, maximum } => write!(
                f,
                "calibration requests {requested} pair evaluations; cap is {maximum}"
            ),
            Self::EmptyDimensionSet => write!(f, "at least one candidate dimension is required"),
            Self::TooManyDimensions { count } => {
                write!(f, "{count} candidate dimensions exceed the supported set")
            }
            Self::DuplicateDimension { dimension } => {
                write!(f, "dimension {dimension} appears more than once")
            }
            Self::InvalidCalibrationSample { dimension, index } => write!(
                f,
                "calibration sample {index} for dimension {dimension} is not finite in [0,1]"
            ),
            Self::CalibrationDimensionMismatch { expected, actual } => write!(
                f,
                "calibration dimension {actual} does not match prior dimension {expected}"
            ),
            Self::CalibrationSampleCount { expected, actual } => write!(
                f,
                "calibration has {actual} samples per dimension; expected {expected}"
            ),
            Self::CalibrationEventCountMismatch { expected, actual } => write!(
                f,
                "calibration event count {actual} does not match observation count {expected}"
            ),
            Self::InvalidObservedOrderingFraction => {
                write!(f, "observed ordering fraction must be finite and in [0,1]")
            }
            Self::EmptyPrior => write!(f, "at least one dimension prior is required"),
            Self::DuplicatePriorDimension { dimension } => {
                write!(f, "prior dimension {dimension} appears more than once")
            }
            Self::PriorDimensionMissingCalibration { dimension } => write!(
                f,
                "prior dimension {dimension} has no calibration samples"
            ),
            Self::InvalidPriorProbability { dimension } => {
                write!(f, "prior probability for dimension {dimension} is invalid")
            }
            Self::ZeroPriorMass => write!(f, "dimension prior has zero total mass"),
            Self::InvalidCalibrationMetadata => {
                write!(f, "calibration metadata is inconsistent or invalid")
            }
            Self::NumericalOverflow => write!(f, "finite-size dimension inference overflowed"),
            Self::Sprinkling(error) => write!(f, "causal-set calibration sprinkling failed: {error}"),
        }
    }
}

impl Error for BayesianDimensionError {}

impl From<CausalSetNdError> for BayesianDimensionError {
    fn from(value: CausalSetNdError) -> Self {
        Self::Sprinkling(value)
    }
}

/// Generate an empirical finite-N likelihood calibration.
///
/// The sampler is fixed-cardinality (a Poisson sprinkling conditioned on N)
/// into a flat Minkowski Alexandrov interval. Its deterministic seeds are:
///
/// `base_seed + dimension * 0xD6E8_FEB8_6659_FD93 + replicate * 0x9E37_79B9_7F4A_7C15`
///
/// This seed scheme and calibration corpus must be frozen before qualification
/// seeds are opened. The pair-evaluation budget is checked before work begins.
pub fn calibrate_minkowski_ordering_fraction(
    event_count: usize,
    proper_time: f64,
    dimensions: &[usize],
    replicates_per_dimension: usize,
    base_seed: u64,
) -> Result<FiniteSizeDimensionCalibration, BayesianDimensionError> {
    validate_event_count(event_count)?;
    if !proper_time.is_finite() || proper_time <= 0.0 {
        return Err(BayesianDimensionError::InvalidProperTime);
    }
    if !(MIN_DIMENSION_CALIBRATION_REPLICATES..=MAX_DIMENSION_CALIBRATION_REPLICATES)
        .contains(&replicates_per_dimension)
    {
        return Err(BayesianDimensionError::InvalidReplicateCount {
            count: replicates_per_dimension,
        });
    }
    validate_dimensions(dimensions)?;
    let pairs = (event_count as u128) * (event_count.saturating_sub(1) as u128) / 2;
    let requested = pairs
        .checked_mul(replicates_per_dimension as u128)
        .and_then(|value| value.checked_mul(dimensions.len() as u128))
        .ok_or(BayesianDimensionError::NumericalOverflow)?;
    if requested > MAX_DIMENSION_CALIBRATION_PAIR_EVALUATIONS {
        return Err(BayesianDimensionError::TooManyPairEvaluations {
            requested,
            maximum: MAX_DIMENSION_CALIBRATION_PAIR_EVALUATIONS,
        });
    }

    let mut samples = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        let mut ordering_fractions = Vec::with_capacity(replicates_per_dimension);
        for replicate in 0..replicates_per_dimension {
            let seed = calibration_seed(base_seed, *dimension, replicate);
            let causet = CausalSetNd::sprinkle_fixed_cardinality(
                seed,
                event_count,
                *dimension,
                proper_time,
            )?;
            let fraction = causet
                .ordering_fraction()
                .ok_or(BayesianDimensionError::NumericalOverflow)?;
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err(BayesianDimensionError::NumericalOverflow);
            }
            ordering_fractions.push(fraction);
        }
        samples.push(DimensionCalibrationSamples {
            spacetime_dimension: *dimension,
            ordering_fractions,
        });
    }

    Ok(FiniteSizeDimensionCalibration {
        event_count,
        proper_time,
        base_seed,
        seed_scheme: "splitmix64-v1-dimensional-stride".to_string(),
        samples_per_dimension: replicates_per_dimension,
        dimensions: samples,
    })
}

/// Estimate a finite-size dimension posterior from an observed ordering
/// fraction and an independent empirical calibration corpus.
///
/// The likelihood is a one-dimensional Gaussian KDE over whole-set
/// Monte-Carlo samples, not a binomial likelihood over supposedly independent
/// pair indicators. The robust Silverman bandwidth uses the smaller of sample
/// standard deviation and IQR/1.349, with a floor set by the finite ordering
/// fraction resolution 1 / choose(N,2).
///
/// If the observation is farther than
/// `MAX_CALIBRATION_SUPPORT_DISTANCE * bandwidth` from every calibration
/// sample for every candidate dimension, no normalized posterior is returned.
pub fn estimate_finite_size_dimension_posterior(
    event_count: usize,
    observed_ordering_fraction: f64,
    calibration: &FiniteSizeDimensionCalibration,
    prior: &[DimensionPrior],
) -> Result<FiniteSizeDimensionPosterior, BayesianDimensionError> {
    validate_event_count(event_count)?;
    if !observed_ordering_fraction.is_finite()
        || !(0.0..=1.0).contains(&observed_ordering_fraction)
    {
        return Err(BayesianDimensionError::InvalidObservedOrderingFraction);
    }
    validate_calibration(calibration)?;
    if calibration.event_count != event_count {
        return Err(BayesianDimensionError::CalibrationEventCountMismatch {
            expected: event_count,
            actual: calibration.event_count,
        });
    }
    if prior.is_empty() {
        return Err(BayesianDimensionError::EmptyPrior);
    }

    let mut prior_dimensions = BTreeSet::new();
    let mut prior_mass = 0.0;
    for item in prior {
        if !(MIN_BAYESIAN_DIMENSION..=MAX_BAYESIAN_DIMENSION)
            .contains(&item.spacetime_dimension)
        {
            return Err(BayesianDimensionError::UnsupportedDimension {
                dimension: item.spacetime_dimension,
            });
        }
        if !prior_dimensions.insert(item.spacetime_dimension) {
            return Err(BayesianDimensionError::DuplicatePriorDimension {
                dimension: item.spacetime_dimension,
            });
        }
        if !item.probability.is_finite()
            || item.probability < 0.0
            || item.probability > 1.0
        {
            return Err(BayesianDimensionError::InvalidPriorProbability {
                dimension: item.spacetime_dimension,
            });
        }
        prior_mass += item.probability;
    }
    if !prior_mass.is_finite() || prior_mass <= 0.0 {
        return Err(BayesianDimensionError::ZeroPriorMass);
    }

    let mut intermediate = Vec::with_capacity(prior.len());
    let mut all_outside_support = true;
    let mut maximum_log_weight = f64::NEG_INFINITY;

    for item in prior {
        let samples = calibration
            .dimensions
            .iter()
            .find(|entry| entry.spacetime_dimension == item.spacetime_dimension)
            .ok_or(BayesianDimensionError::PriorDimensionMissingCalibration {
                dimension: item.spacetime_dimension,
            })?;
        let (mean, standard_deviation, bandwidth) = kde_bandwidth(
            &samples.ordering_fractions,
            event_count,
        )?;
        let nearest = samples
            .ordering_fractions
            .iter()
            .map(|sample| (observed_ordering_fraction - sample).abs() / bandwidth)
            .fold(f64::INFINITY, f64::min);
        let within_support = nearest <= MAX_CALIBRATION_SUPPORT_DISTANCE;
        if normalized_prior > 0.0 {
            all_outside_support &= !within_support;
        }
        let log_likelihood = gaussian_kde_log_density(
            observed_ordering_fraction,
            &samples.ordering_fractions,
            bandwidth,
        )?;
        let normalized_prior = item.probability / prior_mass;
        let log_weight = if normalized_prior > 0.0 {
            normalized_prior.ln() + log_likelihood
        } else {
            f64::NEG_INFINITY
        };
        maximum_log_weight = maximum_log_weight.max(log_weight);
        intermediate.push((
            item.spacetime_dimension,
            normalized_prior,
            log_likelihood,
            mean,
            standard_deviation,
            bandwidth,
            nearest,
            within_support,
            log_weight,
        ));
    }

    if all_outside_support {
        return Ok(FiniteSizeDimensionPosterior {
            status: DimensionInferenceStatus::OutsideCalibrationSupport,
            event_count,
            observed_ordering_fraction,
            calibration_base_seed: calibration.base_seed,
            calibration_replicates_per_dimension: calibration.samples_per_dimension,
            posterior_mean_dimension: None,
            posterior_probability_dimension_at_most_three: None,
            credible_set_95: Vec::new(),
            dimensions: intermediate
                .into_iter()
                .map(|(dimension, prior_probability, log_likelihood, mean, sd, bandwidth, nearest, within, _)| {
                    DimensionPosteriorEntry {
                        spacetime_dimension: dimension,
                        prior_probability,
                        log_likelihood,
                        posterior_probability: None,
                        calibration_mean: mean,
                        calibration_standard_deviation: sd,
                        kde_bandwidth: bandwidth,
                        nearest_sample_distance_bandwidths: nearest,
                        within_calibration_support: within,
                    }
                })
                .collect(),
        });
    }

    if !maximum_log_weight.is_finite() {
        return Err(BayesianDimensionError::NumericalOverflow);
    }
    let scaled_total: f64 = intermediate
        .iter()
        .map(|entry| (entry.8 - maximum_log_weight).exp())
        .sum();
    if !scaled_total.is_finite() || scaled_total <= 0.0 {
        return Err(BayesianDimensionError::NumericalOverflow);
    }

    let mut posterior_entries = Vec::with_capacity(intermediate.len());
    let mut posterior_mean = 0.0;
    let mut probability_at_most_three = 0.0;
    for entry in intermediate {
        let posterior = (entry.8 - maximum_log_weight).exp() / scaled_total;
        posterior_mean += entry.0 as f64 * posterior;
        if entry.0 <= 3 {
            probability_at_most_three += posterior;
        }
        posterior_entries.push(DimensionPosteriorEntry {
            spacetime_dimension: entry.0,
            prior_probability: entry.1,
            log_likelihood: entry.2,
            posterior_probability: Some(posterior),
            calibration_mean: entry.3,
            calibration_standard_deviation: entry.4,
            kde_bandwidth: entry.5,
            nearest_sample_distance_bandwidths: entry.6,
            within_calibration_support: entry.7,
        });
    }

    let mut ranking: Vec<(usize, f64)> = posterior_entries
        .iter()
        .map(|entry| {
            (
                entry.spacetime_dimension,
                entry.posterior_probability.unwrap_or(0.0),
            )
        })
        .collect();
    ranking.sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let mut cumulative = 0.0;
    let mut credible_set_95 = Vec::new();
    for (dimension, probability) in ranking {
        credible_set_95.push(dimension);
        cumulative += probability;
        if cumulative >= 0.95 {
            break;
        }
    }
    credible_set_95.sort_unstable();

    posterior_entries.sort_by_key(|entry| entry.spacetime_dimension);
    Ok(FiniteSizeDimensionPosterior {
        status: DimensionInferenceStatus::CalibratedPosterior,
        event_count,
        observed_ordering_fraction,
        calibration_base_seed: calibration.base_seed,
        calibration_replicates_per_dimension: calibration.samples_per_dimension,
        posterior_mean_dimension: Some(posterior_mean),
        posterior_probability_dimension_at_most_three: Some(probability_at_most_three),
        credible_set_95,
        dimensions: posterior_entries,
    })
}

fn validate_event_count(event_count: usize) -> Result<(), BayesianDimensionError> {
    if event_count < 8 {
        return Err(BayesianDimensionError::InvalidEventCount { event_count });
    }
    if event_count > MAX_DIMENSION_CALIBRATION_EVENTS {
        return Err(BayesianDimensionError::TooManyCalibrationEvents {
            event_count,
            maximum: MAX_DIMENSION_CALIBRATION_EVENTS,
        });
    }
    Ok(())
}

fn validate_dimensions(dimensions: &[usize]) -> Result<(), BayesianDimensionError> {
    if dimensions.is_empty() {
        return Err(BayesianDimensionError::EmptyDimensionSet);
    }
    if dimensions.len() > MAX_BAYESIAN_DIMENSION - MIN_BAYESIAN_DIMENSION + 1 {
        return Err(BayesianDimensionError::TooManyDimensions {
            count: dimensions.len(),
        });
    }
    let mut seen = BTreeSet::new();
    for dimension in dimensions {
        if !(MIN_BAYESIAN_DIMENSION..=MAX_BAYESIAN_DIMENSION).contains(dimension) {
            return Err(BayesianDimensionError::UnsupportedDimension {
                dimension: *dimension,
            });
        }
        if !seen.insert(*dimension) {
            return Err(BayesianDimensionError::DuplicateDimension {
                dimension: *dimension,
            });
        }
    }
    Ok(())
}

fn validate_calibration(
    calibration: &FiniteSizeDimensionCalibration,
) -> Result<(), BayesianDimensionError> {
    validate_event_count(calibration.event_count)?;
    if !calibration.proper_time.is_finite()
        || calibration.proper_time <= 0.0
        || calibration.seed_scheme != "splitmix64-v1-dimensional-stride"
        || calibration.samples_per_dimension < MIN_DIMENSION_CALIBRATION_REPLICATES
        || calibration.samples_per_dimension > MAX_DIMENSION_CALIBRATION_REPLICATES
    {
        return Err(BayesianDimensionError::InvalidCalibrationMetadata);
    }
    let expected_sample_count = calibration.samples_per_dimension;
    let dimensions: Vec<usize> = calibration
        .dimensions
        .iter()
        .map(|entry| entry.spacetime_dimension)
        .collect();
    validate_dimensions(&dimensions)?;
    for entry in &calibration.dimensions {
        if entry.ordering_fractions.len() != expected_sample_count {
            return Err(BayesianDimensionError::CalibrationSampleCount {
                expected: expected_sample_count,
                actual: entry.ordering_fractions.len(),
            });
        }
        for (index, sample) in entry.ordering_fractions.iter().copied().enumerate() {
            if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
                return Err(BayesianDimensionError::InvalidCalibrationSample {
                    dimension: entry.spacetime_dimension,
                    index,
                });
            }
        }
    }
    Ok(())
}

fn calibration_seed(base_seed: u64, dimension: usize, replicate: usize) -> u64 {
    base_seed
        .wrapping_add((dimension as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
        .wrapping_add((replicate as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
}

fn kde_bandwidth(
    samples: &[f64],
    event_count: usize,
) -> Result<(f64, f64, f64), BayesianDimensionError> {
    if samples.len() < MIN_DIMENSION_CALIBRATION_REPLICATES {
        return Err(BayesianDimensionError::InvalidCalibrationMetadata);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    let standard_deviation = variance.sqrt();
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let iqr = percentile(&sorted, 0.75).ok_or(BayesianDimensionError::NumericalOverflow)?
        - percentile(&sorted, 0.25).ok_or(BayesianDimensionError::NumericalOverflow)?;
    let robust_scale = if iqr > 0.0 {
        standard_deviation.min(iqr / IQR_TO_SIGMA)
    } else {
        standard_deviation
    };
    let silverman = SILVERMAN_FACTOR
        * robust_scale
        * (samples.len() as f64).powf(-0.2);
    let pair_count = (event_count as f64) * (event_count.saturating_sub(1) as f64) / 2.0;
    let discrete_resolution = 1.0 / pair_count.max(1.0);
    let bandwidth = silverman.max(discrete_resolution).max(MIN_BANDWIDTH_FLOOR);
    if !mean.is_finite()
        || !standard_deviation.is_finite()
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
    {
        return Err(BayesianDimensionError::NumericalOverflow);
    }
    Ok((mean, standard_deviation, bandwidth))
}

fn gaussian_kde_log_density(
    observed: f64,
    samples: &[f64],
    bandwidth: f64,
) -> Result<f64, BayesianDimensionError> {
    let log_terms: Vec<f64> = samples
        .iter()
        .map(|sample| {
            let standardized = (observed - sample) / bandwidth;
            -0.5 * standardized * standardized - GAUSSIAN_LOG_NORMALIZER - bandwidth.ln()
        })
        .collect();
    let maximum = log_terms.iter().copied().max_by(f64::total_cmp)
        .ok_or(BayesianDimensionError::NumericalOverflow)?;
    if !maximum.is_finite() {
        return Err(BayesianDimensionError::NumericalOverflow);
    }
    let scaled_sum = log_terms
        .iter()
        .map(|term| (*term - maximum).exp())
        .sum::<f64>();
    let log_density = maximum + scaled_sum.ln() - (samples.len() as f64).ln();
    if log_density.is_finite() {
        Ok(log_density)
    } else {
        Err(BayesianDimensionError::NumericalOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prior() -> Vec<DimensionPrior> {
        vec![
            DimensionPrior {
                spacetime_dimension: 2,
                probability: 1.0 / 3.0,
            },
            DimensionPrior {
                spacetime_dimension: 3,
                probability: 1.0 / 3.0,
            },
            DimensionPrior {
                spacetime_dimension: 4,
                probability: 1.0 / 3.0,
            },
        ]
    }

    fn calibration(events: usize, replicates: usize) -> FiniteSizeDimensionCalibration {
        calibrate_minkowski_ordering_fraction(
            events,
            1.0,
            &[2, 3, 4],
            replicates,
            0xB4A9_2026,
        )
        .unwrap()
    }

    #[test]
    fn calibration_is_deterministic_and_keeps_full_set_dependence() {
        let left = calibration(48, 12);
        let right = calibration(48, 12);
        assert_eq!(left, right);
        assert_eq!(left.dimensions.len(), 3);
        assert!(left.dimensions.iter().all(|entry| entry.ordering_fractions.len() == 12));
    }

    #[test]
    fn held_out_minkowski_dimension_four_prefers_four_dimensional_calibration() {
        let corpus = calibration(96, 64);
        let held_out = CausalSetNd::sprinkle_fixed_cardinality(0x51A7, 96, 4, 1.0).unwrap();
        let observed = held_out.ordering_fraction().unwrap();
        let posterior =
            estimate_finite_size_dimension_posterior(96, observed, &corpus, &prior()).unwrap();

        assert_eq!(posterior.status, DimensionInferenceStatus::CalibratedPosterior);
        let p4 = posterior
            .dimensions
            .iter()
            .find(|entry| entry.spacetime_dimension == 4)
            .unwrap()
            .posterior_probability
            .unwrap();
        assert!(p4 > 0.5, "D=4 posterior={p4}, r={observed}");
    }

    #[test]
    fn lorentz_boost_and_event_id_permutation_do_not_change_ordering_posterior() {
        let corpus = calibration(64, 32);
        let causet = CausalSetNd::sprinkle_fixed_cardinality(0xAA55, 64, 3, 1.0).unwrap();
        let base = causet.ordering_fraction().unwrap();
        let boosted = causet.lorentz_boosted(0.7, 0).unwrap();
        let boosted_fraction = boosted.ordering_fraction().unwrap();
        assert!((base - boosted_fraction).abs() < 1.0e-12);

        let original = estimate_finite_size_dimension_posterior(64, base, &corpus, &prior()).unwrap();
        let transformed =
            estimate_finite_size_dimension_posterior(64, boosted_fraction, &corpus, &prior()).unwrap();
        for (left, right) in original.dimensions.iter().zip(transformed.dimensions.iter()) {
            assert_eq!(left.spacetime_dimension, right.spacetime_dimension);
            assert!(
                (left.posterior_probability.unwrap() - right.posterior_probability.unwrap()).abs()
                    < 1.0e-12
            );
        }
    }

    #[test]
    fn posterior_is_prior_sensitive_and_normalized() {
        let corpus = calibration(64, 48);
        let observed = CausalSetNd::sprinkle_fixed_cardinality(0x1919, 64, 3, 1.0)
            .unwrap()
            .ordering_fraction()
            .unwrap();
        let uniform = estimate_finite_size_dimension_posterior(64, observed, &corpus, &prior()).unwrap();
        let d2_prior = vec![
            DimensionPrior { spacetime_dimension: 2, probability: 0.8 },
            DimensionPrior { spacetime_dimension: 3, probability: 0.1 },
            DimensionPrior { spacetime_dimension: 4, probability: 0.1 },
        ];
        let biased = estimate_finite_size_dimension_posterior(64, observed, &corpus, &d2_prior).unwrap();

        let total: f64 = uniform.dimensions.iter().map(|entry| entry.posterior_probability.unwrap()).sum();
        assert!((total - 1.0).abs() < 1.0e-12);
        assert!(
            biased.dimensions[0].posterior_probability.unwrap()
                > uniform.dimensions[0].posterior_probability.unwrap()
        );
    }

    #[test]
    fn outside_all_calibration_supports_returns_unresolved_without_posterior() {
        let corpus = calibration(64, 32);
        let posterior =
            estimate_finite_size_dimension_posterior(64, 1.0, &corpus, &prior()).unwrap();
        assert_eq!(posterior.status, DimensionInferenceStatus::OutsideCalibrationSupport);
        assert_eq!(posterior.posterior_mean_dimension, None);
        assert_eq!(posterior.credible_set_95, Vec::<usize>::new());
        assert!(posterior.dimensions.iter().all(|entry| entry.posterior_probability.is_none()));
    }

    #[test]
    fn prior_probabilities_above_one_are_rejected() {
        let corpus = calibration(32, 8);
        assert!(matches!(
            estimate_finite_size_dimension_posterior(
                32,
                0.5,
                &corpus,
                &[DimensionPrior {
                    spacetime_dimension: 2,
                    probability: 1.5,
                }]
            ),
            Err(BayesianDimensionError::InvalidPriorProbability { dimension: 2 })
        ));
    }

    #[test]
    fn calibration_and_prior_contracts_fail_closed() {
        let corpus = calibration(32, 8);
        assert!(matches!(
            estimate_finite_size_dimension_posterior(64, 0.5, &corpus, &prior()),
            Err(BayesianDimensionError::CalibrationEventCountMismatch { .. })
        ));
        assert!(matches!(
            estimate_finite_size_dimension_posterior(
                32,
                0.5,
                &corpus,
                &[
                    DimensionPrior { spacetime_dimension: 2, probability: 0.5 },
                    DimensionPrior { spacetime_dimension: 2, probability: 0.5 }
                ]
            ),
            Err(BayesianDimensionError::DuplicatePriorDimension { .. })
        ));
    }
}
