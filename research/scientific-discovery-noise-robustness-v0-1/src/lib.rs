//! Deterministic observation-process fixtures for scientific law-discovery tests.
//!
//! This crate deliberately models the measurement process, not a scientific law,
//! candidate selector, or calibration guarantee. Evaluator-owned code chooses a
//! NoiseModel; the policy should only receive the resulting Observation and
//! pre-outcome covariates explicitly allowed by the benchmark protocol.

use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum NoiseModel {
    GaussianHomoscedastic {
        sigma: f64,
    },
    GaussianHeteroscedastic {
        sigma_floor: f64,
        sigma_per_abs_covariate: f64,
    },
    StudentTHeavyTail {
        scale: f64,
        degrees_of_freedom: u16,
    },
    ContaminationOutliers {
        sigma: f64,
        contamination_probability: f64,
        outlier_scale: f64,
    },
    Ar1Residuals {
        stationary_sigma: f64,
        phi: f64,
    },
    CensoredOrMissing {
        sigma: f64,
        missing_probability: f64,
        lower_bound: f64,
        upper_bound: f64,
    },
    ScaleDrift {
        sigma_before: f64,
        sigma_after: f64,
        change_at_step: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Observation {
    Observed(f64),
    Missing,
    /// The latent measurement is known only to be at or below this bound.
    LeftCensored { upper_bound: f64 },
    /// The latent measurement is known only to be at or above this bound.
    RightCensored { lower_bound: f64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoiseError {
    NonFiniteParameter,
    NonPositiveScale,
    NegativeHeteroscedasticSlope,
    InvalidProbability,
    InvalidDegreesOfFreedom,
    InvalidArCoefficient,
    InvalidCensorBounds,
    NonFiniteInput,
    ScaleOverflow,
    NonFiniteOutput,
}

impl fmt::Display for NoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid noise-process input: {self:?}")
    }
}

impl Error for NoiseError {}

impl NoiseModel {
    /// Validate a model before constructing a generator.
    pub fn validate(&self) -> Result<(), NoiseError> {
        let finite = |x: f64| x.is_finite();
        let positive = |x: f64| finite(x) && x > 0.0;
        let probability = |x: f64| finite(x) && (0.0..=1.0).contains(&x);

        match self {
            Self::GaussianHomoscedastic { sigma } => {
                if !finite(*sigma) {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*sigma) {
                    return Err(NoiseError::NonPositiveScale);
                }
            }
            Self::GaussianHeteroscedastic {
                sigma_floor,
                sigma_per_abs_covariate,
            } => {
                if !finite(*sigma_floor) || !finite(*sigma_per_abs_covariate) {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*sigma_floor) {
                    return Err(NoiseError::NonPositiveScale);
                }
                if *sigma_per_abs_covariate < 0.0 {
                    return Err(NoiseError::NegativeHeteroscedasticSlope);
                }
            }
            Self::StudentTHeavyTail {
                scale,
                degrees_of_freedom,
            } => {
                if !finite(*scale) {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*scale) {
                    return Err(NoiseError::NonPositiveScale);
                }
                // Bound work per observation and keep the variance finite.
                if !(3..=256).contains(degrees_of_freedom) {
                    return Err(NoiseError::InvalidDegreesOfFreedom);
                }
            }
            Self::ContaminationOutliers {
                sigma,
                contamination_probability,
                outlier_scale,
            } => {
                if !finite(*sigma)
                    || !finite(*contamination_probability)
                    || !finite(*outlier_scale)
                {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*sigma) || !positive(*outlier_scale) {
                    return Err(NoiseError::NonPositiveScale);
                }
                if !probability(*contamination_probability) {
                    return Err(NoiseError::InvalidProbability);
                }
            }
            Self::Ar1Residuals {
                stationary_sigma,
                phi,
            } => {
                if !finite(*stationary_sigma) || !finite(*phi) {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*stationary_sigma) {
                    return Err(NoiseError::NonPositiveScale);
                }
                if phi.abs() >= 1.0 {
                    return Err(NoiseError::InvalidArCoefficient);
                }
            }
            Self::CensoredOrMissing {
                sigma,
                missing_probability,
                lower_bound,
                upper_bound,
            } => {
                if !finite(*sigma)
                    || !finite(*missing_probability)
                    || !finite(*lower_bound)
                    || !finite(*upper_bound)
                {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*sigma) {
                    return Err(NoiseError::NonPositiveScale);
                }
                if !probability(*missing_probability) {
                    return Err(NoiseError::InvalidProbability);
                }
                if lower_bound >= upper_bound {
                    return Err(NoiseError::InvalidCensorBounds);
                }
            }
            Self::ScaleDrift {
                sigma_before,
                sigma_after,
                ..
            } => {
                if !finite(*sigma_before) || !finite(*sigma_after) {
                    return Err(NoiseError::NonFiniteParameter);
                }
                if !positive(*sigma_before) || !positive(*sigma_after) {
                    return Err(NoiseError::NonPositiveScale);
                }
            }
        }
        Ok(())
    }

    /// Return the known scale for designs where it is well-defined.
    ///
    /// This is an evaluator-side diagnostic, not information to expose to the policy
    /// when the benchmark declares the observation scale hidden.
    pub fn scale_at(&self, covariate: f64, step: usize) -> Option<f64> {
        let scale = match self {
            Self::GaussianHomoscedastic { sigma } => *sigma,
            Self::GaussianHeteroscedastic {
                sigma_floor,
                sigma_per_abs_covariate,
            } => *sigma_floor + *sigma_per_abs_covariate * covariate.abs(),
            Self::StudentTHeavyTail { scale, .. } => *scale,
            Self::ContaminationOutliers { sigma, .. } => *sigma,
            Self::Ar1Residuals {
                stationary_sigma, ..
            } => *stationary_sigma,
            Self::CensoredOrMissing { sigma, .. } => *sigma,
            Self::ScaleDrift {
                sigma_before,
                sigma_after,
                change_at_step,
            } => {
                if step < *change_at_step {
                    *sigma_before
                } else {
                    *sigma_after
                }
            }
        };
        scale.is_finite().then_some(scale)
    }
}

/// Dependency-free deterministic pseudorandom generator.
///
/// Suitable for reproducible synthetic tests only. It is not cryptographic and must
/// not be used for security tokens or secret benchmark labels.
#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        // Xorshift generators have an absorbing zero state.
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self {
            state,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A representable number strictly between zero and one.
    fn unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 12;
        (bits as f64 + 1.0) / (((1_u64 << 52) + 1) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let radius = (-2.0 * self.unit().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.unit();
        let (sin, cos) = angle.sin_cos();
        self.spare_normal = Some(radius * sin);
        radius * cos
    }

    /// Uniform integer in 0..upper_exclusive, using rejection sampling to avoid
    /// modulo bias in Fisher–Yates shuffling.
    fn below(&mut self, upper_exclusive: u64) -> u64 {
        debug_assert!(upper_exclusive > 0);
        let rejection_floor = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= rejection_floor {
                return value % upper_exclusive;
            }
        }
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below((i + 1) as u64) as usize;
            items.swap(i, j);
        }
    }
}


const MAX_DESIGN_POINTS: usize = 1_000_000;

/// Role of a scheduled covariate in a fixed, evaluator-authored diagnostic design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesignRole {
    Discovery,
    /// Repeated reference input used to probe time/order-dependent measurement drift.
    Anchor,
}

/// One pre-outcome measurement request in an interleaved diagnostic schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignPoint {
    pub block: usize,
    pub covariate: f64,
    pub role: DesignRole,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DesignScheduleError {
    EmptyCandidateDesign,
    ZeroAnchorReplicates,
    ZeroBlocks,
    NonFiniteCovariate,
    DuplicateCandidateCovariate,
    TooManyDesignPoints,
}

/// Generate a reproducible, randomized fixed-design control with repeated anchor
/// measurements interleaved into every block.
///
/// This is an evaluation-design helper, not an adaptive experiment policy. Each block
/// contains every candidate covariate once and the anchor exactly anchor_replicates_per_block
/// times; slots are shuffled before their order can be observed. A repeated anchor helps
/// diagnose time/order drift that may otherwise be confounded with changes in covariates.
/// It does not by itself identify the scientific law or guarantee causal attribution.
pub fn interleaved_anchor_schedule(
    candidate_covariates: &[f64],
    anchor_covariate: f64,
    anchor_replicates_per_block: usize,
    blocks: usize,
    seed: u64,
) -> Result<Vec<DesignPoint>, DesignScheduleError> {
    if candidate_covariates.is_empty() {
        return Err(DesignScheduleError::EmptyCandidateDesign);
    }
    if anchor_replicates_per_block == 0 {
        return Err(DesignScheduleError::ZeroAnchorReplicates);
    }
    if blocks == 0 {
        return Err(DesignScheduleError::ZeroBlocks);
    }
    if !anchor_covariate.is_finite()
        || candidate_covariates.iter().any(|value| !value.is_finite())
    {
        return Err(DesignScheduleError::NonFiniteCovariate);
    }
    for index in 0..candidate_covariates.len() {
        if candidate_covariates[..index].contains(&candidate_covariates[index]) {
            return Err(DesignScheduleError::DuplicateCandidateCovariate);
        }
    }
    let per_block = candidate_covariates
        .len()
        .checked_add(anchor_replicates_per_block)
        .ok_or(DesignScheduleError::TooManyDesignPoints)?;
    let total = per_block
        .checked_mul(blocks)
        .ok_or(DesignScheduleError::TooManyDesignPoints)?;
    if total > MAX_DESIGN_POINTS {
        return Err(DesignScheduleError::TooManyDesignPoints);
    }

    let mut rng = DeterministicRng::new(seed);
    let mut schedule = Vec::with_capacity(total);
    for block in 0..blocks {
        let mut points = Vec::with_capacity(per_block);
        points.extend(candidate_covariates.iter().map(|covariate| DesignPoint {
            block,
            covariate: *covariate,
            role: DesignRole::Discovery,
        }));
        points.extend((0..anchor_replicates_per_block).map(|_| DesignPoint {
            block,
            covariate: anchor_covariate,
            role: DesignRole::Anchor,
        }));
        rng.shuffle(&mut points);
        schedule.extend(points);
    }
    Ok(schedule)
}

/// A stateful synthetic observation process.
///
/// The caller supplies the step so change points can be frozen in the benchmark
/// manifest. Only Observation should be handed to the discovery policy; the model
/// and generator remain evaluator-owned configuration.
#[derive(Clone, Debug)]
pub struct NoiseProcess {
    model: NoiseModel,
    rng: DeterministicRng,
    previous_ar_residual: Option<f64>,
}

impl NoiseProcess {
    pub fn new(model: NoiseModel, seed: u64) -> Result<Self, NoiseError> {
        model.validate()?;
        Ok(Self {
            model,
            rng: DeterministicRng::new(seed),
            previous_ar_residual: None,
        })
    }

    /// Sample one observation from the hidden observation process.
    ///
    /// latent_mean is the noiseless response supplied by the evaluator. covariate
    /// must be a public, pre-outcome design covariate if exposed to the policy.
    /// Neither value reveals the noise-family label.
    pub fn observe(
        &mut self,
        latent_mean: f64,
        covariate: f64,
        step: usize,
    ) -> Result<Observation, NoiseError> {
        if !latent_mean.is_finite() || !covariate.is_finite() {
            return Err(NoiseError::NonFiniteInput);
        }

        let residual = match &self.model {
            NoiseModel::GaussianHomoscedastic { sigma } => {
                self.rng.standard_normal() * *sigma
            }
            NoiseModel::GaussianHeteroscedastic {
                sigma_floor,
                sigma_per_abs_covariate,
            } => {
                let scale = *sigma_floor + *sigma_per_abs_covariate * covariate.abs();
                if !scale.is_finite() {
                    return Err(NoiseError::ScaleOverflow);
                }
                self.rng.standard_normal() * scale
            }
            NoiseModel::StudentTHeavyTail {
                scale,
                degrees_of_freedom,
            } => Self::sample_student_t(&mut self.rng, *degrees_of_freedom) * *scale,
            NoiseModel::ContaminationOutliers {
                sigma,
                contamination_probability,
                outlier_scale,
            } => {
                let multiplier = if self.rng.unit() < *contamination_probability {
                    *outlier_scale
                } else {
                    1.0
                };
                self.rng.standard_normal() * *sigma * multiplier
            }
            NoiseModel::Ar1Residuals {
                stationary_sigma,
                phi,
            } => {
                let previous = match self.previous_ar_residual {
                    Some(value) => value,
                    None => self.rng.standard_normal() * *stationary_sigma,
                };
                let innovation_scale = *stationary_sigma * (1.0 - *phi * *phi).sqrt();
                let current = *phi * previous + innovation_scale * self.rng.standard_normal();
                self.previous_ar_residual = Some(current);
                current
            }
            NoiseModel::CensoredOrMissing {
                sigma,
                missing_probability,
                lower_bound,
                upper_bound,
            } => {
                if self.rng.unit() < *missing_probability {
                    return Ok(Observation::Missing);
                }
                let raw = latent_mean + *sigma * self.rng.standard_normal();
                if !raw.is_finite() {
                    return Err(NoiseError::NonFiniteOutput);
                }
                if raw <= *lower_bound {
                    return Ok(Observation::LeftCensored {
                        upper_bound: *lower_bound,
                    });
                }
                if raw >= *upper_bound {
                    return Ok(Observation::RightCensored {
                        lower_bound: *upper_bound,
                    });
                }
                return Ok(Observation::Observed(raw));
            }
            NoiseModel::ScaleDrift {
                sigma_before,
                sigma_after,
                change_at_step,
            } => {
                let scale = if step < *change_at_step {
                    sigma_before
                } else {
                    sigma_after
                };
                self.rng.standard_normal() * *scale
            }
        };

        let observed = latent_mean + residual;
        if !observed.is_finite() {
            return Err(NoiseError::NonFiniteOutput);
        }
        Ok(Observation::Observed(observed))
    }

    fn sample_student_t(rng: &mut DeterministicRng, degrees_of_freedom: u16) -> f64 {
        let numerator = rng.standard_normal();
        let mut chi_square = 0.0;
        for _ in 0..degrees_of_freedom {
            let draw = rng.standard_normal();
            chi_square += draw * draw;
        }
        let denominator = (chi_square / f64::from(degrees_of_freedom))
            .max(f64::MIN_POSITIVE)
            .sqrt();
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<NoiseModel> {
        vec![
            NoiseModel::GaussianHomoscedastic { sigma: 0.2 },
            NoiseModel::GaussianHeteroscedastic {
                sigma_floor: 0.05,
                sigma_per_abs_covariate: 0.1,
            },
            NoiseModel::StudentTHeavyTail {
                scale: 0.2,
                degrees_of_freedom: 3,
            },
            NoiseModel::ContaminationOutliers {
                sigma: 0.1,
                contamination_probability: 0.15,
                outlier_scale: 8.0,
            },
            NoiseModel::Ar1Residuals {
                stationary_sigma: 0.2,
                phi: 0.8,
            },
            NoiseModel::CensoredOrMissing {
                sigma: 0.5,
                missing_probability: 0.2,
                lower_bound: -0.3,
                upper_bound: 0.3,
            },
            NoiseModel::ScaleDrift {
                sigma_before: 0.1,
                sigma_after: 0.5,
                change_at_step: 100,
            },
        ]
    }

    #[test]
    fn all_noise_processes_replay_exactly_from_the_same_seed() {
        for model in models() {
            let mut left = NoiseProcess::new(model.clone(), 0xC0FF_EE12).unwrap();
            let mut right = NoiseProcess::new(model, 0xC0FF_EE12).unwrap();
            for step in 0..512 {
                let covariate = (step as f64 / 37.0).sin();
                assert_eq!(
                    left.observe(1.25 + covariate, covariate, step),
                    right.observe(1.25 + covariate, covariate, step),
                    "replay diverged at step {step}"
                );
            }
        }
    }

    #[test]
    fn invalid_noise_configurations_fail_closed() {
        assert_eq!(
            NoiseProcess::new(NoiseModel::GaussianHomoscedastic { sigma: 0.0 }, 7)
                .unwrap_err(),
            NoiseError::NonPositiveScale
        );
        assert_eq!(
            NoiseProcess::new(
                NoiseModel::StudentTHeavyTail {
                    scale: 1.0,
                    degrees_of_freedom: 1,
                },
                7
            )
            .unwrap_err(),
            NoiseError::InvalidDegreesOfFreedom
        );
        assert_eq!(
            NoiseProcess::new(
                NoiseModel::Ar1Residuals {
                    stationary_sigma: 1.0,
                    phi: 1.0,
                },
                7
            )
            .unwrap_err(),
            NoiseError::InvalidArCoefficient
        );
        assert_eq!(
            NoiseProcess::new(
                NoiseModel::CensoredOrMissing {
                    sigma: 1.0,
                    missing_probability: 0.1,
                    lower_bound: 2.0,
                    upper_bound: 2.0,
                },
                7
            )
            .unwrap_err(),
            NoiseError::InvalidCensorBounds
        );
    }

    #[test]
    fn scale_functions_obey_the_declared_covariate_and_change_point() {
        let hetero = NoiseModel::GaussianHeteroscedastic {
            sigma_floor: 0.2,
            sigma_per_abs_covariate: 0.1,
        };
        assert_eq!(hetero.scale_at(-3.0, 0), Some(0.5));
        assert_eq!(hetero.scale_at(0.0, 0), Some(0.2));

        let drift = NoiseModel::ScaleDrift {
            sigma_before: 0.1,
            sigma_after: 0.7,
            change_at_step: 50,
        };
        assert_eq!(drift.scale_at(0.0, 49), Some(0.1));
        assert_eq!(drift.scale_at(0.0, 50), Some(0.7));
    }

    #[test]
    fn censored_and_missing_observations_are_not_encoded_as_zero() {
        let mut process = NoiseProcess::new(
            NoiseModel::CensoredOrMissing {
                sigma: 0.5,
                missing_probability: 0.2,
                lower_bound: -0.3,
                upper_bound: 0.3,
            },
            0x1234_5678,
        )
        .unwrap();

        let mut observed = 0;
        let mut missing = 0;
        let mut left_censored = 0;
        let mut right_censored = 0;
        for step in 0..2_000 {
            match process.observe(0.0, 0.0, step).unwrap() {
                Observation::Observed(value) => {
                    assert!(value > -0.3 && value < 0.3);
                    observed += 1;
                }
                Observation::Missing => missing += 1,
                Observation::LeftCensored { upper_bound } => {
                    assert_eq!(upper_bound, -0.3);
                    left_censored += 1;
                }
                Observation::RightCensored { lower_bound } => {
                    assert_eq!(lower_bound, 0.3);
                    right_censored += 1;
                }
            }
        }
        assert!(observed > 0);
        assert!(missing > 0);
        assert!(left_censored > 0);
        assert!(right_censored > 0);
    }

    #[test]
    fn autoregressive_residuals_show_positive_lag_one_dependence() {
        let mut process = NoiseProcess::new(
            NoiseModel::Ar1Residuals {
                stationary_sigma: 1.0,
                phi: 0.8,
            },
            0xA11C_E551,
        )
        .unwrap();

        let mut residuals = Vec::with_capacity(2_000);
        for step in 0..2_000 {
            let Observation::Observed(value) = process.observe(0.0, 0.0, step).unwrap() else {
                panic!("AR(1) fixture should always return an observed value");
            };
            residuals.push(value);
        }

        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let numerator = residuals
            .windows(2)
            .map(|pair| (pair[0] - mean) * (pair[1] - mean))
            .sum::<f64>();
        let denominator = residuals.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
        let lag_one_correlation = numerator / denominator;
        assert!(lag_one_correlation > 0.55, "rho1={lag_one_correlation}");
    }



    #[test]
    fn anchor_schedule_is_reproducible_and_interleaves_every_role_per_block() {
        let candidates = [0.1, 0.4, 0.8, 1.2, 2.0];
        let left = interleaved_anchor_schedule(&candidates, 0.25, 2, 7, 0xA11C_E55E).unwrap();
        let right = interleaved_anchor_schedule(&candidates, 0.25, 2, 7, 0xA11C_E55E).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.len(), 7 * (candidates.len() + 2));

        for block in 0..7 {
            let points: Vec<_> = left.iter().filter(|point| point.block == block).collect();
            assert_eq!(points.len(), candidates.len() + 2);
            assert_eq!(
                points.iter().filter(|point| point.role == DesignRole::Anchor).count(),
                2
            );
            for candidate in candidates {
                assert_eq!(
                    points
                        .iter()
                        .filter(|point| {
                            point.role == DesignRole::Discovery && point.covariate == candidate
                        })
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn anchor_schedule_changes_order_with_a_different_seed() {
        let candidates = [0.1, 0.4, 0.8, 1.2, 2.0];
        let left = interleaved_anchor_schedule(&candidates, 0.25, 2, 4, 101).unwrap();
        let right = interleaved_anchor_schedule(&candidates, 0.25, 2, 4, 202).unwrap();
        assert_ne!(left, right);
    }

    #[test]
    fn anchor_schedule_rejects_invalid_or_unbounded_designs() {
        assert_eq!(
            interleaved_anchor_schedule(&[], 0.0, 1, 1, 1).unwrap_err(),
            DesignScheduleError::EmptyCandidateDesign
        );
        assert_eq!(
            interleaved_anchor_schedule(&[1.0], 0.0, 0, 1, 1).unwrap_err(),
            DesignScheduleError::ZeroAnchorReplicates
        );
        assert_eq!(
            interleaved_anchor_schedule(&[1.0], 0.0, 1, 0, 1).unwrap_err(),
            DesignScheduleError::ZeroBlocks
        );
        assert_eq!(
            interleaved_anchor_schedule(&[f64::NAN], 0.0, 1, 1, 1).unwrap_err(),
            DesignScheduleError::NonFiniteCovariate
        );
        assert_eq!(
            interleaved_anchor_schedule(&[1.0, 1.0], 0.0, 1, 1, 1).unwrap_err(),
            DesignScheduleError::DuplicateCandidateCovariate
        );
        assert_eq!(
            interleaved_anchor_schedule(&[1.0], 0.0, MAX_DESIGN_POINTS, 2, 1).unwrap_err(),
            DesignScheduleError::TooManyDesignPoints
        );
    }

    fn sample_variance(values: &[f64]) -> f64 {
        assert!(values.len() >= 2);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64
    }

    #[test]
    fn heteroscedastic_fixture_changes_variance_with_the_covariate() {
        let model = NoiseModel::GaussianHeteroscedastic {
            sigma_floor: 0.05,
            sigma_per_abs_covariate: 0.1,
        };
        let mut low = NoiseProcess::new(model.clone(), 0xAABB_CCDD).unwrap();
        let mut high = NoiseProcess::new(model, 0xAABB_CCDD).unwrap();
        let low_values: Vec<f64> = (0..3_000)
            .map(|step| match low.observe(0.0, 0.0, step).unwrap() {
                Observation::Observed(value) => value,
                other => panic!("unexpected observation: {other:?}"),
            })
            .collect();
        let high_values: Vec<f64> = (0..3_000)
            .map(|step| match high.observe(0.0, 3.0, step).unwrap() {
                Observation::Observed(value) => value,
                other => panic!("unexpected observation: {other:?}"),
            })
            .collect();
        assert!(
            sample_variance(&high_values) > 10.0 * sample_variance(&low_values),
            "high variance should exceed low variance by a wide margin"
        );
    }

    #[test]
    fn student_t_fixture_has_more_extreme_residuals_than_a_gaussian_control() {
        let mut heavy =
            NoiseProcess::new(
                NoiseModel::StudentTHeavyTail {
                    scale: 1.0,
                    degrees_of_freedom: 3,
                },
                0xBEEF_1234,
            )
            .unwrap();
        let mut gaussian =
            NoiseProcess::new(NoiseModel::GaussianHomoscedastic { sigma: 1.0 }, 0xBEEF_1234)
                .unwrap();
        let mut heavy_extremes = 0;
        let mut gaussian_extremes = 0;
        for step in 0..3_000 {
            let heavy_value = match heavy.observe(0.0, 0.0, step).unwrap() {
                Observation::Observed(value) => value,
                other => panic!("unexpected observation: {other:?}"),
            };
            let gaussian_value = match gaussian.observe(0.0, 0.0, step).unwrap() {
                Observation::Observed(value) => value,
                other => panic!("unexpected observation: {other:?}"),
            };
            if heavy_value.abs() > 3.0 {
                heavy_extremes += 1;
            }
            if gaussian_value.abs() > 3.0 {
                gaussian_extremes += 1;
            }
        }
        assert!(heavy_extremes > 50, "heavy-tail extremes={heavy_extremes}");
        assert!(
            heavy_extremes > gaussian_extremes + 30,
            "heavy={heavy_extremes}, gaussian={gaussian_extremes}"
        );
    }

    #[test]
    fn contamination_fixture_produces_a_separate_extreme_residual_population() {
        let mut process = NoiseProcess::new(
            NoiseModel::ContaminationOutliers {
                sigma: 0.1,
                contamination_probability: 0.15,
                outlier_scale: 8.0,
            },
            0xC011_AB1E,
        )
        .unwrap();
        let extremes = (0..2_000)
            .filter(|step| match process.observe(0.0, 0.0, *step).unwrap() {
                Observation::Observed(value) => value.abs() > 0.5,
                other => panic!("unexpected observation: {other:?}"),
            })
            .count();
        assert!(extremes > 40, "contaminated extremes={extremes}");
    }

    #[test]
    fn scale_drift_fixture_changes_variance_at_the_predeclared_step() {
        let mut process = NoiseProcess::new(
            NoiseModel::ScaleDrift {
                sigma_before: 0.1,
                sigma_after: 0.5,
                change_at_step: 2_000,
            },
            0xD12F_0001,
        )
        .unwrap();
        let mut before = Vec::with_capacity(2_000);
        let mut after = Vec::with_capacity(2_000);
        for step in 0..4_000 {
            let Observation::Observed(value) = process.observe(0.0, 0.0, step).unwrap() else {
                panic!("scale-drift fixture should always return an observed value");
            };
            if step < 2_000 {
                before.push(value);
            } else {
                after.push(value);
            }
        }
        assert!(
            sample_variance(&after) > 10.0 * sample_variance(&before),
            "before variance={}, after variance={}",
            sample_variance(&before),
            sample_variance(&after)
        );
    }

    #[test]
    fn non_finite_inputs_and_outputs_are_rejected() {
        let mut process =
            NoiseProcess::new(NoiseModel::GaussianHomoscedastic { sigma: 0.1 }, 123).unwrap();
        assert_eq!(
            process.observe(f64::NAN, 0.0, 0).unwrap_err(),
            NoiseError::NonFiniteInput
        );
        let mut overflow =
            NoiseProcess::new(NoiseModel::GaussianHomoscedastic { sigma: f64::MAX }, 123).unwrap();
        assert_eq!(
            overflow.observe(f64::MAX, 0.0, 1).unwrap_err(),
            NoiseError::NonFiniteOutput
        );
    }
}
