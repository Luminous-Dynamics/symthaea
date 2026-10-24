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
            NoiseModel::GaussianHomoscedastic { sigma } => self.rng.standard_normal() * sigma,
            NoiseModel::GaussianHeteroscedastic {
                sigma_floor,
                sigma_per_abs_covariate,
            } => {
                let scale = sigma_floor + sigma_per_abs_covariate * covariate.abs();
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
