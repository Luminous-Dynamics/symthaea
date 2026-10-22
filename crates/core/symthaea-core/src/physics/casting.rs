// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibrated first-order models for casting solidification.
//!
//! This module intentionally starts with a narrow, empirical model rather than
//! pretending to be a full mold-filling, feeding, microstructure, or thermal-stress
//! solver. It provides an auditable Chvorinov-style fit for a single calibrated
//! process population. The caller is responsible for keeping calibration observations
//! tied to a compatible alloy, mold/tooling family, thermal boundary condition, and
//! measurement protocol.
//!
//! The model is advisory engineering evidence only. It does not qualify a process,
//! infer casting integrity, select a production recipe, or authorize equipment motion.
//! Predictions outside the observed thermal-modulus range are rejected rather than
//! silently extrapolated.
//!
//! The relation fitted here is:
//!
//! solidification_time_s = mold_constant * thermal_modulus_m.powf(exponent)
//!
//! where thermal_modulus_m = casting_volume_m3 / effective_cooling_surface_area_m2.
//! Since thermal modulus is measured in metres, the mold constant has units
//! seconds / metres.powf(exponent).

use std::error::Error;
use std::fmt;

/// Minimum number of observations required for a log-space Chvorinov fit.
///
/// This is a numerical guard against fitting a two-point line, not a process
/// qualification or statistical sufficiency criterion.
pub const MIN_CALIBRATION_OBSERVATIONS: usize = 3;

/// One measured solidification observation used to calibrate the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChvorinovObservation {
    /// Casting volume divided by effective cooling surface area, in metres.
    pub thermal_modulus_m: f64,
    /// Independently measured solidification time, in seconds.
    pub solidification_time_s: f64,
}

impl ChvorinovObservation {
    /// Create an observation.
    ///
    /// Physical validation is performed by ChvorinovModel::calibrate, so malformed
    /// observations cannot enter a fitted model unnoticed.
    pub const fn new(thermal_modulus_m: f64, solidification_time_s: f64) -> Self {
        Self {
            thermal_modulus_m,
            solidification_time_s,
        }
    }
}

/// Errors returned by the first-order casting model.
#[derive(Debug, Clone, PartialEq)]
pub enum CastingModelError {
    /// Too few measurements were provided for a fit.
    NotEnoughObservations {
        /// Number of provided observations.
        actual: usize,
        /// Minimum number required for this numerical fit.
        minimum: usize,
    },
    /// At least one observation was non-finite or non-positive.
    InvalidObservation {
        /// Zero-based index of the invalid observation.
        index: usize,
    },
    /// The observations do not span a usable range of thermal moduli.
    DegenerateCalibrationDomain,
    /// The fit implied a zero or negative solidification exponent.
    NonPositiveExponent,
    /// The fitted parameters or diagnostics were not finite and positive where required.
    InvalidCalibrationParameters,
    /// A requested prediction lies outside the measured calibration domain.
    OutsideCalibrationDomain {
        /// Requested thermal modulus, in metres.
        requested_m: f64,
        /// Smallest thermal modulus used for calibration, in metres.
        minimum_m: f64,
        /// Largest thermal modulus used for calibration, in metres.
        maximum_m: f64,
    },
    /// Casting geometry or the resulting thermal modulus was invalid.
    InvalidGeometry,
    /// Prediction arithmetic overflowed or returned a non-positive value.
    NonFinitePrediction,
}

impl fmt::Display for CastingModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughObservations { actual, minimum } => write!(
                f,
                "casting calibration needs at least {minimum} observations; received {actual}"
            ),
            Self::InvalidObservation { index } => {
                write!(f, "casting observation {index} must contain finite positive values")
            }
            Self::DegenerateCalibrationDomain => {
                f.write_str("casting calibration observations do not span thermal modulus")
            }
            Self::NonPositiveExponent => {
                f.write_str("casting calibration produced a non-positive exponent")
            }
            Self::InvalidCalibrationParameters => {
                f.write_str("casting calibration produced invalid parameters or diagnostics")
            }
            Self::OutsideCalibrationDomain {
                requested_m,
                minimum_m,
                maximum_m,
            } => write!(
                f,
                "thermal modulus {requested_m} m is outside calibrated range [{minimum_m}, {maximum_m}] m"
            ),
            Self::InvalidGeometry => {
                f.write_str("casting volume and cooling surface area must be finite and positive")
            }
            Self::NonFinitePrediction => {
                f.write_str("casting model prediction is not finite and positive")
            }
        }
    }
}

impl Error for CastingModelError {}

/// A Chvorinov-style model calibrated to measured solidification observations.
///
/// Parameters are kept private so a model can only be constructed through the
/// validation/calibration path. The fit is ordinary least squares in log space:
/// ln(t) = ln(B) + n * ln(V/A).
///
/// A good in-sample fit does not establish predictive validity. Use held-out castings
/// and a frozen measurement/inspection protocol before using this model in an
/// engineering decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ChvorinovModel {
    mold_constant_s_per_m_pow_n: f64,
    exponent: f64,
    thermal_modulus_min_m: f64,
    thermal_modulus_max_m: f64,
    fit_r_squared: f64,
    sample_count: usize,
}

impl ChvorinovModel {
    /// Fit a model to measurements from one compatible process population.
    ///
    /// The caller must not combine observations from unlike alloys, mold materials,
    /// tooling states, measurement definitions, or thermal boundary conditions merely
    /// to increase sample count. Fit quality is reported for audit, not treated as
    /// qualification.
    pub fn calibrate(
        observations: &[ChvorinovObservation],
    ) -> Result<Self, CastingModelError> {
        if observations.len() < MIN_CALIBRATION_OBSERVATIONS {
            return Err(CastingModelError::NotEnoughObservations {
                actual: observations.len(),
                minimum: MIN_CALIBRATION_OBSERVATIONS,
            });
        }

        for (index, observation) in observations.iter().enumerate() {
            if !observation.thermal_modulus_m.is_finite()
                || observation.thermal_modulus_m <= 0.0
                || !observation.solidification_time_s.is_finite()
                || observation.solidification_time_s <= 0.0
            {
                return Err(CastingModelError::InvalidObservation { index });
            }
        }

        let min_m = observations
            .iter()
            .map(|observation| observation.thermal_modulus_m)
            .fold(f64::INFINITY, f64::min);
        let max_m = observations
            .iter()
            .map(|observation| observation.thermal_modulus_m)
            .fold(f64::NEG_INFINITY, f64::max);

        let xs: Vec<f64> = observations
            .iter()
            .map(|observation| observation.thermal_modulus_m.ln())
            .collect();
        let ys: Vec<f64> = observations
            .iter()
            .map(|observation| observation.solidification_time_s.ln())
            .collect();

        let count = observations.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / count;
        let mean_y = ys.iter().sum::<f64>() / count;
        let ss_x = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
        if !ss_x.is_finite() || ss_x <= f64::EPSILON {
            return Err(CastingModelError::DegenerateCalibrationDomain);
        }

        let covariance_xy = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
        let exponent = covariance_xy / ss_x;
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(CastingModelError::NonPositiveExponent);
        }

        let log_constant = mean_y - exponent * mean_x;
        let mold_constant_s_per_m_pow_n = log_constant.exp();
        if !mold_constant_s_per_m_pow_n.is_finite()
            || mold_constant_s_per_m_pow_n <= 0.0
        {
            return Err(CastingModelError::InvalidCalibrationParameters);
        }

        let ss_y = ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
        if !ss_y.is_finite() || ss_y <= f64::EPSILON {
            return Err(CastingModelError::DegenerateCalibrationDomain);
        }
        let ss_residual = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (y - (log_constant + exponent * x)).powi(2))
            .sum::<f64>();
        let fit_r_squared = 1.0 - ss_residual / ss_y;
        if !fit_r_squared.is_finite() {
            return Err(CastingModelError::InvalidCalibrationParameters);
        }

        Ok(Self {
            mold_constant_s_per_m_pow_n,
            exponent,
            thermal_modulus_min_m: min_m,
            thermal_modulus_max_m: max_m,
            fit_r_squared,
            sample_count: observations.len(),
        })
    }

    /// Estimate solidification time for a thermal modulus within the calibration range.
    ///
    /// Out-of-domain input is rejected; extrapolation must not be mistaken for a
    /// calibrated estimate. A returned value is not a prediction interval and says
    /// nothing by itself about shrinkage, porosity, microstructure, strength, or acceptance.
    pub fn estimate_solidification_time_s(
        &self,
        thermal_modulus_m: f64,
    ) -> Result<f64, CastingModelError> {
        if !thermal_modulus_m.is_finite() || thermal_modulus_m <= 0.0 {
            return Err(CastingModelError::InvalidGeometry);
        }
        if thermal_modulus_m < self.thermal_modulus_min_m
            || thermal_modulus_m > self.thermal_modulus_max_m
        {
            return Err(CastingModelError::OutsideCalibrationDomain {
                requested_m: thermal_modulus_m,
                minimum_m: self.thermal_modulus_min_m,
                maximum_m: self.thermal_modulus_max_m,
            });
        }

        let estimate = self.mold_constant_s_per_m_pow_n * thermal_modulus_m.powf(self.exponent);
        if !estimate.is_finite() || estimate <= 0.0 {
            return Err(CastingModelError::NonFinitePrediction);
        }
        Ok(estimate)
    }

    /// Estimate solidification time from SI geometry measurements.
    pub fn estimate_from_geometry_s(
        &self,
        volume_m3: f64,
        effective_cooling_surface_area_m2: f64,
    ) -> Result<f64, CastingModelError> {
        let modulus = thermal_modulus_m(volume_m3, effective_cooling_surface_area_m2)?;
        self.estimate_solidification_time_s(modulus)
    }

    /// Return the fitted mold constant in seconds per metre raised to the fitted exponent.
    pub fn mold_constant_s_per_m_pow_n(&self) -> f64 {
        self.mold_constant_s_per_m_pow_n
    }

    /// Return the fitted positive exponent.
    pub fn exponent(&self) -> f64 {
        self.exponent
    }

    /// Return the minimum and maximum observed thermal moduli, in metres.
    pub fn calibration_domain_m(&self) -> (f64, f64) {
        (self.thermal_modulus_min_m, self.thermal_modulus_max_m)
    }

    /// Return the in-sample coefficient of determination in log space.
    ///
    /// This diagnostic is not a confidence level or an estimate of out-of-sample accuracy.
    pub fn fit_r_squared(&self) -> f64 {
        self.fit_r_squared
    }

    /// Return the number of observations used in calibration.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// Calculate the thermal modulus using SI units.
///
/// This is the geometric ratio used by the model, not an assertion that all geometric
/// surfaces exchange heat equally. Apply a documented, physically justified surface
/// definition consistently across calibration and prediction.
pub fn thermal_modulus_m(
    volume_m3: f64,
    effective_cooling_surface_area_m2: f64,
) -> Result<f64, CastingModelError> {
    if !volume_m3.is_finite()
        || volume_m3 <= 0.0
        || !effective_cooling_surface_area_m2.is_finite()
        || effective_cooling_surface_area_m2 <= 0.0
    {
        return Err(CastingModelError::InvalidGeometry);
    }
    let modulus = volume_m3 / effective_cooling_surface_area_m2;
    if !modulus.is_finite() || modulus <= 0.0 {
        return Err(CastingModelError::InvalidGeometry);
    }
    Ok(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_observations() -> Vec<ChvorinovObservation> {
        // Exact synthetic relation: t = 1.2e6 * (V/A)^2.
        // These fixtures test arithmetic only; they are not physical foundry data.
        [0.001_f64, 0.002, 0.004, 0.008, 0.010]
            .into_iter()
            .map(|modulus| ChvorinovObservation::new(modulus, 1.2e6 * modulus.powi(2)))
            .collect()
    }

    #[test]
    fn recovers_exact_synthetic_chvorinov_parameters() {
        let model = ChvorinovModel::calibrate(&synthetic_observations()).unwrap();
        assert!((model.exponent() - 2.0).abs() < 1e-10);
        assert!((model.mold_constant_s_per_m_pow_n() - 1.2e6).abs() < 1e-6);
        assert!(model.fit_r_squared() > 0.999_999_999);
        assert_eq!(model.sample_count(), 5);
    }

    #[test]
    fn predicts_only_inside_observed_modulus_domain() {
        let model = ChvorinovModel::calibrate(&synthetic_observations()).unwrap();
        let predicted = model.estimate_solidification_time_s(0.005).unwrap();
        assert!((predicted - 30.0).abs() < 1e-8);

        assert!(matches!(
            model.estimate_solidification_time_s(0.02),
            Err(CastingModelError::OutsideCalibrationDomain { .. })
        ));
    }

    #[test]
    fn estimates_from_si_geometry() {
        let model = ChvorinovModel::calibrate(&synthetic_observations()).unwrap();
        // 1e-6 m^3 / 1e-3 m^2 = 1e-3 m; predicted time is 1.2 s.
        let predicted = model.estimate_from_geometry_s(1e-6, 1e-3).unwrap();
        assert!((predicted - 1.2).abs() < 1e-9);
    }

    #[test]
    fn rejects_invalid_geometry_and_non_finite_values() {
        assert!(matches!(
            thermal_modulus_m(0.0, 1.0),
            Err(CastingModelError::InvalidGeometry)
        ));
        assert!(matches!(
            thermal_modulus_m(1.0, f64::NAN),
            Err(CastingModelError::InvalidGeometry)
        ));
        let model = ChvorinovModel::calibrate(&synthetic_observations()).unwrap();
        assert!(matches!(
            model.estimate_solidification_time_s(f64::INFINITY),
            Err(CastingModelError::InvalidGeometry)
        ));
    }

    #[test]
    fn rejects_too_few_or_invalid_observations() {
        let too_few = synthetic_observations();
        assert!(matches!(
            ChvorinovModel::calibrate(&too_few[..2]),
            Err(CastingModelError::NotEnoughObservations { .. })
        ));

        let invalid = [
            ChvorinovObservation::new(0.001, 1.0),
            ChvorinovObservation::new(0.002, 2.0),
            ChvorinovObservation::new(0.003, f64::NAN),
        ];
        assert!(matches!(
            ChvorinovModel::calibrate(&invalid),
            Err(CastingModelError::InvalidObservation { index: 2 })
        ));
    }

    #[test]
    fn rejects_degenerate_and_non_positive_fits() {
        let degenerate = [
            ChvorinovObservation::new(0.001, 1.0),
            ChvorinovObservation::new(0.001, 2.0),
            ChvorinovObservation::new(0.001, 3.0),
        ];
        assert!(matches!(
            ChvorinovModel::calibrate(&degenerate),
            Err(CastingModelError::DegenerateCalibrationDomain)
        ));

        let negative_slope = [
            ChvorinovObservation::new(0.001, 3.0),
            ChvorinovObservation::new(0.002, 2.0),
            ChvorinovObservation::new(0.003, 1.0),
        ];
        assert!(matches!(
            ChvorinovModel::calibrate(&negative_slope),
            Err(CastingModelError::NonPositiveExponent)
        ));
    }
}
