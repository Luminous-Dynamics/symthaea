// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit environmental-driver contracts for analytic ecology models.
//!
//! This module does not claim a universal climate-to-ecology law. It provides a
//! small, inspectable parameter bridge whose assumptions and local
//! sensitivities are visible and can be replaced or calibrated by callers.

use crate::error::{ModelError, require_non_negative, require_positive};
use crate::logistic::LogisticModel;

/// Environmental inputs supplied by a climate, habitat, or scenario layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentalDrivers {
    /// Absolute environmental temperature, K.
    pub temperature: f64,
    /// Positive relative productivity index; `1.0` is caller-defined baseline.
    pub productivity: f64,
    /// Non-negative disturbance intensity in caller-defined units.
    pub disturbance: f64,
}

impl EnvironmentalDrivers {
    pub fn try_new(
        temperature: f64,
        productivity: f64,
        disturbance: f64,
    ) -> Result<Self, ModelError> {
        require_positive("temperature", temperature)?;
        require_positive("productivity", productivity)?;
        require_non_negative("disturbance", disturbance)?;
        Ok(Self {
            temperature,
            productivity,
            disturbance,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("temperature", self.temperature)?;
        require_positive("productivity", self.productivity)?;
        require_non_negative("disturbance", self.disturbance)?;
        Ok(())
    }
}

/// Gaussian thermal-performance multiplier with an explicit non-zero floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianThermalResponse {
    pub optimum_temperature: f64,
    pub breadth: f64,
    pub minimum_fraction: f64,
}

impl GaussianThermalResponse {
    pub fn try_new(
        optimum_temperature: f64,
        breadth: f64,
        minimum_fraction: f64,
    ) -> Result<Self, ModelError> {
        let response = Self {
            optimum_temperature,
            breadth,
            minimum_fraction,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("optimum_temperature", self.optimum_temperature)?;
        require_positive("breadth", self.breadth)?;
        require_positive("minimum_fraction", self.minimum_fraction)?;
        if self.minimum_fraction > 1.0 {
            return Err(ModelError::OutOfRange {
                parameter: "minimum_fraction",
                value: self.minimum_fraction,
                min: f64::MIN_POSITIVE,
                max: 1.0,
            });
        }
        Ok(())
    }

    pub fn multiplier(&self, temperature: f64) -> Result<f64, ModelError> {
        Ok(self.multiplier_and_derivative(temperature)?.0)
    }

    /// Return `(multiplier, d multiplier / d temperature)`.
    pub fn multiplier_and_derivative(&self, temperature: f64) -> Result<(f64, f64), ModelError> {
        self.validate()?;
        require_positive("temperature", temperature)?;
        let offset = temperature - self.optimum_temperature;
        let standardized = offset / self.breadth;
        let gaussian = (-0.5 * standardized.powi(2)).exp();
        let active_fraction = 1.0 - self.minimum_fraction;
        let multiplier = self.minimum_fraction + active_fraction * gaussian;
        let derivative = active_fraction * gaussian * (-offset / self.breadth.powi(2));
        Ok((multiplier, derivative))
    }
}

/// Auditable decomposition of an environment-to-logistic transformation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticEnvironmentEvaluation {
    pub model: LogisticModel,
    pub thermal_multiplier: f64,
    pub productivity_multiplier: f64,
    pub disturbance_multiplier: f64,
    /// Local derivative of effective intrinsic growth with respect to K.
    pub growth_temperature_sensitivity: f64,
    /// Local derivative of effective carrying capacity with respect to K.
    pub capacity_temperature_sensitivity: f64,
}

/// Transparent coupling from environmental drivers to logistic parameters.
///
/// - temperature scales both growth and carrying capacity;
/// - productivity scales carrying capacity relative to a named baseline;
/// - disturbance exponentially suppresses intrinsic growth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticEnvironmentCoupling {
    pub baseline: LogisticModel,
    pub thermal_response: GaussianThermalResponse,
    pub reference_productivity: f64,
    pub disturbance_sensitivity: f64,
}

impl LogisticEnvironmentCoupling {
    pub fn try_new(
        baseline: LogisticModel,
        thermal_response: GaussianThermalResponse,
        reference_productivity: f64,
        disturbance_sensitivity: f64,
    ) -> Result<Self, ModelError> {
        baseline.validate()?;
        thermal_response.validate()?;
        require_positive("reference_productivity", reference_productivity)?;
        require_non_negative("disturbance_sensitivity", disturbance_sensitivity)?;
        Ok(Self {
            baseline,
            thermal_response,
            reference_productivity,
            disturbance_sensitivity,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.baseline.validate()?;
        self.thermal_response.validate()?;
        require_positive("reference_productivity", self.reference_productivity)?;
        require_non_negative("disturbance_sensitivity", self.disturbance_sensitivity)?;
        Ok(())
    }

    /// Evaluate all multipliers and local temperature sensitivities.
    pub fn evaluate(
        &self,
        environment: EnvironmentalDrivers,
    ) -> Result<LogisticEnvironmentEvaluation, ModelError> {
        self.validate()?;
        environment.validate()?;

        let (thermal_multiplier, thermal_derivative) = self
            .thermal_response
            .multiplier_and_derivative(environment.temperature)?;
        let disturbance_multiplier =
            (-self.disturbance_sensitivity * environment.disturbance).exp();
        let productivity_multiplier = environment.productivity / self.reference_productivity;

        let model = LogisticModel::try_new(
            self.baseline.intrinsic_growth_rate * thermal_multiplier * disturbance_multiplier,
            self.baseline.carrying_capacity * thermal_multiplier * productivity_multiplier,
        )?;

        Ok(LogisticEnvironmentEvaluation {
            model,
            thermal_multiplier,
            productivity_multiplier,
            disturbance_multiplier,
            growth_temperature_sensitivity: self.baseline.intrinsic_growth_rate
                * disturbance_multiplier
                * thermal_derivative,
            capacity_temperature_sensitivity: self.baseline.carrying_capacity
                * productivity_multiplier
                * thermal_derivative,
        })
    }

    /// Derive a validated logistic model for the supplied environment.
    pub fn effective_model(
        &self,
        environment: EnvironmentalDrivers,
    ) -> Result<LogisticModel, ModelError> {
        Ok(self.evaluate(environment)?.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coupling() -> LogisticEnvironmentCoupling {
        LogisticEnvironmentCoupling::try_new(
            LogisticModel::try_new(0.5, 100.0).unwrap(),
            GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap(),
            1.0,
            0.5,
        )
        .unwrap()
    }

    #[test]
    fn optimum_baseline_environment_recovers_baseline() {
        let evaluation = coupling()
            .evaluate(EnvironmentalDrivers::try_new(293.0, 1.0, 0.0).unwrap())
            .unwrap();
        assert!((evaluation.model.intrinsic_growth_rate - 0.5).abs() < 1e-12);
        assert!((evaluation.model.carrying_capacity - 100.0).abs() < 1e-12);
        assert!((evaluation.thermal_multiplier - 1.0).abs() < 1e-12);
        assert!(evaluation.growth_temperature_sensitivity.abs() < 1e-12);
        assert!(evaluation.capacity_temperature_sensitivity.abs() < 1e-12);
    }

    #[test]
    fn lower_productivity_reduces_capacity_not_thermal_growth() {
        let evaluation = coupling()
            .evaluate(EnvironmentalDrivers::try_new(293.0, 0.5, 0.0).unwrap())
            .unwrap();
        assert!((evaluation.model.intrinsic_growth_rate - 0.5).abs() < 1e-12);
        assert!((evaluation.model.carrying_capacity - 50.0).abs() < 1e-12);
        assert!((evaluation.productivity_multiplier - 0.5).abs() < 1e-12);
    }

    #[test]
    fn disturbance_reduces_growth() {
        let baseline = coupling()
            .evaluate(EnvironmentalDrivers::try_new(293.0, 1.0, 0.0).unwrap())
            .unwrap();
        let disturbed = coupling()
            .evaluate(EnvironmentalDrivers::try_new(293.0, 1.0, 2.0).unwrap())
            .unwrap();
        assert!(disturbed.model.intrinsic_growth_rate < baseline.model.intrinsic_growth_rate);
        assert!(
            (disturbed.model.carrying_capacity - baseline.model.carrying_capacity).abs() < 1e-12
        );
        assert!(disturbed.disturbance_multiplier < 1.0);
    }

    #[test]
    fn thermal_response_is_symmetric_but_derivative_changes_sign() {
        let response = GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap();
        let (cool, cool_derivative) = response.multiplier_and_derivative(283.0).unwrap();
        let (warm, warm_derivative) = response.multiplier_and_derivative(303.0).unwrap();
        assert!((cool - warm).abs() < 1e-12);
        assert!((cool_derivative + warm_derivative).abs() < 1e-12);
        assert!(cool_derivative > 0.0 && warm_derivative < 0.0);
    }

    #[test]
    fn analytic_thermal_derivative_matches_finite_difference() {
        let response = GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap();
        let temperature = 300.0;
        let epsilon = 1e-5;
        let (_, analytic) = response.multiplier_and_derivative(temperature).unwrap();
        let numerical = (response.multiplier(temperature + epsilon).unwrap()
            - response.multiplier(temperature - epsilon).unwrap())
            / (2.0 * epsilon);
        assert!((analytic - numerical).abs() < 1e-9);
    }

    #[test]
    fn manually_invalid_nested_models_are_rejected() {
        let invalid = LogisticEnvironmentCoupling::try_new(
            LogisticModel {
                intrinsic_growth_rate: -1.0,
                carrying_capacity: 100.0,
            },
            GaussianThermalResponse {
                optimum_temperature: 293.0,
                breadth: 10.0,
                minimum_fraction: 0.1,
            },
            1.0,
            0.0,
        );
        assert!(invalid.is_err());
    }
}
