// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit soil-moisture coupling for analytic population models.
//!
//! This module composes the existing temperature/productivity/disturbance
//! coupling with a bounded piecewise-linear soil-moisture response. The input
//! `soil_moisture_fraction` directly matches the dependency-neutral field
//! exported by `symthaea-earth-system::HydrologyDriverSample`; no crate link is
//! required.

use crate::environment::{
    EnvironmentalDrivers, LogisticEnvironmentCoupling, LogisticEnvironmentEvaluation,
};
use crate::error::{ModelError, require_fraction, require_non_negative, require_positive};
use crate::logistic::LogisticModel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydroEnvironmentalDrivers {
    pub climate: EnvironmentalDrivers,
    /// Relative bucket storage in `[0, 1]`.
    pub soil_moisture_fraction: f64,
}

impl HydroEnvironmentalDrivers {
    pub fn try_new(
        climate: EnvironmentalDrivers,
        soil_moisture_fraction: f64,
    ) -> Result<Self, ModelError> {
        climate.validate()?;
        require_fraction("soil_moisture_fraction", soil_moisture_fraction)?;
        Ok(Self {
            climate,
            soil_moisture_fraction,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.climate.validate()?;
        require_fraction("soil_moisture_fraction", self.soil_moisture_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilMoistureResponse {
    /// At or below this fraction, the multiplier is `minimum_fraction`.
    pub wilting_fraction: f64,
    /// At or above this fraction, the multiplier is one.
    pub optimum_fraction: f64,
    /// Non-negative lower bound on the response.
    pub minimum_fraction: f64,
}

impl SoilMoistureResponse {
    pub fn try_new(
        wilting_fraction: f64,
        optimum_fraction: f64,
        minimum_fraction: f64,
    ) -> Result<Self, ModelError> {
        let response = Self {
            wilting_fraction,
            optimum_fraction,
            minimum_fraction,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_fraction("wilting_fraction", self.wilting_fraction)?;
        require_fraction("optimum_fraction", self.optimum_fraction)?;
        require_positive("minimum_fraction", self.minimum_fraction)?;
        if self.minimum_fraction > 1.0 {
            return Err(ModelError::OutOfRange {
                parameter: "minimum_fraction",
                value: self.minimum_fraction,
                min: f64::MIN_POSITIVE,
                max: 1.0,
            });
        }
        if self.wilting_fraction >= self.optimum_fraction {
            return Err(ModelError::OutOfRange {
                parameter: "wilting_fraction",
                value: self.wilting_fraction,
                min: 0.0,
                max: self.optimum_fraction,
            });
        }
        Ok(())
    }

    /// Return `(multiplier, d multiplier / d moisture fraction)`.
    pub fn multiplier_and_derivative(
        &self,
        soil_moisture_fraction: f64,
    ) -> Result<(f64, f64), ModelError> {
        self.validate()?;
        require_fraction("soil_moisture_fraction", soil_moisture_fraction)?;
        if soil_moisture_fraction <= self.wilting_fraction {
            return Ok((self.minimum_fraction, 0.0));
        }
        if soil_moisture_fraction >= self.optimum_fraction {
            return Ok((1.0, 0.0));
        }
        let span = self.optimum_fraction - self.wilting_fraction;
        let position = (soil_moisture_fraction - self.wilting_fraction) / span;
        let active = 1.0 - self.minimum_fraction;
        Ok((self.minimum_fraction + active * position, active / span))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydroLogisticEvaluation {
    pub model: LogisticModel,
    pub climate_evaluation: LogisticEnvironmentEvaluation,
    pub moisture_multiplier: f64,
    pub growth_moisture_multiplier: f64,
    pub capacity_moisture_multiplier: f64,
    pub growth_moisture_sensitivity: f64,
    pub capacity_moisture_sensitivity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydroLogisticEnvironmentCoupling {
    pub climate_coupling: LogisticEnvironmentCoupling,
    pub moisture_response: SoilMoistureResponse,
    /// Exponent applied to the moisture multiplier for intrinsic growth.
    pub growth_exponent: f64,
    /// Exponent applied to the moisture multiplier for carrying capacity.
    pub capacity_exponent: f64,
}

impl HydroLogisticEnvironmentCoupling {
    pub fn try_new(
        climate_coupling: LogisticEnvironmentCoupling,
        moisture_response: SoilMoistureResponse,
        growth_exponent: f64,
        capacity_exponent: f64,
    ) -> Result<Self, ModelError> {
        let coupling = Self {
            climate_coupling,
            moisture_response,
            growth_exponent,
            capacity_exponent,
        };
        coupling.validate()?;
        Ok(coupling)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.climate_coupling.validate()?;
        self.moisture_response.validate()?;
        require_non_negative("growth_exponent", self.growth_exponent)?;
        require_non_negative("capacity_exponent", self.capacity_exponent)
    }

    pub fn evaluate(
        &self,
        drivers: HydroEnvironmentalDrivers,
    ) -> Result<HydroLogisticEvaluation, ModelError> {
        self.validate()?;
        drivers.validate()?;
        let climate_evaluation = self.climate_coupling.evaluate(drivers.climate)?;
        let (moisture_multiplier, moisture_derivative) = self
            .moisture_response
            .multiplier_and_derivative(drivers.soil_moisture_fraction)?;
        let growth_moisture_multiplier = moisture_multiplier.powf(self.growth_exponent);
        let capacity_moisture_multiplier = moisture_multiplier.powf(self.capacity_exponent);
        let model = LogisticModel::try_new(
            climate_evaluation.model.intrinsic_growth_rate * growth_moisture_multiplier,
            climate_evaluation.model.carrying_capacity * capacity_moisture_multiplier,
        )?;
        let growth_moisture_sensitivity = exponent_derivative(
            climate_evaluation.model.intrinsic_growth_rate,
            moisture_multiplier,
            moisture_derivative,
            self.growth_exponent,
        );
        let capacity_moisture_sensitivity = exponent_derivative(
            climate_evaluation.model.carrying_capacity,
            moisture_multiplier,
            moisture_derivative,
            self.capacity_exponent,
        );
        Ok(HydroLogisticEvaluation {
            model,
            climate_evaluation,
            moisture_multiplier,
            growth_moisture_multiplier,
            capacity_moisture_multiplier,
            growth_moisture_sensitivity,
            capacity_moisture_sensitivity,
        })
    }
}

fn exponent_derivative(
    baseline: f64,
    multiplier: f64,
    multiplier_derivative: f64,
    exponent: f64,
) -> f64 {
    if exponent == 0.0 || multiplier_derivative == 0.0 {
        0.0
    } else {
        baseline * exponent * multiplier.powf(exponent - 1.0) * multiplier_derivative
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::GaussianThermalResponse;

    fn coupling() -> HydroLogisticEnvironmentCoupling {
        HydroLogisticEnvironmentCoupling::try_new(
            LogisticEnvironmentCoupling::try_new(
                LogisticModel::try_new(0.5, 100.0).unwrap(),
                GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap(),
                1.0,
                0.5,
            )
            .unwrap(),
            SoilMoistureResponse::try_new(0.2, 0.7, 0.1).unwrap(),
            1.0,
            2.0,
        )
        .unwrap()
    }

    fn drivers(moisture: f64) -> HydroEnvironmentalDrivers {
        HydroEnvironmentalDrivers::try_new(
            EnvironmentalDrivers::try_new(293.0, 1.0, 0.0).unwrap(),
            moisture,
        )
        .unwrap()
    }

    #[test]
    fn optimum_moisture_recovers_climate_only_model() {
        let evaluation = coupling().evaluate(drivers(0.7)).unwrap();
        assert_eq!(evaluation.moisture_multiplier, 1.0);
        assert!((evaluation.model.intrinsic_growth_rate - 0.5).abs() < 1.0e-12);
        assert!((evaluation.model.carrying_capacity - 100.0).abs() < 1.0e-12);
    }

    #[test]
    fn dry_soil_suppresses_capacity_more_strongly_than_growth() {
        let evaluation = coupling().evaluate(drivers(0.2)).unwrap();
        assert!((evaluation.moisture_multiplier - 0.1).abs() < 1.0e-12);
        assert!((evaluation.model.intrinsic_growth_rate - 0.05).abs() < 1.0e-12);
        assert!((evaluation.model.carrying_capacity - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn analytic_moisture_sensitivity_matches_finite_difference() {
        let coupling = coupling();
        let moisture = 0.45;
        let evaluation = coupling.evaluate(drivers(moisture)).unwrap();
        let h = 1.0e-6;
        let upper = coupling.evaluate(drivers(moisture + h)).unwrap();
        let lower = coupling.evaluate(drivers(moisture - h)).unwrap();
        let growth_fd =
            (upper.model.intrinsic_growth_rate - lower.model.intrinsic_growth_rate) / (2.0 * h);
        let capacity_fd =
            (upper.model.carrying_capacity - lower.model.carrying_capacity) / (2.0 * h);
        assert!((evaluation.growth_moisture_sensitivity - growth_fd).abs() < 1.0e-8);
        assert!((evaluation.capacity_moisture_sensitivity - capacity_fd).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_moisture_boundaries_fail_closed() {
        assert!(SoilMoistureResponse::try_new(0.7, 0.2, 0.1).is_err());
        assert!(
            HydroEnvironmentalDrivers::try_new(
                EnvironmentalDrivers::try_new(293.0, 1.0, 0.0).unwrap(),
                1.1,
            )
            .is_err()
        );
        assert!(SoilMoistureResponse::try_new(0.2, 0.7, 0.0).is_err());
    }
}
