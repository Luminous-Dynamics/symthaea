// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit mineral-nutrient limitation layered onto climate and moisture.
//!
//! The response is a bounded Monod-type multiplier with a declared floor and
//! half-saturation stock. It is a transparent parameter bridge, not a universal
//! nutrient-to-demography law. Callers remain responsible for units, uptake
//! feedbacks, species identity, and whether the supplied mineral stock is local,
//! accessible, and temporally aligned with the population model.

use crate::error::{ModelError, require_non_negative, require_positive};
use crate::logistic::LogisticModel;
use crate::moisture_environment::{
    HydroEnvironmentalDrivers, HydroLogisticEnvironmentCoupling, HydroLogisticEvaluation,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientEnvironmentalDrivers {
    pub hydroclimate: HydroEnvironmentalDrivers,
    pub mineral_nutrient: f64,
}

impl NutrientEnvironmentalDrivers {
    pub fn try_new(
        hydroclimate: HydroEnvironmentalDrivers,
        mineral_nutrient: f64,
    ) -> Result<Self, ModelError> {
        hydroclimate.validate()?;
        require_non_negative("mineral_nutrient", mineral_nutrient)?;
        Ok(Self {
            hydroclimate,
            mineral_nutrient,
        })
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.hydroclimate.validate()?;
        require_non_negative("mineral_nutrient", self.mineral_nutrient)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MineralNutrientResponse {
    pub half_saturation: f64,
    pub minimum_fraction: f64,
}

impl MineralNutrientResponse {
    pub fn try_new(half_saturation: f64, minimum_fraction: f64) -> Result<Self, ModelError> {
        let response = Self {
            half_saturation,
            minimum_fraction,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("half_saturation", self.half_saturation)?;
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

    /// Return `(multiplier, derivative with respect to mineral nutrient)`.
    pub fn multiplier_and_derivative(
        &self,
        mineral_nutrient: f64,
    ) -> Result<(f64, f64), ModelError> {
        self.validate()?;
        require_non_negative("mineral_nutrient", mineral_nutrient)?;
        let denominator = self.half_saturation + mineral_nutrient;
        let active = 1.0 - self.minimum_fraction;
        let saturation = mineral_nutrient / denominator;
        let derivative = active * self.half_saturation / denominator.powi(2);
        Ok((self.minimum_fraction + active * saturation, derivative))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientLogisticEvaluation {
    pub model: LogisticModel,
    pub hydroclimate_evaluation: HydroLogisticEvaluation,
    pub nutrient_multiplier: f64,
    pub growth_nutrient_multiplier: f64,
    pub capacity_nutrient_multiplier: f64,
    pub growth_nutrient_sensitivity: f64,
    pub capacity_nutrient_sensitivity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutrientLogisticEnvironmentCoupling {
    pub hydroclimate_coupling: HydroLogisticEnvironmentCoupling,
    pub nutrient_response: MineralNutrientResponse,
    pub growth_exponent: f64,
    pub capacity_exponent: f64,
}

impl NutrientLogisticEnvironmentCoupling {
    pub fn try_new(
        hydroclimate_coupling: HydroLogisticEnvironmentCoupling,
        nutrient_response: MineralNutrientResponse,
        growth_exponent: f64,
        capacity_exponent: f64,
    ) -> Result<Self, ModelError> {
        let coupling = Self {
            hydroclimate_coupling,
            nutrient_response,
            growth_exponent,
            capacity_exponent,
        };
        coupling.validate()?;
        Ok(coupling)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.hydroclimate_coupling.validate()?;
        self.nutrient_response.validate()?;
        require_non_negative("growth_exponent", self.growth_exponent)?;
        require_non_negative("capacity_exponent", self.capacity_exponent)
    }

    pub fn evaluate(
        &self,
        drivers: NutrientEnvironmentalDrivers,
    ) -> Result<NutrientLogisticEvaluation, ModelError> {
        self.validate()?;
        drivers.validate()?;
        let hydroclimate_evaluation = self.hydroclimate_coupling.evaluate(drivers.hydroclimate)?;
        let (nutrient_multiplier, nutrient_derivative) = self
            .nutrient_response
            .multiplier_and_derivative(drivers.mineral_nutrient)?;
        let growth_nutrient_multiplier = nutrient_multiplier.powf(self.growth_exponent);
        let capacity_nutrient_multiplier = nutrient_multiplier.powf(self.capacity_exponent);
        let model = LogisticModel::try_new(
            hydroclimate_evaluation.model.intrinsic_growth_rate * growth_nutrient_multiplier,
            hydroclimate_evaluation.model.carrying_capacity * capacity_nutrient_multiplier,
        )?;
        Ok(NutrientLogisticEvaluation {
            model,
            hydroclimate_evaluation,
            nutrient_multiplier,
            growth_nutrient_multiplier,
            capacity_nutrient_multiplier,
            growth_nutrient_sensitivity: exponent_derivative(
                hydroclimate_evaluation.model.intrinsic_growth_rate,
                nutrient_multiplier,
                nutrient_derivative,
                self.growth_exponent,
            ),
            capacity_nutrient_sensitivity: exponent_derivative(
                hydroclimate_evaluation.model.carrying_capacity,
                nutrient_multiplier,
                nutrient_derivative,
                self.capacity_exponent,
            ),
        })
    }
}

fn exponent_derivative(baseline: f64, multiplier: f64, derivative: f64, exponent: f64) -> f64 {
    if exponent == 0.0 || derivative == 0.0 {
        0.0
    } else {
        baseline * exponent * multiplier.powf(exponent - 1.0) * derivative
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{
        EnvironmentalDrivers, GaussianThermalResponse, LogisticEnvironmentCoupling,
    };
    use crate::moisture_environment::SoilMoistureResponse;

    fn coupling() -> NutrientLogisticEnvironmentCoupling {
        NutrientLogisticEnvironmentCoupling::try_new(
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
                1.0,
            )
            .unwrap(),
            MineralNutrientResponse::try_new(2.0, 0.1).unwrap(),
            1.0,
            2.0,
        )
        .unwrap()
    }

    fn drivers(nutrient: f64) -> NutrientEnvironmentalDrivers {
        NutrientEnvironmentalDrivers::try_new(
            HydroEnvironmentalDrivers::try_new(
                EnvironmentalDrivers::try_new(293.0, 1.0, 0.0).unwrap(),
                0.7,
            )
            .unwrap(),
            nutrient,
        )
        .unwrap()
    }

    #[test]
    fn zero_nutrient_uses_declared_nonzero_floor() {
        let evaluation = coupling().evaluate(drivers(0.0)).unwrap();
        assert!((evaluation.nutrient_multiplier - 0.1).abs() < 1.0e-12);
        assert!((evaluation.model.intrinsic_growth_rate - 0.05).abs() < 1.0e-12);
        assert!((evaluation.model.carrying_capacity - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn response_is_monotone_and_saturating() {
        let low = coupling().evaluate(drivers(1.0)).unwrap();
        let high = coupling().evaluate(drivers(1000.0)).unwrap();
        assert!(high.nutrient_multiplier > low.nutrient_multiplier);
        assert!(high.nutrient_multiplier < 1.0);
        assert!(high.model.carrying_capacity > low.model.carrying_capacity);
    }

    #[test]
    fn analytic_sensitivity_matches_finite_difference() {
        let coupling = coupling();
        let nutrient = 3.0;
        let evaluation = coupling.evaluate(drivers(nutrient)).unwrap();
        let h = 1.0e-6;
        let upper = coupling.evaluate(drivers(nutrient + h)).unwrap();
        let lower = coupling.evaluate(drivers(nutrient - h)).unwrap();
        let growth_fd =
            (upper.model.intrinsic_growth_rate - lower.model.intrinsic_growth_rate) / (2.0 * h);
        let capacity_fd =
            (upper.model.carrying_capacity - lower.model.carrying_capacity) / (2.0 * h);
        assert!((evaluation.growth_nutrient_sensitivity - growth_fd).abs() < 1.0e-9);
        assert!((evaluation.capacity_nutrient_sensitivity - capacity_fd).abs() < 1.0e-7);
    }

    #[test]
    fn invalid_half_saturation_fails_closed() {
        assert!(MineralNutrientResponse::try_new(0.0, 0.1).is_err());
        assert!(NutrientEnvironmentalDrivers::try_new(drivers(1.0).hydroclimate, -1.0,).is_err());
    }
}
