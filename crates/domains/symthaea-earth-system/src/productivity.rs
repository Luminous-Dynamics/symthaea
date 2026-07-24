// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable ecosystem-productivity accounting.
//!
//! This module is a finite-interval ledger rather than a vegetation model. It
//! limits potential gross primary production by an explicit environmental
//! multiplier and a finite mineral-nutrient stock, then partitions assimilated
//! carbon into autotrophic respiration, retained biomass, and litter. Every
//! quantity is caller-unit agnostic as long as carbon and nutrient units are
//! internally consistent.

use crate::error::{ModelError, require_fraction, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductivityLimitation {
    Environmental,
    Nutrient,
    CoLimited,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductivityLedger {
    pub duration: f64,
    pub environmental_multiplier: f64,
    pub environmental_carbon_ceiling: f64,
    pub nutrient_carbon_ceiling: f64,
    pub gross_primary_production: f64,
    pub autotrophic_respiration: f64,
    pub net_primary_production: f64,
    pub retained_biomass_carbon: f64,
    pub litter_carbon: f64,
    pub nutrient_uptake: f64,
    pub remaining_mineral_nutrient: f64,
    pub carbon_budget_residual: f64,
    pub nutrient_budget_residual: f64,
    pub limitation: ProductivityLimitation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcosystemProductivityModel {
    /// Potential gross carbon assimilation per model-time unit.
    pub potential_gross_primary_productivity: f64,
    /// Gross carbon assimilated per unit mineral nutrient acquired.
    pub carbon_per_nutrient: f64,
    /// Fraction of gross production returned through autotrophic respiration.
    pub autotrophic_respiration_fraction: f64,
    /// Fraction of NPP routed directly to litter rather than retained biomass.
    pub litter_fraction_of_npp: f64,
}

impl EcosystemProductivityModel {
    pub fn try_new(
        potential_gross_primary_productivity: f64,
        carbon_per_nutrient: f64,
        autotrophic_respiration_fraction: f64,
        litter_fraction_of_npp: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            potential_gross_primary_productivity,
            carbon_per_nutrient,
            autotrophic_respiration_fraction,
            litter_fraction_of_npp,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_non_negative(
            "potential_gross_primary_productivity",
            self.potential_gross_primary_productivity,
        )?;
        require_positive("carbon_per_nutrient", self.carbon_per_nutrient)?;
        require_fraction(
            "autotrophic_respiration_fraction",
            self.autotrophic_respiration_fraction,
        )?;
        require_fraction("litter_fraction_of_npp", self.litter_fraction_of_npp)
    }

    pub fn account_interval(
        &self,
        duration: f64,
        environmental_multiplier: f64,
        available_mineral_nutrient: f64,
    ) -> Result<ProductivityLedger, ModelError> {
        self.validate()?;
        require_non_negative("duration", duration)?;
        require_fraction("environmental_multiplier", environmental_multiplier)?;
        require_non_negative("available_mineral_nutrient", available_mineral_nutrient)?;

        let environmental_carbon_ceiling =
            self.potential_gross_primary_productivity * environmental_multiplier * duration;
        let nutrient_carbon_ceiling = available_mineral_nutrient * self.carbon_per_nutrient;
        let scale = environmental_carbon_ceiling
            .abs()
            .max(nutrient_carbon_ceiling.abs())
            .max(1.0);
        let tolerance = 1.0e-12 * scale;
        let limitation =
            if (environmental_carbon_ceiling - nutrient_carbon_ceiling).abs() <= tolerance {
                ProductivityLimitation::CoLimited
            } else if environmental_carbon_ceiling < nutrient_carbon_ceiling {
                ProductivityLimitation::Environmental
            } else {
                ProductivityLimitation::Nutrient
            };
        let gross_primary_production = environmental_carbon_ceiling
            .min(nutrient_carbon_ceiling)
            .max(0.0);
        let nutrient_uptake = gross_primary_production / self.carbon_per_nutrient;
        let remaining_mineral_nutrient = (available_mineral_nutrient - nutrient_uptake).max(0.0);
        let autotrophic_respiration =
            gross_primary_production * self.autotrophic_respiration_fraction;
        let net_primary_production = gross_primary_production - autotrophic_respiration;
        let litter_carbon = net_primary_production * self.litter_fraction_of_npp;
        let retained_biomass_carbon = net_primary_production - litter_carbon;
        let carbon_budget_residual = gross_primary_production
            - autotrophic_respiration
            - retained_biomass_carbon
            - litter_carbon;
        let nutrient_budget_residual =
            available_mineral_nutrient - nutrient_uptake - remaining_mineral_nutrient;

        Ok(ProductivityLedger {
            duration,
            environmental_multiplier,
            environmental_carbon_ceiling,
            nutrient_carbon_ceiling,
            gross_primary_production,
            autotrophic_respiration,
            net_primary_production,
            retained_biomass_carbon,
            litter_carbon,
            nutrient_uptake,
            remaining_mineral_nutrient,
            carbon_budget_residual,
            nutrient_budget_residual,
            limitation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> EcosystemProductivityModel {
        EcosystemProductivityModel::try_new(10.0, 20.0, 0.4, 0.25).unwrap()
    }

    #[test]
    fn environmental_limitation_preserves_both_ledgers() {
        let ledger = model().account_interval(2.0, 0.5, 10.0).unwrap();
        assert_eq!(ledger.limitation, ProductivityLimitation::Environmental);
        assert!((ledger.gross_primary_production - 10.0).abs() < 1.0e-12);
        assert!(ledger.carbon_budget_residual.abs() < 1.0e-12);
        assert!(ledger.nutrient_budget_residual.abs() < 1.0e-12);
        assert!((ledger.remaining_mineral_nutrient - 9.5).abs() < 1.0e-12);
    }

    #[test]
    fn finite_nutrient_stock_caps_assimilation() {
        let ledger = model().account_interval(10.0, 1.0, 1.0).unwrap();
        assert_eq!(ledger.limitation, ProductivityLimitation::Nutrient);
        assert!((ledger.gross_primary_production - 20.0).abs() < 1.0e-12);
        assert!(ledger.remaining_mineral_nutrient.abs() < 1.0e-12);
    }

    #[test]
    fn exact_tie_is_reported_as_colimitation() {
        let ledger = model().account_interval(2.0, 1.0, 1.0).unwrap();
        assert_eq!(ledger.limitation, ProductivityLimitation::CoLimited);
    }

    #[test]
    fn zero_duration_is_a_valid_zero_flux_ledger() {
        let ledger = model().account_interval(0.0, 0.7, 2.0).unwrap();
        assert_eq!(ledger.gross_primary_production, 0.0);
        assert_eq!(ledger.nutrient_uptake, 0.0);
        assert_eq!(ledger.remaining_mineral_nutrient, 2.0);
    }
}
