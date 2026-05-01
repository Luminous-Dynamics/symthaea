// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-resolution Earth population model for 8 billion people.
//!
//! # Architecture
//!
//! Three-layer population pyramid:
//!
//! | Layer | What | Count |
//! |-------|------|-------|
//! | 3 | Notable Agents | ~50-200 individually tracked |
//! | 2 | Regional Cohorts | ~2,400 (12 regions × ~200 occupied) |
//! | 1 | EarthRegion | 12 aggregate regions (compatibility layer) |
//!
//! The existing `EarthRegion` struct (earth_regions.rs) is preserved as a
//! compatibility layer via `sync_region_summaries()`.
//!
//! # Key Types
//!
//! - [`RegionCohort`] — statistical aggregate of a demographic slice
//! - [`RegionDemographics`] — cohort container for one Earth region
//! - [`DemographicTransition`] — 5-stage fertility/mortality model (Notestein 1945)
//! - [`MigrationEngine`] — push-pull gravity model for inter-region flows
//! - [`EarthPopulationModel`] — top-level orchestrator
//!
//! # Performance
//!
//! ~2,400 cohort updates at ~1µs each = **2.4ms per tick**.
//! Target: 1000-year run (12,000 ticks) in <10 minutes.

pub mod climate;
pub mod cohort;
pub mod demographic_transition;
pub mod migration;

use crate::earth_regions::EarthRegion;
use crate::stochastic::StochasticEngine;
use crate::viability::ScalingFactors;
use climate::ClimateModel;
use cohort::RegionDemographics;
use demographic_transition::DemographicTransition;
use migration::MigrationEngine;

use serde::{Deserialize, Serialize};

/// Top-level Earth population model.
///
/// Manages the three-layer hierarchy: regions → cohorts → notable agents.
/// Replaces the flat `Vec<EarthRegion>` with a richer demographic model
/// while preserving backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthPopulationModel {
    /// Per-region demographic cohort decomposition.
    pub demographics: Vec<RegionDemographics>,
    /// Demographic transition engine (shared parameters).
    pub transition: DemographicTransition,
    /// Inter-region migration engine.
    pub migration: MigrationEngine,
    /// Climate-economy feedback model (GDP → warming → damage).
    pub climate: ClimateModel,
    /// Total Earth population (cached, recomputed each tick).
    pub total_population: f64,
    /// Whether the model has been initialized from region data.
    pub initialized: bool,
}

impl EarthPopulationModel {
    /// Create from existing `EarthRegion` data.
    ///
    /// Decomposes each region's aggregate population into cohorts using
    /// standard demographic pyramids calibrated to UN WPP 2024 data.
    pub fn from_regions(regions: &[EarthRegion]) -> Self {
        let demographics: Vec<RegionDemographics> = regions
            .iter()
            .enumerate()
            .map(|(i, region)| RegionDemographics::from_region(i as u8, region))
            .collect();

        let total: f64 = demographics.iter().map(|d| d.total_population).sum();

        Self {
            demographics,
            transition: DemographicTransition::default(),
            migration: MigrationEngine::new(regions.len()),
            climate: ClimateModel::new(regions.len()),
            total_population: total,
            initialized: true,
        }
    }

    /// Tick all Earth demographics: aging, births, deaths, transition, migration.
    pub fn tick(
        &mut self,
        regions: &[EarthRegion],
        _scaling: &[ScalingFactors],
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) {
        if !self.initialized || self.demographics.is_empty() {
            return;
        }

        // 1. Demographic tick per region: aging, births, deaths
        for demo in &mut self.demographics {
            demo.tick_demographics(current_tick);
        }

        // 2. Update demographic transition stage per region
        // Urbanization directly reduces fertility (urban women have fewer children)
        for (i, demo) in self.demographics.iter_mut().enumerate() {
            if i < regions.len() {
                let dev_index = self.transition.development_index(&regions[i]);
                let urban = regions[i].urbanization;
                demo.tick_transition_with_urbanization(&self.transition, dev_index, urban);
            }
        }

        // 3. Inter-region migration
        self.migration.tick(&regions, &mut self.demographics, rng);

        // 4. Climate-economy feedback: GDP → emissions → warming → damage
        self.climate.tick(regions);

        // 5. Recompute total
        self.total_population = self.demographics.iter().map(|d| d.total_population).sum();
    }

    /// Sync cohort aggregates back to EarthRegion structs for compatibility.
    /// Applies GDP growth, climate damage, and education improvement.
    pub fn sync_to_regions(&self, regions: &mut [EarthRegion]) {
        for (i, demo) in self.demographics.iter().enumerate() {
            if i < regions.len() {
                regions[i].population = demo.total_population;
                regions[i].education_index = demo.mean_education();
                regions[i].phi_distribution_mean = demo.mean_phi();

                // Regional GDP growth rates from historical data.
                // CITATION: World Bank WDI (2024), Maddison Project Database (2020).
                let year = 1970.0 + self.climate.years_elapsed;
                let annual_gdp_growth = regional_gdp_growth(&regions[i].name, year);
                let monthly_growth = (1.0_f64 + annual_gdp_growth).powf(1.0 / 12.0) - 1.0;
                regions[i].gdp_per_capita *= 1.0 + monthly_growth;

                // Climate-economy feedback: reduce GDP by damage fraction
                let damage = self.climate.damage_fraction(i) / 12.0;
                regions[i].gdp_per_capita *= 1.0 - damage;

                // Education improvement: slow, ~0.003/year
                regions[i].education_index = (regions[i].education_index + 0.00025).min(0.98);

                // Urbanization growth: ~0.4%/year globally (UN DESA).
                // Saturates at 0.95 (some rural population always remains).
                regions[i].urbanization = (regions[i].urbanization + 0.0003).min(0.95);

                // Climate vulnerability slowly increases with warming
                let temp = self.climate.temp_anomaly(i);
                if temp > 2.0 {
                    regions[i].climate_vulnerability =
                        (regions[i].climate_vulnerability + 0.0001 * (temp - 2.0)).min(1.0);
                }
            }
        }
    }
}

/// Historical GDP growth rate by region and year.
/// CITATION: World Bank WDI (2024), Maddison Project Database (2020).
/// Returns annual growth rate (fraction, e.g., 0.08 = 8%).
fn regional_gdp_growth(region_name: &str, year: f64) -> f64 {
    match region_name {
        "East Asia" => {
            if year < 1990.0 {
                0.08
            }
            // Japan/Korea miracle
            else if year < 2010.0 {
                0.09
            }
            // China boom
            else {
                0.055
            } // Maturing
        }
        "South Asia" => {
            if year < 2000.0 {
                0.04
            }
            // Pre-liberalization
            else if year < 2010.0 {
                0.07
            }
            // India IT boom
            else {
                0.06
            } // Continued growth
        }
        "Southeast Asia" => {
            if year < 1997.0 {
                0.07
            }
            // Tiger economies
            else if year < 2000.0 {
                0.02
            }
            // Asian financial crisis
            else {
                0.05
            } // Recovery
        }
        "Central Asia" => 0.04,
        "Middle East & N Africa" => {
            if year < 2010.0 {
                0.04
            } else {
                0.025
            } // Arab Spring + oil price decline
        }
        "Sub-Saharan Africa" => {
            if year < 2000.0 {
                0.02
            }
            // Lost decades
            else {
                0.04
            } // Commodity boom
        }
        "Europe" => {
            if year < 2008.0 {
                0.025
            }
            // Stable growth
            else if year < 2012.0 {
                0.005
            }
            // Great Recession + Euro crisis
            else {
                0.018
            } // Slow recovery
        }
        "North America" => {
            if year < 2008.0 {
                0.03
            } else if year < 2010.0 {
                -0.01
            }
            // Great Recession
            else {
                0.022
            }
        }
        "Latin America" => {
            if year < 1982.0 {
                0.05
            }
            // Pre-debt crisis
            else if year < 1990.0 {
                0.02
            }
            // Lost decade
            else if year < 2014.0 {
                0.04
            }
            // Commodity boom
            else {
                0.015
            } // Stagnation
        }
        "Oceania" => 0.03,
        "Russia/N Asia" => {
            if year < 1991.0 {
                0.03
            }
            // Soviet era
            else if year < 1998.0 {
                -0.05
            }
            // Collapse
            else if year < 2014.0 {
                0.06
            }
            // Oil-driven recovery
            else {
                0.015
            } // Sanctions + stagnation
        }
        _ => 0.03, // default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth_regions::build_earth_regions;

    #[test]
    fn test_model_creation_from_regions() {
        let regions = build_earth_regions();
        let model = EarthPopulationModel::from_regions(&regions);

        assert!(model.initialized);
        assert_eq!(model.demographics.len(), 12);
        // Total should be close to 7.8 billion
        assert!(
            model.total_population > 7000.0,
            "Expected >7000M, got {}",
            model.total_population
        );
        assert!(
            model.total_population < 9000.0,
            "Expected <9000M, got {}",
            model.total_population
        );
    }

    #[test]
    fn test_cohort_decomposition_preserves_population() {
        let regions = build_earth_regions();
        let model = EarthPopulationModel::from_regions(&regions);

        // Sum of all cohort counts per region should match region population
        for (i, demo) in model.demographics.iter().enumerate() {
            let cohort_sum: f64 = demo.cohorts.values().map(|c| c.count).sum();
            let region_pop = regions[i].population;
            assert!(
                (cohort_sum - region_pop).abs() < region_pop * 0.01,
                "Region {} cohort sum {} vs region pop {}",
                i,
                cohort_sum,
                region_pop
            );
        }
    }
}
