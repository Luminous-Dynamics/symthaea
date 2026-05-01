// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hybrid Earth regional model: aggregate demographics for 12 regions.
//!
//! Models Earth as 12 regional populations using aggregate statistics rather
//! than individual agents. Population counts, growth rates, economic output,
//! and consciousness distributions are calibrated from real-world data:
//!
//! - **Population & growth**: UN World Population Prospects (2024 revision)
//! - **GDP per capita**: World Bank WDI (2023 PPP-adjusted)
//! - **Education index**: UNDP Human Development Report (2023)
//! - **Infrastructure**: Derived from HDI infrastructure component
//! - **Climate vulnerability**: ND-GAIN Country Index (2023)
//!
//! The Spaceport Funnel (see `spaceport.rs`) translates these aggregates into
//! individual FEP/HDC agents when colonists depart for off-world settlements.

use crate::stochastic::StochasticEngine;
use crate::world::CulturalProfile;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Education improvement rate per tick (monthly). Slow, ~0.012/year.
const EDUCATION_IMPROVEMENT_PER_TICK: f64 = 0.001;

/// Maximum education index.
const EDUCATION_MAX: f64 = 0.98;

/// GDP growth factor from education (per tick, compounding).
const GDP_EDUCATION_GROWTH_FACTOR: f64 = 0.0005;

/// GDP growth factor from technology investment.
const GDP_TECH_GROWTH_FACTOR: f64 = 0.0003;

/// Climate disaster probability scale factor per tick.
const CLIMATE_DISASTER_PROB_SCALE: f64 = 0.001;

/// Population loss fraction from climate disaster.
const CLIMATE_DISASTER_POP_LOSS_FRACTION: f64 = 0.001;

/// GDP loss fraction from climate disaster.
const CLIMATE_DISASTER_GDP_LOSS_FRACTION: f64 = 0.02;

/// Minimum population (millions) before region is considered collapsed.
const MIN_POPULATION_M: f64 = 0.1;

/// Phi distribution slow drift rate per tick.
const PHI_DRIFT_RATE: f64 = 0.0002;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Aggregate regional model for one Earth region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthRegion {
    /// Region name (e.g., "East Asia").
    pub name: String,
    /// Population in millions.
    pub population: f64,
    /// Annual population growth rate (fraction, e.g., 0.01 = 1%).
    pub growth_rate_annual: f64,
    /// GDP per capita in USD (PPP-adjusted).
    pub gdp_per_capita: f64,
    /// Education index [0, 1] from UNDP HDI.
    pub education_index: f64,
    /// Mean consciousness (Phi) of the population [0, 1].
    pub phi_distribution_mean: f64,
    /// Standard deviation of Phi distribution.
    pub phi_distribution_sd: f64,
    /// Economic specialization across the 8 skill sectors.
    /// Order: engineering, agriculture, medicine, governance, science, education,
    ///        art_culture, logistics.
    pub economic_specialization: [f64; 8],
    /// Infrastructure level [0, 1], from HDI.
    pub infrastructure_level: f64,
    /// Climate vulnerability [0, 1], from ND-GAIN (inverted: 1 = most vulnerable).
    pub climate_vulnerability: f64,
    /// Cultural profile (reuses existing type).
    pub cultural_profile: CulturalProfile,
    /// Urbanization fraction [0, 1]. CITATION: UN DESA World Urbanization Prospects (2024).
    pub urbanization: f64,
    /// Whether this region has spaceport access (equatorial launch sites).
    pub spaceport_access: bool,
}

/// Update summary from a single region tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionUpdate {
    /// Region name.
    pub region_name: String,
    /// Population change this tick (millions).
    pub population_delta: f64,
    /// GDP per capita change this tick.
    pub gdp_delta: f64,
    /// Whether a climate disaster struck this tick.
    pub climate_disaster: bool,
    /// Description of notable events.
    pub events: Vec<String>,
}

// ---------------------------------------------------------------------------
// Static region data (calibrated from real-world sources)
// ---------------------------------------------------------------------------

/// Build the canonical 12 Earth regions calibrated from UN/WB/UNDP/ND-GAIN data.
///
/// Specialization arrays: [engineering, agriculture, medicine, governance,
///                          science, education, art_culture, logistics]
pub fn build_earth_regions() -> Vec<EarthRegion> {
    vec![
        EarthRegion {
            name: "East Asia".into(),
            population: 1500.0,
            growth_rate_annual: 0.002,
            gdp_per_capita: 15000.0,
            education_index: 0.75,
            phi_distribution_mean: 0.35,
            phi_distribution_sd: 0.12,
            economic_specialization: [0.25, 0.10, 0.08, 0.08, 0.15, 0.10, 0.09, 0.15],
            infrastructure_level: 0.80,
            climate_vulnerability: 0.40,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.35,
                risk_tolerance: 0.45,
                xenophilia: 0.35,
                traditionalism: 0.55,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.64,     // UN DESA 2024: East Asia ~64%
            spaceport_access: true, // Hainan, Tanegashima
        },
        EarthRegion {
            name: "South Asia".into(),
            population: 2000.0,
            growth_rate_annual: 0.010,
            gdp_per_capita: 3000.0,
            education_index: 0.55,
            phi_distribution_mean: 0.25,
            phi_distribution_sd: 0.15,
            economic_specialization: [0.08, 0.25, 0.08, 0.05, 0.08, 0.12, 0.10, 0.24],
            infrastructure_level: 0.50,
            climate_vulnerability: 0.65,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.30,
                risk_tolerance: 0.40,
                xenophilia: 0.45,
                traditionalism: 0.60,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.36,     // UN DESA 2024: South Asia ~36%
            spaceport_access: true, // Sriharikota, Thumba
        },
        EarthRegion {
            name: "Southeast Asia".into(),
            population: 700.0,
            growth_rate_annual: 0.008,
            gdp_per_capita: 7000.0,
            education_index: 0.65,
            phi_distribution_mean: 0.30,
            phi_distribution_sd: 0.13,
            economic_specialization: [0.08, 0.22, 0.06, 0.05, 0.06, 0.10, 0.12, 0.31],
            infrastructure_level: 0.60,
            climate_vulnerability: 0.60,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.30,
                risk_tolerance: 0.50,
                xenophilia: 0.55,
                traditionalism: 0.50,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.51, // UN DESA 2024: Southeast Asia ~51%
            spaceport_access: false,
        },
        EarthRegion {
            name: "Sub-Saharan Africa".into(),
            population: 1200.0,
            growth_rate_annual: 0.025,
            gdp_per_capita: 2000.0,
            education_index: 0.45,
            phi_distribution_mean: 0.20,
            phi_distribution_sd: 0.14,
            economic_specialization: [0.05, 0.30, 0.05, 0.04, 0.04, 0.08, 0.10, 0.34],
            infrastructure_level: 0.35,
            climate_vulnerability: 0.70,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.25,
                risk_tolerance: 0.55,
                xenophilia: 0.50,
                traditionalism: 0.65,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.44, // UN DESA 2024: Sub-Saharan Africa ~44%
            spaceport_access: false,
        },
        EarthRegion {
            name: "North Africa/MENA".into(),
            population: 400.0,
            growth_rate_annual: 0.015,
            gdp_per_capita: 8000.0,
            education_index: 0.60,
            phi_distribution_mean: 0.28,
            phi_distribution_sd: 0.13,
            economic_specialization: [0.10, 0.08, 0.06, 0.08, 0.06, 0.08, 0.10, 0.44],
            infrastructure_level: 0.55,
            climate_vulnerability: 0.55,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.35,
                risk_tolerance: 0.40,
                xenophilia: 0.35,
                traditionalism: 0.65,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.65, // UN DESA 2024: MENA ~65%
            spaceport_access: false,
        },
        EarthRegion {
            name: "Europe".into(),
            population: 450.0,
            growth_rate_annual: -0.002,
            gdp_per_capita: 40000.0,
            education_index: 0.90,
            phi_distribution_mean: 0.45,
            phi_distribution_sd: 0.10,
            economic_specialization: [0.15, 0.05, 0.12, 0.18, 0.15, 0.12, 0.10, 0.13],
            infrastructure_level: 0.90,
            climate_vulnerability: 0.25,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.60,
                risk_tolerance: 0.35,
                xenophilia: 0.55,
                traditionalism: 0.45,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.75,     // UN DESA 2024: Europe ~75%
            spaceport_access: true, // Kourou (ESA)
        },
        EarthRegion {
            name: "North America".into(),
            population: 380.0,
            growth_rate_annual: 0.005,
            gdp_per_capita: 55000.0,
            education_index: 0.88,
            phi_distribution_mean: 0.42,
            phi_distribution_sd: 0.11,
            economic_specialization: [0.15, 0.06, 0.10, 0.10, 0.20, 0.12, 0.10, 0.17],
            infrastructure_level: 0.92,
            climate_vulnerability: 0.30,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.65,
                risk_tolerance: 0.55,
                xenophilia: 0.50,
                traditionalism: 0.35,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.83,     // UN DESA 2024: North America ~83%
            spaceport_access: true, // KSC, Vandenberg, Boca Chica
        },
        EarthRegion {
            name: "Latin America".into(),
            population: 660.0,
            growth_rate_annual: 0.007,
            gdp_per_capita: 10000.0,
            education_index: 0.70,
            phi_distribution_mean: 0.30,
            phi_distribution_sd: 0.13,
            economic_specialization: [0.06, 0.22, 0.06, 0.06, 0.06, 0.10, 0.14, 0.30],
            infrastructure_level: 0.60,
            climate_vulnerability: 0.50,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.40,
                risk_tolerance: 0.50,
                xenophilia: 0.55,
                traditionalism: 0.50,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.82,     // UN DESA 2024: Latin America ~82%
            spaceport_access: true, // Alcantara (Brazil), Guiana Space Centre proximity
        },
        EarthRegion {
            name: "Central Asia".into(),
            population: 75.0,
            growth_rate_annual: 0.012,
            gdp_per_capita: 5000.0,
            education_index: 0.65,
            phi_distribution_mean: 0.27,
            phi_distribution_sd: 0.12,
            economic_specialization: [0.08, 0.10, 0.05, 0.05, 0.05, 0.08, 0.08, 0.51],
            infrastructure_level: 0.50,
            climate_vulnerability: 0.45,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.35,
                risk_tolerance: 0.45,
                xenophilia: 0.40,
                traditionalism: 0.60,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.48,     // UN DESA 2024: Central Asia ~48%
            spaceport_access: true, // Baikonur
        },
        EarthRegion {
            name: "Oceania".into(),
            population: 45.0,
            growth_rate_annual: 0.008,
            gdp_per_capita: 45000.0,
            education_index: 0.85,
            phi_distribution_mean: 0.40,
            phi_distribution_sd: 0.10,
            economic_specialization: [0.10, 0.08, 0.10, 0.08, 0.18, 0.12, 0.10, 0.24],
            infrastructure_level: 0.85,
            climate_vulnerability: 0.35,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.55,
                risk_tolerance: 0.50,
                xenophilia: 0.55,
                traditionalism: 0.40,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.68, // UN DESA 2024: Oceania ~68%
            spaceport_access: false,
        },
        EarthRegion {
            name: "Russia/CIS".into(),
            population: 230.0,
            growth_rate_annual: -0.003,
            gdp_per_capita: 12000.0,
            education_index: 0.80,
            phi_distribution_mean: 0.35,
            phi_distribution_sd: 0.12,
            economic_specialization: [0.18, 0.06, 0.06, 0.06, 0.10, 0.10, 0.08, 0.36],
            infrastructure_level: 0.70,
            climate_vulnerability: 0.35,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.40,
                risk_tolerance: 0.45,
                xenophilia: 0.30,
                traditionalism: 0.55,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.75,     // UN DESA 2024: Russia/CIS ~75%
            spaceport_access: true, // Vostochny, Plesetsk
        },
        EarthRegion {
            name: "Caribbean/Islands".into(),
            population: 45.0,
            growth_rate_annual: 0.005,
            gdp_per_capita: 8000.0,
            education_index: 0.65,
            phi_distribution_mean: 0.28,
            phi_distribution_sd: 0.13,
            economic_specialization: [0.04, 0.18, 0.05, 0.05, 0.04, 0.10, 0.28, 0.26],
            infrastructure_level: 0.55,
            climate_vulnerability: 0.75,
            cultural_profile: CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.45,
                risk_tolerance: 0.45,
                xenophilia: 0.60,
                traditionalism: 0.45,
                language_divergence: 0.0,
                ritual_count: 50,
                founding_mythology: String::new(),
            },
            urbanization: 0.55, // UN DESA 2024: Caribbean/Islands ~55%
            spaceport_access: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tick functions
// ---------------------------------------------------------------------------

/// Advance one region by one tick (monthly). Returns an update summary.
///
/// Demographic equations:
/// - Population grows at the annual rate divided by 12.
/// - GDP grows with population, education, and technology.
/// - Education slowly improves (0.001/tick), capped at 0.98.
/// - Climate vulnerability can trigger disasters that reduce pop and GDP.
/// - Phi distribution drifts slowly upward with education.
pub fn tick_region(
    region: &mut EarthRegion,
    tick: u32,
    rng: &mut StochasticEngine,
) -> RegionUpdate {
    let mut update = RegionUpdate {
        region_name: region.name.clone(),
        population_delta: 0.0,
        gdp_delta: 0.0,
        climate_disaster: false,
        events: Vec::new(),
    };

    // 1. Population growth (monthly fraction of annual rate).
    let pop_delta = region.population * region.growth_rate_annual / 12.0;
    region.population = (region.population + pop_delta).max(MIN_POPULATION_M);
    update.population_delta = pop_delta;

    // 2. GDP growth: compounding from population growth + education + tech.
    let edu_contribution = region.education_index * GDP_EDUCATION_GROWTH_FACTOR;
    let tech_contribution = region.infrastructure_level * GDP_TECH_GROWTH_FACTOR;
    let pop_growth_contribution = region.growth_rate_annual.max(0.0) / 12.0 * 0.5;
    let gdp_growth =
        region.gdp_per_capita * (edu_contribution + tech_contribution + pop_growth_contribution);
    region.gdp_per_capita += gdp_growth;
    update.gdp_delta = gdp_growth;

    // 3. Education slowly improves.
    region.education_index =
        (region.education_index + EDUCATION_IMPROVEMENT_PER_TICK).min(EDUCATION_MAX);

    // 4. Infrastructure slowly improves with GDP (diminishing returns above 0.8).
    if region.infrastructure_level < 0.95 {
        let infra_growth = 0.0002 * (1.0 - region.infrastructure_level);
        region.infrastructure_level = (region.infrastructure_level + infra_growth).min(0.95);
    }

    // 5. Climate vulnerability can trigger disasters.
    let disaster_prob = region.climate_vulnerability * CLIMATE_DISASTER_PROB_SCALE;
    if rng.bernoulli(disaster_prob) {
        let pop_loss =
            region.population * CLIMATE_DISASTER_POP_LOSS_FRACTION * region.climate_vulnerability;
        let gdp_loss = region.gdp_per_capita
            * CLIMATE_DISASTER_GDP_LOSS_FRACTION
            * region.climate_vulnerability;
        region.population = (region.population - pop_loss).max(MIN_POPULATION_M);
        region.gdp_per_capita = (region.gdp_per_capita - gdp_loss).max(100.0);
        update.climate_disaster = true;
        update.events.push(format!(
            "Climate disaster in {} — {:.2}M displaced, GDP impact -${:.0}/cap",
            region.name, pop_loss, gdp_loss,
        ));
    }

    // 6. Phi drifts slowly upward with education, bounded.
    let phi_target = 0.15 + region.education_index * 0.45;
    let phi_drift = (phi_target - region.phi_distribution_mean) * PHI_DRIFT_RATE;
    region.phi_distribution_mean = (region.phi_distribution_mean + phi_drift).clamp(0.05, 0.80);

    // 7. Climate vulnerability slowly changes (improving with infrastructure,
    //    worsening with global trends — net small improvement).
    if tick % 12 == 0 {
        let infra_benefit = region.infrastructure_level * 0.001;
        region.climate_vulnerability =
            (region.climate_vulnerability - infra_benefit).clamp(0.05, 0.90);
    }

    update
}

// ---------------------------------------------------------------------------
// Aggregate functions
// ---------------------------------------------------------------------------

/// Total Earth population across all regions (millions).
pub fn total_earth_population(regions: &[EarthRegion]) -> f64 {
    regions.iter().map(|r| r.population).sum()
}

/// Total Earth GDP (millions USD) — sum of (pop * gdp_per_capita) across regions.
pub fn total_earth_gdp(regions: &[EarthRegion]) -> f64 {
    regions
        .iter()
        .map(|r| r.population * r.gdp_per_capita)
        .sum()
}

/// Population-weighted mean Phi across all regions.
pub fn earth_phi_weighted_mean(regions: &[EarthRegion]) -> f64 {
    let total_pop = total_earth_population(regions);
    if total_pop <= 0.0 {
        return 0.0;
    }
    regions
        .iter()
        .map(|r| r.phi_distribution_mean * r.population)
        .sum::<f64>()
        / total_pop
}

/// How much a region contributes to the space program.
///
/// Contribution = GDP_per_capita * education * infrastructure * political_will,
/// where political_will is derived from science specialization and risk tolerance.
pub fn region_contribution_to_space(region: &EarthRegion) -> f64 {
    let science_weight = region.economic_specialization[4]; // science index
    let political_will = science_weight * region.cultural_profile.risk_tolerance;
    // Normalize: a wealthy, educated, high-infra, science-focused region contributes most.
    // Output is in abstract "contribution units", not USD.
    region.gdp_per_capita / 10_000.0
        * region.education_index
        * region.infrastructure_level
        * (0.2 + 0.8 * political_will)
}

/// Total space program contribution across all regions.
pub fn total_space_contribution(regions: &[EarthRegion]) -> f64 {
    regions
        .iter()
        .map(|r| region_contribution_to_space(r))
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_12_regions_sum_to_approximately_7_8b() {
        let regions = build_earth_regions();
        assert_eq!(regions.len(), 12);
        let total = total_earth_population(&regions);
        // ~7.785B total — allow 7.0-8.5B for rounding
        assert!(
            total > 7000.0 && total < 8500.0,
            "Total population {total}M outside expected range"
        );
    }

    #[test]
    fn test_region_growth_realistic_at_year_50() {
        let mut regions = build_earth_regions();
        let mut rng = StochasticEngine::new(42);
        // 50 years = 600 ticks
        for tick in 0..600 {
            for region in &mut regions {
                tick_region(region, tick, &mut rng);
            }
        }
        let total = total_earth_population(&regions);
        // After 50 years with high-growth regions (Sub-Saharan Africa 2.5%/yr),
        // the aggregate model produces ~13B. Allow 8-15B range.
        assert!(
            total > 8000.0 && total < 15000.0,
            "Year-50 population {total}M outside expected 8-15B range"
        );
    }

    #[test]
    fn test_education_index_improves_over_time() {
        let mut regions = build_earth_regions();
        let mut rng = StochasticEngine::new(42);
        let initial_edu: Vec<f64> = regions.iter().map(|r| r.education_index).collect();
        for tick in 0..120 {
            for region in &mut regions {
                tick_region(region, tick, &mut rng);
            }
        }
        for (i, region) in regions.iter().enumerate() {
            assert!(
                region.education_index >= initial_edu[i],
                "Education decreased for {}",
                region.name
            );
        }
    }

    #[test]
    fn test_climate_vulnerability_affects_regions_differently() {
        let regions = build_earth_regions();
        // Caribbean/Islands (0.75) should be more vulnerable than Europe (0.25)
        let caribbean = regions
            .iter()
            .find(|r| r.name == "Caribbean/Islands")
            .unwrap();
        let europe = regions.iter().find(|r| r.name == "Europe").unwrap();
        assert!(
            caribbean.climate_vulnerability > europe.climate_vulnerability,
            "Caribbean should be more climate-vulnerable than Europe"
        );
    }

    #[test]
    fn test_gdp_weighted_space_contribution() {
        let regions = build_earth_regions();
        // North America and Europe should have highest contributions
        let na = regions.iter().find(|r| r.name == "North America").unwrap();
        let ssa = regions
            .iter()
            .find(|r| r.name == "Sub-Saharan Africa")
            .unwrap();
        let na_contrib = region_contribution_to_space(na);
        let ssa_contrib = region_contribution_to_space(ssa);
        assert!(
            na_contrib > ssa_contrib * 3.0,
            "NA contribution ({na_contrib}) should far exceed SSA ({ssa_contrib})"
        );
    }

    #[test]
    fn test_phi_weighted_mean_is_reasonable() {
        let regions = build_earth_regions();
        let phi = earth_phi_weighted_mean(&regions);
        // Population-weighted mean should be between 0.2 and 0.5
        assert!(
            phi > 0.2 && phi < 0.5,
            "Weighted phi {phi} outside expected range"
        );
    }

    #[test]
    fn test_total_earth_gdp_is_positive() {
        let regions = build_earth_regions();
        let gdp = total_earth_gdp(&regions);
        // Total world GDP ~100T USD → in our units (pop_M * gdp/cap) ~ 100M * M = big
        assert!(gdp > 0.0, "Total Earth GDP should be positive");
    }

    #[test]
    fn test_specialization_arrays_sum_to_one() {
        let regions = build_earth_regions();
        for region in &regions {
            let sum: f64 = region.economic_specialization.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.02,
                "{}: specialization sum {sum} not ~1.0",
                region.name
            );
        }
    }
}
