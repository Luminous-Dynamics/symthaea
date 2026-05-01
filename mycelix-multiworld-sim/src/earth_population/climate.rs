// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Climate-economy feedback: GDP → emissions → warming → damage → GDP.
//!
//! Implements a reduced-form climate model inspired by DICE (Nordhaus 2017):
//!
//! 1. Emissions proportional to GDP × carbon intensity
//! 2. Temperature response: ~1.8°C per 1000 GtCO₂ (IPCC AR6 TCRE)
//! 3. Damage function: D(T) = a₂ × T² (quadratic, Nordhaus DICE-2016)
//! 4. Regional amplification: Arctic 2.5×, tropical 0.8×
//!
//! LIMITATION: This is a reduced-form model, not a GCM. It captures the
//! first-order feedback (GDP→warming→damage) but misses tipping points,
//! ocean circulation changes, and nonlinear ice sheet dynamics.
//!
//! CITATION: Nordhaus, W. (2017) "Revisiting the social cost of carbon",
//! PNAS 114(7). IPCC AR6 WG1 Ch5 (TCRE).

use crate::earth_regions::EarthRegion;
use serde::{Deserialize, Serialize};

/// Transient Climate Response to cumulative Emissions (TCRE).
/// CITATION: IPCC AR6 WG1 Chapter 5, Table 5.7.
/// Central estimate: 1.65°C per 1000 GtCO₂ (range 1.0-2.3).
/// We use 1.8 as a slightly conservative central value.
const TCRE_C_PER_1000GT: f64 = 1.8;

/// DICE damage function coefficient.
/// D(T) = DAMAGE_COEFF × T²
/// CITATION: Nordhaus (2017) DICE-2016: a₂ = 0.00236.
const DAMAGE_COEFF: f64 = 0.00236;

/// Initial global carbon intensity (kgCO₂ per $ GDP).
/// For 1970 start: ~0.55 kgCO₂/$ (higher than today's ~0.26).
/// CITATION: IEA World Energy Outlook historical data, Kaya identity.
const INITIAL_CARBON_INTENSITY: f64 = 0.55;

/// Annual decarbonization rate (fraction reduction per year).
/// CALIBRATED: 1.635%/year (optimized via Nelder-Mead from 1.5% baseline).
/// CITATION: IPCC AR6 WG3: historical ~1.5%/year, calibrated to match 1970-2024 data.
const DECARBONIZATION_RATE: f64 = 0.01635;

/// Pre-industrial CO₂ baseline (GtCO₂ already emitted by simulation start ~1970).
/// CITATION: Global Carbon Project: ~900 GtCO₂ cumulative fossil fuel to 1970.
/// TCRE gives: 900 × 1.8/1000 ≈ 1.62°C, but observed 1970 anomaly was ~0°C
/// because TCRE is nonlinear at low cumulative emissions and ocean heat uptake
/// delays surface warming. Use 200 GtCO₂ as effective baseline for temperature.
/// LIMITATION: This is a reduced-form approximation. Real climate has ocean lag.
const BASELINE_CUMULATIVE_GT: f64 = 200.0;

/// Regional temperature amplification factors.
/// CITATION: IPCC AR6 WG1 Ch4: Arctic amplification 2-3×, tropics ~0.8×.
/// Indexed by region order from build_earth_regions().
const REGIONAL_AMPLIFICATION: [f64; 12] = [
    1.0, // East Asia
    0.9, // South Asia (tropical)
    0.8, // Southeast Asia (tropical)
    1.5, // Central Asia (continental)
    1.0, // Middle East & N Africa
    0.8, // Sub-Saharan Africa (tropical)
    1.0, // Europe
    1.2, // North America (mid-latitude)
    0.8, // Latin America (tropical)
    0.8, // Oceania
    2.5, // Russia/N Asia (Arctic amplification)
    1.0, // Other/Small states
];

/// Global climate state tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateModel {
    /// Cumulative CO₂ emissions (GtCO₂) since pre-industrial.
    pub cumulative_emissions_gt: f64,
    /// Current global mean temperature anomaly (°C above pre-industrial).
    pub global_temp_anomaly: f64,
    /// Current carbon intensity (kgCO₂ per $ GDP), declining with technology.
    pub carbon_intensity: f64,
    /// Years elapsed since simulation start.
    pub years_elapsed: f64,
    /// Per-region annual GDP damage fraction from climate.
    pub regional_damage: Vec<f64>,
    /// Per-region effective temperature anomaly.
    pub regional_temp: Vec<f64>,
    /// Annual emissions this tick (GtCO₂/year).
    pub annual_emissions: f64,
}

impl ClimateModel {
    pub fn new(num_regions: usize) -> Self {
        Self {
            cumulative_emissions_gt: BASELINE_CUMULATIVE_GT,
            global_temp_anomaly: BASELINE_CUMULATIVE_GT * TCRE_C_PER_1000GT / 1000.0, // ~4.5°C from 2500 Gt...
            carbon_intensity: INITIAL_CARBON_INTENSITY,
            years_elapsed: 0.0,
            regional_damage: vec![0.0; num_regions],
            regional_temp: vec![0.0; num_regions],
            annual_emissions: 0.0,
        }
    }

    /// Tick the climate model (monthly resolution).
    ///
    /// 1. Compute global GDP → emissions
    /// 2. Accumulate CO₂
    /// 3. Compute temperature from TCRE
    /// 4. Compute per-region damage
    pub fn tick(&mut self, regions: &[EarthRegion]) {
        self.years_elapsed += 1.0 / 12.0;

        // 1. Carbon intensity declines with technology (decarbonization).
        // Rate accelerates over time as renewables become cheaper:
        // 1970-2000: ~1.5%/year (historical baseline)
        // 2000-2020: ~2.0%/year (early renewables + efficiency)
        // 2020+:     ~2.5%/year (solar/wind cost curves, electrification)
        // CITATION: IEA World Energy Outlook 2023, IRENA Renewable Power Costs 2023.
        // Acceleration starts later (year 40 = 2010, year 50 = 2020)
        let effective_rate = if self.years_elapsed < 40.0 {
            DECARBONIZATION_RATE // 1.5% baseline (1970-2010)
        } else if self.years_elapsed < 50.0 {
            DECARBONIZATION_RATE + 0.003 // 1.8% (2010-2020, early renewables)
        } else {
            DECARBONIZATION_RATE + 0.008 // 2.3% (2020+, Paris Agreement effect)
        };
        self.carbon_intensity =
            INITIAL_CARBON_INTENSITY * (1.0 - effective_rate).powf(self.years_elapsed);

        // 2. Global GDP (sum of region GDP × population in millions × 1M)
        let global_gdp: f64 = regions
            .iter()
            .map(|r| r.gdp_per_capita * r.population * 1_000_000.0)
            .sum();

        // 3. Annual emissions (GtCO₂/year) = GDP($) × carbon_intensity(kg/$) / 1e12(kg→Gt)
        self.annual_emissions = global_gdp * self.carbon_intensity / 1e12;

        // 4. Monthly accumulation
        self.cumulative_emissions_gt += self.annual_emissions / 12.0;

        // 5. Temperature from TCRE with ocean heat uptake lag.
        // The transient response is ~60% of equilibrium due to ocean thermal inertia.
        // CITATION: IPCC AR6 WG1 Ch7: TCR/ECS ratio ≈ 0.56 (range 0.4-0.7).
        // We use 0.55 as the realized warming fraction.
        let equilibrium_temp = self.cumulative_emissions_gt * TCRE_C_PER_1000GT / 1000.0;
        let realized_fraction = 0.55;
        // Smooth transition: global_temp tracks realized warming with ~20yr e-folding time
        let target_temp = equilibrium_temp * realized_fraction;
        // CALIBRATED: 5.12-year e-folding (optimized from 20yr via Nelder-Mead)
        let lag_rate = 1.0 / (5.12 * 12.0);
        self.global_temp_anomaly += (target_temp - self.global_temp_anomaly) * lag_rate;

        // 6. Per-region damage (DICE quadratic)
        for (i, region) in regions.iter().enumerate() {
            if i >= self.regional_damage.len() {
                break;
            }

            let amp = if i < REGIONAL_AMPLIFICATION.len() {
                REGIONAL_AMPLIFICATION[i]
            } else {
                1.0
            };

            let regional_temp = self.global_temp_anomaly * amp;
            self.regional_temp[i] = regional_temp;

            // DICE damage: fraction of GDP lost annually
            // D(T) = 0.00236 × T² (Nordhaus 2017)
            self.regional_damage[i] = DAMAGE_COEFF * regional_temp * regional_temp;

            // Climate vulnerability amplifies damage
            let vulnerability_mult = 1.0 + region.climate_vulnerability;
            self.regional_damage[i] *= vulnerability_mult;
        }
    }

    /// Get GDP damage fraction for a region.
    pub fn damage_fraction(&self, region_idx: usize) -> f64 {
        self.regional_damage.get(region_idx).copied().unwrap_or(0.0)
    }

    /// Get temperature anomaly for a region.
    pub fn temp_anomaly(&self, region_idx: usize) -> f64 {
        self.regional_temp
            .get(region_idx)
            .copied()
            .unwrap_or(self.global_temp_anomaly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth_regions::build_earth_regions;

    #[test]
    fn test_temperature_increases_with_emissions() {
        let regions = build_earth_regions();
        let mut climate = ClimateModel::new(regions.len());
        let initial_temp = climate.global_temp_anomaly;

        // Tick for 10 years (120 months)
        for _ in 0..120 {
            climate.tick(&regions);
        }

        assert!(
            climate.global_temp_anomaly > initial_temp,
            "Temperature should increase: {} vs {}",
            climate.global_temp_anomaly,
            initial_temp
        );
    }

    #[test]
    fn test_damage_increases_with_temperature() {
        let regions = build_earth_regions();
        let mut climate = ClimateModel::new(regions.len());

        // Tick to accumulate warming
        for _ in 0..240 {
            // 20 years
            climate.tick(&regions);
        }

        // All regions should have positive damage
        for (i, &damage) in climate.regional_damage.iter().enumerate() {
            assert!(
                damage > 0.0,
                "Region {} should have positive damage: {}",
                i,
                damage
            );
        }
    }

    #[test]
    fn test_arctic_amplification() {
        let regions = build_earth_regions();
        let mut climate = ClimateModel::new(regions.len());

        for _ in 0..120 {
            climate.tick(&regions);
        }

        // Russia/N Asia (idx 10) should have higher temp than tropical regions
        let arctic_temp = climate.regional_temp[10]; // Russia, amp 2.5
        let tropical_temp = climate.regional_temp[2]; // SE Asia, amp 0.8

        assert!(
            arctic_temp > tropical_temp,
            "Arctic should warm more: {} vs {}",
            arctic_temp,
            tropical_temp
        );
        assert!(
            arctic_temp > tropical_temp * 2.0,
            "Arctic amplification should be >2x: {} vs {}",
            arctic_temp,
            tropical_temp
        );
    }

    #[test]
    fn test_carbon_intensity_decreases() {
        let regions = build_earth_regions();
        let mut climate = ClimateModel::new(regions.len());
        let initial_ci = climate.carbon_intensity;

        for _ in 0..120 {
            // 10 years
            climate.tick(&regions);
        }

        assert!(
            climate.carbon_intensity < initial_ci,
            "Carbon intensity should decrease: {} vs {}",
            climate.carbon_intensity,
            initial_ci
        );
        // After 10 years at 1.5%/year: 0.55 × 0.985^10 ≈ 0.473
        assert!(
            climate.carbon_intensity > 0.45 && climate.carbon_intensity < 0.50,
            "Expected ~0.47, got {}",
            climate.carbon_intensity
        );
    }

    #[test]
    fn test_emissions_positive() {
        let regions = build_earth_regions();
        let mut climate = ClimateModel::new(regions.len());
        climate.tick(&regions);

        assert!(
            climate.annual_emissions > 0.0,
            "Annual emissions should be positive: {}",
            climate.annual_emissions
        );
        // Global GDP ~$100T × 0.26 kg/$ / 1e12 ≈ ~26 GtCO₂/year (roughly matches reality ~37 Gt)
        assert!(
            climate.annual_emissions > 10.0 && climate.annual_emissions < 100.0,
            "Emissions should be in plausible range: {}",
            climate.annual_emissions
        );
    }
}
