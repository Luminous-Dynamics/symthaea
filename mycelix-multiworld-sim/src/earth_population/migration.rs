// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inter-region migration using a push-pull gravity model.
//!
//! CITATION: Ravenstein, E. (1885) "The Laws of Migration".
//! Updated: Zipf (1946), Todaro (1969), Lee (1966).
//!
//! Push factors: poverty, climate vulnerability, conflict (from cliodynamics).
//! Pull factors: wages, safety, education opportunity, labor demand.
//! Flow: proportional to push × pull × geometric_mean(populations).
//!
//! LIMITATION: No diaspora networks, no policy barriers (visa systems),
//! no cultural affinity modeling. All regions are equally accessible.

use super::cohort::{CohortSex, EducationBand, FineAgeBand, RegionCohortKey, RegionDemographics};
use crate::earth_regions::EarthRegion;
use crate::stochastic::StochasticEngine;

use serde::{Deserialize, Serialize};

/// Annual global migration as fraction of world population.
/// CITATION: UN DESA (2020): ~281 million international migrants = ~3.6% of 7.8B.
/// We model inter-region migration at ~0.5% annual (subset of total, major flows only).
const _BASE_MIGRATION_RATE: f64 = 0.005;

/// Scale factor to convert push × pull product into actual flow.
/// Calibrated so global migration is ~0.5% of world population annually.
/// Scale factor increased 10× to match UN DESA observed migration rates.
/// Observed: ~281M international migrants in 2020 = ~3.6% of world pop.
/// Target: ~0.3-0.5% annual inter-region flow (subset of total migration).
const FLOW_SCALE: f64 = 0.001;

/// Push-pull migration engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationEngine {
    /// 12×12 annual flow matrix (from, to) in millions.
    pub flow_matrix: Vec<Vec<f64>>,
    /// Number of regions.
    pub num_regions: usize,
    /// Total migration this tick (millions).
    pub total_migration_this_tick: f64,
}

impl MigrationEngine {
    pub fn new(num_regions: usize) -> Self {
        Self {
            flow_matrix: vec![vec![0.0; num_regions]; num_regions],
            num_regions,
            total_migration_this_tick: 0.0,
        }
    }

    /// Compute migration flows and transfer cohort fractions.
    pub fn tick(
        &mut self,
        regions: &[EarthRegion],
        demographics: &mut [RegionDemographics],
        _rng: &mut StochasticEngine,
    ) {
        if regions.len() < 2 {
            return;
        }

        self.total_migration_this_tick = 0.0;

        // Compute push-pull for each pair
        for i in 0..regions.len().min(self.num_regions) {
            for j in 0..regions.len().min(self.num_regions) {
                if i == j {
                    continue;
                }

                let flow =
                    self.compute_flow(&regions[i], &regions[j], &demographics[i], &demographics[j]);
                self.flow_matrix[i][j] = flow;
            }
        }

        // Apply flows: transfer working-age cohort fractions
        // Monthly flow = annual / 12
        let mut transfers: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..self.num_regions.min(regions.len()) {
            for j in 0..self.num_regions.min(regions.len()) {
                if i == j {
                    continue;
                }
                let monthly_flow = self.flow_matrix[i][j] / 12.0;
                if monthly_flow > 0.001 {
                    transfers.push((i, j, monthly_flow));
                }
            }
        }

        for (from, to, amount) in transfers {
            self.transfer_migrants(from, to, amount, demographics);
            self.total_migration_this_tick += amount;
        }
    }

    /// Compute annual migration flow from region i to region j (millions/year).
    ///
    /// Push-pull gravity model:
    /// `flow = push_i × pull_j × sqrt(pop_i × pop_j) × FLOW_SCALE`
    fn compute_flow(
        &self,
        from: &EarthRegion,
        to: &EarthRegion,
        from_demo: &RegionDemographics,
        to_demo: &RegionDemographics,
    ) -> f64 {
        // Push factors (from): what makes people leave
        let push = 0.3 * (1.0 - from.gdp_per_capita / 60_000.0).max(0.0)  // poverty
            + 0.3 * from.climate_vulnerability                               // climate
            + 0.2 * (1.0 - from.infrastructure_level)                        // poor infra
            + 0.2 * (1.0 - from.education_index); // low opportunity

        // Pull factors (to): what attracts people
        let pull = 0.4 * (to.gdp_per_capita / 60_000.0).min(1.0)            // wages
            + 0.3 * (1.0 - to.climate_vulnerability)                          // safety
            + 0.2 * to.education_index                                        // opportunity
            + 0.1 * to.infrastructure_level; // quality of life

        // Gravity: proportional to geometric mean of populations
        let pop_factor = (from_demo.total_population * to_demo.total_population)
            .max(0.0)
            .sqrt();

        (push * pull * pop_factor * FLOW_SCALE).max(0.0)
    }

    /// Transfer migrants from one region to another.
    /// Migrants are biased toward working-age, secondary+ education.
    fn transfer_migrants(
        &self,
        from_idx: usize,
        to_idx: usize,
        amount_millions: f64,
        demographics: &mut [RegionDemographics],
    ) {
        if from_idx >= demographics.len() || to_idx >= demographics.len() {
            return;
        }
        if amount_millions < 0.0001 {
            return;
        }

        // Migrants are predominantly working-age (20-39, bands 4-7)
        // with secondary or tertiary education
        let migrant_bands: [(FineAgeBand, f64); 4] = [
            (FineAgeBand(4), 0.35), // 20-24
            (FineAgeBand(5), 0.30), // 25-29
            (FineAgeBand(6), 0.20), // 30-34
            (FineAgeBand(7), 0.15), // 35-39
        ];

        for &(band, fraction) in &migrant_bands {
            let band_amount = amount_millions * fraction;
            for &sex in &[CohortSex::Male, CohortSex::Female] {
                let sex_amount = band_amount * 0.5;

                // Prefer educated migrants
                let edu = if demographics[from_idx].mean_education() > 0.5 {
                    EducationBand::Secondary
                } else {
                    EducationBand::Primary
                };

                let from_key = RegionCohortKey {
                    region_idx: from_idx as u8,
                    age_band: band,
                    sex,
                    education: edu,
                };

                let to_key = RegionCohortKey {
                    region_idx: to_idx as u8,
                    age_band: band,
                    sex,
                    education: edu,
                };

                // Remove from source
                let actual_transfer =
                    if let Some(cohort) = demographics[from_idx].cohorts.get_mut(&from_key) {
                        let transfer = sex_amount.min(cohort.count * 0.05); // max 5% of cohort per tick
                        cohort.count -= transfer;
                        transfer
                    } else {
                        0.0
                    };

                if actual_transfer > 0.0001 {
                    // Add to destination
                    let dest_cohort = demographics[to_idx].cohorts.entry(to_key).or_default();
                    dest_cohort.count += actual_transfer;
                }
            }
        }

        // Update cached totals
        demographics[from_idx].total_population = demographics[from_idx]
            .cohorts
            .values()
            .map(|c| c.count)
            .sum();
        demographics[to_idx].total_population =
            demographics[to_idx].cohorts.values().map(|c| c.count).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth_population::EarthPopulationModel;
    use crate::earth_regions::build_earth_regions;

    #[test]
    fn test_migration_flows_from_poor_to_rich() {
        let engine = MigrationEngine::new(12);
        let regions = build_earth_regions();
        let model = EarthPopulationModel::from_regions(&regions);

        // South Asia (idx 1, poor) → North America (idx 7, rich)
        let flow_poor_to_rich = engine.compute_flow(
            &regions[1],
            &regions[7],
            &model.demographics[1],
            &model.demographics[7],
        );

        // North America → South Asia (reverse)
        let flow_rich_to_poor = engine.compute_flow(
            &regions[7],
            &regions[1],
            &model.demographics[7],
            &model.demographics[1],
        );

        assert!(
            flow_poor_to_rich > flow_rich_to_poor,
            "Migration should flow from poor to rich: {flow_poor_to_rich} vs {flow_rich_to_poor}"
        );
    }

    #[test]
    fn test_migration_preserves_total_population() {
        let regions = build_earth_regions();
        let mut model = EarthPopulationModel::from_regions(&regions);
        let total_before: f64 = model.demographics.iter().map(|d| d.total_population).sum();

        let mut rng = StochasticEngine::new(42);
        model
            .migration
            .tick(&regions, &mut model.demographics, &mut rng);

        let total_after: f64 = model.demographics.iter().map(|d| d.total_population).sum();

        // Migration should only move people, not create or destroy
        assert!(
            (total_before - total_after).abs() < total_before * 0.001,
            "Population changed: {total_before} → {total_after}"
        );
    }
}
