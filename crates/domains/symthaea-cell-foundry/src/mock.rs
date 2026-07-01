// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mock objects for testing cell foundry components.
//!
//! Provides simulated incubator and cell population models for use in
//! integration tests and demos without real hardware.

use crate::types::{CellState, CellType, CultureEnvironment, DEFAULT_GENE_EXPR_LEN};

/// Mock incubator that simulates environmental drift over time.
///
/// Useful for testing culture controllers: the mock drifts parameters
/// away from their setpoints at a configurable rate.
pub struct MockIncubator {
    /// Current environmental parameters of the simulated incubator.
    pub environment: CultureEnvironment,
    /// Rate at which parameters drift per hour (higher = faster drift).
    pub drift_rate: f64,
}

impl MockIncubator {
    /// Create a mock incubator with standard CO2 culture conditions.
    pub fn default_co2() -> Self {
        Self {
            environment: CultureEnvironment::standard(),
            drift_rate: 0.01,
        }
    }

    /// Create with custom drift rate.
    pub fn with_drift_rate(drift_rate: f64) -> Self {
        Self {
            environment: CultureEnvironment::standard(),
            drift_rate,
        }
    }

    /// Step the incubator forward by dt_hours, applying environmental drift.
    ///
    /// Temperature drifts down (heat loss), CO2 drifts down (diffusion),
    /// pH drifts up (CO2 loss), humidity drifts down (evaporation).
    pub fn step(&mut self, dt_hours: f64) {
        let drift = self.drift_rate * dt_hours;
        self.environment.temperature_celsius -= drift * 0.5; // Heat loss
        self.environment.co2_percent -= drift * 0.3; // CO2 diffusion
        self.environment.ph += drift * 0.1; // pH drift up as CO2 drops
        self.environment.humidity_percent -= drift * 0.2; // Evaporation
        self.environment.media_volume_ml -= drift * 0.05; // Evaporation

        // Clamp to physical limits
        self.environment.temperature_celsius = self.environment.temperature_celsius.max(15.0);
        self.environment.co2_percent = self.environment.co2_percent.max(0.0);
        self.environment.ph = self.environment.ph.clamp(6.0, 9.0);
        self.environment.humidity_percent = self.environment.humidity_percent.max(0.0);
        self.environment.media_volume_ml = self.environment.media_volume_ml.max(0.0);
    }

    /// Apply a correction to the environment (simulates controller output).
    pub fn apply_correction(&mut self, delta_temp: f64, delta_co2: f64, delta_ph: f64) {
        self.environment.temperature_celsius += delta_temp;
        self.environment.co2_percent += delta_co2;
        self.environment.ph += delta_ph;
    }

    /// Check if the environment is within acceptable bounds.
    pub fn is_within_bounds(&self) -> bool {
        let e = &self.environment;
        (35.0..=39.0).contains(&e.temperature_celsius)
            && (3.0..=7.0).contains(&e.co2_percent)
            && (7.0..=7.8).contains(&e.ph)
            && e.humidity_percent > 80.0
    }
}

/// Mock cell population for testing growth and viability dynamics.
pub struct MockCellPopulation {
    /// Individual cell states in this population.
    pub cells: Vec<CellState>,
    /// Net growth rate in doublings per day under ideal conditions.
    pub growth_rate: f64, // Doublings per day
    /// Fraction of cells that die per day under ideal conditions.
    pub death_rate: f64, // Fraction dying per day
}

impl MockCellPopulation {
    /// Create a new population of somatic cells.
    pub fn new(initial_count: usize) -> Self {
        let cells = (0..initial_count)
            .map(|_| CellState::new_somatic())
            .collect();
        Self {
            cells,
            growth_rate: 0.5, // One doubling every 2 days
            death_rate: 0.05, // 5% death per day
        }
    }

    /// Create a population of iPSC cells.
    pub fn new_ipsc(initial_count: usize) -> Self {
        let cells = (0..initial_count).map(|_| CellState::new_ipsc()).collect();
        Self {
            cells,
            growth_rate: 0.7, // iPSCs grow faster
            death_rate: 0.03,
        }
    }

    /// Step the population forward by the given hours.
    ///
    /// Applies growth (new cells) and death (removal) based on environment quality.
    pub fn step(&mut self, hours: f64, env: &CultureEnvironment) {
        let days = hours / 24.0;
        let deviation = env.deviation_from_standard();

        // Environmental stress reduces growth and increases death
        let stress_factor = (1.0 - deviation * 5.0).clamp(0.0, 1.0);
        let effective_growth = self.growth_rate * stress_factor * days;
        let effective_death = self.death_rate * (1.0 + deviation * 10.0) * days;

        // Apply death: reduce viability and remove dead cells
        for cell in &mut self.cells {
            cell.viability -= effective_death;
            cell.viability = cell.viability.clamp(0.0, 1.0);
            cell.age_hours += hours;
        }
        self.cells.retain(|c| c.viability > 0.1);

        // Apply growth: add new cells
        let new_cells = (self.cells.len() as f64 * effective_growth) as usize;
        for _ in 0..new_cells {
            let mut new_cell = CellState {
                cell_type: self
                    .cells
                    .first()
                    .map_or(CellType::Somatic, |c| c.cell_type),
                viability: 0.95,
                pluripotency_score: self.cells.first().map_or(0.0, |c| c.pluripotency_score),
                division_count: 0,
                age_hours: 0.0,
                methylation_score: 0.7,
                gene_expression: vec![0.5; DEFAULT_GENE_EXPR_LEN],
            };
            new_cell.viability = (0.95 * stress_factor).max(0.5);
            self.cells.push(new_cell);
        }
    }

    /// Average viability across all cells.
    pub fn viability(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.cells.iter().map(|c| c.viability).sum::<f64>() / self.cells.len() as f64
    }

    /// Number of living cells.
    pub fn count(&self) -> usize {
        self.cells.len()
    }

    /// Whether any cells remain alive.
    pub fn is_alive(&self) -> bool {
        !self.cells.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_incubator_drifts() {
        let mut incubator = MockIncubator::default_co2();
        let initial_temp = incubator.environment.temperature_celsius;

        incubator.step(10.0); // 10 hours

        assert!(
            incubator.environment.temperature_celsius < initial_temp,
            "Temperature should drift down"
        );
        assert!(
            incubator.environment.co2_percent < 5.0,
            "CO2 should drift down"
        );
    }

    #[test]
    fn test_mock_incubator_correction() {
        let mut incubator = MockIncubator::default_co2();
        incubator.step(10.0); // Let it drift

        let temp_before = incubator.environment.temperature_celsius;
        incubator.apply_correction(1.0, 0.5, -0.1);

        assert!(incubator.environment.temperature_celsius > temp_before);
    }

    #[test]
    fn test_mock_incubator_within_bounds_initially() {
        let incubator = MockIncubator::default_co2();
        assert!(incubator.is_within_bounds());
    }

    #[test]
    fn test_mock_incubator_out_of_bounds_after_drift() {
        let mut incubator = MockIncubator::with_drift_rate(1.0); // Fast drift
        incubator.step(100.0);
        assert!(!incubator.is_within_bounds());
    }

    #[test]
    fn test_mock_population_grows() {
        let mut pop = MockCellPopulation::new(100);
        let env = CultureEnvironment::standard();

        let initial_count = pop.count();
        pop.step(48.0, &env); // 2 days

        assert!(
            pop.count() > initial_count,
            "Population should grow: initial={}, final={}",
            initial_count,
            pop.count()
        );
    }

    #[test]
    fn test_mock_population_viability() {
        let pop = MockCellPopulation::new(100);
        let viability = pop.viability();
        assert!(
            viability > 0.9,
            "Initial viability should be high: {}",
            viability
        );
    }

    #[test]
    fn test_mock_population_dies_in_bad_conditions() {
        let mut pop = MockCellPopulation::new(50);
        let bad_env = CultureEnvironment {
            temperature_celsius: 45.0, // Lethal
            co2_percent: 20.0,
            o2_percent: 0.0,
            ph: 5.0,
            humidity_percent: 10.0,
            media_volume_ml: 0.1,
            passage_number: 0,
        };

        for _ in 0..20 {
            pop.step(24.0, &bad_env); // 1 day per step
        }

        assert!(
            pop.count() < 50,
            "Population should decline in bad conditions: {}",
            pop.count()
        );
    }

    #[test]
    fn test_mock_population_empty() {
        let pop = MockCellPopulation::new(0);
        assert_eq!(pop.count(), 0);
        assert!((pop.viability()).abs() < 1e-9);
        assert!(!pop.is_alive());
    }

    #[test]
    fn test_mock_population_ipsc() {
        let pop = MockCellPopulation::new_ipsc(50);
        assert_eq!(pop.count(), 50);
        assert!(pop.growth_rate > 0.5, "iPSCs should grow faster");
    }

    #[test]
    fn test_incubator_clamps_to_physical_limits() {
        let mut incubator = MockIncubator::with_drift_rate(100.0);
        incubator.step(1000.0);
        assert!(incubator.environment.temperature_celsius >= 15.0);
        assert!(incubator.environment.co2_percent >= 0.0);
        assert!(incubator.environment.ph >= 6.0 && incubator.environment.ph <= 9.0);
        assert!(incubator.environment.humidity_percent >= 0.0);
        assert!(incubator.environment.media_volume_ml >= 0.0);
    }
}
