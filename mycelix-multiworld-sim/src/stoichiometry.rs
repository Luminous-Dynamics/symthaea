// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stoichiometric mass ledger: conservation of matter for closed-loop ECLSS.
//!
//! A generation ship is a strictly closed system. Every atom of carbon,
//! nitrogen, and phosphorus must be accounted for. When mass is lost to
//! the void (hull breach, outgassing), it is gone forever — permanently
//! reducing the ship's carrying capacity.
//!
//! # Chemistry
//!
//! Photosynthesis: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
//! Respiration:    C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O
//! Death:          70kg human → 12.6kg C + 7.0kg H + 45.5kg O + 2.1kg N + 0.7kg P
//!
//! # References
//!
//! - NASA BVAD (2018): Baseline Values and Assumptions Document, Table 4.x
//! - Drysdale et al. (2003): Advanced Life Support Requirements

use serde::{Deserialize, Serialize};

// ============================================================================
// HUMAN BODY COMPOSITION (kg per 70kg adult)
// ============================================================================

/// Carbon content of a 70kg human body (kg).
pub const HUMAN_CARBON_KG: f64 = 12.6;
/// Hydrogen content (kg).
pub const HUMAN_HYDROGEN_KG: f64 = 7.0;
/// Oxygen content (kg).
pub const HUMAN_OXYGEN_KG: f64 = 45.5;
/// Nitrogen content (kg).
pub const HUMAN_NITROGEN_KG: f64 = 2.1;
/// Phosphorus content (kg).
pub const HUMAN_PHOSPHORUS_KG: f64 = 0.7;
/// Total tracked mass per human (kg).
pub const HUMAN_TRACKED_MASS_KG: f64 = 67.9;

// ============================================================================
// DAILY CONSUMPTION (kg per person per day, NASA BVAD)
// ============================================================================

/// O₂ consumption per person per day (kg). NASA BVAD Table 4.1.1
pub const O2_CONSUMPTION_KG_DAY: f64 = 0.84;
/// CO₂ production per person per day (kg).
pub const CO2_PRODUCTION_KG_DAY: f64 = 1.00;
/// Food consumption per person per day (dry kg). NASA BVAD Table 4.2.2
pub const FOOD_DRY_KG_DAY: f64 = 1.83;
/// Water consumption per person per day (kg potable). With 93% recycling: gross 26 kg/day.
pub const WATER_KG_DAY: f64 = 2.50;

// ============================================================================
// FOOD COMPOSITION (per kg dry food)
// ============================================================================

pub const FOOD_CARBON_FRACTION: f64 = 0.45;
pub const FOOD_HYDROGEN_FRACTION: f64 = 0.06;
pub const FOOD_OXYGEN_FRACTION: f64 = 0.30;
pub const FOOD_NITROGEN_FRACTION: f64 = 0.15;

// ============================================================================
// ELEMENTAL LEDGER
// ============================================================================

/// Tracks conservation of biogenic elements in a closed system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementalLedger {
    /// Carbon in the biosphere (kg). Used in food, biomass, CO₂.
    pub carbon_kg: f64,
    /// Hydrogen (kg). In water, food, biomass.
    pub hydrogen_kg: f64,
    /// Oxygen (kg). In atmosphere, water, food.
    pub oxygen_kg: f64,
    /// Nitrogen (kg). In food, biomass, atmosphere (N₂).
    pub nitrogen_kg: f64,
    /// Phosphorus (kg). In bones, DNA, food.
    pub phosphorus_kg: f64,
    /// Water reservoir (kg). Separate tracking for ECLSS.
    pub water_kg: f64,
    /// Total mass lost to void (hull breaches, outgassing). Irreversible.
    pub mass_lost_kg: f64,
    /// Recycling efficiency [0, 1]. ISS achieves ~0.93 for water.
    pub recycling_efficiency: f64,
    /// Carrying capacity derived from limiting element.
    pub max_sustainable_population: usize,
}

impl ElementalLedger {
    /// Initialize for a generation ship with given population.
    ///
    /// Stocks enough consumables for the population + 50% safety margin.
    pub fn for_ship(population: usize, transit_years: f64) -> Self {
        let pop = population as f64;
        let days = transit_years * 365.25;
        let margin = 1.5; // 50% safety margin

        // Total consumption over transit
        let total_food_kg = pop * FOOD_DRY_KG_DAY * days * margin;
        let total_water_kg = pop * WATER_KG_DAY * days * 0.1; // 90% recycled
        let total_o2_kg = pop * O2_CONSUMPTION_KG_DAY * days * 0.1; // 90% recycled

        // Elemental stocks include: biomass (people) + reserves
        let carbon = pop * HUMAN_CARBON_KG + total_food_kg * FOOD_CARBON_FRACTION;
        let hydrogen = pop * HUMAN_HYDROGEN_KG + total_water_kg * 0.111; // H₂O is 11.1% H
        let oxygen = pop * HUMAN_OXYGEN_KG + total_o2_kg + total_water_kg * 0.889;
        let nitrogen = pop * HUMAN_NITROGEN_KG + total_food_kg * FOOD_NITROGEN_FRACTION;
        let phosphorus = pop * HUMAN_PHOSPHORUS_KG * 3.0; // 3× body mass in reserves

        Self {
            carbon_kg: carbon,
            hydrogen_kg: hydrogen,
            oxygen_kg: oxygen,
            nitrogen_kg: nitrogen,
            phosphorus_kg: phosphorus,
            water_kg: total_water_kg + pop * 70.0 * 0.6, // 60% of body mass is water
            mass_lost_kg: 0.0,
            recycling_efficiency: 0.93, // ISS benchmark
            max_sustainable_population: population,
        }
    }

    /// Process one tick (month) of life support chemistry.
    ///
    /// Returns the new carrying capacity based on the limiting element.
    pub fn tick(&mut self, population: usize) -> usize {
        let pop = population as f64;
        let days_per_tick = 30.44; // Average days per month

        // Consumption (population × daily rate × days)
        let food_consumed = pop * FOOD_DRY_KG_DAY * days_per_tick;
        let water_consumed = pop * WATER_KG_DAY * days_per_tick;
        let o2_consumed = pop * O2_CONSUMPTION_KG_DAY * days_per_tick;

        // Recycling: most is recovered, but some is lost
        let loss_fraction = 1.0 - self.recycling_efficiency;

        // Carbon cycle: food consumed → CO₂ → recycled back to food (photosynthesis)
        // Net loss: food_consumed × carbon_fraction × loss_fraction
        let carbon_lost = food_consumed * FOOD_CARBON_FRACTION * loss_fraction;
        self.carbon_kg -= carbon_lost;

        // Nitrogen cycle: similar
        let nitrogen_lost = food_consumed * FOOD_NITROGEN_FRACTION * loss_fraction;
        self.nitrogen_kg -= nitrogen_lost;

        // Oxygen cycle: consumed → CO₂ → recycled
        let oxygen_lost = o2_consumed * loss_fraction;
        self.oxygen_kg -= oxygen_lost;

        // Water cycle: consumed → recycled (ISS 93%)
        let water_lost = water_consumed * loss_fraction;
        self.water_kg -= water_lost;
        self.hydrogen_kg -= water_lost * 0.111;

        // Clamp to zero
        self.carbon_kg = self.carbon_kg.max(0.0);
        self.hydrogen_kg = self.hydrogen_kg.max(0.0);
        self.oxygen_kg = self.oxygen_kg.max(0.0);
        self.nitrogen_kg = self.nitrogen_kg.max(0.0);
        self.phosphorus_kg = self.phosphorus_kg.max(0.0);
        self.water_kg = self.water_kg.max(0.0);

        // Compute carrying capacity from limiting element
        self.max_sustainable_population = self.carrying_capacity();
        self.max_sustainable_population
    }

    /// Process a hull breach: lose atmosphere mass proportional to severity.
    ///
    /// Severity 0.001 = tiny puncture, 0.01 = significant breach.
    pub fn hull_breach(&mut self, severity: f64) {
        // Atmosphere is ~78% N₂, 21% O₂, ~1% other
        let atmosphere_loss = severity * 1000.0; // kg lost (scaled)
        let o2_lost = atmosphere_loss * 0.21;
        let n2_lost = atmosphere_loss * 0.78;

        self.oxygen_kg -= o2_lost;
        self.nitrogen_kg -= n2_lost;
        self.mass_lost_kg += atmosphere_loss;

        self.oxygen_kg = self.oxygen_kg.max(0.0);
        self.nitrogen_kg = self.nitrogen_kg.max(0.0);

        self.max_sustainable_population = self.carrying_capacity();
    }

    /// Process a death: body mass returns to the biosphere for recycling.
    pub fn recycle_body(&mut self) {
        // Recycling a 70kg body returns elements to the system
        // (In reality this is composting, not instantaneous)
        self.carbon_kg += HUMAN_CARBON_KG * self.recycling_efficiency;
        self.nitrogen_kg += HUMAN_NITROGEN_KG * self.recycling_efficiency;
        self.phosphorus_kg += HUMAN_PHOSPHORUS_KG * self.recycling_efficiency;
        // Water and oxygen from body decomposition
        self.water_kg += 42.0 * self.recycling_efficiency; // 60% of 70kg
    }

    /// Compute carrying capacity from the most limiting element.
    fn carrying_capacity(&self) -> usize {
        // Each person needs: C, N, P per month (from food)
        let days = 30.44;
        let c_per_person = FOOD_DRY_KG_DAY * FOOD_CARBON_FRACTION * days;
        let n_per_person = FOOD_DRY_KG_DAY * FOOD_NITROGEN_FRACTION * days;
        let p_per_person_month = HUMAN_PHOSPHORUS_KG / 12.0; // Annual turnover / 12
        let o2_per_person = O2_CONSUMPTION_KG_DAY * days;

        let cap_carbon = (self.carbon_kg / c_per_person) as usize;
        let cap_nitrogen = (self.nitrogen_kg / n_per_person) as usize;
        let cap_phosphorus = (self.phosphorus_kg / p_per_person_month) as usize;
        let cap_oxygen = (self.oxygen_kg / o2_per_person) as usize;
        let cap_water = (self.water_kg / (WATER_KG_DAY * days * 0.07)) as usize; // 7% net loss

        cap_carbon
            .min(cap_nitrogen)
            .min(cap_phosphorus)
            .min(cap_oxygen)
            .min(cap_water)
    }

    /// Which element is the current bottleneck?
    pub fn limiting_element(&self) -> &'static str {
        let days = 30.44;
        let c = self.carbon_kg / (FOOD_DRY_KG_DAY * FOOD_CARBON_FRACTION * days);
        let n = self.nitrogen_kg / (FOOD_DRY_KG_DAY * FOOD_NITROGEN_FRACTION * days);
        let p = self.phosphorus_kg / (HUMAN_PHOSPHORUS_KG / 12.0);
        let o = self.oxygen_kg / (O2_CONSUMPTION_KG_DAY * days);

        let min = c.min(n).min(p).min(o);
        if (min - c).abs() < 0.01 {
            "Carbon"
        } else if (min - n).abs() < 0.01 {
            "Nitrogen"
        } else if (min - p).abs() < 0.01 {
            "Phosphorus"
        } else {
            "Oxygen"
        }
    }

    /// Total tracked mass in the system (kg).
    pub fn total_mass(&self) -> f64 {
        self.carbon_kg + self.hydrogen_kg + self.oxygen_kg + self.nitrogen_kg + self.phosphorus_kg
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ship_initialization() {
        let ledger = ElementalLedger::for_ship(500, 85.0);
        assert!(ledger.carbon_kg > 0.0);
        assert!(ledger.oxygen_kg > 0.0);
        assert!(ledger.nitrogen_kg > 0.0);
        assert!(ledger.phosphorus_kg > 0.0);
        assert!(ledger.water_kg > 0.0);
        assert_eq!(ledger.mass_lost_kg, 0.0);
        assert_eq!(ledger.max_sustainable_population, 500);
    }

    #[test]
    fn tick_consumes_resources() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let initial_carbon = ledger.carbon_kg;
        ledger.tick(500);
        assert!(
            ledger.carbon_kg < initial_carbon,
            "Carbon should decrease: {} → {}",
            initial_carbon,
            ledger.carbon_kg
        );
    }

    #[test]
    fn hull_breach_reduces_capacity() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let initial_o2 = ledger.oxygen_kg;
        ledger.hull_breach(0.01); // Significant breach
        assert!(
            ledger.oxygen_kg < initial_o2,
            "Breach should reduce oxygen: {} → {}",
            initial_o2,
            ledger.oxygen_kg
        );
        assert!(ledger.mass_lost_kg > 0.0);
    }

    #[test]
    fn hull_breach_permanent() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let initial_mass = ledger.total_mass();
        ledger.hull_breach(0.005);
        assert!(
            ledger.total_mass() < initial_mass,
            "Mass should be permanently lost"
        );
        assert!(ledger.mass_lost_kg > 0.0);
    }

    #[test]
    fn body_recycling_returns_elements() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let carbon_before = ledger.carbon_kg;
        ledger.recycle_body();
        assert!(
            ledger.carbon_kg > carbon_before,
            "Body recycling should return carbon"
        );
    }

    #[test]
    fn limiting_element_is_valid() {
        let ledger = ElementalLedger::for_ship(500, 85.0);
        let limiting = ledger.limiting_element();
        assert!(
            ["Carbon", "Nitrogen", "Phosphorus", "Oxygen"].contains(&limiting),
            "Limiting element should be one of CNPO: {}",
            limiting
        );
    }

    #[test]
    fn mass_conservation_over_ticks() {
        let mut ledger = ElementalLedger::for_ship(100, 10.0);
        let initial_total = ledger.total_mass();
        // Run 12 ticks (1 year)
        for _ in 0..12 {
            ledger.tick(100);
        }
        // Mass should decrease (recycling loss) but not catastrophically
        let final_total = ledger.total_mass();
        let loss_fraction = 1.0 - final_total / initial_total;
        assert!(
            loss_fraction < 0.1,
            "Should lose < 10% per year with 93% recycling: {:.1}% lost",
            loss_fraction * 100.0
        );
        assert!(loss_fraction > 0.0, "Should lose something");
    }

    #[test]
    fn capacity_drops_with_severe_depletion() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let initial_cap = ledger.carrying_capacity();
        // Deplete nitrogen to near-zero
        ledger.nitrogen_kg = 1.0; // Almost gone
        let cap = ledger.carrying_capacity();
        assert!(
            cap < initial_cap,
            "Severe nitrogen depletion should reduce capacity: {} → {}",
            initial_cap,
            cap
        );
    }

    #[test]
    fn zero_population_no_consumption() {
        let mut ledger = ElementalLedger::for_ship(500, 85.0);
        let carbon_before = ledger.carbon_kg;
        ledger.tick(0); // No one alive
        assert!(
            (ledger.carbon_kg - carbon_before).abs() < 0.01,
            "Zero pop should consume nothing"
        );
    }
}
