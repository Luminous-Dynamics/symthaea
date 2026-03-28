// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! In-Situ Resource Utilization (ISRU) types for planetary colonies.
//!
//! Defines what each planetary body can produce locally vs. what must be
//! imported. This drives the trade specialization model: colonies that
//! achieve ISRU milestones reduce their import dependency, while their
//! surplus becomes exports for other worlds.
//!
//! # Resource Types
//!
//! Unlike the abstract "materials" resource in early sim versions, this module
//! tracks specific elements: iron, silicon, nitrogen, carbon, etc. Each planet
//! has different abundances, creating genuine trade complementarity.

use crate::earth_colony_protocol::{PlanetaryBody, ResourceManifest};
use serde::{Deserialize, Serialize};

// ============================================================================
// ISRU Capabilities
// ============================================================================

/// What a colony can extract/produce from local resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISRUCapability {
    /// Which body this applies to.
    pub body: PlanetaryBody,
    /// Water extraction rate (kg/day per unit infrastructure).
    pub water_extraction: f64,
    /// Oxygen production from local sources (kg/day per kW).
    pub oxygen_production: f64,
    /// Metal extraction from regolith (kg/day per unit infrastructure).
    pub metal_extraction: f64,
    /// Hydrocarbon collection (Titan only, kg/day).
    pub hydrocarbon_collection: f64,
    /// Nitrogen extraction from atmosphere (Titan/Mars only, kg/day per kW).
    pub nitrogen_extraction: f64,
    /// Whether nuclear power is required (Europa, Titan).
    pub requires_nuclear: bool,
    /// Minimum tech level to unlock ISRU.
    pub min_tech_level: f64,
}

impl ISRUCapability {
    /// Default ISRU profile for a planetary body.
    pub fn for_body(body: PlanetaryBody) -> Self {
        match body {
            PlanetaryBody::Earth => Self {
                body, water_extraction: 1000.0, oxygen_production: 500.0,
                metal_extraction: 100.0, hydrocarbon_collection: 0.0,
                nitrogen_extraction: 100.0, requires_nuclear: false, min_tech_level: 0.0,
            },
            PlanetaryBody::Moon => Self {
                body, water_extraction: 5.0, // Polar ice deposits
                oxygen_production: 20.0, // Ilmenite reduction (FeTiO₃ → Fe + TiO₂ + O₂)
                metal_extraction: 10.0,  // Iron from regolith
                hydrocarbon_collection: 0.0, nitrogen_extraction: 0.0,
                requires_nuclear: false, min_tech_level: 1.5,
            },
            PlanetaryBody::Mars => Self {
                body, water_extraction: 15.0, // Subsurface ice
                oxygen_production: 30.0, // MOXIE-style CO₂ electrolysis
                metal_extraction: 15.0,  // Iron oxide regolith
                hydrocarbon_collection: 0.0,
                nitrogen_extraction: 2.0, // 2.7% N₂ in atmosphere
                requires_nuclear: false, min_tech_level: 1.3,
            },
            PlanetaryBody::Europa => Self {
                body, water_extraction: 500.0, // Unlimited ice
                oxygen_production: 100.0, // Electrolysis
                metal_extraction: 1.0,   // Extremely limited
                hydrocarbon_collection: 0.0,
                nitrogen_extraction: 0.0, // No atmospheric N₂
                requires_nuclear: true, min_tech_level: 2.0,
            },
            PlanetaryBody::Titan => Self {
                body, water_extraction: 200.0, // Ice bedrock
                oxygen_production: 50.0, // Electrolysis
                metal_extraction: 0.0,   // No accessible metals
                hydrocarbon_collection: 1000.0, // Surface methane lakes
                nitrogen_extraction: 500.0, // 94% N₂ atmosphere
                requires_nuclear: true, min_tech_level: 2.0,
            },
            PlanetaryBody::Ceres => Self {
                body, water_extraction: 100.0, // Ice-rich body
                oxygen_production: 30.0,
                metal_extraction: 20.0,  // Carbonaceous chondrite composition
                hydrocarbon_collection: 0.0, nitrogen_extraction: 0.5,
                requires_nuclear: false, min_tech_level: 1.8,
            },
            PlanetaryBody::OrbitalHabitat => Self {
                body, water_extraction: 0.0, oxygen_production: 0.0,
                metal_extraction: 0.0, hydrocarbon_collection: 0.0,
                nitrogen_extraction: 0.0, requires_nuclear: false, min_tech_level: 0.0,
            },
        }
    }

    /// What this body primarily exports (surplus resources).
    pub fn primary_exports(&self) -> Vec<&'static str> {
        match self.body {
            PlanetaryBody::Earth => vec!["knowledge", "rare_earth", "silicon"],
            PlanetaryBody::Moon => vec!["silicon", "aluminum", "oxygen"],
            PlanetaryBody::Mars => vec!["iron", "water", "oxygen"],
            PlanetaryBody::Europa => vec!["water", "oxygen"],
            PlanetaryBody::Titan => vec!["hydrocarbons", "nitrogen", "carbon"],
            PlanetaryBody::Ceres => vec!["iron", "water", "carbon"],
            PlanetaryBody::OrbitalHabitat => vec![],
        }
    }

    /// What this body primarily needs imported.
    pub fn primary_imports(&self) -> Vec<&'static str> {
        match self.body {
            PlanetaryBody::Earth => vec![], // Self-sufficient
            PlanetaryBody::Moon => vec!["nitrogen", "carbon", "food"],
            PlanetaryBody::Mars => vec!["nitrogen", "rare_earth"],
            PlanetaryBody::Europa => vec!["iron", "silicon", "nitrogen", "carbon"],
            PlanetaryBody::Titan => vec!["iron", "silicon", "aluminum", "rare_earth"],
            PlanetaryBody::Ceres => vec!["nitrogen", "food"],
            PlanetaryBody::OrbitalHabitat => vec!["food", "water", "oxygen", "materials"],
        }
    }
}

/// Bootstrap power model: RTG decay for outer system colonies.
///
/// Europa and Titan colonies arrive with RTGs that decay over time.
/// Without fission tech, power output drops exponentially.
/// Pu-238 half-life: 87.7 years. RTG efficiency degrades faster than isotope
/// due to thermocouple degradation (~0.8%/year).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapPower {
    /// Initial RTG power at colony founding (watts electrical).
    pub initial_power_w: f64,
    /// Current effective power (watts), decaying over time.
    pub current_power_w: f64,
    /// Decay rate per tick (monthly). Combines Pu-238 decay + thermocouple degradation.
    /// ~1.6%/year total → ~0.00133/month.
    pub decay_rate_per_tick: f64,
}

impl BootstrapPower {
    /// Standard RTG bootstrap for outer system colonies.
    /// Assumes 4 MMRTGs (~440W total) from the landing craft.
    pub fn outer_system_default() -> Self {
        Self {
            initial_power_w: 440.0,
            current_power_w: 440.0,
            decay_rate_per_tick: 0.0013, // ~1.6%/year
        }
    }

    /// Tick the power model: RTGs decay each month.
    pub fn tick(&mut self) {
        self.current_power_w *= 1.0 - self.decay_rate_per_tick;
    }

    /// Power as fraction of initial (for sim resource scaling).
    pub fn power_fraction(&self) -> f64 {
        if self.initial_power_w > 0.0 {
            self.current_power_w / self.initial_power_w
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isru_requires_nuclear_for_outer_system() {
        assert!(ISRUCapability::for_body(PlanetaryBody::Europa).requires_nuclear);
        assert!(ISRUCapability::for_body(PlanetaryBody::Titan).requires_nuclear);
        assert!(!ISRUCapability::for_body(PlanetaryBody::Mars).requires_nuclear);
    }

    #[test]
    fn test_titan_exports_hydrocarbons() {
        let titan = ISRUCapability::for_body(PlanetaryBody::Titan);
        assert!(titan.primary_exports().contains(&"hydrocarbons"));
        assert!(titan.primary_imports().contains(&"iron"));
    }

    #[test]
    fn test_rtg_decay() {
        let mut rtg = BootstrapPower::outer_system_default();
        assert!((rtg.current_power_w - 440.0).abs() < 0.1);
        // After 10 years (120 ticks), power should be ~93% of initial
        for _ in 0..120 {
            rtg.tick();
        }
        let expected = 440.0 * (1.0 - 0.0013_f64).powi(120);
        assert!((rtg.current_power_w - expected).abs() < 1.0);
        assert!(rtg.power_fraction() > 0.8); // Still > 80% after 10 years
    }

    #[test]
    fn test_complementary_trade() {
        let europa = ISRUCapability::for_body(PlanetaryBody::Europa);
        let titan = ISRUCapability::for_body(PlanetaryBody::Titan);
        // Europa exports water, Titan needs it less (has ice)
        // Titan exports nitrogen, Europa desperately needs it
        assert!(titan.primary_exports().contains(&"nitrogen"));
        assert!(europa.primary_imports().contains(&"nitrogen"));
    }
}
