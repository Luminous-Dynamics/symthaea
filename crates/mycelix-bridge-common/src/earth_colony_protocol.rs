// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Earth Colony Protocol: shared types for planetary colony management.
//!
//! Defines the canonical representation of a colony (any world in the
//! multi-world civilization), its resource manifest, and its status.
//! Used by both the multiworld simulation (for testing/validation) and
//! Mycelix zomes (for production governance/trade).
//!
//! # Integration
//!
//! - **Multiworld sim** imports these types to ensure its simplified models
//!   use the same vocabulary as production Mycelix.
//! - **Mycelix zomes** (governance, finance, space) use these types for
//!   cross-planetary coordination.

use serde::{Deserialize, Serialize};

// ============================================================================
// Planetary Location
// ============================================================================

/// Canonical planetary body identifiers.
///
/// Each variant carries the physical constraints that govern colony survival.
/// These match the multiworld sim's location strings ("Earth", "Moon", etc.)
/// but add type safety and physical parameter access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlanetaryBody {
    Earth,
    Moon,
    Mars,
    /// Jupiter's moon. 5400 krad/yr radiation, 3.7% solar flux.
    Europa,
    /// Saturn's moon. 93.7K surface, 1.467 bar N₂, 0.14g.
    Titan,
    /// Asteroid belt. Variable resources.
    Ceres,
    /// Generic orbital habitat.
    OrbitalHabitat,
}

impl PlanetaryBody {
    /// Solar flux as fraction of Earth's 1361 W/m².
    pub fn solar_flux_fraction(&self) -> f64 {
        match self {
            Self::Earth => 1.0,
            Self::Moon => 1.0,          // Same distance from Sun
            Self::Mars => 0.43,         // 1/1.524² AU
            Self::Europa => 0.037,      // 1/5.2² AU
            Self::Titan => 0.011,       // 1/9.54² AU
            Self::Ceres => 0.13,        // 1/2.77² AU
            Self::OrbitalHabitat => 1.0,
        }
    }

    /// Surface gravity as fraction of Earth's 9.81 m/s².
    pub fn gravity_fraction(&self) -> f64 {
        match self {
            Self::Earth => 1.0,
            Self::Moon => 0.166,
            Self::Mars => 0.38,
            Self::Europa => 0.134,
            Self::Titan => 0.138,
            Self::Ceres => 0.029,
            Self::OrbitalHabitat => 0.0, // microgravity or spin gravity
        }
    }

    /// Whether solar power is viable as primary energy source.
    pub fn solar_viable(&self) -> bool {
        self.solar_flux_fraction() > 0.1
    }

    /// Whether atmosphere provides radiation shielding.
    pub fn has_atmosphere(&self) -> bool {
        matches!(self, Self::Earth | Self::Mars | Self::Titan)
    }

    /// One-way light delay to Earth in seconds (mean).
    pub fn light_delay_to_earth_secs(&self) -> f64 {
        match self {
            Self::Earth => 0.0,
            Self::Moon => 1.28,
            Self::Mars => 750.0,          // ~12.5 min mean
            Self::Europa => 2592.0,       // ~43.2 min mean
            Self::Titan => 4758.0,        // ~79.3 min mean
            Self::Ceres => 1380.0,        // ~23 min mean
            Self::OrbitalHabitat => 0.01, // LEO
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Earth => "Earth",
            Self::Moon => "Moon",
            Self::Mars => "Mars",
            Self::Europa => "Europa",
            Self::Titan => "Titan",
            Self::Ceres => "Ceres",
            Self::OrbitalHabitat => "OrbitalHabitat",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Earth" => Some(Self::Earth),
            "Moon" => Some(Self::Moon),
            "Mars" => Some(Self::Mars),
            "Europa" => Some(Self::Europa),
            "Titan" => Some(Self::Titan),
            "Ceres" => Some(Self::Ceres),
            "OrbitalHabitat" | "Orbital" => Some(Self::OrbitalHabitat),
            _ => None,
        }
    }
}

// ============================================================================
// Colony Status
// ============================================================================

/// Colony lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColonyPhase {
    /// Pre-founding: planning and resource staging.
    Planning,
    /// Active colony, dependent on Earth resupply.
    Dependent,
    /// Partially self-sufficient (some local production).
    Developing,
    /// Fully self-sufficient in basic needs.
    SelfSufficient,
    /// Exports surplus to other colonies.
    Exporting,
    /// Abandoned or destroyed.
    Collapsed,
}

/// Status snapshot of a colony at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonyStatus {
    pub body: PlanetaryBody,
    pub name: String,
    pub phase: ColonyPhase,
    pub population: u32,
    pub founded_tick: u32,
    /// Self-sufficiency score [0.0, 1.0] — 1.0 = fully independent.
    pub self_sufficiency: f64,
    /// Infrastructure level [0.0, 1.0].
    pub infrastructure: f64,
    /// Mean consciousness phi [0.0, 1.0].
    pub mean_phi: f64,
    /// Resource manifest (current state).
    pub resources: ResourceManifest,
}

// ============================================================================
// Resource Manifest
// ============================================================================

/// Specific element/resource tracking for interplanetary trade.
///
/// Replaces abstract "materials" with actual elements that have different
/// abundances per planetary body. This creates genuine trade complementarity:
/// Titan has carbon but no iron; Europa has water but no nitrogen.
///
/// Units: kg per capita per day (calibrated to NASA BVAD 2018).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceManifest {
    // === Life support (NASA BVAD 2018) ===
    /// Food: 1.83 kg/person/day dry (NASA BVAD Table 4.2.2).
    pub food_kg: f64,
    /// Potable water: 2.50 kg/person/day (with 93% recycling, gross 26 kg/day).
    pub water_kg: f64,
    /// Oxygen: 0.84 kg/person/day (NASA BVAD Table 4.1.1).
    pub oxygen_kg: f64,
    /// Nitrogen (atmosphere makeup): 0.01 kg/person/day leakage replacement.
    pub nitrogen_kg: f64,

    // === Energy ===
    /// Electrical power available, kW.
    pub power_kw: f64,
    /// Energy storage, kWh.
    pub energy_stored_kwh: f64,

    // === Construction elements ===
    /// Iron/steel (structural, tools). kg stockpile.
    pub iron_kg: f64,
    /// Silicon (electronics, solar panels). kg stockpile.
    pub silicon_kg: f64,
    /// Aluminum (structures, thermal). kg stockpile.
    pub aluminum_kg: f64,
    /// Carbon (polymers, composites, fuel). kg stockpile.
    pub carbon_kg: f64,
    /// Rare earths (electronics, magnets). kg stockpile.
    pub rare_earth_kg: f64,

    // === Location-specific ===
    /// Hydrocarbons (Titan only): methane/ethane, kg.
    pub hydrocarbons_kg: f64,
    /// Regolith/ice processed for ISRU, kg.
    pub regolith_processed_kg: f64,
}

impl ResourceManifest {
    /// Default abundance profile for a planetary body.
    /// Returns (local_production_multiplier per element).
    pub fn abundance_profile(body: PlanetaryBody) -> Self {
        match body {
            PlanetaryBody::Earth => Self {
                food_kg: 1.0, water_kg: 1.0, oxygen_kg: 1.0, nitrogen_kg: 1.0,
                power_kw: 1.0, energy_stored_kwh: 1.0,
                iron_kg: 1.0, silicon_kg: 1.0, aluminum_kg: 1.0, carbon_kg: 1.0,
                rare_earth_kg: 1.0, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.0,
            },
            PlanetaryBody::Moon => Self {
                food_kg: 0.0, water_kg: 0.1, oxygen_kg: 0.3, nitrogen_kg: 0.0,
                power_kw: 1.0, energy_stored_kwh: 0.5,
                iron_kg: 0.4, silicon_kg: 0.6, aluminum_kg: 0.5, carbon_kg: 0.01,
                rare_earth_kg: 0.2, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.8,
            },
            PlanetaryBody::Mars => Self {
                food_kg: 0.0, water_kg: 0.3, oxygen_kg: 0.2, nitrogen_kg: 0.05,
                power_kw: 0.43, energy_stored_kwh: 0.3,
                iron_kg: 0.7, silicon_kg: 0.5, aluminum_kg: 0.4, carbon_kg: 0.3,
                rare_earth_kg: 0.3, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.6,
            },
            PlanetaryBody::Europa => Self {
                food_kg: 0.0, water_kg: 10.0, oxygen_kg: 5.0, nitrogen_kg: 0.0,
                power_kw: 0.0, energy_stored_kwh: 0.0, // Nuclear only
                iron_kg: 0.05, silicon_kg: 0.02, aluminum_kg: 0.02, carbon_kg: 0.01,
                rare_earth_kg: 0.01, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.1,
            },
            PlanetaryBody::Titan => Self {
                food_kg: 0.0, water_kg: 5.0, oxygen_kg: 2.0, nitrogen_kg: 10.0,
                power_kw: 0.0, energy_stored_kwh: 0.0, // Nuclear only
                iron_kg: 0.0, silicon_kg: 0.0, aluminum_kg: 0.0, carbon_kg: 10.0,
                rare_earth_kg: 0.0, hydrocarbons_kg: 100.0, regolith_processed_kg: 0.3,
            },
            PlanetaryBody::Ceres => Self {
                food_kg: 0.0, water_kg: 3.0, oxygen_kg: 1.0, nitrogen_kg: 0.1,
                power_kw: 0.13, energy_stored_kwh: 0.1,
                iron_kg: 0.5, silicon_kg: 0.4, aluminum_kg: 0.3, carbon_kg: 0.5,
                rare_earth_kg: 0.4, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.7,
            },
            PlanetaryBody::OrbitalHabitat => Self {
                food_kg: 0.0, water_kg: 0.0, oxygen_kg: 0.0, nitrogen_kg: 0.0,
                power_kw: 1.0, energy_stored_kwh: 0.5,
                iron_kg: 0.0, silicon_kg: 0.0, aluminum_kg: 0.0, carbon_kg: 0.0,
                rare_earth_kg: 0.0, hydrocarbons_kg: 0.0, regolith_processed_kg: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planetary_body_roundtrip() {
        for body in [PlanetaryBody::Earth, PlanetaryBody::Europa, PlanetaryBody::Titan] {
            assert_eq!(PlanetaryBody::from_str(body.as_str()), Some(body));
        }
    }

    #[test]
    fn test_solar_viability() {
        assert!(PlanetaryBody::Earth.solar_viable());
        assert!(PlanetaryBody::Mars.solar_viable());
        assert!(!PlanetaryBody::Europa.solar_viable());
        assert!(!PlanetaryBody::Titan.solar_viable());
    }

    #[test]
    fn test_titan_has_atmosphere() {
        assert!(PlanetaryBody::Titan.has_atmosphere());
        assert!(!PlanetaryBody::Europa.has_atmosphere());
        assert!(!PlanetaryBody::Moon.has_atmosphere());
    }

    #[test]
    fn test_abundance_profiles_differ() {
        let titan = ResourceManifest::abundance_profile(PlanetaryBody::Titan);
        let europa = ResourceManifest::abundance_profile(PlanetaryBody::Europa);
        // Titan has carbon, Europa doesn't
        assert!(titan.carbon_kg > europa.carbon_kg);
        // Europa has water, Titan has less
        assert!(europa.water_kg > titan.water_kg);
        // Neither has iron
        assert!(titan.iron_kg < 0.1);
    }
}
