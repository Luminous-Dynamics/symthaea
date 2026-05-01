// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generation Ship: a mobile World for interstellar transit.
//!
//! A generation ship wraps a standard `World` with interstellar physics:
//! - Transit velocity and distance tracking
//! - Growing communication delay with Sol (light-speed)
//! - Cosmic ray flux outside the heliosphere (3× baseline)
//! - Hull integrity degradation from micrometeorites and radiation
//! - Closed-loop ECLSS (no resupply possible)
//! - Cultural speciation after prolonged isolation
//!
//! The ship IS a World — it has agents, economy, governance, knowledge,
//! all existing systems. The GenerationShip struct wraps it with
//! interstellar-specific constraints.
//!
//! # Physics
//!
//! At 0.05c (5% light speed = 14,990 km/s):
//! - Proxima Centauri (4.24 ly): 85 years transit = 1,020 ticks
//! - Barnard's Star (5.96 ly): 119 years = 1,429 ticks
//! - Wolf 359 (7.86 ly): 157 years = 1,885 ticks
//!
//! Communication delay grows linearly: after 10 years at 0.05c,
//! one-way delay is 0.5 light-years ≈ 6 months.
//!
//! # References
//!
//! - Smith, C.M. (2014). Estimation of a genetically viable population
//!   for multigenerational interstellar voyaging. Acta Astronautica.
//! - Marin, F. & Beluffi, C. (2018). Computing the minimal crew for
//!   a multi-generational space journey. JBIS.
//! - Henrich, J. (2004). Demography and cultural evolution: How adaptive
//!   cultural processes can produce maladaptive losses. American Antiquity.

use crate::epistemic_decay::EpistemicState;
use crate::relativistic_dht::RelativisticReconciliation;
use crate::stochastic::StochasticEngine;
use crate::stoichiometry::ElementalLedger;
use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Speed of light in km/s.
const _C_KM_S: f64 = 299_792.458;
/// One light-year in km.
const _LY_KM: f64 = 9.461e12;
/// Ticks per year (12 monthly ticks).
const TICKS_PER_YEAR: f64 = 12.0;
/// Cosmic ray flux multiplier outside heliosphere.
const INTERSTELLAR_COSMIC_RAY_MULTIPLIER: f64 = 3.0;
/// Hull degradation per tick from micrometeorites (fraction).
const HULL_DEGRADATION_PER_TICK: f64 = 0.00005; // ~0.6%/year
/// Hull degradation from cosmic rays (additional, per tick).
const HULL_COSMIC_RAY_DEGRADATION: f64 = 0.00002;
/// Cultural speciation threshold: ticks of isolation before cultures diverge irreversibly.
const CULTURAL_SPECIATION_TICKS: u32 = 600; // ~50 years
/// Minimum viable population for genetic diversity (Smith 2014).
const MINIMUM_VIABLE_POPULATION: usize = 98;
/// Minimum viable population for cultural transmission (Henrich 2004).
const HENRICH_THRESHOLD: usize = 200;

// ============================================================================
// INTERSTELLAR DESTINATIONS
// ============================================================================

/// Target star system for the generation ship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterstellarDestination {
    /// Proxima Centauri: 4.24 ly, potentially habitable planet Proxima b.
    ProximaCentauri,
    /// Barnard's Star: 5.96 ly, cold super-Earth candidate.
    BarnardsStar,
    /// Wolf 359: 7.86 ly, red dwarf with known exoplanets.
    Wolf359,
    /// Custom destination.
    Custom { name: String, distance_ly: f64 },
}

impl InterstellarDestination {
    pub fn name(&self) -> &str {
        match self {
            Self::ProximaCentauri => "Proxima Centauri",
            Self::BarnardsStar => "Barnard's Star",
            Self::Wolf359 => "Wolf 359",
            Self::Custom { name, .. } => name,
        }
    }

    pub fn distance_ly(&self) -> f64 {
        match self {
            Self::ProximaCentauri => 4.24,
            Self::BarnardsStar => 5.96,
            Self::Wolf359 => 7.86,
            Self::Custom { distance_ly, .. } => *distance_ly,
        }
    }
}

// ============================================================================
// SHIP STATUS
// ============================================================================

/// Current phase of the generation ship's journey.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShipPhase {
    /// Under construction, not yet launched.
    Construction,
    /// Accelerating to cruise velocity.
    Acceleration,
    /// Cruising at constant velocity.
    Cruise,
    /// Decelerating for arrival.
    Deceleration,
    /// Arrived at destination, colony founding.
    Arrived,
}

/// Interstellar disaster types unique to generation ships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterstellarDisaster {
    /// Galactic cosmic ray burst (outside heliosphere protection).
    CosmicRayBurst { severity: f64 },
    /// Micrometeorite impact on hull.
    MicrometeoiteImpact { hull_damage: f64 },
    /// Navigation drift requiring course correction fuel.
    NavigationDrift { correction_fuel_fraction: f64 },
    /// Knowledge attrition — critical skills lost (Henrich effect).
    KnowledgeAttrition { skill_loss_fraction: f64 },
    /// Social fracture — factions form during isolation.
    SocialFracture { cohesion_loss: f64 },
}

// ============================================================================
// GENERATION SHIP
// ============================================================================

/// A generation ship: a mobile World transiting between star systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationShip {
    /// The World contained within the ship (agents, economy, governance, etc.).
    pub world_id: u32,
    /// Target star system.
    pub destination: InterstellarDestination,
    /// Cruise velocity as fraction of c (0.01-0.10 realistic).
    pub cruise_velocity_c: f64,
    /// Distance to destination in light-years.
    pub distance_ly: f64,
    /// Total transit time in ticks.
    pub transit_ticks: u32,
    /// Ticks elapsed in transit.
    pub elapsed_ticks: u32,
    /// Current phase.
    pub phase: ShipPhase,
    /// Tick when ship was launched.
    pub launch_tick: u32,
    /// One-way communication delay to Sol in ticks (grows with distance).
    pub comms_delay_ticks: u32,
    /// Fuel remaining as fraction (1.0 = full, 0.0 = empty).
    pub fuel_fraction: f64,
    /// Hull structural integrity (1.0 = perfect, 0.0 = catastrophic failure).
    pub hull_integrity: f64,
    /// Cosmic ray accumulated dose (Sieverts equivalent, abstract).
    pub cosmic_ray_dose: f64,
    /// Whether cultural speciation has occurred.
    pub culturally_speciated: bool,
    /// Ticks of total isolation from Sol.
    pub isolation_ticks: u32,
    /// Disasters survived during transit.
    pub disasters_survived: Vec<InterstellarDisaster>,
    /// Population at launch (for minimum viable tracking).
    pub launch_population: usize,
    /// Stoichiometric mass ledger — conservation of C/H/O/N/P.
    pub mass_ledger: ElementalLedger,
    /// Relativistic timestamp reconciliation for Holochain DHT.
    pub relativistic: RelativisticReconciliation,
    /// Epistemic decay: knowledge degrades from radiation and isolation.
    pub epistemic: EpistemicState,
}

impl GenerationShip {
    /// Create a new generation ship bound for a destination.
    pub fn new(
        world_id: u32,
        destination: InterstellarDestination,
        cruise_velocity_c: f64,
        launch_tick: u32,
        launch_population: usize,
    ) -> Self {
        let distance_ly = destination.distance_ly();
        let transit_years = distance_ly / cruise_velocity_c;
        let transit_ticks = (transit_years * TICKS_PER_YEAR) as u32;

        Self {
            world_id,
            destination,
            cruise_velocity_c,
            distance_ly,
            transit_ticks,
            elapsed_ticks: 0,
            phase: ShipPhase::Acceleration,
            launch_tick,
            comms_delay_ticks: 0,
            fuel_fraction: 1.0,
            hull_integrity: 1.0,
            cosmic_ray_dose: 0.0,
            culturally_speciated: false,
            isolation_ticks: 0,
            disasters_survived: Vec::new(),
            launch_population,
            mass_ledger: ElementalLedger::for_ship(launch_population, transit_years),
            relativistic: RelativisticReconciliation::new(cruise_velocity_c),
            epistemic: EpistemicState::default(),
        }
    }

    /// Tick the ship forward one month.
    ///
    /// Returns any interstellar disasters that occurred this tick.
    pub fn tick(&mut self, rng: &mut StochasticEngine) -> Vec<InterstellarDisaster> {
        if self.phase == ShipPhase::Arrived || self.phase == ShipPhase::Construction {
            return vec![];
        }

        self.elapsed_ticks += 1;
        self.isolation_ticks += 1;
        let mut disasters = Vec::new();

        // Phase transitions
        let accel_ticks = (self.transit_ticks as f64 * 0.1) as u32; // 10% acceleration
        let decel_ticks = (self.transit_ticks as f64 * 0.1) as u32; // 10% deceleration
        if self.elapsed_ticks <= accel_ticks {
            self.phase = ShipPhase::Acceleration;
            self.fuel_fraction -= 0.4 / accel_ticks as f64; // 40% fuel for acceleration
        } else if self.elapsed_ticks >= self.transit_ticks - decel_ticks {
            self.phase = ShipPhase::Deceleration;
            self.fuel_fraction -= 0.4 / decel_ticks as f64; // 40% fuel for deceleration
        } else {
            self.phase = ShipPhase::Cruise;
        }

        // Arrival check
        if self.elapsed_ticks >= self.transit_ticks {
            self.phase = ShipPhase::Arrived;
            return disasters;
        }

        // Communication delay grows with distance
        let distance_traveled_ly =
            self.cruise_velocity_c * (self.elapsed_ticks as f64 / TICKS_PER_YEAR);
        self.comms_delay_ticks = (distance_traveled_ly * TICKS_PER_YEAR) as u32; // 1 tick per light-month

        // Hull degradation
        self.hull_integrity -= HULL_DEGRADATION_PER_TICK;
        self.hull_integrity -= HULL_COSMIC_RAY_DEGRADATION;
        self.hull_integrity = self.hull_integrity.max(0.0);

        // Cosmic ray accumulation (higher outside heliosphere)
        let heliosphere_crossed = self.elapsed_ticks > 24; // ~2 years to exit heliosphere
        let ray_multiplier = if heliosphere_crossed {
            INTERSTELLAR_COSMIC_RAY_MULTIPLIER
        } else {
            1.0
        };
        self.cosmic_ray_dose += 0.001 * ray_multiplier;

        // Cultural speciation check
        if !self.culturally_speciated && self.isolation_ticks >= CULTURAL_SPECIATION_TICKS {
            self.culturally_speciated = true;
        }

        // Stoichiometric mass ledger: tick ECLSS chemistry
        let pop_estimate = self.launch_population; // Approximate (real pop tracked externally)
        self.mass_ledger.tick(pop_estimate);

        // Relativistic clock drift
        let earth_years = self.elapsed_ticks as f64 / TICKS_PER_YEAR;
        self.relativistic.tick(earth_years);

        // Epistemic decay: knowledge degrades from cosmic rays and isolation
        let tech_levels = [1.5; 8]; // Approximate tech levels (real values from World)
        let energy_fraction = self.fuel_fraction.min(1.0); // Energy proxy
        let outside_helio = self.elapsed_ticks > 24; // ~2 years to exit heliosphere
        self.epistemic.tick(
            pop_estimate,
            self.cosmic_ray_dose,
            outside_helio,
            energy_fraction,
            &tech_levels,
        );

        // === INTERSTELLAR DISASTERS ===

        // Cosmic ray burst (1% chance per tick outside heliosphere)
        if heliosphere_crossed && rng.next_f64() < 0.01 {
            let severity = 0.3 + rng.next_f64() * 0.7;
            self.cosmic_ray_dose += severity * 0.01;
            disasters.push(InterstellarDisaster::CosmicRayBurst { severity });
        }

        // Micrometeorite impact (0.5% chance per tick)
        if rng.next_f64() < 0.005 {
            let damage = 0.001 + rng.next_f64() * 0.01;
            self.hull_integrity -= damage;
            self.hull_integrity = self.hull_integrity.max(0.0);
            // Hull breach loses atmosphere mass (stoichiometric consequence)
            self.mass_ledger.hull_breach(damage);
            disasters.push(InterstellarDisaster::MicrometeoiteImpact {
                hull_damage: damage,
            });
        }

        // Navigation drift (0.2% chance per tick during cruise)
        if self.phase == ShipPhase::Cruise && rng.next_f64() < 0.002 {
            let fuel_cost = 0.001 + rng.next_f64() * 0.005;
            self.fuel_fraction -= fuel_cost;
            self.fuel_fraction = self.fuel_fraction.max(0.0);
            disasters.push(InterstellarDisaster::NavigationDrift {
                correction_fuel_fraction: fuel_cost,
            });
        }

        // Knowledge attrition (Henrich effect — 0.3% chance per tick)
        if rng.next_f64() < 0.003 {
            let loss = 0.005 + rng.next_f64() * 0.02;
            disasters.push(InterstellarDisaster::KnowledgeAttrition {
                skill_loss_fraction: loss,
            });
        }

        // Social fracture (0.1% chance per tick, increases with isolation)
        let fracture_prob = 0.001 * (1.0 + self.isolation_ticks as f64 / 600.0);
        if rng.next_f64() < fracture_prob {
            let loss = 0.01 + rng.next_f64() * 0.05;
            disasters.push(InterstellarDisaster::SocialFracture {
                cohesion_loss: loss,
            });
        }

        for d in &disasters {
            self.disasters_survived.push(d.clone());
        }

        disasters
    }

    /// Fraction of transit completed (0.0 to 1.0).
    pub fn transit_progress(&self) -> f64 {
        if self.transit_ticks == 0 {
            return 0.0;
        }
        (self.elapsed_ticks as f64 / self.transit_ticks as f64).min(1.0)
    }

    /// Estimated years remaining until arrival.
    pub fn years_remaining(&self) -> f64 {
        let remaining_ticks = self.transit_ticks.saturating_sub(self.elapsed_ticks);
        remaining_ticks as f64 / TICKS_PER_YEAR
    }

    /// Whether the ship population is below minimum viable.
    pub fn below_minimum_viable(&self, current_pop: usize) -> bool {
        current_pop < MINIMUM_VIABLE_POPULATION
    }

    /// Whether the ship is experiencing the Henrich effect (knowledge loss).
    pub fn henrich_risk(&self, current_pop: usize) -> bool {
        current_pop < HENRICH_THRESHOLD
    }

    /// Communication round-trip delay in years.
    pub fn comms_roundtrip_years(&self) -> f64 {
        2.0 * self.comms_delay_ticks as f64 / TICKS_PER_YEAR
    }

    /// Whether communication with Sol is effectively impossible (>10 year round trip).
    pub fn comms_impossible(&self) -> bool {
        self.comms_roundtrip_years() > 10.0
    }

    /// Generate a status report string.
    pub fn status_report(&self, current_pop: usize) -> String {
        let mut result = format!(
            "=== GENERATION SHIP STATUS ===\n\
             Destination: {} ({:.2} ly)\n\
             Phase: {:?}\n\
             Progress: {:.1}% ({:.1} years remaining)\n\
             Population: {} (launched with {})\n\
             Hull integrity: {:.1}%\n\
             Fuel remaining: {:.1}%\n\
             Cosmic ray dose: {:.3} Sv\n\
             Comms delay: {:.1} years (round trip)\n\
             Isolation: {:.1} years\n\
             Cultural speciation: {}\n\
             Below minimum viable: {}\n\
             Henrich risk: {}\n\
             Disasters survived: {}",
            self.destination.name(),
            self.distance_ly,
            self.phase,
            self.transit_progress() * 100.0,
            self.years_remaining(),
            current_pop,
            self.launch_population,
            self.hull_integrity * 100.0,
            self.fuel_fraction * 100.0,
            self.cosmic_ray_dose,
            self.comms_roundtrip_years(),
            self.isolation_ticks as f64 / TICKS_PER_YEAR,
            if self.culturally_speciated {
                "YES"
            } else {
                "No"
            },
            if self.below_minimum_viable(current_pop) {
                "YES — CRITICAL"
            } else {
                "No"
            },
            if self.henrich_risk(current_pop) {
                "YES — knowledge loss risk"
            } else {
                "No"
            },
            self.disasters_survived.len(),
        );
        result.push_str(&format!(
            "\nMass Ledger:\n  C: {:.0} kg | N: {:.0} kg | O: {:.0} kg | P: {:.0} kg\n  Water: {:.0} kg | Lost to void: {:.1} kg\n  Limiting element: {} | Max sustainable: {}\n  Recycling efficiency: {:.1}%\n\
             Relativistic:\n  {}\n\
             Epistemic:\n  {}\n",
            self.mass_ledger.carbon_kg,
            self.mass_ledger.nitrogen_kg,
            self.mass_ledger.oxygen_kg,
            self.mass_ledger.phosphorus_kg,
            self.mass_ledger.water_kg,
            self.mass_ledger.mass_lost_kg,
            self.mass_ledger.limiting_element(),
            self.mass_ledger.max_sustainable_population,
            self.mass_ledger.recycling_efficiency * 100.0,
            self.relativistic.summary(),
            self.epistemic.summary(),
        ));
        result
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> StochasticEngine {
        StochasticEngine::new(42)
    }

    #[test]
    fn proxima_transit_time() {
        let ship = GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        // 4.24 ly / 0.05c = 84.8 years = ~1018 ticks
        assert!(
            (ship.transit_ticks as f64 - 1017.6).abs() < 2.0,
            "Transit ticks: {}",
            ship.transit_ticks
        );
    }

    #[test]
    fn barnards_transit_time() {
        let ship = GenerationShip::new(1, InterstellarDestination::BarnardsStar, 0.05, 0, 500);
        // 5.96 ly / 0.05c = 119.2 years = ~1430 ticks
        assert!(
            ship.transit_ticks > 1400 && ship.transit_ticks < 1500,
            "Transit ticks: {}",
            ship.transit_ticks
        );
    }

    #[test]
    fn ship_arrives() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        let total = ship.transit_ticks;
        for _ in 0..total + 10 {
            ship.tick(&mut rng);
        }
        assert_eq!(ship.phase, ShipPhase::Arrived);
        assert!((ship.transit_progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn hull_degrades() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        let initial = ship.hull_integrity;
        for _ in 0..120 {
            // 10 years
            ship.tick(&mut rng);
        }
        assert!(
            ship.hull_integrity < initial,
            "Hull should degrade: {} → {}",
            initial,
            ship.hull_integrity
        );
        assert!(
            ship.hull_integrity > 0.9,
            "Hull shouldn't fail in 10 years: {}",
            ship.hull_integrity
        );
    }

    #[test]
    fn comms_delay_grows() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        for _ in 0..120 {
            // 10 years
            ship.tick(&mut rng);
        }
        assert!(ship.comms_delay_ticks > 0, "Comms delay should grow");
        assert!(
            ship.comms_roundtrip_years() > 0.5,
            "Should have >6 month RT delay at 10 years"
        );
    }

    #[test]
    fn cultural_speciation_at_50_years() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        assert!(!ship.culturally_speciated);
        for _ in 0..CULTURAL_SPECIATION_TICKS + 10 {
            ship.tick(&mut rng);
        }
        assert!(
            ship.culturally_speciated,
            "Should speciate after 50 years isolation"
        );
    }

    #[test]
    fn fuel_consumed_during_accel_decel() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        let initial_fuel = ship.fuel_fraction;
        // Run through full transit
        for _ in 0..ship.transit_ticks + 1 {
            ship.tick(&mut rng);
        }
        assert!(ship.fuel_fraction < initial_fuel, "Should consume fuel");
        // Navigation drift may consume additional fuel
        assert!(
            ship.fuel_fraction > -0.1,
            "Shouldn't go deeply negative: {}",
            ship.fuel_fraction
        );
    }

    #[test]
    fn disasters_accumulate() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        for _ in 0..600 {
            // 50 years
            ship.tick(&mut rng);
        }
        assert!(
            !ship.disasters_survived.is_empty(),
            "Should have some disasters in 50 years"
        );
    }

    #[test]
    fn minimum_viable_population() {
        let ship = GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        assert!(!ship.below_minimum_viable(500));
        assert!(!ship.below_minimum_viable(100));
        assert!(ship.below_minimum_viable(50));
    }

    #[test]
    fn henrich_threshold() {
        let ship = GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        assert!(!ship.henrich_risk(500));
        assert!(!ship.henrich_risk(200));
        assert!(ship.henrich_risk(150));
    }

    #[test]
    fn comms_impossible_at_distance() {
        let mut ship =
            GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let mut rng = make_rng();
        // At 0.05c, after ~60 years (720 ticks), distance > 3 ly, RT > 6 years
        for _ in 0..900 {
            // 75 years
            ship.tick(&mut rng);
        }
        assert!(
            ship.comms_roundtrip_years() > 5.0,
            "RT delay: {} years",
            ship.comms_roundtrip_years()
        );
    }

    #[test]
    fn status_report_generates() {
        let ship = GenerationShip::new(1, InterstellarDestination::ProximaCentauri, 0.05, 0, 500);
        let report = ship.status_report(480);
        assert!(report.contains("Proxima Centauri"));
        assert!(report.contains("480"));
    }

    #[test]
    fn custom_destination() {
        let ship = GenerationShip::new(
            1,
            InterstellarDestination::Custom {
                name: "Alpha Centauri A".to_string(),
                distance_ly: 4.37,
            },
            0.10,
            0,
            1000,
        );
        assert_eq!(ship.destination.name(), "Alpha Centauri A");
        // 4.37 ly / 0.10c = 43.7 years = ~524 ticks
        assert!(
            ship.transit_ticks > 500 && ship.transit_ticks < 550,
            "Transit ticks: {}",
            ship.transit_ticks
        );
    }
}
