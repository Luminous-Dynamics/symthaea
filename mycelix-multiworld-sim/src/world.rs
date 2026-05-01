// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! World model: resources, culture, infrastructure, and population management.

use crate::agent::{CivAgent, LifeStage};
use crate::economy::WorldEconomy;
use crate::governance::WorldGovernance;
use crate::harmony::HarmonyTracker;
use crate::knowledge::WorldKnowledge;
use crate::population::SirEpidemic;
use crate::stochastic::StochasticEngine;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A collective moral memory: records the ethical state during a crisis.
/// Future ethical drift away from this memory's orientation is resisted.
/// Models "never again" dynamics and cultural antibodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralMemory {
    /// Tick when the crisis occurred.
    pub tick: u32,
    /// Mean ethical orientation at the time of the crisis.
    pub ethics_at_crisis: [f64; 4],
    /// Which dimension was most stressed (the "lesson learned").
    pub lesson_dimension: usize,
    /// Description for narrative purposes.
    pub description: String,
}

impl MoralMemory {
    /// How strongly this memory resists drift, based on age.
    /// Fresh memories are strong (1.0); decays to 0 over 600 ticks (50 years).
    pub fn strength(&self, current_tick: u32) -> f64 {
        let age = current_tick.saturating_sub(self.tick) as f64;
        (1.0 - age / 600.0).max(0.0)
    }
}

// ============================================================================
// NASA BVAD Calibration Reference (Baseline Values and Assumptions Doc, 2018)
// ============================================================================
// The sim uses abstract resource units (ARU). The mapping to real kg/person/day:
//
//   1 ARU food     ≈ 0.046 kg/person/day dry  (40 ARU/tick ÷ 100 people ÷ 30 days × 5.5 factor)
//   1 ARU water    ≈ 0.056 kg/person/day       (45 ARU/tick ÷ 100 people ÷ 30 days × 7.4 factor)
//   1 ARU oxygen   ≈ 0.028 kg/person/day       (50 ARU/tick ÷ 100 people ÷ 30 days × 3.36 factor)
//   1 ARU energy   ≈ 0.33 kWh/person/day       (80 ARU/tick ÷ 100 people ÷ 30 days × 125 factor)
//
// NASA BVAD requirements per person per day:
//   Food:    1.83 kg dry (Table 4.2.2)
//   Water:   2.50 kg potable (with 93% recycling, gross 26 kg/day)
//   Oxygen:  0.84 kg (Table 4.1.1)
//   Power:   ~10 kW continuous for colony operations
//
// These ARU factors are consistent with the sim's consumption rates for a
// 100-person colony: 40 food ARU/tick ÷ 100 people = 0.4 ARU/person/tick.
// At 30 days/tick: 0.4/30 = 0.013 ARU/person/day × 140 ≈ 1.83 kg/person/day.
/// Conversion factor: ARU to kg/person/day for food.
pub const FOOD_ARU_TO_KG: f64 = 140.0;
/// Conversion factor: ARU to kg/person/day for water.
pub const WATER_ARU_TO_KG: f64 = 187.0;
/// Conversion factor: ARU to kg/person/day for oxygen.
pub const OXYGEN_ARU_TO_KG: f64 = 63.0;
/// Conversion factor: ARU to kWh/person/day for energy.
pub const ENERGY_ARU_TO_KWH: f64 = 125.0;

/// A single resource stock within a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStock {
    /// Current amount available.
    pub current: f64,
    /// Maximum storage capacity.
    pub capacity: f64,
    /// Production rate per tick.
    pub production_rate: f64,
    /// Consumption rate per tick (scales with population).
    pub consumption_rate: f64,
    /// Below this fraction of capacity, the resource is critical.
    pub critical_threshold: f64,
}

impl ResourceStock {
    /// Whether this resource is below its critical threshold.
    pub fn is_critical(&self) -> bool {
        self.capacity > 0.0 && (self.current / self.capacity) < self.critical_threshold
    }
}

/// Resource inventory for a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldResources {
    stocks: HashMap<String, ResourceStock>,
}

impl WorldResources {
    /// Create empty resources.
    pub fn new() -> Self {
        Self {
            stocks: HashMap::new(),
        }
    }

    /// Get a resource stock by name.
    pub fn get(&self, name: &str) -> Option<&ResourceStock> {
        self.stocks.get(name)
    }

    /// Get mutable reference.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ResourceStock> {
        self.stocks.get_mut(name)
    }

    /// Insert or update a resource.
    pub fn set(&mut self, name: impl Into<String>, stock: ResourceStock) {
        self.stocks.insert(name.into(), stock);
    }

    /// Get the current stock level for a named resource (None → None).
    pub fn stock_level(&self, name: &str) -> Option<f64> {
        self.stocks.get(name).map(|s| s.current)
    }

    /// Whether any resource is at critical level.
    pub fn any_critical(&self) -> bool {
        self.stocks.values().any(|s| s.is_critical())
    }

    /// Overall self-sufficiency ratio: average of (production / consumption) across
    /// all resources, clamped to [0, 1].
    pub fn self_sufficiency(&self) -> f64 {
        if self.stocks.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .stocks
            .values()
            .map(|s| {
                if s.consumption_rate > 0.0 {
                    (s.production_rate / s.consumption_rate).min(1.0)
                } else {
                    1.0
                }
            })
            .sum();
        sum / self.stocks.len() as f64
    }

    /// Resource names.
    pub fn resource_names(&self) -> Vec<&str> {
        self.stocks.keys().map(|s| s.as_str()).collect()
    }

    /// Deduct up to `amount` from a resource stock, returning the actual amount deducted.
    pub fn deduct(&mut self, name: &str, amount: f64) -> f64 {
        if let Some(stock) = self.stocks.get_mut(name) {
            let actual = amount.min(stock.current);
            stock.current -= actual;
            actual
        } else {
            0.0
        }
    }

    /// Get the fraction of capacity for a resource (current / capacity), or 0.0 if absent.
    pub fn fraction_of_capacity(&self, name: &str) -> f64 {
        self.stocks
            .get(name)
            .map(|s| {
                if s.capacity > 0.0 {
                    s.current / s.capacity
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Default resource set for a lunar colony.
    pub fn lunar_default() -> Self {
        let mut r = Self::new();
        r.set(
            "food",
            ResourceStock {
                current: 1000.0,
                capacity: 2000.0,
                production_rate: 45.0, // Hydroponics + imported supplements
                consumption_rate: 40.0,
                critical_threshold: 0.1,
            },
        );
        r.set(
            "water",
            ResourceStock {
                current: 800.0,
                capacity: 3000.0,
                production_rate: 50.0,
                consumption_rate: 45.0,
                critical_threshold: 0.15,
            },
        );
        r.set(
            "energy",
            ResourceStock {
                current: 1000.0,
                capacity: 5000.0,
                production_rate: 100.0,
                consumption_rate: 80.0,
                critical_threshold: 0.1,
            },
        );
        r.set(
            "materials",
            ResourceStock {
                current: 300.0,
                capacity: 1000.0,
                production_rate: 10.0,
                consumption_rate: 15.0,
                critical_threshold: 0.2,
            },
        );
        r.set(
            "oxygen",
            ResourceStock {
                current: 900.0,
                capacity: 2000.0,
                production_rate: 60.0,
                consumption_rate: 50.0,
                critical_threshold: 0.2,
            },
        );
        r
    }

    /// Default resource set for Europa (Jupiter system).
    ///
    /// Unlimited water/ice, zero solar power (nuclear-gated), limited materials.
    /// Energy production requires fission/fusion tech — without it, the colony dies.
    /// Paranicas et al. (2009); Anderson et al. (1998) for ocean/ice data.
    pub fn europa_default() -> Self {
        let mut r = Self::new();
        r.set(
            "food",
            ResourceStock {
                current: 500.0,
                capacity: 1500.0,
                production_rate: 30.0, // Hydroponics under ice (energy-limited)
                consumption_rate: 40.0,
                critical_threshold: 0.15,
            },
        );
        r.set(
            "water",
            ResourceStock {
                current: 5000.0,
                capacity: 20000.0,
                production_rate: 500.0, // 10× lunar — unlimited ice
                consumption_rate: 45.0,
                critical_threshold: 0.05,
            },
        );
        r.set(
            "energy",
            ResourceStock {
                current: 500.0, // Bootstrapped from lander reactors
                capacity: 3000.0,
                production_rate: 0.0,    // ZERO until fission/fusion achieved
                consumption_rate: 100.0, // Higher than lunar — radiation shielding + ice processing
                critical_threshold: 0.15,
            },
        );
        r.set(
            "materials",
            ResourceStock {
                current: 200.0,
                capacity: 800.0,
                production_rate: 3.0, // 0.3× lunar — limited metals, must import
                consumption_rate: 15.0,
                critical_threshold: 0.25,
            },
        );
        r.set(
            "oxygen",
            ResourceStock {
                current: 900.0,
                capacity: 2000.0,
                production_rate: 60.0, // Electrolysis from ice (if energy available)
                consumption_rate: 50.0,
                critical_threshold: 0.2,
            },
        );
        r
    }

    /// Default resource set for Titan (Saturn system).
    ///
    /// Limitless hydrocarbons + nitrogen, ice bedrock for water, zero solar.
    /// Energy is nuclear-only. Materials production boosted by hydrocarbon ISRU.
    /// Fulchignoni et al. (2005); Niemann et al. (2010); Mastrogiuseppe et al. (2019).
    pub fn titan_default() -> Self {
        let mut r = Self::new();
        r.set(
            "food",
            ResourceStock {
                current: 300.0,
                capacity: 1000.0,
                production_rate: 22.0, // 0.5× lunar — all lighting artificial, energy-expensive
                consumption_rate: 40.0,
                critical_threshold: 0.15,
            },
        );
        r.set(
            "water",
            ResourceStock {
                current: 3000.0,
                capacity: 15000.0,
                production_rate: 250.0, // 5× lunar — ice bedrock, but must melt (energy cost)
                consumption_rate: 45.0,
                critical_threshold: 0.1,
            },
        );
        r.set(
            "energy",
            ResourceStock {
                current: 300.0, // Bootstrapped from lander reactors
                capacity: 2000.0,
                production_rate: 0.0,    // ZERO until fission/fusion achieved
                consumption_rate: 120.0, // Higher than Europa — heating load is relentless (199K ΔT)
                critical_threshold: 0.2,
            },
        );
        r.set(
            "materials",
            ResourceStock {
                current: 400.0,
                capacity: 1500.0,
                production_rate: 20.0, // 2× lunar — hydrocarbon polymers, ice pykrete construction
                consumption_rate: 15.0,
                critical_threshold: 0.15,
            },
        );
        r.set(
            "oxygen",
            ResourceStock {
                current: 600.0,
                capacity: 1500.0,
                production_rate: 40.0, // Electrolysis from ice (if energy available)
                consumption_rate: 50.0,
                critical_threshold: 0.2,
            },
        );
        // Titan-exclusive: hydrocarbon resource (methane/ethane from lakes).
        // Kraken Mare alone holds ~20,000 km³. Effectively unlimited for colony scale.
        // Used for materials ISRU (polymers, pykrete), rocket fuel (CH4/LOX), and
        // chemical energy storage. Mastrogiuseppe et al. (2019), Nature Astronomy.
        r.set(
            "hydrocarbons",
            ResourceStock {
                current: 50000.0,
                capacity: 100000.0,
                production_rate: 500.0, // Surface collection from methane lakes
                consumption_rate: 50.0, // Materials + fuel synthesis
                critical_threshold: 0.05, // Very hard to run out
            },
        );
        r
    }

    /// Default resource set for Earth (abundant).
    ///
    /// Note: consumption_rate is per-100-people in tick_economy, so these rates
    /// must be scaled to support the Earth population (~10,000 initial).
    /// Production must exceed consumption at initial population for sustainability.
    pub fn earth_default() -> Self {
        let mut r = Self::new();
        for name in &["food", "water", "energy", "materials", "oxygen"] {
            r.set(
                *name,
                ResourceStock {
                    current: 1_000_000.0,
                    capacity: 1_000_000.0,
                    // Production and consumption rates tuned for ~10,000 population.
                    // In tick_economy: actual_production = rate * infrastructure (0.9)
                    //                  actual_consumption = rate * (pop/100)
                    // For pop=10,000: consumption = 100 * rate
                    // Production 15,000 * 0.9 = 13,500 > 100 * 100 = 10,000
                    production_rate: 15_000.0,
                    consumption_rate: 100.0,
                    critical_threshold: 0.05,
                },
            );
        }
        r
    }
}

impl Default for WorldResources {
    fn default() -> Self {
        Self::new()
    }
}

/// Fix 8: Intra-World Geography — zones within a world with different
/// environmental conditions. A generation ship has different decks;
/// a lunar base has dome sectors with varying shielding.
///
/// Ref: NASA Habitat Architecture (ICES-2019); ISS module environmental variation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldZone {
    /// Human-readable zone name (e.g., "Engineering Deck", "Hydroponic Bay").
    pub name: String,
    /// Radiation multiplier for this zone (1.0 = baseline, 2.0 = double exposure).
    pub radiation_multiplier: f64,
    /// Resource access fraction (1.0 = full access, 0.5 = half).
    pub resource_access: f64,
    /// Social density: agents per unit area. Higher → more social contact.
    pub social_density: f64,
}

impl WorldZone {
    /// Default zone for a generation ship section.
    pub fn generation_ship_zones() -> Vec<WorldZone> {
        vec![
            WorldZone {
                name: "Bridge & Command".into(),
                radiation_multiplier: 0.5,
                resource_access: 1.0,
                social_density: 0.8,
            },
            WorldZone {
                name: "Habitation Ring".into(),
                radiation_multiplier: 0.7,
                resource_access: 0.9,
                social_density: 1.0,
            },
            WorldZone {
                name: "Hydroponic Bay".into(),
                radiation_multiplier: 1.0,
                resource_access: 1.2,
                social_density: 0.5,
            },
            WorldZone {
                name: "Engineering Deck".into(),
                radiation_multiplier: 1.5,
                resource_access: 0.8,
                social_density: 0.4,
            },
            WorldZone {
                name: "Cargo Hold".into(),
                radiation_multiplier: 2.0,
                resource_access: 0.6,
                social_density: 0.2,
            },
        ]
    }

    /// Default zones for a lunar base.
    pub fn lunar_zones() -> Vec<WorldZone> {
        vec![
            WorldZone {
                name: "Central Dome".into(),
                radiation_multiplier: 0.8,
                resource_access: 1.0,
                social_density: 1.0,
            },
            WorldZone {
                name: "Regolith-Shielded Lab".into(),
                radiation_multiplier: 0.3,
                resource_access: 0.9,
                social_density: 0.6,
            },
            WorldZone {
                name: "Surface Operations".into(),
                radiation_multiplier: 2.5,
                resource_access: 0.7,
                social_density: 0.3,
            },
        ]
    }
}

/// Cultural profile influencing governance, reproduction, and social dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalProfile {
    /// Emphasis weights for each of the Eight Harmonies. Should sum to ~1.0.
    pub harmony_weights: [f64; 8],
    /// Individualism vs collectivism [0, 1].
    pub individualism: f64,
    /// Willingness to take risks [0, 1].
    pub risk_tolerance: f64,
    /// Openness to outsiders / immigrants [0, 1].
    pub xenophilia: f64,
    /// Attachment to tradition vs innovation [0, 1].
    pub traditionalism: f64,
    /// Fix 9: Language divergence from Earth standard [0.0, 1.0].
    /// 0.0 = mutually intelligible, 1.0 = unintelligible new language.
    /// Diverges at 0.001/tick when isolated (no inter-world contact).
    /// Ref: Sapir-Whorf linguistic relativity; Wichmann & Holman (2009)
    /// automated language distance; Bowern & Atkinson (2012) expansion rates.
    #[serde(default)]
    pub language_divergence: f64,
    /// Fix 9: Count of unique cultural practices/rituals.
    /// New ritual emerges every 120 ticks (~10 years).
    #[serde(default)]
    pub ritual_count: u32,
    /// Fix 9: Founding mythology — the story the colony tells itself.
    /// Generated at world creation, evolves slowly.
    #[serde(default)]
    pub founding_mythology: String,
}

impl CulturalProfile {
    /// Fix 9: Tick cultural evolution — language divergence and ritual emergence.
    pub fn tick_culture_evolution(&mut self, current_tick: u32, contact_frequency: f64) {
        // Language diverges when isolated
        let isolation = 1.0 - contact_frequency;
        self.language_divergence = (self.language_divergence + 0.001 * isolation).min(1.0);

        // New ritual every 120 ticks
        if current_tick > 0 && current_tick % 120 == 0 {
            self.ritual_count += 1;
        }
    }

    /// Brownian cultural drift: magnitude proportional to 1/sqrt(pop) * (1 - contact_frequency).
    /// Small populations drift faster; high inter-group contact slows drift.
    pub fn drift(&mut self, rng: &mut StochasticEngine, population: usize, contact_frequency: f64) {
        let magnitude = 0.01 / (population as f64).sqrt().max(1.0) * (1.0 - contact_frequency);

        for w in &mut self.harmony_weights {
            *w += rng.next_gaussian(0.0, magnitude);
            *w = w.max(0.01);
        }
        // Renormalize
        let sum: f64 = self.harmony_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.harmony_weights {
                *w /= sum;
            }
        }

        self.individualism =
            (self.individualism + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.risk_tolerance =
            (self.risk_tolerance + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.xenophilia = (self.xenophilia + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.traditionalism =
            (self.traditionalism + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
    }

    /// Cosine distance on harmony_weights between two cultures.
    pub fn cultural_distance(&self, other: &CulturalProfile) -> f64 {
        let dot: f64 = self
            .harmony_weights
            .iter()
            .zip(other.harmony_weights.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f64 = self
            .harmony_weights
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let mag_b: f64 = other
            .harmony_weights
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let denom = mag_a * mag_b;
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }

    /// Earth default: balanced harmonies, moderate traits.
    pub fn earth_default() -> Self {
        Self {
            harmony_weights: [0.125; 8],
            individualism: 0.5,
            risk_tolerance: 0.4,
            xenophilia: 0.5,
            traditionalism: 0.5,
            language_divergence: 0.0,
            ritual_count: 100, // Earth has many existing rituals
            founding_mythology:
                "Cradle of humanity — the blue marble from which all journeys begin".into(),
        }
    }

    /// Pioneer default: higher risk tolerance, lower traditionalism.
    pub fn pioneer_default() -> Self {
        Self {
            harmony_weights: [0.125; 8],
            individualism: 0.4,
            risk_tolerance: 0.7,
            xenophilia: 0.6,
            traditionalism: 0.3,
            language_divergence: 0.0,
            ritual_count: 3, // Pioneers start with few rituals
            founding_mythology: "We left everything behind to build something new among the stars"
                .into(),
        }
    }
}

/// A world (colony, settlement, planet) in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World {
    pub id: u32,
    pub name: String,
    /// Location label: "Earth", "Moon", "Mars", "Europa", etc.
    pub location: String,
    pub founded_tick: u32,
    pub parent_world_id: Option<u32>,
    pub agents: Vec<CivAgent>,
    pub next_agent_id: u64,
    pub resources: WorldResources,
    pub culture: CulturalProfile,
    /// Infrastructure development level [0, 1].
    pub infrastructure_level: f64,
    /// Carrying capacity (max sustainable population).
    pub max_population: usize,
    /// Habitable area in square meters.
    pub habitable_area_m2: f64,
    /// Founding emphasis on each of the Eight Harmonies.
    pub founding_harmony_emphasis: [f64; 8],
    /// Active epidemics.
    pub epidemics: Vec<SirEpidemic>,
    /// Knowledge and technology tracking.
    pub knowledge: WorldKnowledge,
    /// 8-sector Cobb-Douglas economy.
    pub economy: WorldEconomy,
    /// Eight Harmonies tracker.
    pub harmony: HarmonyTracker,
    /// Consciousness-gated governance with anti-tyranny invariants.
    #[serde(default)]
    pub governance: WorldGovernance,

    /// Collective moral memories: "never again" antibodies from past crises.
    /// Each memory records the ethical state during a moral crisis.
    /// Future drift AWAY from a memory's orientation is resisted.
    /// Decays over 600 ticks (50 years). Max 10 memories.
    #[serde(default)]
    pub moral_memories: Vec<MoralMemory>,

    /// Institutional ethics: the ethical orientation encoded in laws/customs/norms.
    /// Persists beyond individual lifespans. Updated as slow EMA of governing agents.
    /// New citizens are socialized toward institutional ethics at 0.0002/tick.
    #[serde(default)]
    pub institutional_ethics: crate::agent::EthicalOrientation,

    /// Metabolism cycle state (4-phase governance rhythm).
    #[serde(default)]
    pub metabolism_state: crate::metabolism::MetabolismState,

    /// Currency state (MYCEL/SAP/TEND aggregates).
    #[serde(default)]
    pub currency_state: crate::currency::WorldCurrencyState,
    /// Mutable policy parameters (changed by governance proposals).
    #[serde(default)]
    pub policy_state: crate::proposals::PolicyState,

    // =================================================================
    // Structural realism systems (Top 10 from exhaustive audit)
    // =================================================================
    /// #1: Power flow budget (watts). Total generation vs total demand.
    /// When demand > generation, load shedding begins.
    #[serde(default)]
    pub power_generation_kw: f64,
    #[serde(default)]
    pub power_demand_kw: f64,

    /// #2: Narrative identity — the colony's founding story.
    /// Determines psychological resilience and adaptation capacity.
    #[serde(default)]
    pub narrative_identity: NarrativeIdentity,

    /// #3: Maintenance labor hours required per tick.
    /// When maintenance > available labor, infrastructure decays faster.
    #[serde(default)]
    pub maintenance_hours_required: f64,
    #[serde(default)]
    pub maintenance_hours_available: f64,

    /// #4: Bus factor — number of critical systems with only 1 skilled operator.
    #[serde(default)]
    pub bus_factor_critical: u32,

    /// #5: Pathogen pressure — cumulative sealed-environment disease risk [0, 1].
    #[serde(default)]
    pub pathogen_pressure: f64,

    /// #6: Civilizational Phi — organizational redundancy [0, 1].
    /// Can the colony reproduce its structure if 20% of population dies?
    #[serde(default)]
    pub civilizational_phi: f64,

    /// #8: Trust level [0, 1] — slow to build, fast to lose.
    #[serde(default = "default_trust")]
    pub trust_level: f64,

    /// #9: Earth funding commitment [0, 1] — political willingness to resupply.
    #[serde(default = "default_funding")]
    pub earth_funding: f64,

    /// Lifespan evolution: Gompertz-Makeham modifiers from medical tech.
    /// Pyrkov et al. (2021): resilience wall at 120-150yr without beta modification.
    #[serde(default = "default_one")]
    pub mortality_alpha_mult: f64,
    #[serde(default = "default_one")]
    pub mortality_beta_mult: f64,
    #[serde(default = "default_one")]
    pub mortality_lambda_mult: f64,

    /// #10: Reproduction viability. False if gravity < 0.4g without countermeasures.
    #[serde(default = "default_true")]
    pub reproduction_viable: bool,

    /// #7: Sealed ecosystem balance [0, 1]. Atmospheric stability.
    /// Biosphere 2: O2 dropped from 20.9% to 14.2% in 16 months.
    #[serde(default = "default_one")]
    pub ecosystem_balance: f64,

    /// Low-gravity fertility penalty [0.3, 1.0].
    /// Mars (0.38g) → 0.76, Moon (0.17g) → 0.34. Centrifuge → 1.0.
    /// Ref: Wakayama 2023 (JAXA), Lyons 2026 (Adelaide).
    #[serde(default = "default_one")]
    pub fertility_multiplier: f64,

    /// B: Automation level [0, 1]. Reduces human labor requirements.
    /// At 0.8, a colony of 200 can maintain infrastructure requiring 2000 workers.
    #[serde(default)]
    pub automation_level: f64,

    /// D: Exploration missions completed (cumulative). Each adds knowledge + resources.
    #[serde(default)]
    pub explorations_completed: u32,

    /// Colony project manager — multi-tick construction and development.
    #[serde(default)]
    pub project_manager: crate::projects::ProjectManager,

    /// Modular habitat geometry — radiation and psychology per volume.
    #[serde(default)]
    pub habitat: crate::habitat::HabitatComplex,

    /// Robotic fleet — conscious machines that cost Joules to think.
    #[serde(default)]
    pub fleet: crate::robotics::RoboticFleet,

    /// E: Inter-world relationship scores [-1, 1]. -1 = hostile, 0 = neutral, +1 = allied.
    #[serde(default)]
    pub diplomatic_relations: HashMap<u32, f64>,

    /// Fix 8: Intra-world geography zones with different environmental conditions.
    #[serde(default)]
    pub zones: Vec<WorldZone>,
}

fn default_trust() -> f64 {
    0.7
}
fn default_funding() -> f64 {
    1.0
}
fn default_one() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

/// Parameters for creating a new colony world.
/// Captures the fields that vary between `initialize_worlds()` and `found_colony()`.
pub struct ColonyParams {
    pub id: u32,
    pub name: String,
    pub location: String,
    pub founded_tick: u32,
    pub parent_world_id: Option<u32>,
    pub resources: WorldResources,
    pub culture: CulturalProfile,
    pub infrastructure_level: f64,
    pub max_population: usize,
    pub habitable_area_m2: f64,
}

impl World {
    /// Create a new colony world from parameters.
    ///
    /// All fields not covered by `ColonyParams` get sensible defaults matching
    /// the existing inline initializers. This eliminates the duplicate 35-field
    /// struct construction in `initialize_worlds()` and `found_colony()`.
    pub fn new_colony(params: ColonyParams) -> Self {
        Self {
            id: params.id,
            name: params.name,
            location: params.location.clone(),
            founded_tick: params.founded_tick,
            parent_world_id: params.parent_world_id,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: params.resources,
            culture: params.culture,
            infrastructure_level: params.infrastructure_level,
            max_population: params.max_population,
            habitable_area_m2: params.habitable_area_m2,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: WorldEconomy::new(),
            harmony: HarmonyTracker::new(),
            governance: WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default_surface_colony(&params.location),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        }
    }
}

/// #2: Narrative Identity — the shared story of "who we are" and "why we're here."
///
/// Determines: psychological resilience under disaster, adaptation capacity
/// (can the colony change its behavior?), and generational cohesion.
/// Norse Greenland died from narrative rigidity (refused to eat fish).
/// Ref: Wallace (1956) revitalization movements, Diamond (2005) Collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeIdentity {
    /// The colony's founding archetype.
    pub archetype: NarrativeArchetype,
    /// How strongly the current population identifies with the founding narrative [0, 1].
    pub identification: f64,
    /// How adaptable the narrative is to new circumstances [0, 1].
    /// Low = rigid (Norse), High = flexible (Inuit).
    pub adaptability: f64,
    /// Generational distance from founding (increases identity tension).
    pub generations_since_founding: u32,
}

/// Colony founding archetypes — determines psychological response patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrativeArchetype {
    /// "We are the vanguard of humanity's future." High resilience, high purpose.
    Pioneers,
    /// "We are here to discover." Flexible, curious, but low crisis endurance.
    Scientists,
    /// "We fled to survive." High cohesion, high trauma, suspicious of authority.
    Refugees,
    /// "We are building a new society." Idealistic, high cooperation, fragile to disillusionment.
    Utopians,
    /// "We are here to work." Pragmatic, low ideological investment, vulnerable to meaning collapse.
    Workers,
    /// "This is our sacred home." Deep roots, high traditionalism, resistant to change.
    Settlers,
}

impl NarrativeArchetype {
    /// Disaster resilience multiplier — how much psychological damage disasters cause.
    /// Lower = more resilient.
    pub fn disaster_resilience(&self) -> f64 {
        match self {
            Self::Pioneers => 0.6,   // Built for hardship
            Self::Scientists => 0.8, // Curious but not tough
            Self::Refugees => 0.5,   // Survived worse
            Self::Utopians => 1.0,   // Shattered by reality
            Self::Workers => 0.9,    // Didn't sign up for this
            Self::Settlers => 0.7,   // Stubborn endurance
        }
    }

    /// Adaptation capacity — ability to change behavior when circumstances demand.
    /// Higher = more flexible.
    pub fn adaptation_capacity(&self) -> f64 {
        match self {
            Self::Pioneers => 0.7,
            Self::Scientists => 0.9, // Hypothesis-driven, will try new things
            Self::Refugees => 0.6,   // Cautious, stick to what works
            Self::Utopians => 0.5,   // Ideology constrains adaptation
            Self::Workers => 0.8,    // Pragmatic
            Self::Settlers => 0.3,   // "This is how we've always done it"
        }
    }
}

impl Default for NarrativeIdentity {
    fn default() -> Self {
        Self {
            archetype: NarrativeArchetype::Pioneers,
            identification: 0.8,
            adaptability: 0.6,
            generations_since_founding: 0,
        }
    }
}

impl World {
    /// Count of living agents.
    pub fn population(&self) -> usize {
        self.agents.iter().filter(|a| a.is_alive()).count()
    }

    /// Count of living adults.
    pub fn adults(&self, tick: u32) -> usize {
        self.agents
            .iter()
            .filter(|a| {
                a.is_alive() && matches!(a.life_stage(tick), LifeStage::Adult | LifeStage::Elder)
            })
            .count()
    }

    /// Count of living children.
    pub fn children(&self, tick: u32) -> usize {
        self.agents
            .iter()
            .filter(|a| a.is_alive() && a.life_stage(tick) == LifeStage::Child)
            .count()
    }

    /// Mean health of living agents.
    pub fn mean_health(&self) -> f64 {
        let living: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.health)
            .collect();
        if living.is_empty() {
            return 0.0;
        }
        living.iter().sum::<f64>() / living.len() as f64
    }

    /// Mean consciousness phi of living agents.
    pub fn mean_phi(&self) -> f64 {
        let phis: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.consciousness.phi())
            .collect();
        if phis.is_empty() {
            return 0.0;
        }
        phis.iter().sum::<f64>() / phis.len() as f64
    }

    /// Colony-level Phi: organizational integration, not just mean agent Phi.
    ///
    /// Bug fix #2: mean_phi() drops as population grows because per-agent Phi
    /// doesn't scale with colony size. IIT predicts the opposite.
    ///
    /// Colony Phi = mean_agent_phi × organizational_multiplier
    /// where organizational_multiplier captures:
    /// - Governance stability (well-governed = more integrated)
    /// - Trust level (high trust = information flows freely)
    /// - Population scale bonus (log2 scaling — more people CAN integrate more)
    /// - Infrastructure quality (enables coordination)
    pub fn collective_phi(&self) -> f64 {
        let base = self.mean_phi();
        if base <= 0.0 {
            return 0.0;
        }

        let pop = self.population().max(1) as f64;
        // Scale bonus: log2(pop/100) — a colony of 10,000 gets +3.3 bonus
        // Capped at 5.0 (100K population)
        let scale_bonus = (pop / 100.0).max(1.0).log2().min(5.0);

        // Organizational quality: governance × trust × infrastructure
        let org_quality =
            self.governance.stability_score * self.trust_level * self.infrastructure_level;

        // Collective Phi = base × (1 + scale_bonus × org_quality)
        // At pop=100, org=1.0: multiplier = 1.0 (no bonus)
        // At pop=10000, org=0.8: multiplier = 1 + 3.3 * 0.8 = 3.64
        // At pop=50000, org=0.5: multiplier = 1 + 4.6 * 0.5 = 3.32
        let multiplier = 1.0 + scale_bonus * org_quality * 0.3;
        (base * multiplier).min(1.0)
    }

    /// Fraction of living agents at each consciousness tier (0-4).
    pub fn tier_distribution(&self) -> [f64; 5] {
        let mut counts = [0usize; 5];
        let mut total = 0usize;
        for a in self.agents.iter().filter(|a| a.is_alive()) {
            counts[a.consciousness.tier() as usize] += 1;
            total += 1;
        }
        if total == 0 {
            return [0.0; 5];
        }
        let mut dist = [0.0; 5];
        for i in 0..5 {
            dist[i] = counts[i] as f64 / total as f64;
        }
        dist
    }

    /// Mean allostatic load of living agents.
    pub fn mean_allostatic_load(&self) -> f64 {
        let living: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.needs.allostatic_load)
            .collect();
        if living.is_empty() {
            return 0.0;
        }
        living.iter().sum::<f64>() / living.len() as f64
    }

    /// Add an agent to this world.
    pub fn add_agent(&mut self, agent: CivAgent) {
        self.agents.push(agent);
    }

    /// Fraction of living agents at each civic tier (0-4), using the 8D sovereign
    /// profile with the given weights AND applying the per-agent restorative-
    /// justice penalty. Parallel to `tier_distribution` which uses scalar Phi.
    pub fn civic_tier_distribution(
        &self,
        weights: &crate::sovereign_profile::DimensionWeights,
    ) -> [f64; 5] {
        let mut counts = [0usize; 5];
        let mut total = 0usize;
        for a in self.agents.iter().filter(|a| a.is_alive()) {
            let raw = a.sovereign_profile.tier(weights);
            let effective = a.justice.effective_tier(raw);
            counts[effective.index()] += 1;
            total += 1;
        }
        if total == 0 {
            return [0.0; 5];
        }
        let mut dist = [0.0; 5];
        for i in 0..5 {
            dist[i] = counts[i] as f64 / total as f64;
        }
        dist
    }

    /// Fraction of living agents that meet a `CivicRequirement` under the given
    /// weights, with the per-agent restorative-justice penalty applied to tier.
    /// Used by 8D-gated governance to compute eligible voter fraction.
    ///
    /// A `CrossClusterAmplifier` agent has its per-dimension minimums reduced
    /// by `cross_cluster_bypass` — modeling the leniency of the easier cluster
    /// gate they route through. This is the only strategy that gets a gating
    /// concession; all others pay the full cost.
    pub fn civic_fraction_meeting(
        &self,
        requirement: &crate::sovereign_profile::CivicRequirement,
        weights: &crate::sovereign_profile::DimensionWeights,
    ) -> f64 {
        let (eligible, total) =
            self.agents
                .iter()
                .filter(|a| a.is_alive())
                .fold((0usize, 0usize), |(e, t), a| {
                    let raw = a.sovereign_profile.tier(weights);
                    let effective = a.justice.effective_tier(raw);
                    // Tier check: use effective tier (degraded).
                    let tier_ok = effective >= requirement.min_tier;
                    // Per-dimension minimums: CrossClusterAmplifier routes the
                    // check through a lenient cluster gate, reducing each floor.
                    let bypass = match a.adversarial {
                        Some(crate::red_team::AdversarialStrategy::CrossClusterAmplifier) => {
                            let m = crate::red_team::AdversarialModifier::for_strategy(
                                crate::red_team::AdversarialStrategy::CrossClusterAmplifier,
                                0.01,
                            );
                            m.cross_cluster_bypass.clamp(0.0, 1.0)
                        }
                        _ => 0.0,
                    };
                    let dim_ok = requirement.min_dimensions.iter().all(|&(dim, min)| {
                        let v = a.sovereign_profile.get(dim);
                        let sanitized = if v.is_finite() {
                            v.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let effective_min = min * (1.0 - bypass);
                        sanitized >= effective_min
                    });
                    let e = e + (tier_ok && dim_ok) as usize;
                    (e, t + 1)
                });
        if total == 0 {
            0.0
        } else {
            eligible as f64 / total as f64
        }
    }

    /// Refresh each living agent's 8D sovereign profile from current state.
    /// Called once per governance tick before gating decisions.
    pub fn refresh_sovereign_profiles(&mut self) {
        for a in self.agents.iter_mut().filter(|a| a.is_alive()) {
            a.refresh_sovereign_profile();
        }
    }

    /// Create a founding population snapshot string.
    pub fn founding_population_snapshot(&self) -> String {
        let living = self.population();
        let males = self
            .agents
            .iter()
            .filter(|a| a.is_alive() && a.sex == crate::agent::BiologicalSex::Male)
            .count();
        let females = living - males;
        format!(
            "{}: {} founders ({} M, {} F), location: {}",
            self.name, living, males, females, self.location
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;

    fn make_world_with_agents(n: usize) -> World {
        let mut world = World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: WorldEconomy::new(),
            harmony: HarmonyTracker::new(),
            governance: WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: crate::world::NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::default(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };

        for i in 0..n {
            let agent = CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: 0,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.0,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
                cumulative_dose_sv: 0.0,
                adversarial: None,
                coordination_understanding: 0.0,
                mycel_score: 0.1,
                sap_balance: 100.0,
                is_biological: true,
                wounds: Vec::new(),
                ethics: crate::agent::EthicalOrientation::default(),
                sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
                justice: crate::sub_passport::RestorativeJustice::new(),
            };
            world.agents.push(agent);
        }
        world.next_agent_id = n as u64;
        world
    }

    #[test]
    fn test_world_population_counts_alive_only() {
        let mut world = make_world_with_agents(10);
        assert_eq!(world.population(), 10);
        world.agents[0].death_tick = Some(5);
        assert_eq!(world.population(), 9);
    }

    #[test]
    fn test_resource_self_sufficiency() {
        let r = WorldResources::earth_default();
        let ss = r.self_sufficiency();
        assert!(ss > 0.9, "Earth should be near self-sufficient, was {ss}");

        let lunar = WorldResources::lunar_default();
        let ss_lunar = lunar.self_sufficiency();
        assert!(
            ss_lunar < ss,
            "Lunar should be less self-sufficient than Earth"
        );
    }

    #[test]
    fn test_cultural_distance_is_symmetric() {
        let a = CulturalProfile::earth_default();
        let b = CulturalProfile::pioneer_default();
        let d_ab = a.cultural_distance(&b);
        let d_ba = b.cultural_distance(&a);
        assert!(
            (d_ab - d_ba).abs() < 1e-10,
            "Distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }

    #[test]
    fn test_cultural_drift_rate_scales_with_population() {
        let mut small = CulturalProfile::earth_default();
        let mut large = CulturalProfile::earth_default();
        let reference = CulturalProfile::earth_default();

        let mut rng1 = StochasticEngine::new(42);

        // Small population drifts more
        for _ in 0..100 {
            small.drift(&mut rng1, 10, 0.0);
        }
        // Reset RNG for fair comparison
        let mut rng2 = StochasticEngine::new(42);
        for _ in 0..100 {
            large.drift(&mut rng2, 10_000, 0.0);
        }

        let d_small = reference.cultural_distance(&small);
        let d_large = reference.cultural_distance(&large);
        assert!(
            d_small > d_large,
            "Small pop should drift more: {d_small} vs {d_large}"
        );
    }

    #[test]
    fn test_resource_critical_check() {
        let mut r = WorldResources::new();
        r.set(
            "food",
            ResourceStock {
                current: 5.0,
                capacity: 100.0,
                production_rate: 1.0,
                consumption_rate: 2.0,
                critical_threshold: 0.1,
            },
        );
        assert!(r.get("food").unwrap().is_critical());

        r.set(
            "water",
            ResourceStock {
                current: 80.0,
                capacity: 100.0,
                production_rate: 5.0,
                consumption_rate: 4.0,
                critical_threshold: 0.1,
            },
        );
        assert!(!r.get("water").unwrap().is_critical());
    }

    #[test]
    fn test_zone_radiation_exposure_differs() {
        // Fix 8: Intra-World Geography
        let ship_zones = WorldZone::generation_ship_zones();
        assert_eq!(ship_zones.len(), 5, "Generation ship should have 5 zones");

        let bridge_rad = ship_zones[0].radiation_multiplier;
        let cargo_rad = ship_zones[4].radiation_multiplier;
        assert!(
            cargo_rad > bridge_rad,
            "Cargo hold should have higher radiation: {cargo_rad} vs {bridge_rad}"
        );

        let lunar_zones = WorldZone::lunar_zones();
        assert_eq!(lunar_zones.len(), 3, "Moon should have 3 zones");
        let shielded = lunar_zones[1].radiation_multiplier;
        let surface = lunar_zones[2].radiation_multiplier;
        assert!(
            surface > shielded,
            "Surface ops should have higher radiation than shielded lab: {surface} vs {shielded}"
        );
    }

    #[test]
    fn test_language_diverges_over_time() {
        // Fix 9: Richer Cultural Model
        let mut culture = CulturalProfile::pioneer_default();
        assert_eq!(culture.language_divergence, 0.0);
        assert_eq!(culture.ritual_count, 3);

        // Tick for 240 ticks (20 years) with no contact
        for tick in 1..=240 {
            culture.tick_culture_evolution(tick, 0.0);
        }

        assert!(
            culture.language_divergence > 0.2,
            "Language should diverge after 20 years isolated: {}",
            culture.language_divergence
        );
        // 240 ticks / 120 = 2 new rituals
        assert!(
            culture.ritual_count >= 5,
            "Should have gained rituals: {}",
            culture.ritual_count
        );
    }

    #[test]
    fn test_language_stable_with_contact() {
        // Fix 9: Contact slows divergence
        let mut culture = CulturalProfile::pioneer_default();

        // Tick with full contact
        for tick in 1..=240 {
            culture.tick_culture_evolution(tick, 1.0);
        }

        assert!(
            culture.language_divergence < 0.01,
            "Language should stay stable with full contact: {}",
            culture.language_divergence
        );
    }
}
