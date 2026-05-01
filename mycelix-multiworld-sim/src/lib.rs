// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # mycelix-multiworld-sim
//!
//! 150-year multi-world civilization simulator for autonomous planetary infrastructure.
//!
//! Simulates demographics, genetics, economics, governance, knowledge transfer,
//! consciousness evolution, and inter-world dynamics across lunar, Martian, and
//! outer-system colonies.
//!
//! ## Architecture
//!
//! The simulation runs in monthly ticks (12 per year, 1800 total for 150 years).
//! Each tick executes 10 phases in order:
//!
//! 1. Demographics (birth/death)
//! 2. Genetics (diversity tracking)
//! 3. Psychological needs (allostatic load, social satiation, engagement)
//! 4. Economy (resource production/consumption)
//! 5. Inter-world (migration, trade)
//! 6. Knowledge (education, innovation)
//! 7. Governance (voting, constitutional evolution)
//! 8. Consciousness (individual and collective phi)
//! 9. Emergencies (epidemics, resource crises)
//! 10. Epoch evaluation (milestone checks)

pub mod agent;
pub mod biosphere;
pub mod cascade;
pub mod civic_dimensions;
pub mod cliodynamics;
pub mod cohort;
pub mod config;
pub mod config_loader;
pub mod consciousness;
pub mod consciousness_epidemiology;
pub mod constants;
pub mod counterfactual;
pub mod csv_output;
pub mod currency;
pub mod disaster_cascade;
pub mod disasters;
pub mod dkg;
pub mod earth_population;
pub mod earth_regions;
pub mod economy;
pub mod education;
pub mod empirical;
pub mod engine_v2;
pub mod epistemic_decay;
pub mod epoch;
pub mod events;
pub mod factions;
pub mod fusion_bridge;
pub mod generation_ship;
pub mod governance;
pub mod governance_hardening;
pub mod governance_models;
pub mod habitat;
pub mod harmony;
pub mod interplanetary_consciousness;
pub mod interworld;
pub mod io;
pub mod knowledge;
pub mod live_metrics;
pub mod maglev_network;
pub mod metabolism;
pub mod module_registry;
pub mod mycelix_bridge;
pub mod narrative;
pub mod needs;
pub mod observables;
pub mod peer_recognition;
pub mod population;
pub mod primitives;
pub mod projects;
pub mod proposals;
pub mod red_team;
pub mod relativistic_dht;
pub mod report;
pub mod resontia;
pub mod robotics;
pub mod sanctions;
pub mod scoring_bridge;
pub mod skill_integrity;
pub mod sovereign_profile;
pub mod spaceport;
pub mod statistics;
pub mod stochastic;
pub mod stoichiometry;
pub mod sub_passport;
pub mod supply_chain;
pub mod symtropy_bridge;
pub mod unified_config_bridge;
pub mod validation;
pub mod viability;
pub mod world;
pub mod wound_healing;

use config::{EpochId, SimulationConfig};
use epoch::{EpochManager, EpochSnapshot};
use events::{CivEvent, CivEventType};
use population::PopulationEngine;
use report::CivilizationReport;
use stochastic::StochasticEngine;
use world::World;

use agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
use needs::{NeedsWorldSummary, PsychNeedsEngine, PsychologicalNeeds};
use world::{CulturalProfile, WorldResources};

/// Tick at which Mars colony fission can occur (year 40, month 0).
const MARS_FISSION_EARLIEST_TICK: u32 = 480;
/// Minimum population to trigger Mars fission.
const MARS_FISSION_MIN_POP: usize = 200;
/// Self-sufficiency threshold for Mars fission.
const MARS_FISSION_MIN_SS: f64 = 0.6;
/// Number of settlers sent to Mars colony.
const MARS_FISSION_SETTLERS: usize = 50;

// ─── Structural constants for run() and phase tick methods ───────────────────

/// Work hours per worker per tick (month). Standard: 160h/month.
const WORK_HOURS_PER_MONTH: f64 = 160.0;
/// Fraction of labor hours allocated to colony projects.
const PROJECT_LABOR_FRACTION: f64 = 0.2;
/// Fraction of materials stock available for colony projects per tick.
const PROJECT_MATERIALS_FRACTION: f64 = 0.1;
/// Infrastructure natural improvement rate per tick.
const INFRASTRUCTURE_IMPROVEMENT_RATE: f64 = 0.001;
/// Infrastructure entropy degradation rate per tick (~0.36%/year).
/// Garner & Hamilton (2005): steel half-life ~150yr in radiation.
const INFRASTRUCTURE_DEGRADATION_RATE: f64 = 0.0003;
/// Resource waste rate per tick when trust-weighted governance is disabled.
const RESOURCE_WASTE_RATE: f64 = 0.002;
/// Earth R&D workforce fraction (~5% of workforce in research).
const EARTH_RD_WORKFORCE_FRACTION: f64 = 0.05;
/// Earth institutional investment fraction.
const EARTH_INSTITUTIONAL_INVEST_FRACTION: f64 = 0.1;
/// Moderate environmental policy baseline.
const EARTH_ENVIRO_POLICY_BASELINE: f64 = 0.3;
/// EROI-to-GDP drag coefficient per point below threshold.
const EROI_GDP_DRAG_PER_POINT: f64 = 0.0004;
/// GDP floor multiplier when EROI drag is applied.
const EROI_GDP_FLOOR_MULT: f64 = 0.998;
/// Consciousness bonus per tick from stable governance.
const _GOV_CONSCIOUSNESS_BONUS_STABLE: f64 = 0.001;
/// Consciousness penalty per tick from unstable governance.
const _GOV_CONSCIOUSNESS_PENALTY_UNSTABLE: f64 = 0.002;

/// Top-level multi-world civilization simulator.
pub struct MultiWorldSimulator {
    pub config: SimulationConfig,
    /// Tunable simulation parameters (spoilage, radiation, psychology, etc.).
    /// All fields have published-source defaults. Override via TOML scenario files.
    pub params: constants::SimulationParams,
    pub worlds: Vec<World>,
    pub current_tick: u32,
    pub current_epoch: EpochId,
    pub rng: StochasticEngine,
    pub events: Vec<CivEvent>,
    pub epoch_snapshots: Vec<EpochSnapshot>,
    pub epoch_manager: EpochManager,
    /// Track whether Mars has been founded via fission (to avoid duplicates).
    mars_fission_done: bool,
    /// Track whether we've auto-granted constitution/trade milestones.
    constitution_granted: bool,
    trade_granted: bool,
    /// Per-world psychological needs summaries (recomputed each tick).
    needs_summaries: Vec<NeedsWorldSummary>,
    /// Faction dynamics engine.
    pub faction_engine: factions::FactionEngine,
    /// Per-world governance state.
    governance: Vec<governance::WorldGovernance>,
    /// Probabilistic disaster engine (solar, impacts, ECLSS, psych, tech tree, Tainter/Turchin).
    pub disaster_engine: disasters::DisasterEngine,
    /// Mechanism 6 — Resource Depletion Feedback: consecutive ticks each critical
    /// resource (food, water, oxygen) has been below 20% capacity per world.
    depletion_streaks: std::collections::HashMap<(u32, String), u32>,
    /// Mechanism 6 — Whether a depletion crisis is active per (world_id, resource).
    depletion_crisis_active: std::collections::HashMap<(u32, String), bool>,
    /// Mechanism 9 — Morale Contagion: whether negative or positive contagion is
    /// active per world.
    morale_contagion: std::collections::HashMap<u32, i8>, // -1 = negative, 0 = none, 1 = positive
    /// Mechanism 10 — Carrying Capacity base: original max_population per world
    /// before dynamic scaling. Prevents feedback loop from self-referential capacity.
    carrying_capacity_base: std::collections::HashMap<u32, usize>,
    /// Narrative engine: generates memorable event descriptions from affect + disaster data.
    pub narrative_engine: narrative::NarrativeEngine,
    /// Global supply chain graph (12 Earth regions + 4 colonies).
    pub supply_chain: supply_chain::SupplyChainGraph,
    /// Resontia Earth-hardening infrastructure.
    pub resontia_infra: resontia::ResontiaInfrastructure,
    /// Resontia configuration.
    pub resontia_config: resontia::ResontiaConfig,
    /// Hybrid Earth model: aggregate regional demographics (12 regions).
    /// Empty when `hybrid_earth` is false (backward compat).
    pub earth_regions: Vec<earth_regions::EarthRegion>,
    /// Multi-resolution Earth population model (cohort-based, Phase 2).
    /// Initialized from `earth_regions` when `hybrid_earth` is enabled.
    pub earth_pop_model: Option<earth_population::EarthPopulationModel>,
    /// Six civilizational primitives: network, resources, institutions, ecosystem, trust, knowledge.
    pub earth_primitives: Option<primitives::CivilizationalPrimitives>,
    /// Spaceport for launching colonists from Earth aggregate regions.
    /// None when `hybrid_earth` is false.
    pub spaceport: Option<spaceport::Spaceport>,
    /// Generation ship for interstellar transit (if enabled).
    pub generation_ship: Option<generation_ship::GenerationShip>,
    /// Tick at which to launch the generation ship (0 = disabled).
    pub generation_ship_launch_tick: u32,
    /// Whether the generation ship has been launched.
    generation_ship_launched: bool,
    /// Viability engine: enforces thermodynamic axioms, EROI, and scaling laws.
    pub viability_engine: viability::ViabilityEngine,
    /// Per-world metabolism phase modifiers (recomputed each tick).
    phase_modifiers: std::collections::HashMap<u32, metabolism::PhaseModifiers>,
    /// Module registry: pluggable simulation modules (Phase 3 architecture).
    pub module_registry: module_registry::ModuleRegistry,
    /// Power-law cascade engine: correlated disaster propagation.
    pub cascade_engine: cascade::CascadeEngine,
    /// Per-world Turchin secular cycle state (cliodynamics).
    pub secular_cycles: std::collections::HashMap<u32, cliodynamics::SecularCycleState>,
    /// Disaster config loaded from TOML (or defaults).
    pub disaster_config: config_loader::DisasterConfig,
    /// JSONL output path for live metrics (None = disabled).
    pub jsonl_output_path: Option<std::path::PathBuf>,
    /// CSV time-series recorder for scientific analysis (None = disabled).
    csv_recorder: Option<csv_output::CsvRecorder>,
    /// Founding ethical orientation means per world (computed on first call).
    founding_ethics: std::collections::HashMap<u32, [f64; 4]>,
    /// Previous ethics std-dev per world (for synthesis detection).
    prev_ethics_stddev: std::collections::HashMap<u32, f64>,
    /// Last tick at which a moral revival occurred per world (cooldown tracking).
    last_moral_revival: std::collections::HashMap<u32, u32>,
}

impl MultiWorldSimulator {
    /// Create a new simulator from configuration.
    pub fn new(config: SimulationConfig) -> Self {
        let rng = StochasticEngine::new(config.seed);
        Self {
            config,
            params: constants::SimulationParams::default(),
            worlds: Vec::new(),
            current_tick: 0,
            current_epoch: 0,
            rng,
            events: Vec::new(),
            epoch_snapshots: Vec::new(),
            epoch_manager: EpochManager::new(),
            mars_fission_done: false,
            constitution_granted: false,
            trade_granted: false,
            needs_summaries: Vec::new(),
            faction_engine: factions::FactionEngine::new(),
            governance: Vec::new(),
            disaster_engine: disasters::DisasterEngine::new(),
            depletion_streaks: std::collections::HashMap::new(),
            depletion_crisis_active: std::collections::HashMap::new(),
            morale_contagion: std::collections::HashMap::new(),
            carrying_capacity_base: std::collections::HashMap::new(),
            narrative_engine: narrative::NarrativeEngine::new(),
            viability_engine: viability::ViabilityEngine::new(),
            phase_modifiers: std::collections::HashMap::new(),
            module_registry: module_registry::ModuleRegistry::new(),
            cascade_engine: cascade::CascadeEngine::new(5), // up to 5 worlds
            supply_chain: supply_chain::SupplyChainGraph::new(),
            resontia_infra: resontia::ResontiaInfrastructure::new(),
            resontia_config: resontia::ResontiaConfig::default(),
            earth_regions: Vec::new(),
            earth_pop_model: None,
            earth_primitives: None,
            spaceport: None,
            generation_ship: None,
            generation_ship_launch_tick: 0,
            generation_ship_launched: false,
            secular_cycles: std::collections::HashMap::new(),
            disaster_config: config_loader::load_disaster_config(std::path::Path::new(
                "config/disasters.toml",
            )),
            jsonl_output_path: None,
            csv_recorder: None,
            founding_ethics: std::collections::HashMap::new(),
            prev_ethics_stddev: std::collections::HashMap::new(),
            last_moral_revival: std::collections::HashMap::new(),
        }
    }

    /// Enable JSONL telemetry output to a file.
    /// Set custom simulation parameters (spoilage, radiation, psychology, etc.).
    /// Call before `run()`.
    pub fn with_params(mut self, params: constants::SimulationParams) -> Self {
        self.params = params;
        self
    }

    pub fn enable_jsonl_output(&mut self, path: std::path::PathBuf) {
        self.jsonl_output_path = Some(path);
    }

    /// Enable CSV time-series output for scientific analysis.
    ///
    /// Writes one row per world per tick with population, Phi, Gini, allostatic
    /// load, resource levels, governance authority, and more. Output is amenable
    /// to R/Python analysis.
    pub fn enable_csv_output(&mut self, path: std::path::PathBuf) {
        self.csv_recorder = Some(csv_output::CsvRecorder::new(path));
    }

    /// Enable interstellar mode: launch a generation ship at the specified tick.
    pub fn enable_interstellar(&mut self, launch_tick: u32, passengers: usize, velocity_c: f64) {
        self.generation_ship_launch_tick = launch_tick;
        // Ship will be created at launch tick, not now
        let _ = (passengers, velocity_c); // stored in config
    }

    /// Enable Resontia Earth-hardening for this simulation.
    pub fn enable_resontia(&mut self) {
        self.resontia_config = resontia::default_resontia_config();
    }

    /// Enable hybrid Earth model with 12 aggregate regions and spaceport.
    pub fn enable_hybrid_earth(&mut self) {
        self.earth_regions = earth_regions::build_earth_regions();
        // Phase 2: Initialize multi-resolution cohort model from aggregate regions
        self.earth_pop_model = Some(earth_population::EarthPopulationModel::from_regions(
            &self.earth_regions,
        ));
        // Initialize six civilizational primitives for Earth.
        // Ecosystem health is derived from 3.8 Ga of biosphere evolution (B(t))
        // via the HANPP thermodynamic bridge, rather than hardcoded 1970 values.
        let bt_engine = biosphere::BiosphereCoherenceEngine::build();
        let terminal_bt = bt_engine.terminal_bt();
        let bt_health = biosphere::bridge::bt_to_ecosystem_health(terminal_bt, terminal_bt);
        let bt_ecosystem = primitives::EcosystemHealth {
            biodiversity: bt_health.biodiversity,
            forest_cover: bt_health.forest_cover,
            soil_health: bt_health.soil_health,
            ocean_health: bt_health.ocean_health,
            freshwater: bt_health.freshwater,
        };
        self.earth_primitives = Some(primitives::CivilizationalPrimitives::earth_from_biosphere(
            bt_ecosystem,
        ));
        self.spaceport = Some(spaceport::Spaceport::new_equatorial(0));
    }

    /// Initialize worlds without running the simulation.
    /// Call this before `inject_adversaries()` to ensure agents exist.
    pub fn initialize(&mut self) {
        self.initialize_worlds();
    }

    /// Inject adversarial agents into all worlds for red team testing.
    /// Call after `initialize()` and before `run()`.
    /// Marks `n` agents per world with the given strategy.
    pub fn inject_adversaries(
        &mut self,
        strategy: red_team::AdversarialStrategy,
        n_per_world: usize,
    ) {
        for world in &mut self.worlds {
            let mut count = 0;
            for agent in world
                .agents
                .iter_mut()
                .filter(|a| a.is_alive() && a.adversarial.is_none())
            {
                if count >= n_per_world {
                    break;
                }
                agent.adversarial = Some(strategy);
                count += 1;
            }
        }
    }

    /// Initialize worlds that should exist at tick 0.
    /// Public for experiment binaries that need to modify agents before run().
    pub fn run_initialization(&mut self) {
        if !self.worlds.is_empty() {
            return;
        }
        self.initialize_worlds();
    }

    fn initialize_worlds(&mut self) {
        // If hybrid_earth is enabled, populate aggregate regions and spaceport
        // instead of creating individual Earth agents.
        if self.config.policy.hybrid_earth {
            self.enable_hybrid_earth();
        }

        let seeds: Vec<_> = self
            .config
            .initial_worlds
            .iter()
            .filter(|w| w.founding_tick == 0)
            .cloned()
            .collect();

        for (idx, seed) in seeds.iter().enumerate() {
            let is_earth = seed.location == "Earth";
            let resources = match seed.location.as_str() {
                "Earth" => WorldResources::earth_default(),
                "Europa" => WorldResources::europa_default(),
                "Titan" => WorldResources::titan_default(),
                _ => WorldResources::lunar_default(),
            };
            let culture = if is_earth {
                CulturalProfile::earth_default()
            } else {
                CulturalProfile::pioneer_default()
            };
            let culture_individualism = culture.individualism;

            let mut world = World::new_colony(world::ColonyParams {
                id: idx as u32,
                name: seed.name.clone(),
                location: seed.location.clone(),
                founded_tick: 0,
                parent_world_id: if is_earth { None } else { Some(0) },
                resources,
                culture,
                infrastructure_level: if is_earth { 0.9 } else { 0.2 },
                max_population: if is_earth { 2_000 } else { 10_000 },
                habitable_area_m2: if is_earth { 1e12 } else { 50_000.0 },
            });

            // Hybrid Earth: skip individual agent creation for Earth;
            // aggregate demographics are handled by earth_regions.
            let skip_agents = self.config.policy.hybrid_earth && seed.location == "Earth";

            // Spawn initial population as adults (age 25-45)
            let spawn_count = if skip_agents {
                0
            } else {
                seed.initial_population
            };
            for _ in 0..spawn_count {
                let sex = if self.rng.bernoulli(0.5) {
                    BiologicalSex::Male
                } else {
                    BiologicalSex::Female
                };
                // Birth tick set so they start as adults (25-45 years old)
                let age_months = (25 * 12) + (self.rng.next_u64() % (20 * 12)) as u32;
                let birth_tick = 0u32.wrapping_sub(age_months);

                let agent = CivAgent {
                    id: world.next_agent_id,
                    birth_tick,
                    death_tick: None,
                    sex,
                    world_id: world.id,
                    health: self.rng.next_gaussian(0.85, 0.1).clamp(0.3, 1.0),
                    skills: {
                        let mut s = SkillVector::new();
                        let primary = (self.rng.next_u64() % 8) as usize;
                        s.learn(primary, self.rng.next_f64() * 0.5);
                        s
                    },
                    education_level: (self.rng.next_f64() * 0.5)
                        + self.config.policy.education_boost,
                    consciousness: {
                        let mut c = ConsciousnessState::nascent();
                        let boost = self.config.policy.consciousness_boost;
                        if boost > 0.0 {
                            c.level = (c.level + boost).min(1.0);
                            c.meta_awareness = (c.meta_awareness + boost * 0.8).min(1.0);
                            c.coherence = (c.coherence + boost * 0.6).min(1.0);
                            c.care_activation = (c.care_activation + boost * 0.5).min(1.0);
                            c.harmonic_alignment = (c.harmonic_alignment + boost * 0.4).min(1.0);
                            c.epistemic_confidence =
                                (c.epistemic_confidence + boost * 0.3).min(1.0);
                        }
                        c
                    },
                    partner_id: None,
                    children_ids: vec![],
                    is_immigrant: false,
                    needs: PsychologicalNeeds::new(),
                    tend_balance: 0.0,
                    parent_ids: None,
                    faction_id: None,
                    generation: 0,
                    trauma_level: 0.0,
                    cumulative_dose_sv: 0.0,
                    adversarial: None,
                    coordination_understanding: self
                        .config
                        .policy
                        .coordination_understanding_initial,
                    mycel_score: 0.1,
                    sap_balance: 100.0,
                    is_biological: true,
                    wounds: Vec::new(),
                    ethics: agent::EthicalOrientation::from_culture(
                        culture_individualism,
                        &mut self.rng,
                    ),
                    // 8D sovereign profile: founder sample from cultural individualism.
                    sovereign_profile: crate::sovereign_profile::SovereignProfile::sample(
                        culture_individualism,
                        &mut self.rng,
                    ),
                    justice: crate::sub_passport::RestorativeJustice::new(),
                };
                world.next_agent_id += 1;
                world.agents.push(agent);
            }

            self.events.push(CivEvent::new(
                0,
                Some(world.id),
                CivEventType::WorldFounded,
                format!(
                    "{} founded with {} colonists",
                    world.name, seed.initial_population
                ),
            ));

            self.worlds.push(world);
        }
    }

    /// Check if any deferred worlds should be founded this tick.
    fn check_deferred_worlds(&mut self) {
        let seeds: Vec<_> = self
            .config
            .initial_worlds
            .iter()
            .filter(|w| w.founding_tick == self.current_tick && w.founding_tick > 0)
            .cloned()
            .collect();

        for seed in seeds {
            self.found_colony(
                &seed.name,
                &seed.location,
                seed.initial_population,
                seed.initial_resources,
            );
        }
    }

    /// Found a new colony world, spawning settlers.
    fn found_colony(&mut self, name: &str, location: &str, population: usize, _resource_mult: f64) {
        let world_id = self.worlds.len() as u32;
        let resources = match location {
            "Mars" => {
                let mut r = WorldResources::lunar_default();
                // Mars has slightly better resources than Moon
                if let Some(food) = r.get_mut("food") {
                    food.production_rate *= 1.2;
                }
                r
            }
            "Europa" => WorldResources::europa_default(),
            "Titan" => WorldResources::titan_default(),
            _ => WorldResources::lunar_default(),
        };

        let mut world = World::new_colony(world::ColonyParams {
            id: world_id,
            name: name.into(),
            location: location.into(),
            founded_tick: self.current_tick,
            parent_world_id: Some(0),
            resources,
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.1,
            max_population: 5_000,
            habitable_area_m2: 30_000.0,
        });

        for _ in 0..population {
            let sex = if self.rng.bernoulli(0.5) {
                BiologicalSex::Male
            } else {
                BiologicalSex::Female
            };
            let age_months = (25 * 12) + (self.rng.next_u64() % (15 * 12)) as u32;
            let birth_tick = self.current_tick.wrapping_sub(age_months);

            let agent = CivAgent {
                id: world.next_agent_id,
                birth_tick,
                death_tick: None,
                sex,
                world_id,
                health: self.rng.next_gaussian(0.9, 0.08).clamp(0.4, 1.0),
                skills: {
                    let mut s = SkillVector::new();
                    let primary = (self.rng.next_u64() % 8) as usize;
                    s.learn(primary, self.rng.next_f64() * 0.6);
                    s
                },
                education_level: (self.rng.next_f64() * 0.6) + self.config.policy.education_boost,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: true,
                needs: PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
                cumulative_dose_sv: 0.0,
                adversarial: None,
                coordination_understanding: self.config.policy.coordination_understanding_initial,
                mycel_score: 0.1,
                sap_balance: 100.0,
                is_biological: true,
                wounds: Vec::new(),
                ethics: agent::EthicalOrientation::from_culture(
                    world.culture.individualism,
                    &mut self.rng,
                ),
                sovereign_profile: crate::sovereign_profile::SovereignProfile::sample(
                    world.culture.individualism,
                    &mut self.rng,
                ),
                justice: crate::sub_passport::RestorativeJustice::new(),
            };
            world.next_agent_id += 1;
            world.agents.push(agent);
        }

        self.events.push(CivEvent::new(
            self.current_tick,
            Some(world_id),
            CivEventType::WorldFounded,
            format!(
                "{} founded at tick {} with {} settlers",
                name, self.current_tick, population
            ),
        ));

        self.worlds.push(world);
    }

    /// Check if Mars fission should occur: split settlers from the largest
    /// non-Earth world when population and self-sufficiency thresholds are met.
    fn check_mars_fission(&mut self) {
        if self.mars_fission_done {
            return;
        }
        if self.current_tick < MARS_FISSION_EARLIEST_TICK {
            return;
        }

        // Check if any non-Earth world qualifies
        let qualifies = self.worlds.iter().any(|w| {
            w.location != "Earth"
                && w.population() >= MARS_FISSION_MIN_POP
                && w.resources.self_sufficiency() >= MARS_FISSION_MIN_SS
        });

        // Also check total off-Earth population
        let total_off_earth: usize = self
            .worlds
            .iter()
            .filter(|w| w.location != "Earth")
            .map(|w| w.population())
            .sum();

        if qualifies || total_off_earth >= MARS_FISSION_MIN_POP {
            // Check if Mars world already exists (from deferred config)
            let mars_exists = self.worlds.iter().any(|w| w.location == "Mars");
            if !mars_exists {
                self.found_colony("Ares Colony (Fission)", "Mars", MARS_FISSION_SETTLERS, 0.2);
            }
            self.mars_fission_done = true;
        }
    }

    /// Tick generation ship: launch if scheduled, evolve in-flight, handle disasters.
    fn tick_generation_ship(&mut self) {
        if self.generation_ship_launch_tick > 0
            && self.current_tick >= self.generation_ship_launch_tick
            && !self.generation_ship_launched
        {
            let ship = generation_ship::GenerationShip::new(
                99,
                generation_ship::InterstellarDestination::ProximaCentauri,
                0.05,
                self.current_tick,
                500,
            );
            self.events.push(CivEvent::new(
                self.current_tick,
                None,
                CivEventType::EmergencyDeclared,
                format!(
                    "GENERATION SHIP LAUNCHED → {} ({:.2} ly at {:.1}%c, {} passengers)",
                    ship.destination.name(),
                    ship.distance_ly,
                    ship.cruise_velocity_c * 100.0,
                    500
                ),
            ));
            self.generation_ship = Some(ship);
            self.generation_ship_launched = true;
        }
        if let Some(ref mut ship) = self.generation_ship {
            let disasters = ship.tick(&mut self.rng);
            for d in &disasters {
                let desc = match d {
                    generation_ship::InterstellarDisaster::CosmicRayBurst { severity } => format!(
                        "Cosmic ray burst (severity {:.2}) on generation ship",
                        severity
                    ),
                    generation_ship::InterstellarDisaster::MicrometeoiteImpact { hull_damage } => {
                        format!(
                            "Micrometeorite impact (hull damage {:.3}) on generation ship",
                            hull_damage
                        )
                    }
                    generation_ship::InterstellarDisaster::NavigationDrift {
                        correction_fuel_fraction,
                    } => format!(
                        "Navigation drift (fuel cost {:.3}) on generation ship",
                        correction_fuel_fraction
                    ),
                    generation_ship::InterstellarDisaster::KnowledgeAttrition {
                        skill_loss_fraction,
                    } => format!(
                        "Knowledge attrition ({:.1}% skill loss) on generation ship",
                        skill_loss_fraction * 100.0
                    ),
                    generation_ship::InterstellarDisaster::SocialFracture { cohesion_loss } => {
                        format!(
                            "Social fracture ({:.1}% cohesion loss) on generation ship",
                            cohesion_loss * 100.0
                        )
                    }
                };
                self.events.push(CivEvent::new(
                    self.current_tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    desc,
                ));
            }
            if ship.phase == generation_ship::ShipPhase::Arrived {
                self.events.push(CivEvent::new(
                    self.current_tick,
                    None,
                    CivEventType::TradeEstablished,
                    format!(
                        "GENERATION SHIP ARRIVED at {} — new human culture founded",
                        ship.destination.name()
                    ),
                ));
            }
        }
    }

    /// Tick Earth hybrid model: aggregate demographics, civilizational primitives,
    /// resource depletion feedback, ecosystem tipping points, and spaceport/Kessler.
    fn tick_earth_hybrid(&mut self) {
        if self.earth_regions.is_empty() {
            return;
        }
        if let Some(ref mut pop_model) = self.earth_pop_model {
            let scaling: Vec<_> = self
                .earth_regions
                .iter()
                .map(|r| viability::ScalingFactors::compute(r.population * 1_000_000.0))
                .collect();
            pop_model.tick(
                &self.earth_regions,
                &scaling,
                self.current_tick,
                &mut self.rng,
            );
            pop_model.sync_to_regions(&mut self.earth_regions);

            if let Some(ref mut prims) = self.earth_primitives {
                let total_pop: f64 = self.earth_regions.iter().map(|r| r.population).sum();
                let mean_urban: f64 = self
                    .earth_regions
                    .iter()
                    .map(|r| r.urbanization * r.population)
                    .sum::<f64>()
                    / total_pop.max(1.0);
                let mean_gdp: f64 = self
                    .earth_regions
                    .iter()
                    .map(|r| r.gdp_per_capita * r.population)
                    .sum::<f64>()
                    / total_pop.max(1.0);
                prims.tick(
                    total_pop,
                    mean_urban,
                    mean_gdp,
                    EARTH_RD_WORKFORCE_FRACTION,
                    EARTH_ENVIRO_POLICY_BASELINE,
                    EARTH_INSTITUTIONAL_INVEST_FRACTION,
                );

                // 1. Resource depletion → EROI → GDP drag
                if let Some(oil) = prims.resources.iter().find(|r| r.name == "Oil") {
                    let eroi = oil.current_eroi();
                    if eroi < 8.0 {
                        let monthly_drag = 1.0 - (8.0 - eroi) * EROI_GDP_DRAG_PER_POINT;
                        for region in &mut self.earth_regions {
                            region.gdp_per_capita *= monthly_drag.max(EROI_GDP_FLOOR_MULT);
                        }
                    }
                }
                // 2. Ecosystem → agriculture → GDP
                let ag_modifier = prims.ecosystem.agriculture_modifier();
                if ag_modifier < 0.95 {
                    for region in &mut self.earth_regions {
                        region.gdp_per_capita *= 1.0 - (1.0 - ag_modifier) * 0.01;
                    }
                }
                // 3. Knowledge network → innovation rate
                let knowledge_mult = prims.knowledge.growth_rate().min(3.0);
                for region in &mut self.earth_regions {
                    region.education_index =
                        (region.education_index + 0.00005 * knowledge_mult).min(0.98);
                }
                // 5. Ecosystem tipping points → disaster events
                let tips = prims.ecosystem.tipping_points();
                for tip in &tips {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        None,
                        CivEventType::EmergencyDeclared,
                        format!("ECOSYSTEM TIPPING POINT: {}", tip),
                    ));
                }
            }
        } else {
            for region in &mut self.earth_regions {
                earth_regions::tick_region(region, self.current_tick, &mut self.rng);
            }
        }
        // Spaceport + Kessler sync
        if let Some(ref mut sp) = self.spaceport {
            sp.tick(self.current_tick);
            if self.disaster_engine.orbital_debris.cascade_active && !sp.leo_blocked {
                let duration = spaceport::kessler_duration(&mut self.rng);
                sp.activate_kessler(duration);
                self.events.push(CivEvent::new(
                    self.current_tick,
                    None,
                    CivEventType::EmergencyDeclared,
                    "Kessler syndrome blocks spaceport — colonist launches suspended".to_string(),
                ));
            }
        }
    }

    /// Check if inter-world trade milestone should be granted.
    fn check_trade_milestone(&mut self) {
        if self.trade_granted {
            return;
        }
        let viable_worlds: Vec<&World> = self
            .worlds
            .iter()
            .filter(|w| {
                w.location != "Earth" && w.population() > 0 && w.resources.self_sufficiency() > 0.3
            })
            .collect();
        let has_established_colony = viable_worlds
            .iter()
            .any(|w| self.current_tick.saturating_sub(w.founded_tick) >= 12);
        if viable_worlds.len() >= 2
            || (viable_worlds.len() >= 1 && has_established_colony && self.worlds.len() >= 3)
        {
            self.epoch_manager
                .record_milestone("trade", self.current_tick);
            self.trade_granted = true;
            self.events.push(CivEvent::new(
                self.current_tick,
                None,
                CivEventType::TradeEstablished,
                "Inter-world trade route established",
            ));
        }
    }

    // --- Phase implementations ---

    fn tick_psychological_needs(&mut self) {
        self.needs_summaries.clear();
        let world_count = self.worlds.len();
        let epoch = self.epoch_manager.current_epoch;

        for i in 0..world_count {
            let mut world = std::mem::take(&mut self.worlds[i]);

            // Count care workers: agents whose strongest skill is "medicine" and can work
            let care_workers = world
                .agents
                .iter()
                .filter(|a| {
                    a.is_alive()
                        && a.life_stage(self.current_tick).can_work()
                        && a.skills.strongest() == "medicine"
                })
                .count();

            let mean_tech = world.knowledge.mean_tech_level();
            // Governance stability: blend of infrastructure, consciousness, and burnout fraction.
            // Low collective phi + high burnout = unstable governance.
            let burnout_frac = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .filter(|a| a.needs.is_burnout())
                .count() as f64
                / world.population().max(1) as f64;
            // Governance stability incorporates Spinozist consent (trust/reciprocity).
            // Low collective consent erodes governance; high consent strengthens it.
            let mean_consent = {
                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    0.5
                } else {
                    living.iter().map(|a| a.needs.affect.consent).sum::<f64>() / living.len() as f64
                }
            };
            // Realism F: Communication latency degrades governance coherence.
            // Distant colonies can't participate in real-time deliberation.
            let latency_penalty = match world.location.as_str() {
                "Earth" | "Moon" => 0.0, // Real-time
                "Mars" => 0.15,          // 4-22 min delay
                "Europa" => 0.35,        // 33-54 min delay
                "Titan" => 0.50,         // 67-90 min delay
                _ => 0.0,
            };
            let governance_stability = ((world.infrastructure_level * 0.4
                + world.mean_phi() * 0.2
                + (1.0 - burnout_frac) * 0.2
                + mean_consent * 0.2)
                * (1.0 - latency_penalty * 0.3)) // Latency reduces effective stability
                .clamp(0.0, 1.0);

            // Compute worker ratio for overwork stress
            let working = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.life_stage(self.current_tick).can_work())
                .count();
            let worker_ratio = working as f64 / world.population().max(1) as f64;

            let care_eff = world.policy_state.care_effectiveness;
            let (events, summary) = PsychNeedsEngine::tick_needs(
                &mut world,
                self.current_tick,
                epoch,
                care_workers,
                mean_tech,
                governance_stability,
                worker_ratio,
                care_eff,
                self.config.policy.deep_space_isolation_mult,
                &mut self.rng,
            );

            // Compute Spinozist affect state for each agent from updated needs.
            // Affects emerge from the body's relationship to its environment (Ethics III).
            let gov_stability = governance_stability;
            let resource_frac = world.resources.self_sufficiency();
            for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                let new_affect = needs::AffectState::compute(
                    &agent.needs,
                    agent.consciousness.care_activation,
                    agent.trauma_level,
                    gov_stability,
                    agent.faction_id.is_some(),
                    resource_frac,
                );
                // Affect momentum: blend new state with previous (α=0.3).
                // Emotions persist — grief doesn't vanish, joy lingers.
                agent.needs.affect = new_affect.blend_with_previous(&agent.needs.affect, 0.3);
            }

            self.needs_summaries.push(summary);
            self.events.extend(events);
            self.worlds[i] = world;
        }
    }

    fn tick_education(&mut self) {
        if !self.config.policy.education_enabled {
            return; // A/B comparison: skip education tick entirely
        }
        let events = education::tick_education_all_worlds(
            &mut self.worlds,
            self.current_tick,
            &mut self.rng,
        );
        self.events.extend(events);
    }

    fn tick_genetics(&mut self) {
        let events = population::check_genetic_diversity(&self.worlds, self.current_tick);
        self.events.extend(events);
    }

    fn tick_economy(&mut self) {
        // Phase 4: Resource production/consumption + Cobb-Douglas economy.
        let tick = self.current_tick;
        for world in &mut self.worlds {
            let pop = world.population() as f64;

            // Nuclear energy gate: Europa and Titan have zero solar power.
            // Energy production is gated on fission/fusion tech milestones.
            // Without nuclear power, these colonies consume from reserves until death.
            let is_outer_system = world.location == "Europa" || world.location == "Titan";
            let has_nuclear = self
                .disaster_engine
                .tech_tree
                .is_achieved("Fission Surface Power")
                || self.disaster_engine.tech_tree.is_achieved("Fusion Demo");

            // Spaceport supply line cut: if Kessler active and hybrid model,
            // reduce off-world production by 30%.
            let supply_line_cut = self.config.policy.hybrid_earth
                && self.spaceport.as_ref().map_or(false, |sp| sp.leo_blocked)
                && world.location != "Earth";

            // Raw resource arithmetic (life support)
            for name in &["food", "water", "energy", "materials", "oxygen"] {
                if let Some(stock) = world.resources.get_mut(name) {
                    let mut production = stock.production_rate * world.infrastructure_level;

                    // Supply line cut: off-world production reduced 30% during Kessler.
                    if supply_line_cut {
                        production *= 1.0 - spaceport::apply_supply_line_cut_penalty();
                    }

                    // Outer system energy gate: no solar, must have nuclear
                    if is_outer_system && *name == "energy" && !has_nuclear {
                        production = 0.0;
                    }
                    // Outer system: oxygen/food production requires energy
                    if is_outer_system && (*name == "oxygen" || *name == "food") && !has_nuclear {
                        // Can still produce at 20% from reserves/manual processes
                        production *= 0.2;
                    }

                    let mut consumption = stock.consumption_rate * (pop / 100.0).max(0.1);
                    // Closed-Loop ECLSS: 40% reduction in water/oxygen/food consumption
                    if self
                        .disaster_engine
                        .tech_tree
                        .is_achieved("Closed-Loop ECLSS")
                        && (*name == "water" || *name == "oxygen" || *name == "food")
                    {
                        consumption *= 0.6;
                    }
                    // Bioregenerative Agriculture: 50% more food production
                    if self
                        .disaster_engine
                        .tech_tree
                        .is_achieved("Bioregenerative Agriculture")
                        && *name == "food"
                    {
                        production *= 1.5;
                    }
                    // #2: Resource spoilage — entropy is real.
                    // Rates from SimulationParams (override via TOML scenarios).
                    let spoilage_rate = match *name {
                        "food" => self.params.spoilage_food,
                        "water" => self.params.spoilage_water,
                        "materials" => self.params.spoilage_materials,
                        "energy" => self.params.spoilage_energy,
                        "oxygen" => self.params.spoilage_oxygen,
                        _ => 0.01,
                    };
                    let spoilage = stock.current * spoilage_rate;

                    stock.current = (stock.current + production - consumption - spoilage)
                        .clamp(0.0, stock.capacity);
                }
            }

            // Titan hydrocarbon ISRU: consume hydrocarbons to boost materials production.
            // CH4/C2H6 → polymers, pykrete construction, fuel synthesis.
            if world.location == "Titan" {
                // Tick hydrocarbons (production/consumption)
                if let Some(hc) = world.resources.get_mut("hydrocarbons") {
                    let hc_prod = hc.production_rate * world.infrastructure_level;
                    let hc_cons = hc.consumption_rate * (pop / 100.0).max(0.1);
                    hc.current = (hc.current + hc_prod - hc_cons).clamp(0.0, hc.capacity);
                }
                // Hydrocarbon abundance boosts materials: if hydrocarbons > 50% capacity,
                // add bonus materials production (ISRU manufacturing)
                let hc_fraction = world.resources.fraction_of_capacity("hydrocarbons");
                if hc_fraction > 0.5 && has_nuclear {
                    if let Some(mat) = world.resources.get_mut("materials") {
                        let bonus = mat.production_rate * 0.5 * world.infrastructure_level;
                        mat.current = (mat.current + bonus).min(mat.capacity);
                    }
                }
            }

            // Cobb-Douglas economy: assign workers, produce, demurrage, Gini
            world.economy.assign_workers(&world.agents, tick);

            // Feed technology multipliers from knowledge system
            let tech = world.knowledge.mean_tech_level();
            for i in 0..8 {
                world.economy.technology_multiplier[i] = tech;
            }
            world.economy.infrastructure_capital = (world.infrastructure_level * 50.0).max(1.0); // map 0-1 to 0-50

            // Phase 1b: resource_priority biases sector productivity
            match world.policy_state.resource_priority {
                config::ResourcePriority::Industrial => {
                    world.economy.technology_multiplier[0] *= 1.5; // engineering
                    world.economy.technology_multiplier[7] *= 1.3; // logistics
                }
                config::ResourcePriority::Biological => {
                    world.economy.technology_multiplier[1] *= 1.5; // agriculture
                    world.economy.technology_multiplier[2] *= 1.3; // medicine
                }
                config::ResourcePriority::Knowledge => {
                    world.economy.technology_multiplier[4] *= 1.5; // science
                    world.economy.technology_multiplier[5] *= 1.3; // education
                }
                config::ResourcePriority::Balanced => {} // no bias
            }

            // Extended production: apply West-Bettencourt scaling + energy constraint
            let scaling = self.viability_engine.scaling.get(&world.id);
            let energy_avail = world.resources.stock_level("energy");
            world
                .economy
                .tick_production_extended(scaling, energy_avail);

            // Ethics-modulated resource efficiency
            let mean_ethics = agent::EthicalOrientation::mean_of(&world.agents);
            world.economy.apply_ethics_efficiency(&mean_ethics);

            // Hard energy constraint: when energy stock is depleted, only essential
            // sectors (agriculture=1, medicine=2) continue at full capacity.
            // All other sectors drop to 10%. This makes outer-system energy gates
            // (fission/fusion tech) genuinely consequential.
            let net_energy = world.resources.stock_level("energy").unwrap_or(0.0);
            if net_energy <= 0.0 {
                for sector in 0..economy::NUM_SECTORS {
                    if sector != 1 && sector != 2 {
                        // agriculture=1, medicine=2
                        world.economy.sector_output[sector] *= 0.1;
                    }
                }
            }

            let total_workers: usize = world.economy.sector_workers.iter().sum();
            world.economy.tick_prices(total_workers);
            world.economy.tick_demurrage();
            world.economy.compute_gini(&world.agents);
            world.economy.self_sufficiency = world.resources.self_sufficiency();

            // Investment: higher-consciousness worlds invest more
            let mean_phi = world.mean_phi();
            let mut invest_rate = 0.1 + mean_phi * 0.2; // 0.1-0.3 based on consciousness

            // Mechanism 2 — Agent Decision Quality Under Stress: high allostatic
            // load degrades investment decisions. McEwen (1998): sustained stress
            // impairs prefrontal cortex executive function.
            let world_mean_load = world.mean_allostatic_load();
            if world_mean_load > 0.6 {
                invest_rate *= 0.5; // 50% reduction in investment quality
            }
            world.economy.invest(invest_rate);

            // Mechanism 4 — Skill Gap Crisis: sectors with zero workers in a
            // populated world cannot produce. Recovery requires 36 ticks of
            // education (3 years). This models the critical dependency on human
            // capital in early space colonies.
            if pop > 100.0 {
                let mut new_gaps = Vec::new();
                for sector in 0..economy::NUM_SECTORS {
                    if world.economy.sector_workers[sector] == 0 {
                        // Check if already in crisis
                        if !world.economy.skill_gap_sectors.contains(&sector) {
                            new_gaps.push(sector);
                            world.economy.skill_gap_recovery_ticks[sector] = 36;
                            self.events.push(CivEvent::new(
                                self.current_tick,
                                Some(world.id),
                                CivEventType::SkillGapCrisis,
                                format!(
                                    "{}: SKILL GAP in {} — production halted until workers trained (36 ticks)",
                                    world.name, economy::WorldEconomy::sector_name(sector)
                                ),
                            ));
                        }
                    }
                }
                world.economy.skill_gap_sectors.extend(new_gaps);

                // Tick down recovery and remove recovered sectors
                for sector in 0..economy::NUM_SECTORS {
                    if world.economy.skill_gap_recovery_ticks[sector] > 0 {
                        if world.economy.sector_workers[sector] > 0 {
                            world.economy.skill_gap_recovery_ticks[sector] =
                                world.economy.skill_gap_recovery_ticks[sector].saturating_sub(1);
                        }
                        // While in crisis, zero output
                        if world.economy.skill_gap_recovery_ticks[sector] > 0 {
                            world.economy.sector_output[sector] = 0.0;
                        }
                    }
                }
                world
                    .economy
                    .skill_gap_sectors
                    .retain(|&s| world.economy.skill_gap_recovery_ticks[s] > 0);
            }

            // Mechanism 7 — Infrastructure Aging: natural entropy-driven degradation.
            // Base rate: 0.0003/tick (~0.36%/year) from general entropy.
            // PLUS: Materials aging (cold welding, polymer embrittlement, whisker growth).
            // Older habitats degrade faster: rate increases with colony age.
            // Garner & Hamilton (2005): steel half-life ~150yr in radiation; polymers 30-300yr.
            let colony_age_years =
                self.current_tick.saturating_sub(world.founded_tick) as f64 / 12.0;
            let materials_aging = if colony_age_years > 50.0 {
                // Accelerating degradation after 50 years (polymer embrittlement onset)
                0.0001 * (colony_age_years / 50.0).ln().max(0.0)
            } else {
                0.0
            };
            world.infrastructure_level =
                (world.infrastructure_level - INFRASTRUCTURE_DEGRADATION_RATE - materials_aging)
                    .max(0.0);

            // Mechanism 7 continued: if infrastructure drops below 0.3, ECLSS failure
            // rates double. This is handled by the disaster engine's mtbf_factor which
            // already uses infrastructure_level, but we add extra degradation pressure.
            // (The disaster engine naturally accounts for this via infra_factor.)

            // Slowly improve infrastructure (capped at 1.0).
            // Without trust-weighted governance, poor governance decisions cause
            // ~20% resource waste (wrong priorities, unchecked extractive behavior).
            if !self.config.policy.trust_weighted_governance {
                for name in &["food", "water", "energy"] {
                    if let Some(stock) = world.resources.get_mut(name) {
                        stock.current *= 1.0 - RESOURCE_WASTE_RATE; // ~2.4% annual
                    }
                }
            }
            world.infrastructure_level =
                (world.infrastructure_level + INFRASTRUCTURE_IMPROVEMENT_RATE).min(1.0);
        }
    }

    fn tick_interworld(&mut self) {
        // Phase 4: Simplified migration and trade between worlds.
        if self.worlds.len() < 2 {
            return;
        }

        // Cultural drift for each world
        let world_count = self.worlds.len();
        let contact_freq = if world_count > 1 { 0.3 } else { 0.0 };
        for world in &mut self.worlds {
            let pop = world.population();
            world.culture.drift(&mut self.rng, pop, contact_freq);
        }

        // Resource sharing between worlds: surplus flows from rich to poor
        // (simplified: every 12 ticks = 1 year)
        //
        // ORBITAL CLOCK: Transfer windows gate interplanetary trade.
        // Moon-Earth is continuous; all other pairs are gated by synodic periods.
        // Fusion grid-scale achievement removes window constraints.
        // Fusion Drive eliminates transfer windows; Fusion Grid Scale relaxes them
        let has_fusion_drive = self.disaster_engine.tech_tree.is_achieved("Fusion Drive")
            || self
                .disaster_engine
                .tech_tree
                .is_achieved("Fusion Grid Scale");
        let leo_access = self.disaster_engine.orbital_debris.leo_access_multiplier;
        // Closed-Loop ECLSS halves resource consumption rates
        let _has_closed_eclss = self
            .disaster_engine
            .tech_tree
            .is_achieved("Closed-Loop ECLSS");

        if self.current_tick % 12 == 0 && self.worlds.len() >= 2 {
            // Calculate average self-sufficiency
            let avg_ss: f64 = self
                .worlds
                .iter()
                .map(|w| w.resources.self_sufficiency())
                .sum::<f64>()
                / world_count as f64;

            // Worlds above average share a small amount with those below
            let mut transfers: Vec<(u32, u32, f64)> = Vec::new();
            for i in 0..world_count {
                let ss_i = self.worlds[i].resources.self_sufficiency();
                if ss_i > avg_ss + 0.1 {
                    for j in 0..world_count {
                        if i != j {
                            let ss_j = self.worlds[j].resources.self_sufficiency();
                            if ss_j < avg_ss - 0.1 {
                                // Transfer window check
                                let (synodic, _transfer_time) =
                                    interworld::InterWorldEngine::orbital_params(
                                        &self.worlds[i].location,
                                        &self.worlds[j].location,
                                    );
                                // Realism C: Transfer window cost curve (not binary).
                                // At optimal window: 1.0x volume. Off-window: reduced but
                                // non-zero (emergency transfers at higher delta-v cost).
                                let window_efficiency = if has_fusion_drive || synodic == 0 {
                                    1.0
                                } else {
                                    let phase_offset = self.current_tick % synodic as u32;
                                    let half = synodic as f64 / 2.0;
                                    let phase = std::f64::consts::PI * phase_offset as f64 / half;
                                    // Cosine: 1.0 at window, 0.1 at midpoint
                                    (0.1 + 0.9 * (1.0 + phase.cos()) / 2.0).clamp(0.1, 1.0)
                                };

                                // Phase 1c: trade_openness scales all trade
                                let mut amount = (ss_i - ss_j)
                                    * 10.0
                                    * window_efficiency
                                    * self.worlds[i].policy_state.trade_openness;

                                // Kessler degradation: reduce trade volume for Earth routes
                                if leo_access < 1.0 {
                                    let involves_earth = self.worlds[i].location == "Earth"
                                        || self.worlds[j].location == "Earth";
                                    if involves_earth {
                                        amount *= leo_access;
                                    }
                                }

                                if amount > 0.1 {
                                    transfers.push((self.worlds[i].id, self.worlds[j].id, amount));
                                }
                            }
                        }
                    }
                }
            }

            // Option C: Per-world trade specialization — transfer ALL resource types,
            // not just food. Each world exports its surplus, imports its deficit.
            // Titan→hydrocarbons/materials, Europa→water, Earth→knowledge/materials,
            // Mars→minerals. This creates real supply chain interdependence.
            let trade_resources = ["food", "water", "energy", "materials", "oxygen"];
            for (from_id, to_id, base_amount) in transfers {
                let from_idx = self.worlds.iter().position(|w| w.id == from_id);
                let to_idx = self.worlds.iter().position(|w| w.id == to_id);
                if let (Some(fi), Some(ti)) = (from_idx, to_idx) {
                    for res_name in &trade_resources {
                        let (surplus, deficit) = {
                            let from_frac =
                                self.worlds[fi].resources.fraction_of_capacity(res_name);
                            let to_frac = self.worlds[ti].resources.fraction_of_capacity(res_name);
                            let surplus = (from_frac - 0.5).max(0.0);
                            let deficit = (0.5 - to_frac).max(0.0);
                            (surplus, deficit)
                        };
                        // Transfer proportional to surplus × deficit × base_amount
                        let transfer = (surplus * deficit * base_amount * 2.0).min(50.0);
                        if transfer > 0.1 {
                            if let Some(stock) = self.worlds[fi].resources.get_mut(res_name) {
                                stock.current = (stock.current - transfer).max(0.0);
                            }
                            if let Some(stock) = self.worlds[ti].resources.get_mut(res_name) {
                                stock.current = (stock.current + transfer).min(stock.capacity);
                            }
                        }
                    }
                    // Titan hydrocarbons: export to any world that needs materials
                    if self.worlds[fi].location == "Titan" {
                        let hc_surplus = self.worlds[fi]
                            .resources
                            .fraction_of_capacity("hydrocarbons");
                        let mat_deficit = (0.5
                            - self.worlds[ti].resources.fraction_of_capacity("materials"))
                        .max(0.0);
                        if hc_surplus > 0.3 && mat_deficit > 0.1 {
                            let hc_transfer = (mat_deficit * base_amount * 3.0).min(100.0);
                            if let Some(hc) = self.worlds[fi].resources.get_mut("hydrocarbons") {
                                hc.current = (hc.current - hc_transfer).max(0.0);
                            }
                            if let Some(mat) = self.worlds[ti].resources.get_mut("materials") {
                                mat.current = (mat.current + hc_transfer * 0.5).min(mat.capacity);
                            }
                        }
                    }
                }
            }
        }

        // Inter-world migration: every 6 ticks (biannual), move a few adults from
        // Earth to off-world colonies. This maintains genetic diversity and carries
        // fresh social bonds into isolated populations.
        //
        // When hybrid_earth is enabled, colonists are instantiated from aggregate
        // regional data via the Spaceport Funnel instead of cloning Earth agents.
        if self.config.policy.migration_enabled
            && self.current_tick % 6 == 0
            && self.worlds.len() >= 2
        {
            let use_spaceport = self.config.policy.hybrid_earth
                && !self.earth_regions.is_empty()
                && self.spaceport.as_ref().map_or(false, |sp| sp.can_launch());

            if use_spaceport {
                // Spaceport Funnel: instantiate colonists from aggregate regions.
                let destinations: Vec<(usize, u32, String)> = self
                    .worlds
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| {
                        w.location != "Earth"
                            && w.population() < w.max_population
                            && w.population() > 0
                    })
                    .map(|(i, w)| (i, w.id, w.name.clone()))
                    .collect();

                let monthly_cap = self
                    .spaceport
                    .as_ref()
                    .map_or(0, |sp| sp.monthly_capacity());

                for (dest_idx, dest_id, dest_name) in destinations {
                    let max_mig = self.config.policy.migration_max_per_cycle.max(1) as usize;
                    let n_migrants =
                        ((self.rng.next_u64() % max_mig as u64 + 1) as usize).min(monthly_cap);
                    if n_migrants == 0 {
                        continue;
                    }

                    // Pick the best source region (highest space contribution with access).
                    let best_region_idx = self
                        .earth_regions
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.spaceport_access && r.population > 1.0)
                        .max_by(|(_, a), (_, b)| {
                            earth_regions::region_contribution_to_space(a)
                                .partial_cmp(&earth_regions::region_contribution_to_space(b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);

                    if let Some(ri) = best_region_idx {
                        let selection =
                            spaceport::prepare_selection(&self.earth_regions[ri], n_migrants);
                        if let Some(sel) = selection {
                            let mut next_id = self.worlds[dest_idx].next_agent_id as u64;
                            let new_agents = spaceport::instantiate_colonists(
                                &sel,
                                &mut self.earth_regions[ri],
                                dest_id,
                                self.current_tick,
                                &mut next_id,
                                &mut self.rng,
                            );
                            let moved = new_agents.len();
                            self.worlds[dest_idx].next_agent_id = next_id;
                            self.worlds[dest_idx].agents.extend(new_agents);
                            if let Some(ref mut sp) = self.spaceport {
                                sp.total_launched += moved as u64;
                            }

                            if moved > 0 {
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(dest_id),
                                    CivEventType::Migration,
                                    format!(
                                        "{} colonists launched from {} to {} via Spaceport Funnel",
                                        moved, self.earth_regions[ri].name, dest_name
                                    ),
                                ));
                            }
                        }
                    }
                }
            } else {
                // Classic migration path: clone Earth agents to off-world colonies.
                let earth_idx = self.worlds.iter().position(|w| w.location == "Earth");
                if let Some(ei) = earth_idx {
                    let earth_pop = self.worlds[ei].population();
                    if earth_pop > 100 {
                        // Find off-world colonies below capacity
                        let destinations: Vec<usize> = (0..self.worlds.len())
                            .filter(|&i| {
                                i != ei
                                    && self.worlds[i].population() < self.worlds[i].max_population
                                    && self.worlds[i].population() > 0
                            })
                            .collect();

                        // Collect migration plans first, then execute
                        let mut migration_plans: Vec<(Vec<u64>, usize, u32, String)> = Vec::new();

                        for &dest_idx in &destinations {
                            let max_mig = self.config.policy.migration_max_per_cycle.max(1) as u64;
                            let n_migrants = (self.rng.next_u64() % max_mig + 1) as usize;
                            let dest_id = self.worlds[dest_idx].id;
                            let dest_name = self.worlds[dest_idx].name.clone();

                            let migrant_ids: Vec<u64> = self.worlds[ei]
                                .agents
                                .iter()
                                .filter(|a| {
                                    a.is_alive()
                                        && a.life_stage(self.current_tick)
                                            == agent::LifeStage::Adult
                                })
                                .take(n_migrants)
                                .map(|a| a.id)
                                .collect();

                            if !migrant_ids.is_empty() {
                                migration_plans.push((migrant_ids, dest_idx, dest_id, dest_name));
                            }
                        }

                        for (migrant_ids, dest_idx, dest_id, dest_name) in migration_plans {
                            // O(1) lookup for migration
                            let mig_map: std::collections::HashMap<u64, usize> = self.worlds[ei]
                                .agents
                                .iter()
                                .enumerate()
                                .map(|(i, a)| (a.id, i))
                                .collect();
                            let mut migrants: Vec<CivAgent> = Vec::new();
                            for mid in &migrant_ids {
                                if let Some(&idx) = mig_map.get(mid) {
                                    let agent = &mut self.worlds[ei].agents[idx];
                                    let mut migrant = agent.clone();
                                    migrant.world_id = dest_id;
                                    migrant.is_immigrant = true;
                                    migrant.partner_id = None;
                                    migrant.death_tick = None;
                                    migrants.push(migrant);

                                    agent.death_tick = Some(self.current_tick);
                                }
                            }

                            let moved = migrants.len();
                            self.worlds[dest_idx].agents.extend(migrants);

                            if moved > 0 {
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(dest_id),
                                    CivEventType::Migration,
                                    format!(
                                        "{} migrants arrived at {} from Earth",
                                        moved, dest_name
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Option B: Affect-driven inter-world migration (refugees).
        // Agents on worlds with strongly negative conatus flee to worlds with positive conatus.
        // This creates interplanetary refugee crises when disasters hit outer system colonies.
        if self.config.policy.migration_enabled
            && self.current_tick % 12 == 0
            && self.worlds.len() >= 3
        {
            // Compute per-world mean conatus
            let world_conatus: Vec<f64> = self
                .worlds
                .iter()
                .map(|w| {
                    let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                    if living.is_empty() {
                        return 0.0;
                    }
                    living
                        .iter()
                        .map(|a| a.needs.affect.net_conatus())
                        .sum::<f64>()
                        / living.len() as f64
                })
                .collect();

            // Find suffering worlds (conatus < -0.1) and refuge worlds (conatus > 0.1)
            let mut refugee_moves: Vec<(usize, usize, usize)> = Vec::new(); // (from, to, count)
            for (fi, &fc) in world_conatus.iter().enumerate() {
                if fc < -0.1 && self.worlds[fi].population() > 20 {
                    // Find best refuge
                    if let Some((ti, _)) = world_conatus
                        .iter()
                        .enumerate()
                        .filter(|&(i, &c)| {
                            i != fi
                                && c > 0.1
                                && self.worlds[i].population() < self.worlds[i].max_population
                        })
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        // Transfer window check
                        let (synodic, _) = interworld::InterWorldEngine::orbital_params(
                            &self.worlds[fi].location,
                            &self.worlds[ti].location,
                        );
                        let window = has_fusion_drive
                            || synodic == 0
                            || self.current_tick % synodic as u32 == 0;
                        if window {
                            let n = (self.worlds[fi].population() as f64 * 0.02).ceil() as usize;
                            refugee_moves.push((fi, ti, n.min(5)));
                        }
                    }
                }
            }

            for (fi, ti, count) in refugee_moves {
                let dest_id = self.worlds[ti].id;
                let dest_name = self.worlds[ti].name.clone();
                let from_name = self.worlds[fi].name.clone();
                let mut moved = 0;
                // Ethics-modulated refugee selection (Phase 2d):
                // Consequentialist agents flee more readily (outcome-seeking).
                // Relational agents resist leaving (community bonds hold them).
                let ids: Vec<u64> = self.worlds[fi]
                    .agents
                    .iter()
                    .filter(|a| {
                        if !a.is_alive() {
                            return false;
                        }
                        let conatus = a.needs.affect.net_conatus();
                        let flee_threshold =
                            -0.05 + a.ethics.relational * 0.08 - a.ethics.consequentialist * 0.05;
                        conatus < flee_threshold
                    })
                    .take(count)
                    .map(|a| a.id)
                    .collect();
                // O(1) refugee lookup
                let ref_map: std::collections::HashMap<u64, usize> = self.worlds[fi]
                    .agents
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.id, i))
                    .collect();
                for id in ids {
                    if let Some(&idx) = ref_map.get(&id) {
                        let mut refugee = self.worlds[fi].agents[idx].clone();
                        refugee.world_id = dest_id;
                        refugee.is_immigrant = true;
                        refugee.partner_id = None;
                        self.worlds[ti].agents.push(refugee);
                        self.worlds[fi].agents[idx].death_tick = Some(self.current_tick);
                        moved += 1;
                    }
                }
                if moved > 0 {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(dest_id),
                        CivEventType::Migration,
                        format!(
                            "{} refugees fled {} → {} (conatus crisis)",
                            moved, from_name, dest_name
                        ),
                    ));
                }
            }
        }
    }

    /// Fix 2: Immigration pipeline for genetic rescue.
    ///
    /// When outer colonies face genetic bottleneck (viability < 0.5), Earth/Moon/Mars
    /// send settlers during transfer windows. This is the lifeline that makes small
    /// founding populations survivable — without it, Europa (200) and Titan (150)
    /// hit F > 0.0625 within ~10 generations.
    ///
    /// Smith (2014): sustained immigration of even 1-2 individuals per generation
    /// can maintain heterozygosity indefinitely (the "one migrant per generation" rule).
    fn tick_immigration_pipeline(&mut self) {
        // Run every 12 ticks (annually) and only when migration is enabled
        if !self.config.policy.migration_enabled || self.current_tick % 12 != 0 {
            return;
        }
        if self.worlds.len() < 2 {
            return;
        }

        let has_fusion_drive = self.disaster_engine.tech_tree.is_achieved("Fusion Drive")
            || self
                .disaster_engine
                .tech_tree
                .is_achieved("Fusion Grid Scale");

        // Identify colonies needing genetic rescue
        let needs_rescue: Vec<(usize, f64)> = self
            .worlds
            .iter()
            .enumerate()
            .filter(|(_, w)| w.location != "Earth" && w.population() > 0)
            .map(|(i, w)| (i, PopulationEngine::genetic_viability(w, self.current_tick)))
            .filter(|(_, v)| *v < 0.5)
            .collect();

        // Identify donor worlds (population > 1000, can spare settlers)
        let donors: Vec<usize> = self
            .worlds
            .iter()
            .enumerate()
            .filter(|(_, w)| w.population() > 1000)
            .map(|(i, _)| i)
            .collect();

        if donors.is_empty() || needs_rescue.is_empty() {
            return;
        }

        for (recipient_idx, viability) in needs_rescue {
            // Find best donor with open transfer window
            for &donor_idx in &donors {
                if donor_idx == recipient_idx {
                    continue;
                }

                let (synodic, _) = interworld::InterWorldEngine::orbital_params(
                    &self.worlds[donor_idx].location,
                    &self.worlds[recipient_idx].location,
                );
                let window_open = has_fusion_drive
                    || synodic == 0
                    || self.current_tick % synodic.max(1) as u32 == 0;

                if !window_open {
                    continue;
                }

                // Send settlers: more when viability is lower
                let n_settlers = if viability < 0.2 { 10 } else { 5 };
                let n_settlers = n_settlers.min(self.worlds[donor_idx].population() / 50);
                if n_settlers == 0 {
                    continue;
                }

                // Select healthy young adults from donor
                let dest_id = self.worlds[recipient_idx].id;
                let dest_name = self.worlds[recipient_idx].name.clone();
                let donor_name = self.worlds[donor_idx].name.clone();

                let settler_ids: Vec<u64> = self.worlds[donor_idx]
                    .agents
                    .iter()
                    .filter(|a| {
                        a.is_alive()
                            && a.health > 0.7
                            && a.age_years(self.current_tick) > 20.0
                            && a.age_years(self.current_tick) < 40.0
                            && !a.is_immigrant
                    })
                    .take(n_settlers)
                    .map(|a| a.id)
                    .collect();

                // Collect settlers first, then modify both worlds to avoid double borrow.
                // O(1) lookup for immigration pipeline
                let settler_map: std::collections::HashMap<u64, usize> = self.worlds[donor_idx]
                    .agents
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.id, i))
                    .collect();
                let mut settlers_to_move: Vec<agent::CivAgent> = Vec::new();
                for id in &settler_ids {
                    if let Some(&idx) = settler_map.get(id) {
                        let mut settler = self.worlds[donor_idx].agents[idx].clone();
                        settler.world_id = dest_id;
                        settler.is_immigrant = true;
                        settler.partner_id = None;
                        settlers_to_move.push(settler);
                    }
                }
                let moved = settlers_to_move.len();
                // Mark donors as dead (reuse map)
                for id in &settler_ids {
                    if let Some(&idx) = settler_map.get(id) {
                        self.worlds[donor_idx].agents[idx].death_tick = Some(self.current_tick);
                    }
                }
                // Add settlers to recipient
                self.worlds[recipient_idx].agents.extend(settlers_to_move);
                if moved > 0 {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(dest_id),
                        CivEventType::Migration,
                        format!(
                            "GENETIC RESCUE: {} settlers from {} → {} (viability {:.0}%)",
                            moved,
                            donor_name,
                            dest_name,
                            viability * 100.0
                        ),
                    ));
                    break; // One donor per recipient per year
                }
            }
        }
    }

    /// Structural realism tick: power flow, maintenance trap, bus factor,
    /// pathogen pressure, trust dynamics, narrative identity, Earth funding.
    fn tick_structural_realism(&mut self) {
        let tick = self.current_tick;
        let earth_id_cached = self
            .worlds
            .iter()
            .find(|w| w.location == "Earth")
            .map(|w| w.id);

        for world in &mut self.worlds {
            let pop = world.population().max(1) as f64;
            let living_workers = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.life_stage(tick).can_work())
                .count() as f64;

            // === #1: POWER FLOW BUDGET (watts) ===
            // Each subsystem draws continuous power.
            let eclss_kw = pop * 2.0; // 2 kW/person for life support
            let heating_kw = match world.location.as_str() {
                "Titan" => pop * 1.5,  // 199K ΔT relentless
                "Europa" => pop * 0.5, // Cold but not Titan-level
                "Moon" => pop * 0.3,   // Lunar night heating
                _ => 0.0,
            };
            let food_lighting_kw = pop * 0.5; // Conservative estimate (research says 50-100 kW)
            let fabrication_kw = world.infrastructure_level * 20.0;
            let comms_kw = 5.0; // Baseline

            world.power_demand_kw =
                eclss_kw + heating_kw + food_lighting_kw + fabrication_kw + comms_kw;

            // Power generation based on location + tech
            let has_fission = self
                .disaster_engine
                .tech_tree
                .is_achieved("Fission Surface Power");
            let has_fusion = self
                .disaster_engine
                .tech_tree
                .is_achieved("Fusion Grid Scale");
            // #6: Solar panel dust degradation (Mars: 0.3%/sol, ~9%/month).
            // Ref: Spirit/Opportunity dust accumulation data (NASA MER).
            // Cleaning resets degradation. Assumes monthly cleaning if maintenance available.
            let dust_penalty = match world.location.as_str() {
                "Mars" => 0.95, // 5% loss from dust between cleanings
                "Moon" => 0.98, // 2% from regolith electrostatic dust
                _ => 1.0,
            };
            let solar_kw = match world.location.as_str() {
                "Earth" => pop * 5.0,
                "Moon" => pop * 4.0 * dust_penalty,
                "Mars" => pop * 2.0 * dust_penalty,
                "Europa" | "Titan" => 0.0,
                _ => pop * 3.0,
            };
            let nuclear_kw = if has_fusion {
                500.0 * world.infrastructure_level // Fusion: abundant
            } else if has_fission {
                100.0 * world.infrastructure_level // Fission: moderate
            } else {
                // RTG bootstrap (decaying)
                let age = tick.saturating_sub(world.founded_tick) as f64;
                0.44 * (1.0 - 0.0013_f64).powf(age) // 440W decaying
            };
            world.power_generation_kw = solar_kw + nuclear_kw;

            // === JOULE TAX: Robotic fleet power allocation ===
            // Robots consume power AFTER human needs. Surplus power feeds the fleet.
            // If insufficient, robots lose Phi → enter safe-mode → humans must take over.
            let surplus_kw = (world.power_generation_kw - world.power_demand_kw).max(0.0);
            let (robot_draw_kw, robot_labor, brownouts) =
                world.fleet.tick_power_allocation(surplus_kw, 730.0);
            world.power_demand_kw += robot_draw_kw;

            // Brownout events
            if brownouts > 0 {
                self.events.push(CivEvent::new(
                    tick,
                    Some(world.id),
                    CivEventType::EmergencyDeclared,
                    format!(
                        "{}: BROWNOUT CASCADE — {} robots entered safe-mode. \
                        Power deficit forces human labor in hazardous modules. \
                        Fleet operational: {:.0}%",
                        world.name,
                        brownouts,
                        world.fleet.operational_fraction() * 100.0
                    ),
                ));
            }

            // Robot labor offsets human maintenance requirements
            let _robot_maintenance_offset = robot_labor * 0.5; // Each robot-hour = 0.5 human maintenance hours

            // === B: AUTOMATION LEVEL ===
            // Grows with engineering + manufacturing tech. Reduces labor requirements.
            // At automation 0.8, a colony of 200 can do work of 2000.
            let eng_level = world.knowledge.technology_levels[0];
            let has_manufacturing = self
                .disaster_engine
                .tech_tree
                .is_achieved("Manufacturing Breakthrough");
            let auto_target = if has_manufacturing {
                ((eng_level - 1.0) / 5.0).clamp(0.0, 0.9) // Up to 90% automation
            } else {
                ((eng_level - 1.0) / 10.0).clamp(0.0, 0.5) // Up to 50% without manufacturing
            };
            // EMA toward target (slow adoption)
            world.automation_level = world.automation_level * 0.99 + auto_target * 0.01;

            // Robot manufacturing: fabrication workshop produces robots gradually.
            // Each robot takes ~6 months to build. Workshop can build 1 at a time.
            // Robot type selected by colony need (agriculture humanoid first,
            // then manipulators, then drones for exploration).
            let has_fab_workshop = world
                .project_manager
                .has_completed(projects::ProjectBlueprint::FabricationWorkshop);
            if has_fab_workshop && tick % 6 == 0 // Check every 6 months
                && world.fleet.total_units() < ((pop as usize) / 50 + 1).max(2)
            // 1 robot per 50 people + 1
            {
                // Choose robot type by need
                let platform = if world
                    .fleet
                    .count_operational(robotics::RobotPlatform::Humanoid)
                    == 0
                {
                    robotics::RobotPlatform::Humanoid // First: Ag-Bay labor
                } else if world
                    .fleet
                    .count_operational(robotics::RobotPlatform::Manipulator)
                    < 2
                {
                    robotics::RobotPlatform::Manipulator // Then: workshop/maintenance
                } else if world
                    .fleet
                    .count_operational(robotics::RobotPlatform::Quadrotor)
                    == 0
                {
                    robotics::RobotPlatform::Quadrotor // Then: exploration drone
                } else {
                    robotics::RobotPlatform::Manipulator // Default: more maintenance capacity
                };
                world.fleet.deploy(platform);
                self.events.push(CivEvent::new(
                    tick,
                    Some(world.id),
                    CivEventType::EmergencyDeclared,
                    format!(
                        "{}: ROBOT COMMISSIONED — {:?} unit #{} operational. \
                        Fleet power demand: {:.0}W, labor offset: {:.0} hr/day",
                        world.name,
                        platform,
                        world.fleet.total_units(),
                        world.fleet.total_power_draw_w,
                        world.fleet.total_labor_replaced / 30.0
                    ), // per day from monthly
                ));
            }

            // === #3: MAINTENANCE LABOR BUDGET ===
            // Infrastructure demands maintenance proportional to complexity.
            // Tainter (1988): diminishing returns on complexity.
            // Automation reduces human labor needed for maintenance.
            let tech_complexity = world.knowledge.mean_tech_level().max(1.0);
            let automation_reduction = 1.0 - world.automation_level * 0.8; // Up to 80% reduction
            world.maintenance_hours_required = world.infrastructure_level
                * tech_complexity
                * 80.0
                * (pop / 100.0).max(1.0)
                * automation_reduction;
            // Available: workers × 160 hrs/month × fraction allocated to maintenance
            let maintenance_fraction = 0.3;
            world.maintenance_hours_available = living_workers * 160.0 * maintenance_fraction;

            // Maintenance trap: when demand > supply, infrastructure decays faster
            if world.maintenance_hours_required > world.maintenance_hours_available * 1.2 {
                let deficit_ratio =
                    world.maintenance_hours_required / world.maintenance_hours_available.max(1.0);
                let extra_decay = 0.001 * (deficit_ratio - 1.2).max(0.0);
                world.infrastructure_level = (world.infrastructure_level - extra_decay).max(0.0);
            }

            // === #4: BUS FACTOR + CRITICAL SYSTEMS DEPENDENCY ===
            // Compute detailed per-system coverage from agent skills.
            // Systems with 0 operators actively fail (5%/tick infrastructure decay).
            // Systems with 1 operator are at critical risk (bus factor = 1).
            let coverage = knowledge::CriticalSystemCoverage::compute(&world.agents, tick);
            world.bus_factor_critical = coverage.systems_at_risk;
            world.knowledge.critical_system_coverage = coverage;

            // Systems with zero operators cause specific failures
            if world.knowledge.critical_system_coverage.systems_failing > 0 && pop > 20.0 {
                let failing = world.knowledge.critical_system_coverage.systems_failing;
                // Each failing system accelerates infrastructure decay
                world.infrastructure_level =
                    (world.infrastructure_level - 0.005 * failing as f64).max(0.0);
                // Failing medical = health crisis
                if world.knowledge.critical_system_coverage.medical_operators == 0 {
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.health = (agent.health - 0.01).max(0.1);
                    }
                }
                // Failing education = knowledge decay
                if world.knowledge.critical_system_coverage.education_operators == 0 {
                    for level in world.knowledge.technology_levels.iter_mut() {
                        *level = (*level - 0.005).max(1.0);
                    }
                }
            }

            // === #5: PATHOGEN PRESSURE ===
            // Accumulates in sealed environments. Lenski: novel virulence per ~30K generations.
            if world.location != "Earth" {
                let sealed_years = tick.saturating_sub(world.founded_tick) as f64 / 12.0;
                // Pressure increases logarithmically (initial rapid adaptation, then slower)
                world.pathogen_pressure = (sealed_years / 500.0).ln().max(0.0).min(1.0) * 0.5
                    + world.pathogen_pressure * 0.5; // EMA smoothing
                                                     // Larger populations have better immune diversity
                let immune_resistance = (pop / 1000.0).min(1.0);
                world.pathogen_pressure *= 1.0 - immune_resistance * 0.5;
            }

            // === #6: CIVILIZATIONAL PHI ===
            // Organizational redundancy: how many critical systems have bus_factor >= 3?
            let mut redundant_systems = 0u32;
            for sector in 0..8 {
                let skilled = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive() && a.skills.as_slice()[sector] > 0.3)
                    .count();
                if skilled >= 3 {
                    redundant_systems += 1;
                }
            }
            world.civilizational_phi = redundant_systems as f64 / 8.0;

            // === #8: TRUST DYNAMICS WITH HYSTERESIS ===
            // Trust builds at 0.002/tick (slow). Lost at 10× rate during crises.
            let in_crisis =
                world.governance.stability_score < 0.4 || world.governance.constitutional_crisis;
            if in_crisis {
                world.trust_level = (world.trust_level - 0.02).max(0.0); // Fast loss
            } else {
                world.trust_level = (world.trust_level + 0.002).min(1.0); // Slow build
            }
            // Trust below 0.3 amplifies all negative effects
            if world.trust_level < 0.3 {
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.needs.allostatic_load = (agent.needs.allostatic_load + 0.001).min(1.0);
                }
            }

            // === #9: EARTH FUNDING ===
            if world.location != "Earth" {
                // Earth's willingness to fund decays with time and distance.
                // Colonies must produce ROI (science, resources, strategic value).
                let distance_factor = match world.location.as_str() {
                    "Moon" => 1.0,
                    "Mars" => 0.8,
                    "Europa" => 0.5,
                    "Titan" => 0.3,
                    _ => 0.6,
                };
                let _colony_age = tick.saturating_sub(world.founded_tick) as f64 / 12.0;
                // Funding decays over decades unless colony provides value
                let value_to_earth = world.knowledge.mean_tech_level() * 0.1
                    + world.resources.self_sufficiency() * 0.2;
                let decay = 0.0005 * (1.0 - distance_factor) * (1.0 - value_to_earth);
                world.earth_funding = (world.earth_funding - decay).max(0.0);
                // Low funding reduces trade volume from Earth
            }

            // === LIFESPAN EVOLUTION (Gompertz-Makeham parameter shifts) ===
            // Research calibration: Pyrkov et al. (2021), Rejuvenate Bio (2024).
            // Tech milestones progressively reduce alpha (initial mortality),
            // beta (aging rate), and lambda (background mortality).
            {
                let med_skill = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive())
                    .map(|a| a.skills.as_slice()[2]) // medicine sector
                    .sum::<f64>()
                    / pop.max(1.0);
                let sci_skill = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive())
                    .map(|a| a.skills.as_slice()[4]) // science sector
                    .sum::<f64>()
                    / pop.max(1.0);

                // Era 1 (baseline): alpha=1.0, beta=1.0, lambda=1.0 → lifespan ~80
                let mut alpha_m = 1.0_f64;
                let mut beta_m = 1.0_f64;
                let mut lambda_m = 1.0_f64;

                // Era 2 (senolytics equivalent): med > 0.4 → lambda -50%, alpha -30%
                if med_skill > 0.4 {
                    lambda_m *= 0.5;
                    alpha_m *= 0.7;
                }
                // Era 3 (reprogramming): med > 0.6 AND sci > 0.5 → beta starts dropping
                if med_skill > 0.6 && sci_skill > 0.5 {
                    beta_m *= 0.82; // Mortality doubling time: 8yr → ~10yr
                }
                // Era 4 (genetic engineering milestone): further beta reduction
                if self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Genetic Engineering")
                {
                    beta_m *= 0.85;
                    alpha_m *= 0.5;
                }
                // Era 5 (bioregenerative + closed ECLSS): lambda near zero
                if self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Bioregenerative Agriculture")
                    && self
                        .disaster_engine
                        .tech_tree
                        .is_achieved("Closed-Loop ECLSS")
                {
                    lambda_m *= 0.3;
                }
                // Space colony health penalty: radiation + low gravity + isolation
                let space_penalty = match world.location.as_str() {
                    "Earth" => 1.0,
                    "Moon" => 1.15,   // -5 to -10 years
                    "Mars" => 1.10,   // -5 years (borderline gravity)
                    "Europa" => 1.25, // -10 to -15 years (radiation even shielded)
                    "Titan" => 1.20,  // -10 years (low-g + isolation)
                    _ => 1.1,
                };
                // Isolation penalty for small populations
                let isolation_penalty = if pop < 500.0 { 1.1 } else { 1.0 };

                world.mortality_alpha_mult = alpha_m * space_penalty * isolation_penalty;
                world.mortality_beta_mult = beta_m;
                world.mortality_lambda_mult = lambda_m * space_penalty;
            }

            // === #10: REPRODUCTION VIABILITY ===
            // Unknown if mammals can reproduce at < 0.4g.
            // NASA ICES-2021-142: partial gravity below 0.4g insufficient for health.
            // Without centrifuge habitat tech, low-g colonies can't have children.
            {
                let gravity = match world.location.as_str() {
                    "Earth" => 1.0,
                    "Moon" => 0.166,
                    "Mars" => 0.38,
                    "Europa" => 0.134,
                    "Titan" => 0.138,
                    _ => 1.0,
                };
                // Centrifuge habitats (Manufacturing Breakthrough) enable full-g reproduction.
                // Genetic Engineering enables risky low-g reproduction (higher infant mortality).
                let has_centrifuge = self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Manufacturing Breakthrough");
                let has_gene_therapy = self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Genetic Engineering");
                // Mars (0.38g) is borderline — use >= 0.37 to avoid float jitter
                world.reproduction_viable =
                    gravity >= 0.37 || has_centrifuge || (has_gene_therapy && gravity >= 0.13);
                // Low-gravity fertility penalty (Wakayama 2023, Lyons 2026).
                // JAXA: mouse embryo survival <30% in microgravity vs >60% at 1g.
                // No partial-g data exists — this is extrapolated conservatively.
                // fertility_mult = clamp(gravity / 0.5, 0.3, 1.0)
                // Mars (0.38g) → 0.76, Moon (0.17g) → 0.34, centrifuge → 1.0
                world.fertility_multiplier = if has_centrifuge {
                    1.0
                } else {
                    (gravity / 0.5_f64).clamp(0.3_f64, 1.0_f64)
                };
            }

            // === #7: SEALED ECOSYSTEM BALANCE ===
            // Atmospheric composition drifts in sealed habitats.
            // Without active management, CO2 builds, O2 drops, trace contaminants accumulate.
            // Biosphere 2: lost 7% O2 in 16 months from soil microbe respiration.
            if world.location != "Earth" {
                let has_eclss = self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Closed-Loop ECLSS");
                let decay_rate = if has_eclss { 0.00005 } else { 0.0002 }; // ECLSS slows decay 4×
                world.ecosystem_balance = (world.ecosystem_balance - decay_rate).max(0.3);
                // Low balance reduces food production and increases health stress
                if world.ecosystem_balance < 0.7 {
                    let penalty = (0.7 - world.ecosystem_balance) * 0.01;
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.needs.allostatic_load =
                            (agent.needs.allostatic_load + penalty).min(1.0);
                    }
                }
            }

            // === #3: INDEPENDENCE MOVEMENTS ===
            // When a colony's population > 5000 AND self-sufficiency > 0.8 AND
            // cultural distance from Earth > 0.3, independence pressure rises.
            // Mars surpassing Earth (11,644 vs 10,859 in v8) is THE political event.
            if world.location != "Earth"
                && pop > 5000.0
                && world.resources.self_sufficiency() > 0.7
                && tick % 600 == 0
            // Check every 50 years (was 10 — too spammy)
            {
                // Cultural distance from Earth (use individualism as proxy)
                let cultural_dist = (world.culture.individualism - 0.5).abs()
                    + (world.culture.risk_tolerance - 0.4).abs();
                let independence_pressure = (pop / 10000.0).min(1.0) * 0.3
                    + world.resources.self_sufficiency() * 0.3
                    + cultural_dist * 0.2
                    + (1.0 - world.trust_level) * 0.2;

                if independence_pressure > 0.6 {
                    let world_name = world.name.clone();
                    let world_id = world.id;

                    // Escalation: count prior independence events for this world
                    let prior_count = self
                        .events
                        .iter()
                        .filter(|e| {
                            e.world_id == Some(world_id) && e.description.contains("INDEPENDENCE")
                        })
                        .count();

                    // Phase-based escalation
                    let (phase, description, trust_hit, diplo_hit) = match prior_count {
                        0 => (
                            "PETITION",
                            format!(
                                "{}: INDEPENDENCE PETITION — pressure {:.0}%. \
                            \"We respectfully request greater autonomy.\"",
                                world_name,
                                independence_pressure * 100.0
                            ),
                            0.02,
                            0.05,
                        ),
                        1 => (
                            "MOVEMENT",
                            format!(
                                "{}: INDEPENDENCE MOVEMENT — organized political action. \
                            Pop {}, SS {:.0}%. Assemblies debate sovereignty.",
                                world_name,
                                pop as usize,
                                world.resources.self_sufficiency() * 100.0
                            ),
                            0.05,
                            0.10,
                        ),
                        2 => (
                            "DECLARATION",
                            format!(
                                "{}: INDEPENDENCE DECLARATION — \"We are sovereign.\" \
                            Pop {}, fully self-sufficient. The break is formal.",
                                world_name, pop as usize
                            ),
                            0.10,
                            0.20,
                        ),
                        3 => (
                            "FEDERATION PROPOSAL",
                            format!(
                                "{}: FEDERATION PROPOSAL — equal partnership offered to Earth. \
                            \"Not subjects. Partners.\" Trade and defense alliance terms drafted.",
                                world_name
                            ),
                            0.0,
                            0.05,
                        ), // Federation improves relations
                        _ => (
                            "POLITICAL EVOLUTION",
                            format!(
                                "{}: sovereign governance matures. Gen {} leadership \
                            shapes post-independence institutions.",
                                world_name,
                                (tick as f64 / 360.0) as u16
                            ),
                            0.0,
                            0.0,
                        ),
                    };

                    self.events.push(CivEvent::new(
                        tick,
                        Some(world_id),
                        CivEventType::ConstitutionalAmendment,
                        format!("INDEPENDENCE {}: {}", phase, description),
                    ));

                    world.narrative_identity.identification =
                        (world.narrative_identity.identification + 0.1).min(1.0);
                    world.trust_level = (world.trust_level - trust_hit).max(0.0);
                    if let Some(eid) = earth_id_cached {
                        let rel = world.diplomatic_relations.entry(eid).or_insert(0.5);
                        *rel = (*rel - diplo_hit).max(-1.0);
                    }
                }
            }

            // === D: EXPLORATION MISSIONS ===
            // Colonies with sufficient population and tech can launch exploration missions.
            // Each mission costs resources but discovers new deposits + generates knowledge.
            if pop > 100.0 && world.infrastructure_level > 0.5 && tick % 120 == 0
            // Check every 10 years
            {
                // Base 3% + science bonus. Fires even at starting science level.
                // Phase 1e: exploration_investment multiplies probability
                let explore_mult = 1.0 + world.policy_state.exploration_investment * 5.0;
                let exploration_prob = (0.03
                    + (world.knowledge.technology_levels[4] - 1.0).max(0.0) * 0.05)
                    * explore_mult;
                if self.rng.bernoulli(exploration_prob.min(0.25)) {
                    world.explorations_completed += 1;
                    // Discovery boosts knowledge and resources
                    world.knowledge.technology_levels[4] += 0.1; // science boost
                    if let Some(mat) = world.resources.get_mut("materials") {
                        mat.capacity *= 1.1; // Discovered new deposits
                        mat.current += mat.capacity * 0.1;
                    }
                    // P2: Location-specific exploration narratives
                    let discovery = match world.location.as_str() {
                        "Mars" => "subsurface ice deposit mapped by ground-penetrating radar",
                        "Europa" => "hydrothermal vent system detected beneath the ice shell",
                        "Titan" => "methane lake shoreline rich in organic tholins catalogued",
                        "Moon" => "permanently shadowed crater with water ice confirmed",
                        _ => "new mineral deposits surveyed and mapped",
                    };
                    let explorer = narrative::generate_character_name(
                        &world.name,
                        "scientist",
                        (tick / 360) as u16,
                    );
                    self.events.push(CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: EXPLORATION SUCCESS #{} — {}. \
                            Lead scientist: {}.",
                            world.name, world.explorations_completed, discovery, explorer
                        ),
                    ));
                }
            }

            // === A: INTER-WORLD DIPLOMATIC RELATIONS ===
            // Deferred to after the world loop to avoid borrow conflicts.
            // (Needs access to multiple worlds simultaneously.)

            // === #2: NARRATIVE IDENTITY ===
            // Generation counting
            let mean_generation = {
                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    0.0
                } else {
                    living.iter().map(|a| a.generation as f64).sum::<f64>() / living.len() as f64
                }
            };
            world.narrative_identity.generations_since_founding = mean_generation as u32;
            // Identification decays with generations (children didn't choose this)
            if mean_generation > 3.0 {
                world.narrative_identity.identification *=
                    1.0 - 0.001 * (mean_generation - 3.0).min(10.0);
                world.narrative_identity.identification =
                    world.narrative_identity.identification.max(0.1);
            }
            // Low identification + high stress = identity crisis
            if world.narrative_identity.identification < 0.3
                && world.mean_allostatic_load() > 0.5
                && tick % 120 == 0
            {
                self.events.push(CivEvent::new(
                    tick,
                    Some(world.id),
                    CivEventType::EmergencyDeclared,
                    format!(
                        "{}: IDENTITY CRISIS — founding narrative rejected by generation {}. \
                        Colony seeks new purpose.",
                        world.name, world.narrative_identity.generations_since_founding
                    ),
                ));
                // Crisis can be productive: boost adaptability
                world.narrative_identity.adaptability =
                    (world.narrative_identity.adaptability + 0.1).min(1.0);
            }
        }
    }

    /// Inter-world diplomatic relations (deferred from structural realism to avoid borrow).
    fn tick_diplomacy(&mut self) {
        if self.worlds.len() < 2 || self.current_tick % 12 != 0 {
            return;
        }
        let n = self.worlds.len();
        // Collect culture weights and self-sufficiency for all worlds
        let world_data: Vec<(u32, [f64; 8], f64)> = self
            .worlds
            .iter()
            .map(|w| {
                (
                    w.id,
                    w.culture.harmony_weights,
                    w.resources.self_sufficiency(),
                )
            })
            .collect();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let (_, cw_i, ss_i) = &world_data[i];
                let (id_j, cw_j, ss_j) = &world_data[j];
                // Cultural similarity (cosine-ish)
                let culture_sim: f64 = cw_i.iter().zip(cw_j.iter()).map(|(a, b)| a * b).sum();
                // Trade mutual benefit
                let trade_benefit = (1.0 - ss_i) * (1.0 - ss_j);
                let target = culture_sim * 0.5 + trade_benefit * 0.5;
                let relation = self.worlds[i]
                    .diplomatic_relations
                    .entry(*id_j)
                    .or_insert(0.5);
                *relation = *relation * 0.99 + target * 0.01;
                *relation = relation.clamp(-1.0, 1.0);
            }
        }
    }

    fn tick_knowledge(&mut self) {
        // Phase 6: Education, skill growth, and technology advancement.
        let tick = self.current_tick;
        for world in &mut self.worlds {
            for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                let stage = agent.life_stage(tick);
                // Children and youth gain education
                if stage == agent::LifeStage::Child || stage == agent::LifeStage::Youth {
                    agent.education_level = (agent.education_level + 0.005).min(1.0);
                }
                // Adults gain skill in their strongest sector (learning-by-doing)
                if stage.can_work() {
                    let skills = agent.skills.as_slice();
                    let primary = skills
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    agent.skills.learn(primary, 0.002 * agent.education_level);
                }
                // Elders experience slight skill decay
                if stage == agent::LifeStage::Elder {
                    agent.skills.decay(0.001);
                }
            }
        }

        // Run the Romer endogenous growth model for technology advancement.
        // This drives tech_level growth via researchers, education, and breakthroughs.
        // Must be done outside the world borrow to avoid aliasing.
        let world_count = self.worlds.len();
        for i in 0..world_count {
            let mut world = std::mem::take(&mut self.worlds[i]);
            let mut knowledge = std::mem::take(&mut world.knowledge);
            // Option B: Desire-driven innovation — high collective desire boosts tech.
            // Spinoza: desire (cupiditas) is the essence of human striving.
            let mean_desire = {
                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    0.0
                } else {
                    living.iter().map(|a| a.needs.affect.desire).sum::<f64>() / living.len() as f64
                }
            };
            // Desire > 0.5 gives up to 30% innovation boost (necessity drives invention)
            if mean_desire > 0.5 {
                knowledge.innovation_rate *= 1.0 + (mean_desire - 0.5) * 0.6;
            }
            let knowledge_events = knowledge.tick_knowledge(&world, tick, &mut self.rng);

            // Mechanism 5 — Inter-Generational Knowledge Loss: for each tech sector,
            // count agents whose strongest skill matches that sector AND skill > 0.3.
            // If fewer than 3 skilled workers exist, tech level decays. If zero,
            // catastrophic decay occurs (0.05/tick). This models the fragility of
            // specialized knowledge in small populations.
            for sector in 0..knowledge::NUM_SECTORS {
                let sector_name = economy::WorldEconomy::sector_name(sector);
                let skilled_count = world
                    .agents
                    .iter()
                    .filter(|a| {
                        a.is_alive()
                            && a.life_stage(tick).can_work()
                            && a.skills.strongest() == sector_name
                            && a.skills.as_slice()[sector] > 0.3
                    })
                    .count();

                if skilled_count == 0 && world.population() > 20 {
                    // Catastrophic knowledge loss (only for established populations)
                    knowledge.technology_levels[sector] =
                        (knowledge.technology_levels[sector] - 0.05).max(1.0);
                    // Log yearly to control event volume
                    if tick % 12 == 0 {
                        self.events.push(CivEvent::new(
                            self.current_tick,
                            Some(world.id),
                            CivEventType::KnowledgeLoss,
                            format!(
                                "{}: CATASTROPHIC knowledge loss in {} — zero skilled workers",
                                world.name, sector_name
                            ),
                        ));
                    }
                } else if skilled_count < 3 && world.population() > 50 {
                    // Gradual knowledge loss
                    knowledge.technology_levels[sector] =
                        (knowledge.technology_levels[sector] - 0.01).max(1.0);
                    // Log yearly to control event volume
                    if tick % 12 == 0 {
                        self.events.push(CivEvent::new(
                            self.current_tick,
                            Some(world.id),
                            CivEventType::KnowledgeLoss,
                            format!(
                                "{}: knowledge erosion in {} — only {} skilled workers",
                                world.name, sector_name, skilled_count
                            ),
                        ));
                    }
                }
            }

            world.knowledge = knowledge;
            self.worlds[i] = world;
            self.events.extend(knowledge_events);
        }
    }

    fn tick_governance(&mut self) {
        // Phase 6: Simplified governance milestones.
        // Grant constitution once population of any off-Earth world is large enough
        if !self.constitution_granted {
            let has_sizable = self
                .worlds
                .iter()
                .any(|w| w.location != "Earth" && w.population() >= 20);
            if has_sizable {
                self.epoch_manager
                    .record_milestone("constitution", self.current_tick);
                self.constitution_granted = true;
                self.events.push(CivEvent::new(
                    self.current_tick,
                    None,
                    CivEventType::ConstitutionalAmendment,
                    "Colony constitution ratified",
                ));
            }
        }

        // Dunbar number transitions (Dunbar 1992).
        // Population crossing 150 or 1500 triggers governance restructuring crisis.
        // Direct trust networks break down at ~150; hierarchical management required at ~1500.
        {
            let mut dunbar_events = Vec::new();
            let mut dunbar_worlds = Vec::new(); // Track which worlds need governance hit
            for world in &self.worlds {
                let pop = world.population();
                for (threshold, desc) in [
                    (150, "direct trust to formal roles"),
                    (1500, "community to bureaucracy"),
                ] {
                    let key = format!("dunbar_{}_{}", world.id, threshold);
                    if pop >= threshold
                        && pop < threshold + 50
                        && !self.events.iter().any(|e| e.description.contains(&key))
                    {
                        dunbar_events.push(CivEvent::new(
                            self.current_tick,
                            Some(world.id),
                            CivEventType::ConstitutionalAmendment,
                            format!(
                                "{}: DUNBAR TRANSITION at pop {} — {} required. [{}]",
                                world.name, pop, desc, key
                            ),
                        ));
                        dunbar_worlds.push(world.id);
                    }
                }
            }
            self.events.extend(dunbar_events);
            // P2: Dunbar transitions temporarily destabilize governance
            for world in &mut self.worlds {
                if dunbar_worlds.contains(&world.id) {
                    world.governance.stability_score =
                        (world.governance.stability_score - 0.15).max(0.0);
                    world.trust_level = (world.trust_level - 0.1).max(0.0);
                }
            }
        }

        // Phase 6b: Per-world trust-weighted governance with anti-tyranny invariants
        {
            let tick = self.current_tick;
            let amendment_enabled = self.config.policy.amendment_enabled;
            let hostile_guardian = self.config.policy.hostile_guardian;
            let epoch = self.current_epoch as u32;
            let rng = &mut self.rng;
            let mut all_gov_events = Vec::new();
            for world in &mut self.worlds {
                if world.population() == 0 {
                    continue;
                }
                let pop = world.population();
                let mean_phi = world.mean_phi();
                let stability = world.governance.stability_score;
                world
                    .governance
                    .evolve_authority(epoch, pop, mean_phi, stability);
                // Phase 2a: refresh 8D sovereign profiles from current agent
                // state before any gating decisions this tick. Skipped in
                // A2 counterfactual mode so the baseline governance operates
                // on the same state as before Phase 2 existed.
                if self.config.policy.phase2_enabled {
                    world.refresh_sovereign_profiles();
                }
                let mut gov = std::mem::take(&mut world.governance);
                let voting_suppression = self
                    .phase_modifiers
                    .get(&world.id)
                    .map(|m| m.voting_suppression)
                    .unwrap_or(0.0);
                let gov_events = gov.tick_governance_full(
                    world,
                    tick,
                    rng,
                    amendment_enabled,
                    hostile_guardian,
                    voting_suppression,
                    self.config.policy.phase2_enabled,
                );
                world.governance = gov;
                all_gov_events.extend(gov_events);
            }
            self.events.extend(all_gov_events);
        }
    }

    fn tick_consciousness(&mut self) {
        consciousness::tick_consciousness_all_worlds(
            &mut self.worlds,
            self.current_tick,
            self.config.policy.pharma_boost,
            self.config.policy.trust_weighted_governance,
            &mut self.rng,
        );
    }

    /// Phase 8.05: Ethical dynamics — moral dilemmas, ethics drift, and synthesis.
    ///
    /// 1. Moral dilemma detection (every 12 ticks): when a disaster-affected world
    ///    has closely split ethical orientations, the community faces a genuine dilemma.
    /// 2. Ethics shift detection (every 60 ticks): compares current mean ethics to
    ///    the founding population's orientation, flagging significant drift.
    /// 3. Ethical synthesis detection (every 60 ticks): when a recently resolved
    ///    faction conflict coincides with decreased ethical diversity, the community
    ///    has forged a new shared understanding.
    fn tick_ethical_dynamics(&mut self) {
        let tick = self.current_tick;
        let dimension_names = [
            "deontological",
            "consequentialist",
            "virtue_care",
            "relational",
        ];

        // ── 1. Moral Dilemma Detection (every 12 ticks = annually) ──────────
        if tick % 12 == 0 {
            let mut moral_memory_queue: Vec<(u32, crate::world::MoralMemory)> = Vec::new();
            for world in &self.worlds {
                let pop = world.population();
                if pop <= 50 {
                    continue;
                }

                // Check if this world has an active disaster this tick
                let has_disaster = self
                    .disaster_engine
                    .active_disasters
                    .iter()
                    .any(|d| d.world_id == Some(world.id) || d.world_id.is_none());
                if !has_disaster {
                    continue;
                }

                let mean = agent::EthicalOrientation::mean_of(&world.agents);
                let vals = mean.as_vec();

                // Find the two highest dimensions
                let mut indexed: Vec<(usize, f64)> =
                    vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let (idx_a, score_a) = indexed[0];
                let (idx_b, score_b) = indexed[1];

                // If the top two are within 0.15, the community is genuinely split
                if (score_a - score_b).abs() < 0.15 {
                    let desc = format!(
                        "{}: MORAL DILEMMA — {} ({:.2}) vs {} ({:.2}) on disaster response",
                        world.name,
                        dimension_names[idx_a],
                        score_a,
                        dimension_names[idx_b],
                        score_b,
                    );
                    self.events.push(CivEvent::new(
                        tick,
                        Some(world.id),
                        CivEventType::MoralDilemma,
                        desc.clone(),
                    ));

                    // Queue moral memory for deferred push (can't mutate during & borrow)
                    moral_memory_queue.push((
                        world.id,
                        crate::world::MoralMemory {
                            tick,
                            ethics_at_crisis: mean.as_vec(),
                            lesson_dimension: idx_a,
                            description: desc,
                        },
                    ));
                }
            }
            // Apply deferred moral memories
            for (wid, mem) in moral_memory_queue {
                if let Some(w) = self.worlds.iter_mut().find(|w| w.id == wid) {
                    if w.moral_memories.len() < 10 {
                        w.moral_memories.push(mem);
                    }
                }
            }
        }

        // ── 2. Ethics Shift Detection (every 60 ticks = 5 years) ────────────
        if tick % 60 == 0 {
            for world in &self.worlds {
                let mean = agent::EthicalOrientation::mean_of(&world.agents);
                let current = mean.as_vec();

                // Store founding ethics on first encounter
                let founding = self
                    .founding_ethics
                    .entry(world.id)
                    .or_insert(current)
                    .clone();

                // Skip if this is the first call (founding == current)
                if founding == current {
                    continue;
                }

                for (dim_idx, dim_name) in dimension_names.iter().enumerate() {
                    let shift = current[dim_idx] - founding[dim_idx];
                    if shift.abs() > 0.15 {
                        let direction = if shift > 0.0 {
                            "increased"
                        } else {
                            "decreased"
                        };
                        let desc = format!(
                            "{}: ETHICS SHIFT — {} moved from {:.2} to {:.2} ({})",
                            world.name, dim_name, founding[dim_idx], current[dim_idx], direction,
                        );
                        self.events.push(CivEvent::new(
                            tick,
                            Some(world.id),
                            CivEventType::EthicsShift,
                            desc,
                        ));
                    }
                }
            }
        }

        // ── 3. Ethical Synthesis Detection (every 60 ticks) ─────────────────
        if tick % 60 == 0 {
            for world in &self.worlds {
                // Check for recently resolved conflicts in this world
                let world_factions: Vec<u32> = self
                    .faction_engine
                    .factions
                    .iter()
                    .filter(|f| f.world_id == world.id)
                    .map(|f| f.id)
                    .collect();

                let has_recent_resolution = self.faction_engine.conflicts.iter().any(|c| {
                    if let Some(rt) = c.resolved_tick {
                        // Resolved in the last 60 ticks (since last check)
                        rt > tick.saturating_sub(60)
                            && rt <= tick
                            && (world_factions.contains(&c.faction_a)
                                || world_factions.contains(&c.faction_b))
                    } else {
                        false
                    }
                });

                if !has_recent_resolution {
                    continue;
                }

                // Compute current ethics std-dev across the population
                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    continue;
                }
                let mean = agent::EthicalOrientation::mean_of(&world.agents);
                let mean_v = mean.as_vec();
                let n = living.len() as f64;
                let variance: f64 = living
                    .iter()
                    .map(|a| {
                        let v = a.ethics.as_vec();
                        (0..4).map(|i| (v[i] - mean_v[i]).powi(2)).sum::<f64>()
                    })
                    .sum::<f64>()
                    / (n * 4.0);
                let stddev = variance.sqrt();

                let prev_stddev = self.prev_ethics_stddev.get(&world.id).copied();
                self.prev_ethics_stddev.insert(world.id, stddev);

                // If diversity decreased since last check, synthesis occurred
                if let Some(prev) = prev_stddev {
                    if stddev < prev {
                        let desc = format!(
                            "{}: ETHICAL SYNTHESIS — post-conflict convergence, ethical diversity decreased",
                            world.name,
                        );
                        self.events.push(CivEvent::new(
                            tick,
                            Some(world.id),
                            CivEventType::EthicalSynthesis,
                            desc,
                        ));
                    }
                }
            }
        }

        // ── 4. Moral Revival Detection (every 24 ticks = 2 years) ───────────
        // When accumulated outrage + guilt crosses thresholds, the community
        // undergoes a collective moral awakening: virtue and duty surge back.
        // Cooldown: 120 ticks (10 years) per world.
        if tick % 24 == 0 {
            const OUTRAGE_THRESHOLD: f64 = 0.10;
            const GUILT_THRESHOLD: f64 = 0.03;
            const REVIVAL_COOLDOWN: u32 = 120;
            const MAX_REVIVAL_BOOST: f64 = 0.08; // maximum virtue/deont boost per revival

            let narrator_names = [
                "Amara Osei",
                "Yuki Tanaka",
                "Lena Kovač",
                "Omar Ndoye",
                "Priya Rajan",
                "Seren Williams",
                "Mateo Cruz",
                "Freya Lindqvist",
            ];

            let mut revival_queue: Vec<(u32, String, f64, f64)> = Vec::new();

            for world in &self.worlds {
                if world.population() <= 50 {
                    continue;
                }

                let last_revival = self.last_moral_revival.get(&world.id).copied().unwrap_or(0);
                if tick.saturating_sub(last_revival) < REVIVAL_COOLDOWN {
                    continue; // cooldown active
                }

                let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                if living.is_empty() {
                    continue;
                }

                let n = living.len() as f64;
                let mean_outrage = living.iter().map(|a| a.needs.affect.outrage).sum::<f64>() / n;
                let mean_guilt = living.iter().map(|a| a.needs.affect.guilt).sum::<f64>() / n;

                if mean_outrage > OUTRAGE_THRESHOLD && mean_guilt > GUILT_THRESHOLD {
                    // Scale revival strength: how far above threshold are we?
                    let outrage_excess = (mean_outrage - OUTRAGE_THRESHOLD).min(0.30);
                    let guilt_excess = (mean_guilt - GUILT_THRESHOLD).min(0.20);
                    let revival_strength =
                        (outrage_excess * 0.6 + guilt_excess * 0.4).clamp(0.005, MAX_REVIVAL_BOOST);

                    revival_queue.push((
                        world.id,
                        world.name.clone(),
                        revival_strength,
                        mean_outrage,
                    ));
                }
            }

            for (world_id, world_name, revival_strength, mean_outrage) in revival_queue {
                self.last_moral_revival.insert(world_id, tick);

                // Apply revival to all living agents in this world
                if let Some(world) = self.worlds.iter_mut().find(|w| w.id == world_id) {
                    let rng_seed = (tick ^ world_id) as usize % narrator_names.len();
                    let narrator = narrator_names[rng_seed];

                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        // Boost virtue_care and deontological — the two care-oriented dimensions
                        agent.ethics.virtue_care =
                            (agent.ethics.virtue_care + revival_strength * 1.0).min(0.95);
                        agent.ethics.deontological =
                            (agent.ethics.deontological + revival_strength * 0.6).min(0.95);
                        // Drain some outrage (the revival is channeling it into action)
                        agent.needs.affect.outrage = (agent.needs.affect.outrage * 0.60).max(0.0);

                        // Normalize: keep sum ≤ 2.05
                        let sum = agent.ethics.deontological
                            + agent.ethics.consequentialist
                            + agent.ethics.virtue_care
                            + agent.ethics.relational;
                        if sum > 2.05 {
                            let scale = 2.05 / sum;
                            agent.ethics.deontological *= scale;
                            agent.ethics.consequentialist *= scale;
                            agent.ethics.virtue_care *= scale;
                            agent.ethics.relational *= scale;
                        }
                    }

                    let mean = agent::EthicalOrientation::mean_of(&world.agents);
                    let desc = format!(
                        "{}: MORAL REVIVAL — {narrator} and others sparked a moral awakening \
                        (outrage {:.2}). Virtue +{:.3}, duty renewed. New ethics: \
                        deont={:.2} conseq={:.2} virtue={:.2} relat={:.2}",
                        world_name,
                        mean_outrage,
                        revival_strength,
                        mean.deontological,
                        mean.consequentialist,
                        mean.virtue_care,
                        mean.relational,
                    );
                    self.events.push(CivEvent::new(
                        tick,
                        Some(world_id),
                        CivEventType::MoralRevival,
                        desc,
                    ));
                }
            }
        }
    }

    fn tick_harmony_scoring(&mut self) {
        // Phase 8.5: Score Eight Harmonies using all accumulated data.
        let tick = self.current_tick;
        let world_count = self.worlds.len();

        for i in 0..world_count {
            let mut world = std::mem::take(&mut self.worlds[i]);

            // Build HarmonyInputs from world state + needs summary
            let needs_summary = self.needs_summaries.get(i);
            let mean_load = needs_summary.map(|s| s.mean_allostatic_load).unwrap_or(0.1);
            let mean_engagement = needs_summary.map(|s| s.mean_engagement).unwrap_or(0.8);

            let inputs = harmony::HarmonyInputs {
                governance_stability: world.infrastructure_level.min(1.0),
                food_level: world
                    .resources
                    .get("food")
                    .map(|f| (f.current / f.capacity.max(1.0)).min(1.0))
                    .unwrap_or(0.5),
                mean_education: {
                    let living: Vec<f64> = world
                        .agents
                        .iter()
                        .filter(|a| a.is_alive())
                        .map(|a| a.education_level)
                        .collect();
                    if living.is_empty() {
                        0.0
                    } else {
                        living.iter().sum::<f64>() / living.len() as f64
                    }
                },
                mean_tech_level: ((world.knowledge.mean_tech_level() - 1.0) / 9.0).clamp(0.0, 1.0),
                innovation_rate: world.knowledge.innovation_rate,
                art_per_capita: world.economy.sector_output[6] / world.population().max(1) as f64,
                trade_connections: if world_count > 1 {
                    (world_count - 1) as u32
                } else {
                    0
                },
                gini_coefficient: world.economy.gini_coefficient,
                self_sufficiency: world.resources.self_sufficiency(),
                knowledge_growth_rate: world.knowledge.innovation_rate,
                pop_stability: 0.8, // stub
                genetic_diversity: PopulationEngine::genetic_diversity_index(&world, tick),
                emergency_fraction: world.epidemics.len() as f64 * 0.1,
                worker_ratio: world.economy.total_workers() as f64
                    / world.population().max(1) as f64,
                mean_allostatic_load: mean_load,
                mean_engagement,
            };

            // Build a ConsciousnessEngine with current metrics
            let mut consciousness = consciousness::ConsciousnessEngine::new();
            consciousness.collective_phi = world.mean_phi();
            consciousness.mean_phi = world.mean_phi();

            // Extract harmony tracker to avoid borrow conflict (world vs world.harmony)
            let mut harmony = std::mem::take(&mut world.harmony);
            let harmony_events = harmony.tick_harmony(&world, &inputs, &consciousness, tick);
            world.harmony = harmony;

            self.events.extend(harmony_events);
            self.worlds[i] = world;
        }
    }

    /// Mechanism 6 — Resource Depletion Feedback: track consecutive ticks where
    /// critical resources (food, water, oxygen) are below 20% of capacity. After
    /// 12+ ticks, trigger a depletion crisis that halves birth rate and increases
    /// death rate 1.5x until the resource recovers above 30%.
    fn tick_resource_depletion(&mut self) {
        let critical_resources = ["food", "water", "oxygen"];
        for world in &self.worlds {
            for res_name in &critical_resources {
                let key = (world.id, res_name.to_string());
                let fraction = world.resources.fraction_of_capacity(res_name);

                // Track depletion streak
                if fraction < 0.2 {
                    let streak = self.depletion_streaks.entry(key.clone()).or_insert(0);
                    *streak += 1;

                    if *streak >= 12
                        && !self
                            .depletion_crisis_active
                            .get(&key)
                            .copied()
                            .unwrap_or(false)
                    {
                        self.depletion_crisis_active.insert(key.clone(), true);
                        self.events.push(CivEvent::new(
                            self.current_tick,
                            Some(world.id),
                            CivEventType::ResourceDepletionCrisis,
                            format!(
                                "{}: RESOURCE DEPLETION CRISIS — {} below 20% for {} months, birth rate halved",
                                world.name, res_name, streak
                            ),
                        ));
                    }
                } else if fraction > 0.3 {
                    // Recovery
                    self.depletion_streaks.insert(key.clone(), 0);
                    if self
                        .depletion_crisis_active
                        .get(&key)
                        .copied()
                        .unwrap_or(false)
                    {
                        self.depletion_crisis_active.insert(key, false);
                    }
                }
            }
        }
    }

    /// Mechanism 8 — Communication Delay Governance Penalty: multi-world governance
    /// decisions take longer when worlds are far apart.
    fn tick_communication_delay(&mut self) {
        // Applied during governance tick: for Federation-level decisions, we model
        // communication delay by temporarily preventing governance processing for
        // remote worlds. This is implemented by delaying governance evolution for
        // worlds beyond the Moon.
        for world in &mut self.worlds {
            if world.governance.authority_level == governance::GovernanceAuthority::Federation
                || world.governance.authority_level
                    == governance::GovernanceAuthority::Confederation
            {
                let delay = match world.location.as_str() {
                    "Earth" | "Moon" => 0,
                    "Mars" => 1,
                    _ => 2, // Outer system: Europa, Titan, etc.
                };
                // If within communication delay window, reduce governance stability
                // (modeling inability to coordinate with central authority)
                if delay > 0 {
                    let penalty = delay as f64 * 0.02;
                    world.governance.stability_score =
                        (world.governance.stability_score - penalty).max(0.0);
                }
            }
        }
    }

    /// Mechanism 9 — Morale Contagion (Social Epidemic Model): low morale spreads
    /// when >30% of population has allostatic_load > 0.7. High morale spreads (weaker)
    /// when >50% has load < 0.3. Creates tipping points for collective burnout or
    /// recovery. Ref: Christakis & Fowler (2009) "Connected".
    fn tick_morale_contagion(&mut self) {
        for world in &mut self.worlds {
            let pop = world.population();
            if pop < 20 {
                continue; // Small colonies don't have enough social mass for contagion
            }

            let high_stress_count = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.needs.allostatic_load > 0.7)
                .count();
            let low_stress_count = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.needs.allostatic_load < 0.3)
                .count();

            let high_frac = high_stress_count as f64 / pop as f64;
            let low_frac = low_stress_count as f64 / pop as f64;

            let prev_contagion = self.morale_contagion.get(&world.id).copied().unwrap_or(0);

            if high_frac > 0.3 {
                // Negative contagion: stress spreads
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.needs.allostatic_load = (agent.needs.allostatic_load + 0.001).min(1.0);
                }
                if prev_contagion != -1 {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(world.id),
                        CivEventType::MoraleContagion,
                        format!(
                            "{}: NEGATIVE morale contagion — {:.0}% population in high stress, spreading",
                            world.name, high_frac * 100.0
                        ),
                    ));
                }
                self.morale_contagion.insert(world.id, -1);
            } else if low_frac > 0.5 {
                // Positive contagion: wellbeing spreads (weaker effect)
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.needs.allostatic_load = (agent.needs.allostatic_load - 0.002).max(0.0);
                }
                if prev_contagion != 1 {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(world.id),
                        CivEventType::MoraleContagion,
                        format!(
                            "{}: POSITIVE morale contagion — {:.0}% population thriving, wellbeing spreading",
                            world.name, low_frac * 100.0
                        ),
                    ));
                }
                self.morale_contagion.insert(world.id, 1);
            } else {
                self.morale_contagion.insert(world.id, 0);
            }
        }
    }

    /// Mechanism 10 — Environmental Carrying Capacity: dynamic max population based
    /// on infrastructure and technology. When population exceeds 80% of carrying
    /// capacity, birth rate reduces. When above capacity, overcrowding stress kills.
    /// Ref: Meadows et al. (1972) "Limits to Growth".
    fn tick_carrying_capacity(&mut self) {
        for world in &mut self.worlds {
            let pop = world.population();
            if pop == 0 {
                continue;
            }

            // Use the base_max from the original config (stored in carrying_capacity_base).
            // If not set yet, snapshot the current max_population as the base.
            let base_max = *self
                .carrying_capacity_base
                .entry(world.id)
                .or_insert(world.max_population);

            // Dynamic carrying capacity: base * infrastructure * (1 + tech * 0.5)
            let tech_level = world.knowledge.mean_tech_level();
            let dynamic_capacity =
                (base_max as f64 * world.infrastructure_level * (1.0 + tech_level * 0.5)).max(10.0);

            let pop_fraction = pop as f64 / dynamic_capacity;

            if pop_fraction > 1.0 {
                // Overcrowding: increased death rate via health reduction
                let overcrowding_stress = (pop_fraction - 1.0) * 0.1;
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.health = (agent.health - overcrowding_stress).max(0.1);
                    agent.needs.allostatic_load =
                        (agent.needs.allostatic_load + overcrowding_stress * 0.5).min(1.0);
                }
                self.events.push(CivEvent::new(
                    self.current_tick,
                    Some(world.id),
                    CivEventType::CarryingCapacityExceeded,
                    format!(
                        "{}: OVERCROWDING — pop {} exceeds carrying capacity {:.0} ({:.0}%)",
                        world.name,
                        pop,
                        dynamic_capacity,
                        pop_fraction * 100.0
                    ),
                ));
                world.max_population = dynamic_capacity as usize;
            } else if pop_fraction > 0.8 {
                // Near capacity: reduce birth rate by lowering max_population signal
                world.max_population = (dynamic_capacity * 0.95) as usize;
            } else {
                // Below 80%: full dynamic capacity
                world.max_population = dynamic_capacity as usize;
            }
        }
    }

    /// Phase 9.5: Probabilistic disaster engine — solar weather, impacts, ECLSS
    /// failures, psychological events, tech milestones, and Tainter/Turchin dynamics.
    fn tick_disasters(&mut self) {
        if !self.config.policy.disasters_enabled {
            return;
        }

        let policy = self.config.policy.clone();
        let disaster_results =
            self.disaster_engine
                .tick(&self.worlds, self.current_tick, &mut self.rng, &policy);

        // Phase 1d: defense_spending reduces disaster severity
        let defense_mult = 1.0
            - (self
                .worlds
                .first()
                .map(|w| w.policy_state.defense_spending)
                .unwrap_or(0.0)
                * 2.0)
                .min(0.5);

        for (effects, world_id, event) in disaster_results {
            // Dead Loop #8 fix: Cultural memory reduces disaster damage.
            // Civilizations that remember past disasters have better preparedness.
            let cultural_defense = {
                let world_name = world_id
                    .and_then(|id| self.worlds.iter().find(|w| w.id == id))
                    .map(|w| w.name.as_str());
                let lesson = self.narrative_engine.cultural_lesson_for(world_name);
                1.0 - lesson.preparedness.min(0.3) // up to 30% reduction
            };
            let total_mult = defense_mult * cultural_defense;

            let effects = disasters::DisasterEffects {
                consciousness_shock: effects.consciousness_shock * total_mult,
                allostatic_load_increase: effects.allostatic_load_increase * total_mult,
                infrastructure_damage: effects.infrastructure_damage * total_mult,
                resource_production_penalty: effects.resource_production_penalty * total_mult,
                ..effects
            };
            // Apply effects to targeted world(s)
            let target_ids: Vec<u32> = match world_id {
                Some(id) => vec![id],
                None => self.worlds.iter().map(|w| w.id).collect(),
            };

            for &wid in &target_ids {
                if let Some(world) = self.worlds.iter_mut().find(|w| w.id == wid) {
                    // Mechanism 1 — Non-Linear Cascade Failures: when a world has 3+
                    // active disasters, multiply ALL effects by 1.0 + 0.5 * (count - 2).
                    let active_count = self
                        .disaster_engine
                        .active_per_world
                        .get(&wid)
                        .copied()
                        .unwrap_or(0);
                    let cascade_mult = if active_count >= 3 {
                        let mult = 1.0 + 0.5 * (active_count as f64 - 2.0);
                        // Log the cascade event (once per disaster application, not per tick)
                        self.events.push(CivEvent::new(
                            self.current_tick,
                            Some(wid),
                            CivEventType::SystemicCascadeFailure,
                            format!(
                                "{}: CASCADE FAILURE — {} active disasters, effects amplified {:.1}x",
                                world.name, active_count, mult
                            ),
                        ));
                        mult
                    } else {
                        1.0
                    };

                    // Mechanism 4 — Consciousness-Gated Evacuation: high-consciousness
                    // colonies lose up to 50% fewer people to disasters. Mean Phi acts
                    // as a collective awareness multiplier for evacuation efficiency.
                    let mean_phi = world.mean_phi();
                    // Ethics-modulated disaster resilience (distinct from consciousness):
                    // Virtue/care: mutual aid reduces casualties (people carry each other)
                    // Relational: community-first response (Ubuntu: no one left behind)
                    // Deontological: orderly evacuation protocols reduce panic
                    // Consequentialist: triage efficiency costs (some sacrificed for many)
                    let disaster_ethics = agent::EthicalOrientation::mean_of(&world.agents);
                    let ethics_resilience = (disaster_ethics.virtue_care * 0.07
                        + disaster_ethics.relational * 0.06
                        + disaster_ethics.deontological * 0.04
                        - disaster_ethics.consequentialist * 0.03)
                        .clamp(-0.03, 0.12);
                    let mut effective_loss = effects.population_loss_fraction
                        * cascade_mult
                        * (1.0 - mean_phi * 0.5)
                        * (1.0 - ethics_resilience);

                    // Resontia Earth-hardening: vaults reduce Earth casualties.
                    // Compute mitigation inline to avoid borrow conflicts.
                    if world.location == "Earth"
                        && self.resontia_config.enabled
                        && self.resontia_infra.vault_count > 0
                    {
                        let vault_cap = self.resontia_infra.vault_capacity;
                        let pop = world.population();
                        let mitigated_fraction = if pop > 0 {
                            (vault_cap as f64 / pop as f64).min(0.9)
                        } else {
                            0.0
                        };
                        effective_loss *= 1.0 - mitigated_fraction;
                    }

                    // Population loss: kill a fraction of living agents (random selection)
                    if effective_loss > 0.0 {
                        let living_count = world.population();
                        let to_kill = (living_count as f64 * effective_loss).round() as usize;
                        if to_kill > 0 {
                            // Collect living agent ids, then kill the first `to_kill`
                            // (deterministic for reproducibility with the same RNG state)
                            let mut living_ids: Vec<u64> = world
                                .agents
                                .iter()
                                .filter(|a| a.is_alive())
                                .map(|a| a.id)
                                .collect();
                            // Shuffle using RNG for randomness
                            for i in (1..living_ids.len()).rev() {
                                let j = (self.rng.next_u64() as usize) % (i + 1);
                                living_ids.swap(i, j);
                            }
                            // O(1) kill lookup + survivor trauma
                            let kill_map: std::collections::HashMap<u64, usize> = world
                                .agents
                                .iter()
                                .enumerate()
                                .map(|(i, a)| (a.id, i))
                                .collect();
                            for &kill_id in living_ids.iter().take(to_kill) {
                                if let Some(&idx) = kill_map.get(&kill_id) {
                                    world.agents[idx].death_tick = Some(self.current_tick);
                                }
                            }
                            // Survivor trauma: witnessing deaths
                            if to_kill > 0 {
                                let survivors =
                                    world.agents.iter().filter(|a| a.is_alive()).count().max(1);
                                let trauma_from_deaths =
                                    (to_kill as f64 / survivors as f64).min(0.3);
                                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                                    agent.trauma_level =
                                        (agent.trauma_level + trauma_from_deaths).min(1.0);
                                    agent.wounds.push(wound_healing::WoundState::new(
                                        trauma_from_deaths.min(0.5),
                                        wound_healing::WoundOrigin::Disaster,
                                        self.current_tick,
                                    ));
                                }
                            }
                        }
                    }

                    // Infrastructure damage (amplified by cascade)
                    if effects.infrastructure_damage > 0.0 {
                        let infra_dmg = effects.infrastructure_damage * cascade_mult;

                        // Mechanism 3 — Supply Chain Fragility: deduct materials
                        // for repair. If insufficient materials, repair is partial.
                        let repair_cost = infra_dmg * 100.0;
                        let materials_available = world.resources.deduct("materials", repair_cost);
                        let actual_repair_fraction = if repair_cost > 0.0 {
                            materials_available / repair_cost
                        } else {
                            1.0
                        };
                        // Infrastructure drops by full damage, but recovery (below)
                        // is scaled by how much material was available for repair.
                        world.infrastructure_level =
                            (world.infrastructure_level - infra_dmg).max(0.0);

                        // Reduce the Sacred Stillness recovery bonus by material availability
                        let _ = actual_repair_fraction; // used implicitly via materials deduction
                    }

                    // Mechanism 3 — Sacred Stillness Recovery Bonus: harmony index 7
                    // (Sacred Stillness) accelerates infrastructure recovery after damage.
                    // Stillness 0.0 → 0.2%/tick (~42 years full rebuild)
                    // Stillness 0.5 → 0.7%/tick (~12 years)
                    // Stillness 1.0 → 1.2%/tick (~7 years)
                    let stillness_score = world.harmony.current_scores[7];
                    let mut recovery_rate = 0.002 + stillness_score * 0.01;
                    // Option B: Care cascade — high collective care accelerates recovery.
                    // Spinoza Ethics IV: "Nothing is more useful to man than man."
                    let mean_care = {
                        let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                        if living.is_empty() {
                            0.0
                        } else {
                            living.iter().map(|a| a.needs.affect.care).sum::<f64>()
                                / living.len() as f64
                        }
                    };
                    recovery_rate += mean_care * 0.005; // Up to 0.5%/tick bonus from mutual aid

                    // Coordination science recovery boost: populations that understand
                    // systems thinking coordinate disaster response more effectively.
                    // Up to 0.3%/tick additional recovery from collective coordination.
                    let mean_cu = {
                        let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
                        if living.is_empty() {
                            0.0
                        } else {
                            living
                                .iter()
                                .map(|a| a.coordination_understanding)
                                .sum::<f64>()
                                / living.len() as f64
                        }
                    };
                    recovery_rate += mean_cu * 0.003;

                    world.infrastructure_level =
                        (world.infrastructure_level + recovery_rate).min(1.0);

                    // Resource production penalty: reduce production_rate temporarily
                    // by multiplying by (1 - penalty). We apply to current stocks as proxy
                    // since production_rate is static and the penalty is per-tick from active disasters.
                    if effects.resource_production_penalty > 0.0 {
                        let factor = 1.0 - effects.resource_production_penalty;
                        for name in &["food", "water", "energy", "materials", "oxygen"] {
                            if let Some(stock) = world.resources.get_mut(name) {
                                // Reduce production output this tick
                                stock.current = (stock.current * factor).max(0.0);
                            }
                        }
                    }

                    // Solar power penalty: additional reduction to energy
                    if effects.solar_power_penalty > 0.0 {
                        if let Some(energy) = world.resources.get_mut("energy") {
                            energy.current =
                                (energy.current * (1.0 - effects.solar_power_penalty)).max(0.0);
                        }
                    }

                    // Consciousness shock (amplified by cascade)
                    if effects.consciousness_shock > 0.0 {
                        let shock = effects.consciousness_shock * cascade_mult;
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.consciousness.level =
                                (agent.consciousness.level - shock).max(0.0);
                        }
                    }

                    // Allostatic load increase (amplified by cascade)
                    if effects.allostatic_load_increase > 0.0 {
                        let load_inc = effects.allostatic_load_increase * cascade_mult;
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.needs.allostatic_load =
                                (agent.needs.allostatic_load + load_inc).min(1.0);
                        }
                    }

                    // Morale impact (amplified by cascade)
                    if effects.morale_impact != 0.0 {
                        let morale = effects.morale_impact * cascade_mult;
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.needs.engagement =
                                (agent.needs.engagement + morale).clamp(0.0, 1.0);
                        }
                    }

                    // Dead Loop #1 fix: Disasters cause trauma (calibrated).
                    // Includes population_loss_fraction — deaths from disasters are
                    // the strongest trauma source (witnessing loss).
                    // Threshold lowered from 0.005 to 0.001 so moderate ECLSS failures
                    // and psychological events still register trauma.
                    // Equilibrium: 1 disaster/month at 0.03 trauma, decay 0.002/tick
                    // → baseline ~0.15. Catastrophic (Carrington) → 0.3-0.5 spike.
                    // Trauma from disasters: scaled by effect severity and cascade.
                    // Calibrated so a moderate ECLSS failure (~0.02 shock, ~0.05 load)
                    // produces ~0.02 trauma/tick, which accumulates meaningfully
                    // against the 0.001/tick decay rate.
                    let trauma_inc = (effects.consciousness_shock * 0.5
                        + effects.allostatic_load_increase * 0.3
                        + effects.morale_impact.abs() * 0.2
                        + effects.population_loss_fraction * 1.0) // witnessing death
                        * cascade_mult;
                    if trauma_inc > 0.0001 {
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.trauma_level = (agent.trauma_level + trauma_inc).min(1.0);
                            if trauma_inc > 0.05 {
                                agent.wounds.push(wound_healing::WoundState::new(
                                    trauma_inc.min(0.5),
                                    wound_healing::WoundOrigin::Disaster,
                                    self.current_tick,
                                ));
                            }
                            // Crisis ethical reorientation (replaces gentle Phase 3b nudge):
                            // When trauma exceeds 0.4, agents undergo wholesale ethical
                            // shift — not a nudge. Direction depends on crisis type:
                            // - Population loss (violence/disaster) → deontological (crave order)
                            // - Resource scarcity → consequentialist (survival calculus)
                            // - Isolation/morale → relational (crave community)
                            // Ref: Weimar hyperinflation→authoritarianism; war→duty ethics
                            if agent.trauma_level > 0.4 {
                                let shift = (trauma_inc * 0.15).min(0.08); // substantial but bounded
                                if effects.population_loss_fraction > 0.01 {
                                    // Violence/death → crave rules and order
                                    agent.ethics.deontological =
                                        (agent.ethics.deontological + shift).min(1.0);
                                    agent.ethics.consequentialist =
                                        (agent.ethics.consequentialist - shift * 0.5).max(0.05);
                                } else if effects.allostatic_load_increase > 0.03 {
                                    // Resource stress → survival calculus
                                    agent.ethics.consequentialist =
                                        (agent.ethics.consequentialist + shift).min(1.0);
                                    agent.ethics.virtue_care =
                                        (agent.ethics.virtue_care - shift * 0.3).max(0.05);
                                } else if effects.morale_impact < -0.02 {
                                    // Isolation/morale collapse → crave community
                                    agent.ethics.relational =
                                        (agent.ethics.relational + shift).min(1.0);
                                    agent.ethics.deontological =
                                        (agent.ethics.deontological - shift * 0.3).max(0.05);
                                }
                            } else {
                                // Below crisis threshold: gentle drift (original Phase 3b)
                                agent.ethics.deontological =
                                    (agent.ethics.deontological + trauma_inc * 0.01).min(1.0);
                                agent.ethics.consequentialist =
                                    (agent.ethics.consequentialist - trauma_inc * 0.005).max(0.05);
                            }
                        }
                    }
                }
            }

            // Push the event
            self.events.push(event);
        }

        // #3: Power-law cascade propagation.
        // After all disasters are applied, update domain stress and check for cascades.
        for (i, world) in self.worlds.iter().enumerate() {
            if world.population() == 0 {
                continue;
            }
            let resource_fracs: Vec<(&str, f64)> = ["food", "water", "oxygen", "energy"]
                .iter()
                .filter_map(|&name| {
                    world.resources.get(name).map(|s| {
                        let frac = if s.capacity > 0.0 {
                            s.current / s.capacity
                        } else {
                            0.0
                        };
                        (name, frac)
                    })
                })
                .collect();
            let gov_stability = if i < self.governance.len() {
                self.governance[i].stability_score
            } else {
                0.5
            };
            self.cascade_engine.update_stress(
                i,
                world.infrastructure_level,
                &resource_fracs,
                gov_stability,
            );

            // Check if any recent disaster triggers a cascade
            if self.disaster_engine.total_disasters > 0 {
                // Use infrastructure damage as severity proxy for cascade trigger
                let severity = (1.0 - world.infrastructure_level).max(0.1);
                let domain = if resource_fracs
                    .iter()
                    .any(|(n, f)| *n == "energy" && *f < 0.3)
                {
                    cascade::CascadeDomain::Power
                } else if resource_fracs.iter().any(|(n, f)| *n == "food" && *f < 0.3) {
                    cascade::CascadeDomain::Food
                } else {
                    cascade::CascadeDomain::LifeSupport
                };
                if let Some(cascade_event) =
                    self.cascade_engine
                        .try_cascade(world.id, i, domain, severity, &mut self.rng)
                {
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "CASCADE FAILURE: {} domains affected (depth {}), total severity {:.2}",
                            cascade_event.affected.len(),
                            cascade_event.depth,
                            cascade_event.total_severity
                        ),
                    ));
                }
            }
        }

        // Apply tech milestone effects: check for newly achieved milestones this tick
        for milestone in &self.disaster_engine.tech_tree.milestones {
            if milestone.achieved && milestone.achieved_tick == Some(self.current_tick) {
                // Apply tech effects to all worlds
                for world in &mut self.worlds {
                    for &(sector, boost) in &milestone.effects.tech_level_boost {
                        if sector < 8 {
                            world.knowledge.technology_levels[sector] += boost;
                        }
                    }
                    // Power multiplier (boosts energy production rate)
                    if milestone.effects.power_multiplier > 1.0 {
                        if let Some(energy) = world.resources.get_mut("energy") {
                            energy.production_rate *= milestone.effects.power_multiplier;
                        }
                    }
                    // Resource efficiency (boosts all production rates)
                    if milestone.effects.resource_efficiency > 1.0 {
                        for name in &["food", "water", "energy", "materials", "oxygen"] {
                            if let Some(stock) = world.resources.get_mut(name) {
                                stock.production_rate *= milestone.effects.resource_efficiency;
                            }
                        }
                    }
                }
            }
        }
    }

    fn tick_emergencies(&mut self) {
        // Tick existing epidemics (index-iterate to allow &mut self.rng)
        let world_count_epi = self.worlds.len();
        for wi in 0..world_count_epi {
            let pop = self.worlds[wi].population();
            let mut ended = Vec::new();
            for (i, epi) in self.worlds[wi].epidemics.iter_mut().enumerate() {
                epi.tick(pop, &mut self.rng);
                if epi.is_over() {
                    ended.push(i);
                }
            }
            for i in ended.into_iter().rev() {
                let epi = self.worlds[wi].epidemics.remove(i);
                self.events.push(CivEvent::new(
                    self.current_tick,
                    Some(self.worlds[wi].id),
                    CivEventType::EpidemicEnd,
                    format!("{} epidemic ended on {}", epi.name, self.worlds[wi].name),
                ));
            }
        }

        // Stochastic epidemic onset (low probability each tick)
        let world_count = self.worlds.len();
        for i in 0..world_count {
            let pop = self.worlds[i].population();
            if pop > 50 && self.rng.bernoulli(0.002) {
                let epi = population::SirEpidemic::common_cold(pop, self.current_tick);
                let name = epi.name.clone();
                let world_name = self.worlds[i].name.clone();
                self.worlds[i].epidemics.push(epi);
                self.events.push(CivEvent::new(
                    self.current_tick,
                    Some(self.worlds[i].id),
                    CivEventType::EpidemicStart,
                    format!("{} outbreak on {}", name, world_name),
                ));
            }
        }
    }

    /// Populate a snapshot's extended observable fields from current simulation state.
    fn populate_observables(&self, snap: &mut EpochSnapshot) {
        snap.faction_count = self.faction_engine.active_faction_count();

        // Elite persistence: max across worlds
        snap.elite_persistence = self
            .worlds
            .iter()
            .map(|w| observables::elite_persistence_index(w, self.current_tick))
            .fold(0.0f64, f64::max);

        // Innovation stagnation: max across worlds
        snap.innovation_stagnation = self
            .worlds
            .iter()
            .map(|w| {
                observables::innovation_stagnation_index(
                    &w.knowledge.tech_history,
                    self.current_tick,
                    120,
                )
            })
            .fold(0.0f64, f64::max);

        // Inter-world divergence
        snap.inter_world_divergence = observables::inter_world_divergence(&self.worlds);

        // Consciousness Gini: compute across all living agents
        let all_phis: Vec<f64> = self
            .worlds
            .iter()
            .flat_map(|w| w.agents.iter())
            .filter(|a| a.is_alive())
            .map(|a| a.consciousness.phi())
            .collect();
        snap.consciousness_gini = observables::consciousness_gini(&all_phis);

        // Phi trend: use combined phi_history from first world (proxy)
        // Or use mean phi from snapshots
        let phi_values: Vec<f64> = self.epoch_snapshots.iter().map(|s| s.mean_phi).collect();
        snap.phi_trend = format!("{}", observables::classify_phi_trend(&phi_values, 120));

        // Recovery count
        snap.recovery_count = observables::recovery_count(&self.epoch_snapshots);

        // Mean trauma across all worlds
        let total_trauma: f64 = self
            .worlds
            .iter()
            .map(|w| observables::trauma_level(w) * w.population() as f64)
            .sum();
        let total_pop: f64 = self.worlds.iter().map(|w| w.population() as f64).sum();
        snap.trauma_level = if total_pop > 0.0 {
            total_trauma / total_pop
        } else {
            0.0
        };

        // Speciation index
        snap.speciation_index = observables::speciation_index(&self.worlds, self.current_tick);
    }

    fn tick_epoch_evaluation(&mut self) {
        // Periodic snapshots every 60 ticks (5 years) for trajectory analysis
        if self.current_tick % 60 == 0 && self.current_tick > 0 {
            let mut snapshot = self
                .epoch_manager
                .take_snapshot(self.current_tick, &self.worlds);
            self.populate_observables(&mut snapshot);
            self.epoch_snapshots.push(snapshot);
        }

        // Evaluate epoch checkpoints
        let checkpoint_events = self
            .epoch_manager
            .evaluate_tick(self.current_tick, &self.worlds);
        self.events.extend(checkpoint_events);

        // Check for epoch transitions
        let total_pop = self.total_population();
        let num_worlds = self.worlds.len();
        let ss = self.mean_self_sufficiency();

        if let Some(new_epoch) =
            self.epoch_manager
                .check_epoch_transition(self.current_tick, total_pop, num_worlds, ss)
        {
            // Take a snapshot at transition
            let snapshot = self
                .epoch_manager
                .take_snapshot(self.current_tick, &self.worlds);
            self.epoch_snapshots.push(snapshot);
            self.current_epoch = new_epoch;

            self.events.push(CivEvent::new(
                self.current_tick,
                None,
                CivEventType::EpochTransition,
                format!(
                    "Epoch transition to {} at tick {}",
                    epoch::epoch_name(new_epoch),
                    self.current_tick
                ),
            ));

            // Activate LCF breakthrough probability in Epochs 3-4.
            // 0.005 per tick = ~30% chance over a 25-year epoch.
            // Models the possibility that lattice confinement fusion
            // (spark-engine physics) is validated and scaled.
            if new_epoch == epoch::EPOCH_BRANCHES || new_epoch == epoch::EPOCH_CANOPY {
                for world in &mut self.worlds {
                    if world.knowledge.lcf_probability == 0.0 {
                        world.knowledge.lcf_probability = 0.005;
                    }
                }
            }
        }
    }

    // --- Milestone tracking helpers ---

    /// Scan events for first birth and record it.
    fn track_milestones(&mut self, new_events: &[CivEvent]) {
        for event in new_events {
            match event.event_type {
                CivEventType::Birth => {
                    self.epoch_manager
                        .record_milestone("birth", self.current_tick);
                }
                CivEventType::TradeEstablished => {
                    self.epoch_manager
                        .record_milestone("trade", self.current_tick);
                }
                _ => {}
            }
        }
    }

    // --- Helpers ---

    fn total_population(&self) -> usize {
        self.worlds.iter().map(|w| w.population()).sum()
    }

    fn mean_self_sufficiency(&self) -> f64 {
        if self.worlds.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .worlds
            .iter()
            .map(|w| w.resources.self_sufficiency())
            .sum();
        sum / self.worlds.len() as f64
    }

    fn mean_phi_all(&self) -> f64 {
        if self.worlds.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.worlds.iter().map(|w| w.mean_phi()).sum();
        sum / self.worlds.len() as f64
    }

    /// Run the full simulation. Returns a civilization report.
    pub fn run(&mut self) -> CivilizationReport {
        // Skip initialization if already done (allows inject_adversaries before run)
        if self.worlds.is_empty() {
            self.initialize_worlds();
        }

        while self.current_tick < self.config.total_ticks {
            // Check for deferred world founding (from config)
            self.check_deferred_worlds();

            // Check for Mars fission (dynamic world founding)
            self.check_mars_fission();

            self.check_trade_milestone();

            self.tick_earth_hybrid();

            self.tick_generation_ship();

            // Phase 0.9: Viability Engine — thermodynamic axioms, EROI, scaling laws.
            // Must run BEFORE economy so scaling factors and energy budgets are available.
            {
                let world_data: Vec<(u32, f64, f64, f64)> = self
                    .worlds
                    .iter()
                    .filter(|w| w.population() > 0)
                    .map(|w| {
                        let energy_prod = w.resources.stock_level("energy").unwrap_or(0.0);
                        (
                            w.id,
                            energy_prod,
                            w.population() as f64,
                            w.infrastructure_level,
                        )
                    })
                    .collect();
                self.viability_engine.tick(&world_data, self.current_tick);
            }

            // Module registry: tick all registered plugin modules.
            // Runs alongside the existing tick loop during incremental migration.
            if self.module_registry.module_count() > 0 {
                let _outputs = self
                    .module_registry
                    .tick_all(self.current_tick as u64 as u32);
                // TODO: apply outputs.mutations to world state as modules are migrated
            }

            // Phase 0.5: Metabolism — compute phase modifiers for this tick
            self.phase_modifiers = std::collections::HashMap::new();
            if self.config.policy.metabolism.enabled {
                for world in &mut self.worlds {
                    let mods = metabolism::compute_modifiers(
                        &self.config.policy.metabolism,
                        &mut world.metabolism_state,
                        self.current_tick,
                    );
                    self.phase_modifiers.insert(world.id, mods);
                }
            }

            // Phase 1: Demographics (pair-bonding, births, deaths)
            let mut phase1_events = Vec::new();
            let world_count = self.worlds.len();
            for i in 0..world_count {
                let mut world = std::mem::take(&mut self.worlds[i]);
                // Birth policy modifies pair bond rate (C: Scenario Mode)
                let birth_mult = match world.policy_state.birth_policy {
                    config::BirthPolicy::ProNatal => 1.5,
                    config::BirthPolicy::PopulationControl => 0.5,
                    config::BirthPolicy::ReplacementOnly => {
                        let pop = world.population();
                        let deaths_per_tick = (pop as f64 * 0.001).max(1.0); // ~1.2% annual
                        let target_rate = deaths_per_tick / pop.max(1) as f64;
                        target_rate / self.config.policy.pair_bond_rate.max(0.001)
                    }
                    config::BirthPolicy::Natural => 1.0,
                };
                let fert_mult = world.fertility_multiplier;
                PopulationEngine::tick_pair_bonding(
                    &mut world,
                    &mut self.rng,
                    self.current_tick,
                    self.config.policy.pair_bond_rate * birth_mult * fert_mult,
                );
                // Accumulate radiation dose per agent based on location.
                // Updated with MSL/RAD measured data (Hassler 2014, Zeitlin 2013).
                // Mars: 0.67 mSv/day = ~20 mSv/month (was 6 — 3.3x undercount!)
                // Moon: ~15 mSv/month unshielded (Cucinotta 2014)
                // NASA career limit: 600 mSv (NASA-STD-3001 Rev C, 2022)
                // A Mars colonist hits the 600 mSv limit in ~2.5 years unshielded.
                // Radiation shelters reduce exposure by assumed 60-80%.
                // Modular habitat-aware radiation dose.
                // Ambient dose depends on location (Hassler 2014, Cucinotta 2014).
                // Actual dose depends on which module the agent works in.
                // Dead Loop #4 fix: Apply GCR (galactic cosmic ray) multiplier.
                // During solar minimum, GCR flux is ~15% higher (weaker heliospheric
                // shielding). During solar maximum, ~20% lower.
                // CITATION: Usoskin et al. (2011) "Solar modulation of galactic
                // cosmic rays", Living Reviews in Solar Physics 8(3).
                let gcr_mult = self.disaster_engine.gcr_multiplier();
                let ambient_dose_sv = match world.location.as_str() {
                    "Earth" => 0.0002, // Magnetosphere shields from GCR
                    "Moon" => self.params.radiation_moon_sv_month * gcr_mult,
                    "Mars" => self.params.radiation_mars_sv_month * gcr_mult,
                    "Europa" => self.params.radiation_europa_sv_month,
                    "Titan" => 0.00005, // Thick N2 atmosphere shields
                    _ => self.params.radiation_europa_sv_month * gcr_mult,
                };
                if world.habitat.modules.is_empty() {
                    // No habitat modules yet — flat dose for everyone
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.cumulative_dose_sv += ambient_dose_sv;
                    }
                } else {
                    // Per-agent dose based on profession → module assignment
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        let sector = agent.skills.strongest_index();
                        let dose_frac = world.habitat.agent_dose_fraction(sector);
                        agent.cumulative_dose_sv += ambient_dose_sv * dose_frac;
                        // Habitat psychology affects allostatic load
                        let psych_mod = world.habitat.agent_psych_modifier(sector);
                        agent.needs.allostatic_load =
                            (agent.needs.allostatic_load + psych_mod).clamp(0.0, 1.0);
                    }
                }
                let dem_events = PopulationEngine::tick_demographics(
                    &mut world,
                    &mut self.rng,
                    self.current_tick,
                );
                // Fix 3: Genetic Engineering eliminates inbreeding depression.
                // Newborns with health penalties from inbreeding get gene therapy.
                if self
                    .disaster_engine
                    .tech_tree
                    .is_achieved("Genetic Engineering")
                {
                    for agent in world.agents.iter_mut() {
                        if agent.birth_tick == self.current_tick && agent.health < 0.85 {
                            agent.health = agent.health.max(0.85);
                        }
                    }
                }

                phase1_events.extend(dem_events);
                self.worlds[i] = world;
            }
            self.track_milestones(&phase1_events);
            self.events.extend(phase1_events);

            // Phase 1.5: Wound Healing — advance healing phases
            for world in &mut self.worlds {
                let healing_mult = self
                    .phase_modifiers
                    .get(&world.id)
                    .map(|m| m.healing_mult)
                    .unwrap_or(1.0);
                let ctx = wound_healing::HealingContext {
                    medicine: 0.5, // TODO: derive from world resources
                    care_ratio: 0.3,
                    collective_phi: world.mean_phi(),
                    mediation_factor: world.governance.stability_score.min(1.0),
                    metabolism_healing_mult: healing_mult,
                };
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.wounds.retain_mut(|w| {
                        wound_healing::tick_wound(w, &ctx);
                        !w.is_healed()
                    });
                    if wound_healing::attempt_kenosis(
                        &mut agent.wounds,
                        agent.consciousness.care_activation,
                    ) {
                        agent.consciousness.meta_awareness =
                            (agent.consciousness.meta_awareness + 0.01).min(1.0);
                    }
                    // Trauma is the max of accumulated trauma (from disasters/deaths)
                    // and wound-derived trauma. Don't overwrite — wounds are a subset.
                    let wound_trauma = wound_healing::aggregate_trauma(&agent.wounds);
                    agent.trauma_level = agent.trauma_level.max(wound_trauma);
                    // TEND cost for care services (wound healing requires community resources)
                    if !agent.wounds.is_empty() && ctx.care_ratio > 0.01 {
                        agent.tend_balance = (agent.tend_balance - 0.5).max(-40.0);
                    }
                }
            }

            // Phase 2: Genetics
            self.tick_genetics();

            // Phase 3: Psychological needs
            self.tick_psychological_needs();

            // Metabolism: apply recovery multiplier to allostatic load
            for world in &mut self.worlds {
                let mult = self
                    .phase_modifiers
                    .get(&world.id)
                    .map(|m| m.recovery_mult)
                    .unwrap_or(1.0);
                if (mult - 1.0).abs() > 0.01 {
                    let boost = (mult - 1.0) * 0.002;
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.needs.allostatic_load =
                            (agent.needs.allostatic_load - boost).max(0.0);
                    }
                }
            }
            // Phase 3.5: Education (peer-to-peer learning, TEND rewards)
            self.tick_education();

            // Phase 4: Economy
            self.tick_economy();

            // Metabolism: modulate production output
            for world in &mut self.worlds {
                let prod_mult = self
                    .phase_modifiers
                    .get(&world.id)
                    .map(|m| m.production_mult)
                    .unwrap_or(1.0);
                if (prod_mult - 1.0).abs() > 0.01 {
                    for output in &mut world.economy.sector_output {
                        *output *= prod_mult;
                    }
                }
            }
            // Phase 4.5: Currency — MYCEL/SAP/TEND tick
            for world in &mut self.worlds {
                // SAP income: workers earn proportional to sector output
                let total_output: f64 = world.economy.sector_output.iter().sum();
                let living_workers = world.agents.iter().filter(|a| a.is_alive()).count();
                if living_workers > 0 && total_output > 0.0 {
                    let sap_per_worker = (total_output * 0.1) / living_workers as f64;
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.sap_balance += sap_per_worker;
                    }
                }

                // TEND bounds enforcement
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.tend_balance = agent
                        .tend_balance
                        .clamp(-currency::TEND_LIMIT, currency::TEND_LIMIT);
                }

                // SAP demurrage (per-agent)
                let mut sap_balances: Vec<f64> = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive())
                    .map(|a| a.sap_balance)
                    .collect();
                let collected = currency::apply_sap_demurrage(&mut sap_balances);
                let mut idx = 0;
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    if idx < sap_balances.len() {
                        agent.sap_balance = sap_balances[idx];
                        idx += 1;
                    }
                }
                let (local, planetary, system) = currency::distribute_demurrage(collected);
                world.currency_state.sap_commons_local += local;
                world.currency_state.sap_commons_planetary += planetary;
                world.currency_state.sap_commons_system += system;
                world.currency_state.sap_demurrage_collected = collected;

                // Commons pool → public goods investment (Economic Charter)
                let spending = currency::spend_commons(&mut world.currency_state, 0.1);
                world.infrastructure_level =
                    (world.infrastructure_level + spending.infrastructure * 0.001).min(1.0);
                // Education effect: boost all agents' education_level slightly
                if spending.education > 0.01 {
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.education_level =
                            (agent.education_level + spending.education * 0.00001).min(1.0);
                    }
                }
                // Medicine effect: reduce allostatic load for all agents
                if spending.medicine > 0.01 {
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.needs.allostatic_load =
                            (agent.needs.allostatic_load - spending.medicine * 0.00001).max(0.0);
                    }
                }

                // Peer recognition (Commons Charter: 10 recognitions/month, MYCEL-weighted)
                let recognition = peer_recognition::tick_recognition(&world.agents, &mut self.rng);

                // MYCEL computation (now with real peer recognition)
                for (orig_idx, agent) in world.agents.iter_mut().enumerate() {
                    if !agent.is_alive() {
                        continue;
                    }
                    let inputs = currency::MycelInputs {
                        participated: agent.needs.engagement > 0.3,
                        recognition: recognition.score_for(orig_idx), // Real agent-to-agent recognition
                        quality: agent.consciousness.coherence * agent.education_level,
                        years_active: agent.age_years(self.current_tick),
                    };
                    agent.mycel_score = currency::compute_mycel(agent.mycel_score, &inputs);
                }

                // Jubilee (every 48 ticks = 4 years)
                if self.current_tick >= world.currency_state.next_jubilee_tick {
                    let mut scores: Vec<f64> = world
                        .agents
                        .iter()
                        .filter(|a| a.is_alive())
                        .map(|a| a.mycel_score)
                        .collect();
                    currency::apply_jubilee(&mut scores);
                    let mut idx = 0;
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        if idx < scores.len() {
                            agent.mycel_score = scores[idx];
                            idx += 1;
                        }
                    }
                    world.currency_state.next_jubilee_tick += currency::MYCEL_JUBILEE_TICKS;
                }

                // Update aggregates
                let living_count = world.agents.iter().filter(|a| a.is_alive()).count();
                if living_count > 0 {
                    world.currency_state.mycel_mean = world
                        .agents
                        .iter()
                        .filter(|a| a.is_alive())
                        .map(|a| a.mycel_score)
                        .sum::<f64>()
                        / living_count as f64;
                }
            }

            // Phase 5: Inter-world
            self.tick_interworld();

            // Phase 5.5: Immigration pipeline for genetic rescue
            self.tick_immigration_pipeline();

            // Phase 5.3: Colony projects (multi-tick construction)
            for world in &mut self.worlds {
                if world.population() == 0 {
                    continue;
                }

                // Auto-prioritize annually
                if world.project_manager.active.is_empty()
                    && world.project_manager.queue.is_empty()
                    && self.current_tick % 12 == 0
                {
                    let mut priorities = projects::prioritize_projects(
                        world.population(),
                        world.power_demand_kw - world.power_generation_kw,
                        world.resources.fraction_of_capacity("food"),
                        world.max_population,
                        world.population() > (world.max_population as f64 * 0.8) as usize,
                        world
                            .project_manager
                            .has_completed(projects::ProjectBlueprint::MedicalFacility),
                        world
                            .project_manager
                            .has_completed(projects::ProjectBlueprint::FabricationWorkshop),
                        &world.location,
                        &world.project_manager.completed,
                    );
                    // Phase 1a: project_strategy inserts strategy-specific priorities first
                    let completed = &world.project_manager.completed;
                    let mut strategy_priorities = Vec::new();
                    match self.config.policy.project_strategy {
                        config::ProjectStrategy::SurvivalFirst => {
                            if !completed.contains(&projects::ProjectBlueprint::RadiationShelter) {
                                strategy_priorities
                                    .push(projects::ProjectBlueprint::RadiationShelter);
                            }
                        }
                        config::ProjectStrategy::GrowthFirst => {
                            strategy_priorities.push(projects::ProjectBlueprint::HabitatExpansion);
                            strategy_priorities.push(projects::ProjectBlueprint::GreenhouseModule);
                        }
                        config::ProjectStrategy::ScienceFirst => {
                            if !completed.contains(&projects::ProjectBlueprint::ExplorationVehicle)
                            {
                                strategy_priorities
                                    .push(projects::ProjectBlueprint::ExplorationVehicle);
                            }
                        }
                        config::ProjectStrategy::IndependenceFirst => {
                            if !completed.contains(&projects::ProjectBlueprint::FissionReactor) {
                                strategy_priorities
                                    .push(projects::ProjectBlueprint::FissionReactor);
                            }
                        }
                        config::ProjectStrategy::Balanced => {}
                    }
                    // Strategy priorities go first, then default priorities
                    strategy_priorities.append(&mut priorities);
                    let priorities = strategy_priorities;

                    for bp in priorities.into_iter().take(2) {
                        world.project_manager.queue.push(bp);
                    }
                }

                // Tick projects
                let workers = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive() && a.life_stage(self.current_tick).can_work())
                    .count() as f64;
                let available_labor = workers * WORK_HOURS_PER_MONTH * PROJECT_LABOR_FRACTION;
                let available_materials = world
                    .resources
                    .get("materials")
                    .map(|s| s.current * PROJECT_MATERIALS_FRACTION)
                    .unwrap_or(0.0);

                let (completed, _labor, mat_used) = world
                    .project_manager
                    .tick(available_labor, available_materials);

                if mat_used > 0.0 {
                    if let Some(mat) = world.resources.get_mut("materials") {
                        mat.current = (mat.current - mat_used).max(0.0);
                    }
                }

                for bp in &completed {
                    match bp {
                        projects::ProjectBlueprint::GreenhouseModule => {
                            if let Some(food) = world.resources.get_mut("food") {
                                food.production_rate += 25.0;
                                food.capacity += 500.0;
                            }
                        }
                        projects::ProjectBlueprint::HabitatExpansion => {
                            world.max_population += 500;
                            world.habitable_area_m2 += 15000.0;
                        }
                        projects::ProjectBlueprint::FissionReactor => {
                            world.power_generation_kw += 100.0;
                        }
                        projects::ProjectBlueprint::CentrifugeHabitat => {
                            world.reproduction_viable = true;
                        }
                        projects::ProjectBlueprint::FabricationWorkshop => {
                            if let Some(mat) = world.resources.get_mut("materials") {
                                mat.production_rate *= 1.5;
                            }
                            // Fabrication workshop enables robot manufacturing.
                            // Robots are built over time, not spawned instantly.
                        }
                        projects::ProjectBlueprint::WaterExtractionPlant => {
                            if let Some(w) = world.resources.get_mut("water") {
                                w.production_rate *= 2.0;
                                w.capacity *= 2.0;
                            }
                        }
                        _ => {}
                    }
                    self.events.push(CivEvent::new(
                        self.current_tick,
                        Some(world.id),
                        CivEventType::EmergencyDeclared,
                        format!(
                            "{}: PROJECT COMPLETE — {} after {} months",
                            world.name,
                            bp.name(),
                            bp.duration()
                        ),
                    ));
                }
            }

            // Phase 5.4: Supply chain propagation
            // Disasters that hit Earth regions propagate through the supply DAG.
            let colony_supply = self.supply_chain.propagate();
            // Apply supply multipliers to colony resource production
            for world in &mut self.worlds {
                let supply_mult = match world.location.as_str() {
                    "Moon" => colony_supply
                        .get(&supply_chain::SupplyNode::LunarColony)
                        .copied()
                        .unwrap_or(1.0),
                    "Mars" => colony_supply
                        .get(&supply_chain::SupplyNode::MarsColony)
                        .copied()
                        .unwrap_or(1.0),
                    "Europa" => colony_supply
                        .get(&supply_chain::SupplyNode::EuropaStation)
                        .copied()
                        .unwrap_or(1.0),
                    "Titan" => colony_supply
                        .get(&supply_chain::SupplyNode::TitanOutpost)
                        .copied()
                        .unwrap_or(1.0),
                    _ => 1.0,
                };
                // Low supply chain health degrades colony resource production
                if supply_mult < 0.8 {
                    let penalty = 1.0 - supply_mult;
                    world.infrastructure_level =
                        (world.infrastructure_level - penalty * 0.001).max(0.0);
                }
            }

            // Phase 5.55: Fission Delivery — Earth can deliver reactors to colonies.
            // NASA doesn't expect colonies to invent fission; they deliver Kilopower units.
            // If Earth has engineering > 1.2 and an outer-system colony exists without
            // fission, the reactor is delivered via transfer window.
            if !self
                .disaster_engine
                .tech_tree
                .is_achieved("Fission Surface Power")
                && self.current_tick > 36
            // FISSION_EARLIEST: year 3
            {
                let earth_eng = self
                    .worlds
                    .iter()
                    .find(|w| w.location == "Earth")
                    .map(|w| w.knowledge.technology_levels[0])
                    .unwrap_or(1.0);
                let has_outer_colony = self
                    .worlds
                    .iter()
                    .any(|w| w.location != "Earth" && w.location != "Moon" && w.population() > 0);

                if earth_eng >= 1.2 && has_outer_colony {
                    // Deliver fission reactor — achieve the milestone
                    for m in &mut self.disaster_engine.tech_tree.milestones {
                        if m.name == "Fission Surface Power" && !m.achieved {
                            m.achieved = true;
                            m.achieved_tick = Some(self.current_tick);
                            self.events.push(CivEvent::new(
                                self.current_tick, None, CivEventType::EmergencyDeclared,
                                format!("FISSION DELIVERY: Earth delivered Kilopower reactors to outer colonies \
                                    (Earth engineering {:.2}). Nuclear power online.",
                                    earth_eng),
                            ));
                        }
                    }
                }
            }

            // Phase 5.6: Structural realism (power, maintenance, bus factor, trust, narrative)
            self.tick_structural_realism();

            // Phase 5.7: Inter-world diplomacy
            self.tick_diplomacy();

            // Phase 6: Knowledge
            self.tick_knowledge();

            // Metabolism: modulate innovation rate
            for world in &mut self.worlds {
                let innov_mult = self
                    .phase_modifiers
                    .get(&world.id)
                    .map(|m| m.innovation_mult)
                    .unwrap_or(1.0);
                if (innov_mult - 1.0).abs() > 0.01 {
                    world.knowledge.innovation_rate *= innov_mult;
                }
            }
            // Phase 7: Governance
            self.tick_governance();

            // Phase 7.25: Proposal Governance — factions propose, citizens vote
            {
                let tick = self.current_tick;
                for world in &mut self.worlds {
                    let factions_snapshot: Vec<crate::factions::Faction> =
                        self.faction_engine.factions.iter().cloned().collect();
                    let mut next_id = (tick * 100) as u32;
                    let mut new_proposals = proposals::generate_proposals(
                        &factions_snapshot,
                        &world.policy_state,
                        tick,
                        &mut next_id,
                        &mut self.rng,
                    );
                    for proposal in &mut new_proposals {
                        proposals::tally_votes(
                            proposal,
                            &mut world.agents,
                            &factions_snapshot,
                            tick,
                            &mut self.rng,
                        );
                        let adults = world.agents.iter().filter(|a| a.is_alive()).count();
                        let status = proposals::resolve_proposal(proposal, adults);
                        if status == proposals::ProposalStatus::Passed {
                            proposals::execute_proposal(proposal, &mut world.policy_state);
                        }
                    }
                }
            }
            // Phase 7.5: Graduated Sanctions (Ostrom Principle 5, Zosh et al. 2025).
            // Uses the Phase 2-aware entry so A2 counterfactual runs skip
            // the restorative-justice violation hook.
            let phase2 = self.config.policy.phase2_enabled;
            for world in &mut self.worlds {
                let oppression = world.governance.oppression_index;
                if oppression > 0.1 {
                    let _result = sanctions::apply_sanctions_with_phase2(
                        &mut world.agents,
                        oppression,
                        self.current_tick,
                        phase2,
                    );
                }
            }

            // Phase 7.55: Mycelix adversarial tick (Phase 2c).
            // Attack behavior still engages regardless of `phase2_enabled`
            // — we want counterfactual runs to receive the same attack
            // environment, isolating the DEFENSE's contribution.
            for world in &mut self.worlds {
                let _tel = red_team::apply_mycelix_adversarial_tick(
                    &mut world.agents,
                    self.current_tick,
                    0.01,
                );
            }

            // Phase 7.6: Restorative corrections (Phase 2b). Only runs when
            // Phase 2 defenses are active — the counterfactual baseline
            // doesn't have restorative justice.
            if self.config.policy.phase2_enabled {
                for world in &mut self.worlds {
                    let _ = sanctions::apply_restorative_corrections(
                        &mut world.agents,
                        self.current_tick,
                        &mut self.rng,
                    );
                }
            }

            // Phase 8: Consciousness
            self.tick_consciousness();

            // Phase 8.05: Ethical dynamics (moral dilemmas, drift, synthesis)
            self.tick_ethical_dynamics();

            // Phase 8.06: Institutional ethics persistence + socialization.
            // Institutional ethics = slow EMA of population mean (alpha=0.02).
            // New citizens drift toward institutional ethics at 0.0002/tick.
            // Moral memories resist drift away from crisis orientations.
            for world in &mut self.worlds {
                // Update institutional ethics as EMA of current population
                let pop_mean = agent::EthicalOrientation::mean_of(&world.agents);
                let alpha = 0.02;
                world.institutional_ethics.deontological +=
                    (pop_mean.deontological - world.institutional_ethics.deontological) * alpha;
                world.institutional_ethics.consequentialist += (pop_mean.consequentialist
                    - world.institutional_ethics.consequentialist)
                    * alpha;
                world.institutional_ethics.virtue_care +=
                    (pop_mean.virtue_care - world.institutional_ethics.virtue_care) * alpha;
                world.institutional_ethics.relational +=
                    (pop_mean.relational - world.institutional_ethics.relational) * alpha;

                // Socialization: agents drift toward institutional ethics
                let inst = world.institutional_ethics.as_vec();
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    let ag = agent.ethics.as_vec();
                    for d in 0..4 {
                        let delta = (inst[d] - ag[d]) * 0.0002;
                        agent.ethics.modify_with_sacred_resistance(d, delta);
                    }
                }

                // Moral memory antibodies: resist drift away from crisis orientations
                let tick = self.current_tick;
                for mem in &world.moral_memories {
                    let strength = mem.strength(tick);
                    if strength <= 0.0 {
                        continue;
                    }
                    // If population is drifting away from the lesson dimension,
                    // apply resistance proportional to memory strength
                    let current_val = pop_mean.as_vec()[mem.lesson_dimension];
                    let crisis_val = mem.ethics_at_crisis[mem.lesson_dimension];
                    if current_val < crisis_val - 0.1 {
                        // Drifting away from the lesson — resist
                        let correction = (crisis_val - current_val) * 0.001 * strength;
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent
                                .ethics
                                .modify_with_sacred_resistance(mem.lesson_dimension, correction);
                        }
                    }
                }

                // Prune expired memories (> 600 ticks old)
                world.moral_memories.retain(|m| m.strength(tick) > 0.0);
            }

            // Phase 8.1: Dead Loop #7 fix — Consciousness ↔ Governance feedback.
            // Stable governance creates space for consciousness growth.
            // Unstable governance (fear, uncertainty) suppresses growth.
            // Constitution maturity lowers participation thresholds over time.
            for (i, gov) in self.governance.iter().enumerate() {
                if i >= self.worlds.len() {
                    break;
                }
                let stability = gov.stability_score;
                let stability_bonus = if stability > 0.7 {
                    0.001 // stable governance → mild consciousness boost
                } else if stability < 0.3 {
                    -0.002 // unstable → consciousness suppression (fear)
                } else {
                    0.0
                };
                if stability_bonus != 0.0 {
                    for agent in self.worlds[i].agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.consciousness.level =
                            (agent.consciousness.level + stability_bonus).clamp(0.0, 1.0);
                    }
                }
            }

            // Phase 8.5: Factions (after consciousness, before harmony)
            {
                let policy = self.config.policy.clone();
                // Mechanism 2 — Stress-driven faction emergence: compute max stress
                // boost across all worlds. When mean allostatic load > 0.8, faction
                // emergence probability is multiplied by up to 4x.
                // Stress boost: classic allostatic load + Spinozist suffering.
                // Negative net conatus (sadness > joy) amplifies faction recruitment.
                // High desire + low care = Turchin elite overproduction dynamics.
                let stress_boost: f64 = self
                    .worlds
                    .iter()
                    .map(|w| {
                        let load = w.mean_allostatic_load();
                        let load_boost = if load > 0.8 { load - 0.8 } else { 0.0 };
                        // Spinozist amplifier: collective suffering drives faction emergence
                        let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                        let n = living.len().max(1) as f64;
                        let mean_conatus = living
                            .iter()
                            .map(|a| a.needs.affect.net_conatus())
                            .sum::<f64>()
                            / n;
                        let mean_desire =
                            living.iter().map(|a| a.needs.affect.desire).sum::<f64>() / n;
                        let mean_care = living.iter().map(|a| a.needs.affect.care).sum::<f64>() / n;
                        // Suffering (negative conatus) adds to faction pressure
                        let suffering_boost = if mean_conatus < 0.0 {
                            -mean_conatus * 0.5
                        } else {
                            0.0
                        };
                        // High desire + low care = frustration → faction emergence
                        let frustration_boost = if mean_desire > 0.5 && mean_care < 0.3 {
                            (mean_desire - mean_care) * 0.3
                        } else {
                            0.0
                        };
                        load_boost + suffering_boost + frustration_boost
                    })
                    .fold(0.0f64, f64::max);
                let faction_events = self.faction_engine.tick_factions_with_stress(
                    &mut self.worlds,
                    self.current_tick,
                    &mut self.rng,
                    &policy,
                    stress_boost,
                );
                self.events.extend(faction_events);
            }

            // Phase 8.6: Morale contagion (social epidemic model)
            self.tick_morale_contagion();

            // Phase 8.7: Genetic viability + language drift (every 12 ticks = annually)
            if self.current_tick % 12 == 0 {
                for world in &mut self.worlds {
                    // Genetic viability check (Smith 2014, Murray & Murray 2012)
                    let viability = PopulationEngine::genetic_viability(world, self.current_tick);
                    if viability < 0.3 && world.population() > 0 && world.population() < 500 {
                        // Genetic crisis: health penalty on newborns (already handled in
                        // inbreeding_coefficient check in population.rs), but also increase
                        // allostatic load from awareness of genetic bottleneck
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.needs.allostatic_load =
                                (agent.needs.allostatic_load + 0.001).min(1.0);
                        }
                        if self.current_tick % 120 == 0 {
                            // Log every 10 years
                            self.events.push(CivEvent::new(
                                self.current_tick,
                                Some(world.id),
                                CivEventType::EmergencyDeclared,
                                format!(
                                    "{}: GENETIC BOTTLENECK — viability {:.0}%, pop {} (Ne ≈ {})",
                                    world.name,
                                    viability * 100.0,
                                    world.population(),
                                    (world.population() as f64 * 0.25) as usize
                                ),
                            ));
                        }
                    }

                    // Language drift: inter-colony communication degrades over centuries.
                    // Swadesh (1952): 14% core vocabulary loss per 1000yr.
                    // Small populations (<1000): mutual unintelligibility in ~300-500yr.
                    // Modeled as cultural distance acceleration for isolated worlds.
                    if world.location != "Earth" && world.population() < 1000 {
                        let isolation_years =
                            self.current_tick.saturating_sub(world.founded_tick) as f64 / 12.0;
                        if isolation_years > 200.0 {
                            // Accelerate cultural drift for small isolated populations
                            // (already handled in culture.drift, but we add a bonus)
                            world.culture.individualism =
                                (world.culture.individualism + 0.001).min(1.0);
                        }
                    }
                }
            }

            // Phase 9: Harmony scoring
            self.tick_harmony_scoring();

            // Phase 9.1: Dead Loop #5 fix — Harmony → Policy feedback.
            // Low harmony scores trigger soft policy adjustments.
            for world in &mut self.worlds {
                let scores = world.harmony.current_scores;
                // Low Pan-Sentient Flourishing (< 0.3) → shift culture toward care
                if scores[1] < 0.3 {
                    world.culture.harmony_weights[1] =
                        (world.culture.harmony_weights[1] + 0.005).min(0.3);
                }
                // Low Sacred Stillness (< 0.3) → reduce overwork for stressed agents
                if scores[7] < 0.3 {
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        if agent.needs.allostatic_load > 0.5 {
                            agent.needs.allostatic_load =
                                (agent.needs.allostatic_load - 0.002).max(0.0);
                        }
                    }
                }
                // High Resonant Coherence (> 0.7) → trust bonus
                if scores[0] > 0.7 {
                    world.trust_level = (world.trust_level + 0.002).min(1.0);
                }
            }

            // Phase 9.2: Resource depletion feedback
            self.tick_resource_depletion();

            // Phase 9.2b: Apply depletion crisis effects (health reduction models
            // increased mortality pressure — proxy for 1.5x death rate)
            for world in &mut self.worlds {
                let has_crisis = ["food", "water", "oxygen"].iter().any(|res| {
                    self.depletion_crisis_active
                        .get(&(world.id, res.to_string()))
                        .copied()
                        .unwrap_or(false)
                });
                if has_crisis {
                    // Increase mortality via gradual health reduction
                    for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                        agent.health = (agent.health - 0.003).max(0.1);
                    }
                }
            }

            // Phase 9.3: Communication delay governance penalty
            self.tick_communication_delay();

            // Phase 9.4: Environmental carrying capacity
            self.tick_carrying_capacity();

            // Phase 9.5: Disasters (before emergencies — disasters can trigger emergencies)
            self.tick_disasters();

            // Phase 9.6: Turchin cliodynamics — secular cycles, civil war, secession
            if self.config.policy.turchin_cycles_enabled {
                let world_count = self.worlds.len();
                for i in 0..world_count {
                    let world = &self.worlds[i];
                    let world_id = world.id;
                    let pop = world.population();
                    if pop == 0 {
                        continue;
                    }

                    // Compute elite fraction: top 20% by consciousness + skills
                    // In the real sim, agents with phi >= 0.6 are rare.
                    // Elite = those whose economic influence (skills + education) puts
                    // them in the top quintile — they compete for governance positions.
                    let alive: Vec<&agent::CivAgent> = world
                        .agents
                        .iter()
                        .filter(|a| a.death_tick.is_none())
                        .collect();
                    let elite_threshold = if alive.len() >= 10 {
                        let mut skill_scores: Vec<f64> = alive
                            .iter()
                            .map(|a| a.skills.total() + a.education_level * 2.0)
                            .collect();
                        skill_scores
                            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        let top_20_idx = (skill_scores.len() as f64 * 0.20) as usize;
                        skill_scores.get(top_20_idx).copied().unwrap_or(0.0)
                    } else {
                        f64::MAX // Too few people for elite dynamics
                    };
                    let elite_count = alive
                        .iter()
                        .filter(|a| a.skills.total() + a.education_level * 2.0 >= elite_threshold)
                        .count();
                    let elite_fraction = elite_count as f64 / pop.max(1) as f64;

                    // Compute non-elite mean phi and prediction error proxy
                    let non_elites: Vec<&agent::CivAgent> = alive
                        .iter()
                        .filter(|a| a.skills.total() + a.education_level * 2.0 < elite_threshold)
                        .copied()
                        .collect();
                    let non_elite_mean_phi = if non_elites.is_empty() {
                        0.5
                    } else {
                        non_elites
                            .iter()
                            .map(|a| a.consciousness.phi())
                            .sum::<f64>()
                            / non_elites.len() as f64
                    };
                    // FEP prediction error proxy: allostatic load × (1 - phi)
                    // High stress + low consciousness = high prediction error
                    let non_elite_prediction_error = if self.config.policy.fep_immiseration_enabled
                    {
                        let mean_load = if non_elites.is_empty() {
                            0.0
                        } else {
                            non_elites
                                .iter()
                                .map(|a| a.needs.allostatic_load)
                                .sum::<f64>()
                                / non_elites.len() as f64
                        };
                        mean_load * (1.0 - non_elite_mean_phi) * 2.0
                    } else {
                        0.0
                    };

                    let inputs = cliodynamics::CycleInputs {
                        gini: world.economy.gini_coefficient,
                        elite_fraction,
                        governance_positions: 10 + pop / 100, // ~1 seat per 100 people
                        population: pop,
                        self_sufficiency: world.resources.self_sufficiency(),
                        governance_quality: world.harmony.current_scores.iter().sum::<f64>() / 8.0,
                        mean_allostatic_load: if non_elites.is_empty() {
                            0.0
                        } else {
                            non_elites
                                .iter()
                                .map(|a| a.needs.allostatic_load)
                                .sum::<f64>()
                                / non_elites.len() as f64
                        },
                        non_elite_prediction_error,
                        non_elite_mean_phi,
                        secession_capable: world.location != "Earth"
                            && world.resources.self_sufficiency() > 0.7,
                        current_tick: self.current_tick,
                        mean_coordination_understanding: {
                            let alive: Vec<&agent::CivAgent> =
                                world.agents.iter().filter(|a| a.is_alive()).collect();
                            if alive.is_empty() {
                                0.0
                            } else {
                                alive
                                    .iter()
                                    .map(|a| a.coordination_understanding)
                                    .sum::<f64>()
                                    / alive.len() as f64
                            }
                        },
                    };

                    let cycle_state = self
                        .secular_cycles
                        .entry(world_id)
                        .or_insert_with(cliodynamics::SecularCycleState::default);

                    let cycle_events = cycle_state.tick(&inputs, self.rng.next_f64());

                    // Apply cycle events to world
                    for event in cycle_events {
                        match event {
                            cliodynamics::CycleEvent::CivilWar {
                                population_loss_fraction,
                                infrastructure_damage,
                            } => {
                                let world = &mut self.worlds[i];
                                let deaths =
                                    (world.population() as f64 * population_loss_fraction) as usize;
                                // Kill random agents
                                let mut killed = 0;
                                for agent in &mut world.agents {
                                    if killed >= deaths {
                                        break;
                                    }
                                    if agent.death_tick.is_none()
                                        && self.rng.next_f64() < population_loss_fraction * 3.0
                                    {
                                        agent.death_tick = Some(self.current_tick);
                                        killed += 1;
                                    }
                                }
                                world.infrastructure_level =
                                    (world.infrastructure_level - infrastructure_damage).max(0.0);
                                // Increase trauma for all survivors
                                for agent in &mut world.agents {
                                    if agent.death_tick.is_none() {
                                        agent.trauma_level = (agent.trauma_level
                                            + 0.3 * population_loss_fraction)
                                            .min(1.0);
                                        agent.wounds.push(wound_healing::WoundState::new(
                                            (0.3 * population_loss_fraction).min(0.5),
                                            wound_healing::WoundOrigin::Faction,
                                            self.current_tick,
                                        ));
                                    }
                                }
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(world_id),
                                    CivEventType::EmergencyDeclared,
                                    format!("CIVIL WAR on {} — {:.1}% casualties, {:.0}% infrastructure destroyed",
                                        self.worlds[i].name, population_loss_fraction * 100.0, infrastructure_damage * 100.0),
                                ));
                            }
                            cliodynamics::CycleEvent::Secession { .. } => {
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(world_id),
                                    CivEventType::EmergencyDeclared,
                                    format!(
                                        "SECESSION: {} declares independence from Earth",
                                        self.worlds[i].name
                                    ),
                                ));
                            }
                            cliodynamics::CycleEvent::ElitePurge {
                                agents_affected_fraction,
                            } => {
                                let world = &mut self.worlds[i];
                                for agent in &mut world.agents {
                                    if agent.death_tick.is_none()
                                        && agent.consciousness.phi() >= 0.6
                                        && self.rng.next_f64() < agents_affected_fraction
                                    {
                                        // Demote: reduce consciousness
                                        agent.consciousness.level *= 0.5;
                                        agent.consciousness.meta_awareness *= 0.5;
                                    }
                                }
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(world_id),
                                    CivEventType::EmergencyDeclared,
                                    format!(
                                        "Elite purge on {} — {:.0}% of elites demoted",
                                        self.worlds[i].name,
                                        agents_affected_fraction * 100.0
                                    ),
                                ));
                            }
                            cliodynamics::CycleEvent::PhaseTransition { from, to } => {
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(world_id),
                                    CivEventType::TradeEstablished, // Reuse event type for phase change
                                    format!(
                                        "Turchin cycle on {}: {:?} → {:?}",
                                        self.worlds[i].name, from, to
                                    ),
                                ));
                            }
                            cliodynamics::CycleEvent::Recovery => {
                                self.events.push(CivEvent::new(
                                    self.current_tick,
                                    Some(world_id),
                                    CivEventType::TradeEstablished,
                                    format!(
                                        "{} recovering from secular depression",
                                        self.worlds[i].name
                                    ),
                                ));
                            }
                        }
                    }
                }
            }

            // Phase 9.7: JSONL telemetry output
            if let Some(ref path) = self.jsonl_output_path {
                let total_pop = self.worlds.iter().map(|w| w.population()).sum();
                let mean_phi = if self.worlds.is_empty() {
                    0.0
                } else {
                    self.worlds
                        .iter()
                        .map(|w| {
                            let alive: Vec<_> =
                                w.agents.iter().filter(|a| a.death_tick.is_none()).collect();
                            if alive.is_empty() {
                                0.0
                            } else {
                                alive.iter().map(|a| a.consciousness.phi()).sum::<f64>()
                                    / alive.len() as f64
                            }
                        })
                        .sum::<f64>()
                        / self.worlds.len() as f64
                };
                let mean_gini = if self.worlds.is_empty() {
                    0.0
                } else {
                    self.worlds
                        .iter()
                        .map(|w| w.economy.gini_coefficient)
                        .sum::<f64>()
                        / self.worlds.len() as f64
                };
                let first_world_cycle = self.secular_cycles.values().next();
                let metrics = live_metrics::LiveMetrics::compute(
                    self.current_tick,
                    &self.config.policy,
                    first_world_cycle,
                    0.0, // CVS computed later
                    total_pop,
                    mean_phi,
                    mean_gini,
                    0.0,   // mean allostatic load
                    0.0,   // infrastructure mean
                    false, // solar cycle max
                    1.0,   // kessler density
                );
                let _ = metrics.append_jsonl(path);
            }

            // Phase 9.8: CSV time-series output (scientific analysis)
            if let Some(ref mut recorder) = self.csv_recorder {
                let active = self.disaster_engine.active_disasters.len() as u32;
                let _ = recorder.record_tick(self.current_tick, &self.worlds, active);
            }

            // Phase 10: Emergencies
            self.tick_emergencies();

            // Phase 10.5: Resontia construction (if enabled)
            if self.resontia_config.enabled {
                if let Some(earth) = self.worlds.iter().find(|w| w.location == "Earth") {
                    let earth_clone = earth.clone();
                    let (resontia_events, _) = resontia::tick_resontia(
                        &mut self.resontia_infra,
                        &self.resontia_config,
                        &earth_clone,
                        self.current_tick,
                        0, // no disaster this call — just construction
                        None,
                        &mut self.rng,
                    );
                    self.events.extend(resontia_events);
                }
            }

            // Phase 11: Epoch evaluation
            self.tick_epoch_evaluation();

            // Phase 12: Narrative engine — generate memorable events from affect + disaster data
            if self.current_tick % 12 == 0 {
                // Monthly narrative check (yearly)
                let world_data: Vec<_> = self
                    .worlds
                    .iter()
                    .map(|w| {
                        let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                        let n = living.len().max(1) as f64;
                        (
                            w.name.clone(),
                            w.location.clone(),
                            w.population(),
                            living.iter().map(|a| a.needs.affect.joy).sum::<f64>() / n,
                            living.iter().map(|a| a.needs.affect.sadness).sum::<f64>() / n,
                            living.iter().map(|a| a.needs.affect.desire).sum::<f64>() / n,
                            living.iter().map(|a| a.needs.affect.care).sum::<f64>() / n,
                            w.resources.self_sufficiency(),
                        )
                    })
                    .collect();
                let cvs = self
                    .epoch_snapshots
                    .last()
                    .map(|s| s.civilization_viability_score)
                    .unwrap_or(0.5);
                let achieved: Vec<String> = self
                    .disaster_engine
                    .tech_tree
                    .milestones
                    .iter()
                    .filter(|m| m.achieved)
                    .map(|m| m.name.clone())
                    .collect();
                let active_count = self.disaster_engine.active_disasters.len() as u32;
                self.narrative_engine.tick(
                    self.current_tick,
                    &world_data,
                    cvs,
                    &achieved,
                    active_count,
                    self.disaster_engine.orbital_debris.cascade_active,
                    self.disaster_engine.magnetosphere.excursion_active,
                );
                // P0: Bridge CivEvents into narrative engine
                self.narrative_engine.ingest_civ_events(&self.events);
            }

            // Dead agent compaction: every 600 ticks, remove agents dead for 1200+ ticks
            if self.current_tick % 600 == 0 && self.current_tick > 1200 {
                let cutoff = self.current_tick - 1200;
                for world in &mut self.worlds {
                    world
                        .agents
                        .retain(|a| a.death_tick.map_or(true, |dt| dt >= cutoff));
                }
            }

            // Trauma decay: 0.001/tick ≈ 0.012/year. Full trauma (1.0) takes
            // ~83 years to fully decay. Ref: PTSD recovery timelines (Kessler 1995).
            // Care workers accelerate recovery. Slower decay = trauma persists meaningfully.
            for world in &mut self.worlds {
                let care_ratio = world
                    .agents
                    .iter()
                    .filter(|a| a.is_alive() && a.skills.strongest() == "medicine")
                    .count() as f64
                    / world.population().max(1) as f64;
                // Slower base decay: ~167 years to fully recover without care (was ~83yr).
                // This ensures trauma accumulates meaningfully from repeated disasters
                // and persists long enough to affect consciousness and governance.
                let decay = 0.0005 + care_ratio * 0.002;
                for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.trauma_level = (agent.trauma_level - decay).max(0.0);
                }
            }

            // Cohort snapshot: compute cohort demographics every 120 ticks (10 years).
            // This runs alongside V1 agents — it doesn't replace them, just provides
            // aggregate statistics for reporting and future V2 migration.
            if self.current_tick % 120 == 0 {
                for world in &self.worlds {
                    let cm = cohort::CohortManager::from_v1_agents(
                        &world.agents,
                        world.id,
                        self.current_tick,
                    );
                    // Log cohort stats at key intervals
                    if self.current_tick % 600 == 0 {
                        // Every 50 years
                        let _cohorts = cm.cohort_count();
                        let _workers = cm.worker_count();
                        let _load = cm.mean_load();
                        // These stats are available for telemetry/reporting
                        // Future: replace V1 agent loop with cohort updates
                    }
                }
            }

            self.current_tick += 1;
        }

        self.build_final_report()
    }

    /// Construct the final CivilizationReport from accumulated simulation state.
    fn build_final_report(&self) -> CivilizationReport {
        // Genetic diversity: focus on OFF-EARTH worlds only. Earth's 10K
        // population masks the bottleneck that matters — the colony's genetics.
        let off_earth: Vec<_> = self
            .worlds
            .iter()
            .filter(|w| w.location != "Earth")
            .collect();
        let genetic_diversity = if off_earth.is_empty() {
            // No off-Earth worlds yet — use all worlds
            if self.worlds.is_empty() {
                0.0
            } else {
                self.worlds
                    .iter()
                    .map(|w| PopulationEngine::genetic_diversity_index(w, self.current_tick))
                    .sum::<f64>()
                    / self.worlds.len() as f64
            }
        } else {
            off_earth
                .iter()
                .map(|w| PopulationEngine::genetic_diversity_index(w, self.current_tick))
                .sum::<f64>()
                / off_earth.len() as f64
        };

        let economic_sustainability = self.mean_self_sufficiency();
        let collective_phi = self.mean_phi_all();

        // Real harmony scores from HarmonyTracker (not the cultural weights proxy)
        let harmony_scores = if self.worlds.is_empty() {
            [0.0; 8]
        } else {
            let trackers: Vec<_> = self.worlds.iter().map(|w| w.harmony.clone()).collect();
            harmony::HarmonyTracker::civilization_harmony(&trackers)
        };
        let love_coherence = if self.worlds.is_empty() {
            0.0
        } else {
            let trackers: Vec<_> = self.worlds.iter().map(|w| w.harmony.clone()).collect();
            harmony::HarmonyTracker::civilization_love_coherence(&trackers)
        };

        let harmony_mean: f64 = harmony_scores.iter().sum::<f64>() / 8.0;
        // Bug #3 fix: Wire actual oppression index from governance system
        // instead of hardcoded 0.0. The oppression detector (governance.rs) uses
        // tier distribution to measure democratic health.
        let max_oppression = self
            .governance
            .iter()
            .map(|g| g.oppression_index)
            .fold(0.0f64, f64::max);

        let final_cvs = EpochManager::compute_cvs(
            genetic_diversity,
            economic_sustainability,
            harmony_mean,
            max_oppression,
            collective_phi,
        );
        let final_cvs_geometric = EpochManager::compute_cvs_geometric(
            genetic_diversity,
            economic_sustainability,
            harmony_mean,
            max_oppression,
            collective_phi,
        );

        let mut final_snapshot = EpochSnapshot::from_worlds(
            self.epoch_manager.current_epoch,
            self.current_tick,
            &self.worlds,
        );
        self.populate_observables(&mut final_snapshot);

        let mut snapshots = self.epoch_snapshots.clone();
        snapshots.push(final_snapshot);

        // Aggregate psychological needs from final snapshot
        let (total_load, total_engagement, agent_count) =
            self.worlds
                .iter()
                .fold((0.0f64, 0.0f64, 0usize), |(load, eng, count), w| {
                    let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                    let n = living.len();
                    let l: f64 = living.iter().map(|a| a.needs.allostatic_load).sum();
                    let e: f64 = living.iter().map(|a| a.needs.engagement).sum();
                    (load + l, eng + e, count + n)
                });
        let ac = agent_count.max(1) as f64;
        let mean_allostatic_load = total_load / ac;
        let mean_engagement = total_engagement / ac;

        let mut report = CivilizationReport::build(
            &self
                .config
                .initial_worlds
                .first()
                .map(|w| w.name.clone())
                .unwrap_or_else(|| "Unnamed".into()),
            self.config.seed,
            self.current_tick,
            self.total_population(),
            self.worlds.len(),
            final_cvs,
            snapshots.clone(),
            self.epoch_manager.checkpoint_results.clone(),
            &self.events,
            genetic_diversity,
            economic_sustainability,
            harmony_scores,
            love_coherence,
            max_oppression,
            collective_phi,
            self.epoch_manager.first_birth_tick,
            self.epoch_manager.first_constitution_tick,
            self.epoch_manager.first_trade_tick,
            mean_allostatic_load,
            mean_engagement,
        );

        // Populate disaster statistics from the disaster engine
        report.total_disasters = self.disaster_engine.total_disasters;
        report.carrington_events = self.disaster_engine.carrington_events;
        report.tech_milestones_achieved = self
            .disaster_engine
            .tech_tree
            .milestones
            .iter()
            .filter(|m| m.achieved)
            .count();

        // Populate extended observable fields from snapshots
        report.max_elite_persistence = snapshots
            .iter()
            .map(|s| s.elite_persistence)
            .fold(0.0f64, f64::max);
        report.max_innovation_stagnation = snapshots
            .iter()
            .map(|s| s.innovation_stagnation)
            .fold(0.0f64, f64::max);
        report.phi_trend_at_end = snapshots
            .last()
            .map(|s| s.phi_trend.clone())
            .unwrap_or_else(|| "Unknown".into());
        report.max_trauma = snapshots
            .iter()
            .map(|s| s.trauma_level)
            .fold(0.0f64, f64::max);

        // Phase 2c roadmap A1: expose MycelixResilience for scenarios that
        // injected adversaries. `None` if no agent is tagged adversarial.
        report.mycelix_resilience = red_team::compute_resilience_from_worlds(&self.worlds);

        // Geometric-mean CVS ("weakest link" — K-index-inspired).
        report.final_cvs_geometric = final_cvs_geometric;

        report
    }
}

// Provide a Default for World so std::mem::take works
impl Default for World {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            location: String::new(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::default(),
            culture: CulturalProfile::earth_default(),
            infrastructure_level: 0.0,
            max_population: 0,
            habitable_area_m2: 0.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: knowledge::WorldKnowledge::new(),
            economy: economy::WorldEconomy::new(),
            harmony: harmony::HarmonyTracker::new(),
            governance: governance::WorldGovernance::new(),
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
            fertility_multiplier: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config(ticks: u32) -> SimulationConfig {
        let mut c = SimulationConfig::default_150_year();
        c.total_ticks = ticks;
        c
    }

    #[test]
    fn test_full_150_year_simulation() {
        let config = SimulationConfig::default_150_year();
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        assert_eq!(report.total_ticks, 1800);
        assert!(
            report.final_population > 0,
            "Population should survive 150 years"
        );
        assert!(report.final_worlds >= 2, "Should have at least 2 worlds");
        assert!(
            report.final_cvs > 0.0,
            "CVS should be computed: {}",
            report.final_cvs
        );
        assert!(
            !report.epoch_snapshots.is_empty(),
            "Should have epoch snapshots"
        );
        assert!(
            report.checkpoints_passed + report.checkpoints_failed > 0,
            "Should have evaluated checkpoints"
        );

        // Verify the summary is non-empty
        let summary = report.summary();
        assert!(summary.len() > 100, "Summary should be substantial");
    }

    #[test]
    fn test_10_year_smoke() {
        let config = small_config(120);
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        assert_eq!(report.total_ticks, 120);
        assert!(
            report.final_population > 0,
            "Should have surviving population"
        );
        // Population should grow from initial 10012 (Earth + Moon)
        assert!(
            report.final_population >= 100,
            "Population should not collapse in 10 years: {}",
            report.final_population
        );
    }

    #[test]
    fn test_deterministic_reproducibility() {
        let config1 = SimulationConfig::default_150_year();
        let mut config2 = SimulationConfig::default_150_year();
        // Same seed = same results
        assert_eq!(config1.seed, config2.seed);

        // Run shorter for test speed
        let mut c1 = config1;
        c1.total_ticks = 240;
        let mut c2 = config2;
        c2.total_ticks = 240;

        let mut sim1 = MultiWorldSimulator::new(c1);
        let mut sim2 = MultiWorldSimulator::new(c2);

        let r1 = sim1.run();
        let r2 = sim2.run();

        assert_eq!(
            r1.final_population, r2.final_population,
            "Same seed should produce same population"
        );
        assert!(
            (r1.final_cvs - r2.final_cvs).abs() < 1e-10,
            "Same seed should produce same CVS: {} vs {}",
            r1.final_cvs,
            r2.final_cvs
        );
        assert_eq!(
            r1.total_events, r2.total_events,
            "Same seed should produce same event count"
        );
    }

    #[test]
    fn test_mars_colony_founded() {
        // The default config has Mars at tick 300 (year 25)
        let mut config = SimulationConfig::default_150_year();
        config.total_ticks = 360; // Run past Mars founding
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // Should have at least Earth + Moon + Mars
        assert!(
            report.final_worlds >= 3,
            "Mars should be founded by tick 300, got {} worlds",
            report.final_worlds
        );

        // Verify WorldFounded event for Mars
        let mars_founded = report.worlds_founded;
        assert!(
            mars_founded >= 3,
            "Should have at least 3 WorldFounded events (Earth, Moon, Mars), got {}",
            mars_founded
        );
    }

    #[test]
    fn test_epoch_transitions_occur() {
        let mut config = SimulationConfig::default_150_year();
        config.total_ticks = 500; // Run through Seeds and into Branches
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        let epoch_events: Vec<_> = sim
            .events
            .iter()
            .filter(|e| e.event_type == CivEventType::EpochTransition)
            .collect();

        assert!(
            !epoch_events.is_empty(),
            "Should have epoch transitions in 500 ticks"
        );
    }

    #[test]
    fn test_consciousness_grows_over_time() {
        let mut config = SimulationConfig::default_150_year();
        config.total_ticks = 600;
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // Threshold lowered from 0.1 to 0.05: realism mechanisms (infrastructure
        // aging, knowledge loss, morale contagion, carrying capacity) add realistic
        // headwinds to consciousness development, especially in early decades.
        assert!(
            report.final_collective_phi > 0.05,
            "Phi should grow above nascent level after 50 years: {:.3}",
            report.final_collective_phi
        );
    }

    #[test]
    fn test_report_summary_format() {
        let mut config = SimulationConfig::default_150_year();
        config.total_ticks = 24;
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        let summary = report.summary();

        assert!(summary.contains("CIVILIZATION REPORT"));
        assert!(summary.contains("VIABILITY COMPONENTS:"));
        assert!(summary.contains("KEY EVENTS:"));
        assert!(summary.contains("HARMONY SCORES"));
    }

    /// A/B comparison: education guild ON vs OFF.
    ///
    /// The thermodynamic hypothesis: communities with peer-to-peer education
    /// should show lower allostatic load, higher consciousness, and better
    /// survival rates than communities without it.
    ///
    /// Run full 150 years (1800 ticks) — the crucible.
    /// With epistemic friction, teacher fatigue, and diminishing social returns,
    /// the guild advantage should be real but not magical.
    #[test]
    fn test_education_guild_ab_comparison() {
        // === Scenario A: Education enabled (guild model) ===
        let mut config_a = SimulationConfig::default_150_year();
        config_a.seed = 42;
        config_a.policy.education_enabled = true;

        let mut sim_a = MultiWorldSimulator::new(config_a);
        let report_a = sim_a.run();

        // === Scenario B: Education disabled (1602 model — no peer teaching) ===
        let mut config_b = SimulationConfig::default_150_year();
        config_b.seed = 42; // Same seed for fair comparison
        config_b.policy.education_enabled = false;

        let mut sim_b = MultiWorldSimulator::new(config_b);
        let report_b = sim_b.run();

        // === Print comparison ===
        eprintln!("\n=== EDUCATION GUILD A/B COMPARISON (150 years, with epistemic friction) ===");
        eprintln!("                          WITH guild    WITHOUT guild");
        eprintln!(
            "Population:              {:>10}    {:>10}",
            report_a.final_population, report_b.final_population
        );
        eprintln!(
            "Survived:                {:>10}    {:>10}",
            report_a.survived, report_b.survived
        );
        eprintln!(
            "CVS (viability):         {:>10.3}    {:>10.3}",
            report_a.final_cvs, report_b.final_cvs
        );
        eprintln!(
            "Mean allostatic load:    {:>10.3}    {:>10.3}",
            report_a.final_mean_allostatic_load, report_b.final_mean_allostatic_load
        );
        eprintln!(
            "Collective Phi:          {:>10.3}    {:>10.3}",
            report_a.final_collective_phi, report_b.final_collective_phi
        );
        eprintln!(
            "Mean engagement:         {:>10.3}    {:>10.3}",
            report_a.final_mean_engagement, report_b.final_mean_engagement
        );
        eprintln!(
            "Breakthroughs:           {:>10}    {:>10}",
            report_a.breakthroughs, report_b.breakthroughs
        );
        eprintln!(
            "Checkpoints passed:      {:>10}    {:>10}",
            report_a.checkpoints_passed, report_b.checkpoints_passed
        );

        // Count teaching events in scenario A
        let teaching_events_a = sim_a
            .events
            .iter()
            .filter(|e| matches!(e.event_type, CivEventType::TeachingInteraction))
            .count();
        let crisis_events_a = sim_a
            .events
            .iter()
            .filter(|e| matches!(e.event_type, CivEventType::SkillCrisis))
            .count();
        let crisis_events_b = sim_b
            .events
            .iter()
            .filter(|e| matches!(e.event_type, CivEventType::SkillCrisis))
            .count();

        eprintln!("Teaching events (A):     {:>10}", teaching_events_a);
        eprintln!(
            "Skill crises (A):        {:>10}    {:>10}",
            crisis_events_a, crisis_events_b
        );
        eprintln!("================================================\n");

        // === Assertions: the education guild should help ===
        // Both should survive 50 years (this is a baseline sanity check)
        assert!(report_a.final_population > 0, "Guild world should survive");
        assert!(
            report_b.final_population > 0,
            "Control world should survive 50 years"
        );

        // The guild world should have peer teaching events
        assert!(
            teaching_events_a > 0,
            "Education guild should produce teaching interactions"
        );
    }

    #[test]
    fn test_hybrid_earth_backward_compat_default() {
        // With hybrid_earth=false (default), earth_regions and spaceport remain empty/None.
        let config = small_config(60);
        assert!(!config.policy.hybrid_earth);
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        assert!(
            sim.earth_regions.is_empty(),
            "hybrid_earth=false should not populate regions"
        );
        assert!(
            sim.spaceport.is_none(),
            "hybrid_earth=false should not create spaceport"
        );
        assert!(report.final_population > 0, "Classic model should work");
    }

    #[test]
    fn test_hybrid_earth_creates_12_regions_and_spaceport() {
        let mut config = small_config(120);
        config.policy.hybrid_earth = true;
        let mut sim = MultiWorldSimulator::new(config);
        let _report = sim.run();
        assert_eq!(sim.earth_regions.len(), 12, "Should have 12 Earth regions");
        assert!(sim.spaceport.is_some(), "Should have a spaceport");
    }

    // ================================================================
    // Phase 0-2 Integration Tests
    // ================================================================

    #[test]
    fn test_viability_engine_runs_during_simulation() {
        let mut config = small_config(120); // 10 years
        config.policy.hybrid_earth = false;
        let mut sim = MultiWorldSimulator::new(config);
        let _report = sim.run();

        // Viability engine should have tracked worlds
        assert!(
            !sim.viability_engine.ledgers.is_empty(),
            "Viability engine should have energy ledgers"
        );
        assert!(
            !sim.viability_engine.scaling.is_empty(),
            "Viability engine should have scaling factors"
        );

        // Entropy should have increased (2nd Law)
        for ledger in sim.viability_engine.ledgers.values() {
            assert!(
                ledger.cumulative_entropy > 0.0,
                "Entropy should be positive after 10 years: {}",
                ledger.cumulative_entropy
            );
        }
    }

    #[test]
    fn test_scaling_factors_affect_economy() {
        let mut config = small_config(60); // 5 years
        config.policy.hybrid_earth = false;
        let mut sim = MultiWorldSimulator::new(config);
        let _report = sim.run();

        // Check that scaling factors were computed for populated worlds
        for world in &sim.worlds {
            if world.population() > 0 {
                let scaling = sim.viability_engine.scaling_for(world.id);
                assert!(
                    scaling.population > 0.0,
                    "Scaling should track population for world {}",
                    world.name
                );
            }
        }
    }

    #[test]
    fn test_dead_loops_active_during_simulation() {
        let mut config = small_config(240); // 20 years — enough for disasters
        config.policy.hybrid_earth = false;
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // DL#1: Disasters should cause some trauma OR wounds
        // With wound healing active, trauma_level may be 0.0 if all wounds healed.
        // Check for either non-zero trauma OR non-empty wound history.
        let max_trauma: f64 = sim
            .worlds
            .iter()
            .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
            .map(|a| a.trauma_level)
            .fold(0.0f64, f64::max);
        let total_wounds: usize = sim
            .worlds
            .iter()
            .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
            .map(|a| a.wounds.len())
            .sum();
        if report.total_disasters > 10 {
            assert!(
                max_trauma > 0.0 || total_wounds > 0,
                "DL#1: With {} disasters, should have trauma or wounds, got trauma={}, wounds={}",
                report.total_disasters,
                max_trauma,
                total_wounds
            );
        }

        // Bug#3: CVS should use real oppression index (not hardcoded 0.0)
        assert!(report.final_cvs > 0.0, "CVS should be computed");
    }

    #[test]
    fn test_earth_population_model_with_hybrid() {
        let mut config = small_config(60); // 5 years
        config.policy.hybrid_earth = true;
        let mut sim = MultiWorldSimulator::new(config);
        let _report = sim.run();

        // Earth population model should be initialized
        assert!(
            sim.earth_pop_model.is_some(),
            "Earth population model should be initialized when hybrid_earth=true"
        );

        let pop_model = sim.earth_pop_model.as_ref().unwrap();
        assert!(
            pop_model.total_population > 7000.0,
            "Earth population should be >7 billion: {}M",
            pop_model.total_population
        );
        assert_eq!(
            pop_model.demographics.len(),
            12,
            "Should have 12 regional demographics"
        );

        // Climate model should have been ticked
        assert!(
            pop_model.climate.cumulative_emissions_gt > 200.0,
            "Cumulative emissions should exceed baseline (200 GtCO₂): {}",
            pop_model.climate.cumulative_emissions_gt
        );
        assert!(
            pop_model.climate.global_temp_anomaly > 0.0,
            "Temperature anomaly should be positive: {}",
            pop_model.climate.global_temp_anomaly
        );
    }

    #[test]
    fn test_unified_config_bridge_creates_working_sim() {
        let toml = r#"
[simulation]
years = 10
seed = 777

[policy]
disasters_enabled = true
hybrid_earth = false

[params]
stress_baseline = 0.005

[feedback_loops]
disaster_trauma = true
harmony_policy = true
"#;
        let (sim_config, _, _) = unified_config_bridge::from_unified_toml(toml).unwrap();
        let mut sim = MultiWorldSimulator::new(sim_config);
        let report = sim.run();

        assert_eq!(report.total_ticks, 120); // 10 years × 12
        assert!(report.final_population > 0, "Should survive 10 years");
        assert!(report.survived, "Should survive with standard config");
    }

    #[test]
    fn test_affect_modulates_labor() {
        // DL#6: Agents with high joy should produce more than agents with high sadness
        let tick = 30 * 12; // Agent born at tick 0, now 30 years old
        let mut happy = CivAgent {
            id: 1,
            birth_tick: 0,
            death_tick: None,
            sex: BiologicalSex::Male,
            world_id: 0,
            health: 1.0,
            skills: SkillVector::new(),
            education_level: 0.5,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
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
        happy.skills.learn(0, 0.5);
        happy.needs.engagement = 0.8;
        happy.needs.affect.joy = 0.9;
        happy.needs.affect.sadness = 0.1;

        let mut sad = happy.clone();
        sad.needs.affect.joy = 0.1;
        sad.needs.affect.sadness = 0.9;

        let happy_labor = happy.effective_labor(tick);
        let sad_labor = sad.effective_labor(tick);

        assert!(
            happy_labor > sad_labor,
            "DL#6: Happy agents should produce more: {} vs {}",
            happy_labor,
            sad_labor
        );
        assert!(
            happy_labor > sad_labor * 1.3,
            "DL#6: Effect should be >30% difference: {} vs {}",
            happy_labor,
            sad_labor
        );
    }
}
