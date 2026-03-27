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
pub mod config;
pub mod consciousness;
pub mod disasters;
pub mod economy;
pub mod education;
pub mod epoch;
pub mod events;
pub mod factions;
pub mod governance;
pub mod harmony;
pub mod interworld;
pub mod knowledge;
pub mod needs;
pub mod observables;
pub mod population;
pub mod report;
pub mod stochastic;
pub mod world;

use config::{EpochId, SimulationConfig};
use epoch::{EpochManager, EpochSnapshot};
use events::{CivEvent, CivEventType};
use population::PopulationEngine;
use report::CivilizationReport;
use stochastic::StochasticEngine;
use world::World;

use agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
use education::EducationEngine;
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

/// Top-level multi-world civilization simulator.
pub struct MultiWorldSimulator {
    pub config: SimulationConfig,
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
}

impl MultiWorldSimulator {
    /// Create a new simulator from configuration.
    pub fn new(config: SimulationConfig) -> Self {
        let rng = StochasticEngine::new(config.seed);
        Self {
            config,
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
        }
    }

    /// Initialize worlds that should exist at tick 0.
    fn initialize_worlds(&mut self) {
        let seeds: Vec<_> = self
            .config
            .initial_worlds
            .iter()
            .filter(|w| w.founding_tick == 0)
            .cloned()
            .collect();

        for (idx, seed) in seeds.iter().enumerate() {
            let resources = match seed.location.as_str() {
                "Earth" => WorldResources::earth_default(),
                _ => WorldResources::lunar_default(),
            };
            let culture = match seed.location.as_str() {
                "Earth" => CulturalProfile::earth_default(),
                _ => CulturalProfile::pioneer_default(),
            };

            let mut world = World {
                id: idx as u32,
                name: seed.name.clone(),
                location: seed.location.clone(),
                founded_tick: 0,
                parent_world_id: if seed.location == "Earth" {
                    None
                } else {
                    Some(0)
                },
                agents: Vec::new(),
                next_agent_id: 0,
                resources,
                culture,
                infrastructure_level: if seed.location == "Earth" {
                    0.9
                } else {
                    0.2
                },
                max_population: if seed.location == "Earth" {
                    2_000 // Representative sample cap — Earth is supply depot
                } else {
                    10_000
                },
                habitable_area_m2: if seed.location == "Earth" {
                    1e12
                } else {
                    50_000.0
                },
                founding_harmony_emphasis: [0.125; 8],
                epidemics: Vec::new(),
                knowledge: knowledge::WorldKnowledge::new(),
                economy: economy::WorldEconomy::new(),
                harmony: harmony::HarmonyTracker::new(),
                governance: governance::WorldGovernance::new(),
            };

            // Spawn initial population as adults (age 25-45)
            for _ in 0..seed.initial_population {
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
                    education_level: self.rng.next_f64() * 0.5,
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
                };
                world.next_agent_id += 1;
                world.agents.push(agent);
            }

            self.events.push(CivEvent::new(
                0,
                Some(world.id),
                CivEventType::WorldFounded,
                format!("{} founded with {} colonists", world.name, seed.initial_population),
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
    fn found_colony(
        &mut self,
        name: &str,
        location: &str,
        population: usize,
        _resource_mult: f64,
    ) {
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
            _ => WorldResources::lunar_default(),
        };

        let mut world = World {
            id: world_id,
            name: name.into(),
            location: location.into(),
            founded_tick: self.current_tick,
            parent_world_id: Some(0),
            agents: Vec::new(),
            next_agent_id: 0,
            resources,
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.1,
            max_population: 5_000,
            habitable_area_m2: 30_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: knowledge::WorldKnowledge::new(),
            economy: economy::WorldEconomy::new(),
            harmony: harmony::HarmonyTracker::new(),
            governance: governance::WorldGovernance::new(),
        };

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
                education_level: self.rng.next_f64() * 0.6,
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
            };
            world.next_agent_id += 1;
            world.agents.push(agent);
        }

        self.events.push(CivEvent::new(
            self.current_tick,
            Some(world_id),
            CivEventType::WorldFounded,
            format!("{} founded at tick {} with {} settlers", name, self.current_tick, population),
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
        let qualifies = self
            .worlds
            .iter()
            .any(|w| {
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
            let governance_stability = (world.infrastructure_level * 0.5
                + world.mean_phi() * 0.3
                + (1.0 - burnout_frac) * 0.2)
                .clamp(0.0, 1.0);

            // Compute worker ratio for overwork stress
            let working = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.life_stage(self.current_tick).can_work())
                .count();
            let worker_ratio = working as f64 / world.population().max(1) as f64;

            let (events, summary) = PsychNeedsEngine::tick_needs(
                &mut world,
                self.current_tick,
                epoch,
                care_workers,
                mean_tech,
                governance_stability,
                worker_ratio,
                self.config.policy.care_effectiveness,
                self.config.policy.deep_space_isolation_mult,
                &mut self.rng,
            );

            self.needs_summaries.push(summary);
            self.events.extend(events);
            self.worlds[i] = world;
        }
    }

    fn tick_education(&mut self) {
        if !self.config.policy.education_enabled {
            return; // A/B comparison: skip education tick entirely
        }
        let world_count = self.worlds.len();
        for i in 0..world_count {
            let mut world = std::mem::take(&mut self.worlds[i]);
            let (events, _summary) =
                EducationEngine::tick(&mut world, self.current_tick, &mut self.rng);
            self.events.extend(events);
            self.worlds[i] = world;
        }
    }

    fn tick_genetics(&mut self) {
        for world in &self.worlds {
            let div = PopulationEngine::genetic_diversity_index(world, self.current_tick);
            if div < 0.5 {
                self.events.push(CivEvent::new(
                    self.current_tick,
                    Some(world.id),
                    CivEventType::GeneticAlert,
                    format!("{}: genetic diversity critical ({div:.3})", world.name),
                ));
            }
        }
    }

    fn tick_economy(&mut self) {
        // Phase 4: Resource production/consumption + Cobb-Douglas economy.
        let tick = self.current_tick;
        for world in &mut self.worlds {
            let pop = world.population() as f64;

            // Raw resource arithmetic (life support)
            for name in &["food", "water", "energy", "materials", "oxygen"] {
                if let Some(stock) = world.resources.get_mut(name) {
                    let production = stock.production_rate * world.infrastructure_level;
                    let consumption = stock.consumption_rate * (pop / 100.0).max(0.1);
                    stock.current = (stock.current + production - consumption)
                        .clamp(0.0, stock.capacity);
                }
            }

            // Cobb-Douglas economy: assign workers, produce, demurrage, Gini
            world.economy.assign_workers(&world.agents, tick);

            // Feed technology multipliers from knowledge system
            let tech = world.knowledge.mean_tech_level();
            for i in 0..8 {
                world.economy.technology_multiplier[i] = tech;
            }
            world.economy.infrastructure_capital =
                (world.infrastructure_level * 50.0).max(1.0); // map 0-1 to 0-50

            world.economy.tick_production();
            world.economy.tick_demurrage();
            world.economy.compute_gini(&world.agents);
            world.economy.self_sufficiency = world.resources.self_sufficiency();

            // Investment: higher-consciousness worlds invest more
            let mean_phi = world.mean_phi();
            let invest_rate = 0.1 + mean_phi * 0.2; // 0.1-0.3 based on consciousness
            world.economy.invest(invest_rate);

            // Slowly improve infrastructure (capped at 1.0).
            // Without consciousness gating, poor governance decisions cause
            // ~20% resource waste (wrong priorities, unchecked extractive behavior).
            if !self.config.policy.consciousness_gating_enabled {
                for name in &["food", "water", "energy"] {
                    if let Some(stock) = world.resources.get_mut(name) {
                        stock.current *= 0.998; // 0.2% waste per tick = ~2.4% annual
                    }
                }
            }
            world.infrastructure_level =
                (world.infrastructure_level + 0.001).min(1.0);
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
                                transfers.push((
                                    self.worlds[i].id,
                                    self.worlds[j].id,
                                    (ss_i - ss_j) * 10.0,
                                ));
                            }
                        }
                    }
                }
            }

            for (from_id, to_id, amount) in transfers {
                if let Some(from_world) = self.worlds.iter_mut().find(|w| w.id == from_id) {
                    if let Some(food) = from_world.resources.get_mut("food") {
                        food.current = (food.current - amount).max(0.0);
                    }
                }
                if let Some(to_world) = self.worlds.iter_mut().find(|w| w.id == to_id) {
                    if let Some(food) = to_world.resources.get_mut("food") {
                        food.current = (food.current + amount).min(food.capacity);
                    }
                }
            }
        }

        // Inter-world migration: every 6 ticks (biannual), move a few adults from
        // Earth to off-world colonies. This maintains genetic diversity and carries
        // fresh social bonds into isolated populations.
        if self.config.policy.migration_enabled
            && self.current_tick % 6 == 0
            && self.worlds.len() >= 2
        {
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
                                    && a.life_stage(self.current_tick) == agent::LifeStage::Adult
                            })
                            .take(n_migrants)
                            .map(|a| a.id)
                            .collect();

                        if !migrant_ids.is_empty() {
                            migration_plans.push((migrant_ids, dest_idx, dest_id, dest_name));
                        }
                    }

                    for (migrant_ids, dest_idx, dest_id, dest_name) in migration_plans {
                        let mut migrants: Vec<CivAgent> = Vec::new();
                        for mid in &migrant_ids {
                            if let Some(agent) = self.worlds[ei]
                                .agents
                                .iter_mut()
                                .find(|a| a.id == *mid)
                            {
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
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
            let knowledge_events = knowledge.tick_knowledge(&world, tick, &mut self.rng);
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
                self.epoch_manager.record_milestone("constitution", self.current_tick);
                self.constitution_granted = true;
                self.events.push(CivEvent::new(
                    self.current_tick,
                    None,
                    CivEventType::ConstitutionalAmendment,
                    "Colony constitution ratified",
                ));
            }
        }

        // Phase 6b: Per-world consciousness-gated governance with anti-tyranny invariants
        {
            let tick = self.current_tick;
            let amendment_enabled = self.config.policy.amendment_enabled;
            let epoch = self.current_epoch as u32;
            let rng = &mut self.rng;
            let mut all_gov_events = Vec::new();
            for world in &mut self.worlds {
                if world.population() == 0 {
                    continue;
                }
                let pop = world.population();
                world.governance.evolve_authority(epoch, pop);
                let mut gov = std::mem::take(&mut world.governance);
                let gov_events =
                    gov.tick_governance_with_policy(world, tick, rng, amendment_enabled);
                world.governance = gov;
                all_gov_events.extend(gov_events);
            }
            self.events.extend(all_gov_events);
        }
    }

    fn tick_consciousness(&mut self) {
        // Phase 7: Gradual consciousness growth for agents.
        let tick = self.current_tick;
        for world in &mut self.worlds {
            let mean_edu: f64 = {
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
            };

            for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                let stage = agent.life_stage(tick);
                let growth_rate = match stage {
                    agent::LifeStage::Child => 0.001,
                    agent::LifeStage::Youth => 0.003,
                    agent::LifeStage::Adult => 0.002,
                    agent::LifeStage::Elder => 0.001,
                };

                // Consciousness decay — earned, not permanent.
                // Children and youth are still developing, so no decay for them.
                // Allostatic load amplifies decay: stressed agents lose consciousness faster.
                let base_decay = if stage == agent::LifeStage::Adult || stage == agent::LifeStage::Elder {
                    0.0015
                } else {
                    0.0
                };
                let decay = base_decay * (1.0 + agent.needs.allostatic_load);
                let c = &mut agent.consciousness;
                c.level = (c.level - decay).max(0.0);
                c.meta_awareness = (c.meta_awareness - decay).max(0.0);
                c.coherence = (c.coherence - decay).max(0.0);
                c.care_activation = (c.care_activation - decay).max(0.0);
                c.harmonic_alignment = (c.harmonic_alignment - decay).max(0.0);
                c.epistemic_confidence = (c.epistemic_confidence - decay).max(0.0);

                // Education, community, and pharmaceutical therapeutics amplify consciousness.
                // Psychedelic therapeutics (psilocybin sessions) boost neuroplasticity,
                // increasing care_activation and meta_awareness growth by ~30%.
                // This models the "consciousness maintenance" effect described in
                // the astropharmacy resource catalog.
                let pharma_boost = if world.infrastructure_level > 0.3 {
                    self.config.policy.pharma_boost
                } else {
                    0.0
                };
                // Without consciousness gating, there's no institutional incentive
                // for consciousness development — growth rate drops by 50%.
                let gating_factor = if self.config.policy.consciousness_gating_enabled {
                    1.0
                } else {
                    0.5
                };
                let amplifier = (1.0 + mean_edu * 0.5 + pharma_boost) * gating_factor;
                // Burnout caps consciousness growth (symthaea-psych-bench pattern).
                let burnout_penalty = if agent.needs.is_burnout() { 0.35 } else { 1.0 };
                let amplifier = amplifier * burnout_penalty;
                // Social satiation boosts care_activation growth.
                if agent.needs.social_satiation > 0.5 {
                    c.care_activation = (c.care_activation + 0.0005).min(1.0);
                }
                // Diminishing returns: growth slows as dimension approaches cap.
                // effective_growth = growth_rate * (1 - dim/cap) to prevent saturation.
                let cap = 0.85;
                let dr = |dim: f64, rate: f64| -> f64 {
                    let headroom = (1.0 - dim / cap).max(0.0);
                    rate * amplifier * headroom
                };
                c.level = (c.level + dr(c.level, growth_rate)).min(cap);
                c.meta_awareness = (c.meta_awareness + dr(c.meta_awareness, growth_rate * 0.8)).min(cap);
                c.coherence = (c.coherence + dr(c.coherence, growth_rate * 0.6)).min(cap);
                c.care_activation = (c.care_activation + dr(c.care_activation, growth_rate * 0.7)).min(cap);
                c.harmonic_alignment =
                    (c.harmonic_alignment + dr(c.harmonic_alignment, growth_rate * 0.5)).min(cap);
                c.epistemic_confidence =
                    (c.epistemic_confidence + dr(c.epistemic_confidence, growth_rate * 0.4)).min(cap);

                // Overall phi ceiling: moral humility (no agent achieves perfection)
                // If phi exceeds 0.95, attenuate all dimensions proportionally.
                let phi = c.phi();
                if phi > 0.95 {
                    let scale = 0.95 / phi;
                    c.level *= scale;
                    c.meta_awareness *= scale;
                    c.coherence *= scale;
                    c.care_activation *= scale;
                    c.harmonic_alignment *= scale;
                    c.epistemic_confidence *= scale;
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
                art_per_capita: world.economy.sector_output[6]
                    / world.population().max(1) as f64,
                trade_connections: if world_count > 1 { (world_count - 1) as u32 } else { 0 },
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

        for (effects, world_id, event) in disaster_results {
            // Apply effects to targeted world(s)
            let target_ids: Vec<u32> = match world_id {
                Some(id) => vec![id],
                None => self.worlds.iter().map(|w| w.id).collect(),
            };

            for &wid in &target_ids {
                if let Some(world) = self.worlds.iter_mut().find(|w| w.id == wid) {
                    // Population loss: kill a fraction of living agents (random selection)
                    if effects.population_loss_fraction > 0.0 {
                        let living_count = world.population();
                        let to_kill =
                            (living_count as f64 * effects.population_loss_fraction).round()
                                as usize;
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
                            for &kill_id in living_ids.iter().take(to_kill) {
                                if let Some(agent) =
                                    world.agents.iter_mut().find(|a| a.id == kill_id)
                                {
                                    agent.death_tick = Some(self.current_tick);
                                }
                            }
                        }
                    }

                    // Infrastructure damage
                    if effects.infrastructure_damage > 0.0 {
                        world.infrastructure_level = (world.infrastructure_level
                            - effects.infrastructure_damage)
                            .max(0.0);
                    }

                    // Resource production penalty: reduce production_rate temporarily
                    // by multiplying by (1 - penalty). We apply to current stocks as proxy
                    // since production_rate is static and the penalty is per-tick from active disasters.
                    if effects.resource_production_penalty > 0.0 {
                        let factor = 1.0 - effects.resource_production_penalty;
                        for name in &["food", "water", "energy", "materials", "oxygen"] {
                            if let Some(stock) = world.resources.get_mut(name) {
                                // Reduce production output this tick
                                stock.current =
                                    (stock.current * factor).max(0.0);
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

                    // Consciousness shock: reduce all living agents' consciousness level
                    if effects.consciousness_shock > 0.0 {
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.consciousness.level = (agent.consciousness.level
                                - effects.consciousness_shock)
                                .max(0.0);
                        }
                    }

                    // Allostatic load increase
                    if effects.allostatic_load_increase > 0.0 {
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.needs.allostatic_load = (agent.needs.allostatic_load
                                + effects.allostatic_load_increase)
                                .min(1.0);
                        }
                    }

                    // Morale impact: affects engagement
                    if effects.morale_impact != 0.0 {
                        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
                            agent.needs.engagement = (agent.needs.engagement
                                + effects.morale_impact)
                                .clamp(0.0, 1.0);
                        }
                    }
                }
            }

            // Push the event
            self.events.push(event);
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
        snap.elite_persistence = self.worlds.iter()
            .map(|w| observables::elite_persistence_index(w, self.current_tick))
            .fold(0.0f64, f64::max);

        // Innovation stagnation: max across worlds
        snap.innovation_stagnation = self.worlds.iter()
            .map(|w| observables::innovation_stagnation_index(&w.knowledge.tech_history, self.current_tick, 120))
            .fold(0.0f64, f64::max);

        // Inter-world divergence
        snap.inter_world_divergence = observables::inter_world_divergence(&self.worlds);

        // Consciousness Gini: compute across all living agents
        let all_phis: Vec<f64> = self.worlds.iter()
            .flat_map(|w| w.agents.iter())
            .filter(|a| a.is_alive())
            .map(|a| a.consciousness.phi())
            .collect();
        snap.consciousness_gini = observables::consciousness_gini(&all_phis);

        // Phi trend: use combined phi_history from first world (proxy)
        // Or use mean phi from snapshots
        let phi_values: Vec<f64> = self.epoch_snapshots.iter()
            .map(|s| s.mean_phi)
            .collect();
        snap.phi_trend = format!("{}", observables::classify_phi_trend(&phi_values, 120));

        // Recovery count
        snap.recovery_count = observables::recovery_count(&self.epoch_snapshots);

        // Mean trauma across all worlds
        let total_trauma: f64 = self.worlds.iter()
            .map(|w| observables::trauma_level(w) * w.population() as f64)
            .sum();
        let total_pop: f64 = self.worlds.iter().map(|w| w.population() as f64).sum();
        snap.trauma_level = if total_pop > 0.0 { total_trauma / total_pop } else { 0.0 };

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
        let checkpoint_events = self.epoch_manager.evaluate_tick(self.current_tick, &self.worlds);
        self.events.extend(checkpoint_events);

        // Check for epoch transitions
        let total_pop = self.total_population();
        let num_worlds = self.worlds.len();
        let ss = self.mean_self_sufficiency();

        if let Some(new_epoch) = self.epoch_manager.check_epoch_transition(
            self.current_tick,
            total_pop,
            num_worlds,
            ss,
        ) {
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
        let sum: f64 = self.worlds.iter().map(|w| w.resources.self_sufficiency()).sum();
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
        self.initialize_worlds();

        while self.current_tick < self.config.total_ticks {
            // Check for deferred world founding (from config)
            self.check_deferred_worlds();

            // Check for Mars fission (dynamic world founding)
            self.check_mars_fission();

            // Grant trade milestone when 2+ non-Earth worlds with population exist
            // and both have self_sufficiency > 0.3. Trade between Earth and a tiny
            // colony is trivial — real trade requires established colonies.
            if !self.trade_granted {
                let viable_worlds: Vec<&World> = self
                    .worlds
                    .iter()
                    .filter(|w| {
                        w.location != "Earth"
                            && w.population() > 0
                            && w.resources.self_sufficiency() > 0.3
                    })
                    .collect();
                // Need at least 2 off-Earth colonies, OR 1 off-Earth with SS>0.3
                // plus Earth, but only after colony is established (12+ ticks old)
                let has_established_colony = viable_worlds.iter().any(|w| {
                    self.current_tick.saturating_sub(w.founded_tick) >= 12
                });
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

            // Phase 1: Demographics (pair-bonding, births, deaths)
            let mut phase1_events = Vec::new();
            let world_count = self.worlds.len();
            for i in 0..world_count {
                let mut world = std::mem::take(&mut self.worlds[i]);
                PopulationEngine::tick_pair_bonding(
                    &mut world,
                    &mut self.rng,
                    self.current_tick,
                    self.config.policy.pair_bond_rate,
                );
                let dem_events =
                    PopulationEngine::tick_demographics(&mut world, &mut self.rng, self.current_tick);
                phase1_events.extend(dem_events);
                self.worlds[i] = world;
            }
            self.track_milestones(&phase1_events);
            self.events.extend(phase1_events);

            // Phase 2: Genetics
            self.tick_genetics();

            // Phase 3: Psychological needs
            self.tick_psychological_needs();

            // Phase 3.5: Education (peer-to-peer learning, TEND rewards)
            self.tick_education();

            // Phase 4: Economy
            self.tick_economy();

            // Phase 5: Inter-world
            self.tick_interworld();

            // Phase 6: Knowledge
            self.tick_knowledge();

            // Phase 7: Governance
            self.tick_governance();

            // Phase 8: Consciousness
            self.tick_consciousness();

            // Phase 8.5: Factions (after consciousness, before harmony)
            {
                let policy = self.config.policy.clone();
                let faction_events = self.faction_engine.tick_factions(
                    &mut self.worlds,
                    self.current_tick,
                    &mut self.rng,
                    &policy,
                );
                self.events.extend(faction_events);
            }

            // Phase 9: Harmony scoring
            self.tick_harmony_scoring();

            // Phase 9.5: Disasters (before emergencies — disasters can trigger emergencies)
            self.tick_disasters();

            // Phase 10: Emergencies
            self.tick_emergencies();

            // Phase 11: Epoch evaluation
            self.tick_epoch_evaluation();

            // Dead agent compaction: every 600 ticks, remove agents dead for 1200+ ticks
            if self.current_tick % 600 == 0 && self.current_tick > 1200 {
                let cutoff = self.current_tick - 1200;
                for world in &mut self.worlds {
                    world.agents.retain(|a| {
                        a.death_tick.map_or(true, |dt| dt >= cutoff)
                    });
                }
            }

            // Trauma decay for high-trauma agents: increased faction recruitment
            // (handled in factions.rs tick_recruitment via agent.trauma_level)

            self.current_tick += 1;
        }

        self.build_final_report()
    }

    /// Construct the final CivilizationReport from accumulated simulation state.
    fn build_final_report(&self) -> CivilizationReport {
        // Genetic diversity: focus on OFF-EARTH worlds only. Earth's 10K
        // population masks the bottleneck that matters — the colony's genetics.
        let off_earth: Vec<_> = self.worlds.iter().filter(|w| w.location != "Earth").collect();
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
            harmony::HarmonyTracker::civilization_harmony(
                &trackers,
            )
        };
        let love_coherence = if self.worlds.is_empty() {
            0.0
        } else {
            let trackers: Vec<_> = self.worlds.iter().map(|w| w.harmony.clone()).collect();
            harmony::HarmonyTracker::civilization_love_coherence(&trackers)
        };

        let harmony_mean: f64 = harmony_scores.iter().sum::<f64>() / 8.0;
        let max_oppression = 0.0; // no oppression subsystem yet

        let final_cvs = EpochManager::compute_cvs(
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
        let (total_load, total_engagement, agent_count) = self.worlds.iter().fold(
            (0.0f64, 0.0f64, 0usize),
            |(load, eng, count), w| {
                let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                let n = living.len();
                let l: f64 = living.iter().map(|a| a.needs.allostatic_load).sum();
                let e: f64 = living.iter().map(|a| a.needs.engagement).sum();
                (load + l, eng + e, count + n)
            },
        );
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
        report.max_elite_persistence = snapshots.iter()
            .map(|s| s.elite_persistence)
            .fold(0.0f64, f64::max);
        report.max_innovation_stagnation = snapshots.iter()
            .map(|s| s.innovation_stagnation)
            .fold(0.0f64, f64::max);
        report.phi_trend_at_end = snapshots.last()
            .map(|s| s.phi_trend.clone())
            .unwrap_or_else(|| "Unknown".into());
        report.max_trauma = snapshots.iter()
            .map(|s| s.trauma_level)
            .fold(0.0f64, f64::max);

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
        assert!(report.final_population > 0, "Population should survive 150 years");
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
        assert!(report.final_population > 0, "Should have surviving population");
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
            r1.total_events,
            r2.total_events,
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
        let mars_founded = report
            .worlds_founded;
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

        assert!(
            report.final_collective_phi > 0.1,
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
        eprintln!("Population:              {:>10}    {:>10}", report_a.final_population, report_b.final_population);
        eprintln!("Survived:                {:>10}    {:>10}", report_a.survived, report_b.survived);
        eprintln!("CVS (viability):         {:>10.3}    {:>10.3}", report_a.final_cvs, report_b.final_cvs);
        eprintln!("Mean allostatic load:    {:>10.3}    {:>10.3}", report_a.final_mean_allostatic_load, report_b.final_mean_allostatic_load);
        eprintln!("Collective Phi:          {:>10.3}    {:>10.3}", report_a.final_collective_phi, report_b.final_collective_phi);
        eprintln!("Mean engagement:         {:>10.3}    {:>10.3}", report_a.final_mean_engagement, report_b.final_mean_engagement);
        eprintln!("Breakthroughs:           {:>10}    {:>10}", report_a.breakthroughs, report_b.breakthroughs);
        eprintln!("Checkpoints passed:      {:>10}    {:>10}", report_a.checkpoints_passed, report_b.checkpoints_passed);

        // Count teaching events in scenario A
        let teaching_events_a = sim_a.events.iter()
            .filter(|e| matches!(e.event_type, CivEventType::TeachingInteraction))
            .count();
        let crisis_events_a = sim_a.events.iter()
            .filter(|e| matches!(e.event_type, CivEventType::SkillCrisis))
            .count();
        let crisis_events_b = sim_b.events.iter()
            .filter(|e| matches!(e.event_type, CivEventType::SkillCrisis))
            .count();

        eprintln!("Teaching events (A):     {:>10}", teaching_events_a);
        eprintln!("Skill crises (A):        {:>10}    {:>10}", crisis_events_a, crisis_events_b);
        eprintln!("================================================\n");

        // === Assertions: the education guild should help ===
        // Both should survive 50 years (this is a baseline sanity check)
        assert!(report_a.final_population > 0, "Guild world should survive");
        assert!(report_b.final_population > 0, "Control world should survive 50 years");

        // The guild world should have peer teaching events
        assert!(
            teaching_events_a > 0,
            "Education guild should produce teaching interactions"
        );
    }
}
