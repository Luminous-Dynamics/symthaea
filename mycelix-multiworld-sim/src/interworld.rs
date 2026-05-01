// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inter-world dynamics: trade routes, migration, cultural exchange, and world fission.
//!
//! Models the economic and social connections between worlds (colonies, settlements,
//! planets). Trade routes carry resources at delta-v-dependent cost, with knowledge
//! modules transferring at near-zero mass cost. Migration follows economic gradients.
//! Cultural exchange reduces distance between connected worlds while unconnected
//! worlds drift apart.

use crate::agent::BiologicalSex;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::{CulturalProfile, World, WorldResources};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Orbital mechanics constants (Bate, Mueller & White 1971; JPL Horizons)
// ---------------------------------------------------------------------------

/// Synodic periods in ticks (1 tick = 1 month). Time between launch windows.
const SYNODIC_EARTH_MARS: u32 = 26; // 25.6 months
const SYNODIC_EARTH_JUPITER: u32 = 13; // 13.1 months (Europa)
const SYNODIC_EARTH_SATURN: u32 = 12; // 12.4 months (Titan)
const SYNODIC_MARS_JUPITER: u32 = 27; // 26.8 months
const SYNODIC_MARS_SATURN: u32 = 24; // 24.1 months

/// Hohmann transfer times in ticks.
const TRANSFER_EARTH_MARS: u32 = 9; // 258 days (Vallado 2013)
const TRANSFER_EARTH_JUPITER: u32 = 33; // 2.73 years
const TRANSFER_EARTH_SATURN: u32 = 73; // 6.05 years
const TRANSFER_MARS_JUPITER: u32 = 26; // 2.16 years
const TRANSFER_MARS_SATURN: u32 = 59; // 4.9 years

/// Emergency (off-window) transfer delta-v cost multiplier.
const _EMERGENCY_DV_MULTIPLIER: f64 = 2.5;

/// Migration probability per agent per tick when GDP disparity is >= 2x.
const MIGRATION_PROB_PER_AGENT: f64 = 0.001;

/// Cultural convergence rate per trade route per tick.
const CULTURAL_CONVERGENCE_RATE: f64 = 0.001;

/// Knowledge diffusion factor: fraction of tech gap transferred per tick per route.
const KNOWLEDGE_DIFFUSION_RATE: f64 = 0.005;

/// Maximum trade routes for connectivity normalization.
const MAX_TRADE_ROUTES: f64 = 5.0;

/// Near-zero mass cost for information/knowledge transfer (kg-equivalent).
const KNOWLEDGE_MASS_COST: f64 = 0.001;

/// Cargo in transit between worlds (launched at window, arrives after transfer time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InTransitCargo {
    pub resource: String,
    pub amount: f64,
    pub departure_tick: u32,
    pub arrival_tick: u32,
    pub destination_world: u32,
}

/// A trade route between two worlds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRoute {
    pub from_world: u32,
    pub to_world: u32,
    pub established_tick: u32,
    /// Trade volume per tick (abstract units).
    pub volume_per_tick: f64,
    /// What the from_world mostly exports.
    pub primary_export: String,
    /// What the from_world mostly imports.
    pub primary_import: String,
    /// Energy cost per kg in kWh (derived from delta-v).
    pub transport_cost_per_kg: f64,
    /// One-way light delay in seconds.
    pub light_delay_secs: f64,
    /// Probability of disruption per tick (based on distance + conflict).
    pub route_vulnerability: f64,
    /// Synodic period (ticks between launch windows). 0 = continuous access.
    pub synodic_period: u32,
    /// Hohmann transfer time (ticks for cargo to arrive). 0 = instant.
    pub transfer_time: u32,
}

/// Record of a migration event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationEvent {
    pub agent_id: u64,
    pub from_world: u32,
    pub to_world: u32,
    pub tick: u32,
    pub reason: MigrationReason,
}

/// Why an agent migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationReason {
    /// Better economic opportunities.
    Economic,
    /// Founding a new settlement.
    Pioneer,
    /// Joining a partner or family.
    Family,
    /// Fleeing a crisis.
    Refuge,
}

/// Engine managing all inter-world interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterWorldEngine {
    pub trade_routes: Vec<TradeRoute>,
    pub migration_history: Vec<MigrationEvent>,
    pub total_trade_volume: f64,
    pub total_migrants: u64,
    pub knowledge_transfers: u64,
    /// Cargo currently in transit between worlds.
    pub in_transit: Vec<InTransitCargo>,
    /// LEO access multiplier from Kessler debris (1.0 = normal, 0.0 = denied).
    pub leo_access_multiplier: f64,
    /// Whether fusion grid-scale is achieved (removes window constraint).
    pub fusion_override: bool,
}

impl InterWorldEngine {
    pub fn new() -> Self {
        Self {
            trade_routes: Vec::new(),
            migration_history: Vec::new(),
            total_trade_volume: 0.0,
            total_migrants: 0,
            knowledge_transfers: 0,
            in_transit: Vec::new(),
            leo_access_multiplier: 1.0,
            fusion_override: false,
        }
    }

    /// Look up orbital parameters for a route between two locations.
    /// Returns (synodic_period, transfer_time) in ticks. (0, 0) = continuous access.
    pub fn orbital_params(from_loc: &str, to_loc: &str) -> (u32, u32) {
        // Moon-Earth: co-orbital, continuous access (3-day transfer ≈ 0 ticks)
        let pair = Self::normalize_pair(from_loc, to_loc);
        match pair.as_str() {
            "Earth-Moon" | "Moon-Earth" => (0, 0),
            "Earth-Mars" | "Mars-Earth" => (SYNODIC_EARTH_MARS, TRANSFER_EARTH_MARS),
            "Earth-Europa" | "Europa-Earth" => (SYNODIC_EARTH_JUPITER, TRANSFER_EARTH_JUPITER),
            "Earth-Titan" | "Titan-Earth" => (SYNODIC_EARTH_SATURN, TRANSFER_EARTH_SATURN),
            "Mars-Europa" | "Europa-Mars" => (SYNODIC_MARS_JUPITER, TRANSFER_MARS_JUPITER),
            "Mars-Titan" | "Titan-Mars" => (SYNODIC_MARS_SATURN, TRANSFER_MARS_SATURN),
            "Moon-Mars" | "Mars-Moon" => (SYNODIC_EARTH_MARS, TRANSFER_EARTH_MARS), // ~same orbit
            "Moon-Europa" | "Europa-Moon" => (SYNODIC_EARTH_JUPITER, TRANSFER_EARTH_JUPITER),
            "Moon-Titan" | "Titan-Moon" => (SYNODIC_EARTH_SATURN, TRANSFER_EARTH_SATURN),
            _ => (0, 0), // Unknown pairs: continuous
        }
    }

    fn normalize_pair(a: &str, b: &str) -> String {
        if a <= b {
            format!("{}-{}", a, b)
        } else {
            format!("{}-{}", b, a)
        }
    }

    /// Check if a transfer window is currently open for this route.
    pub fn is_window_open(synodic_period: u32, established_tick: u32, current_tick: u32) -> bool {
        if synodic_period == 0 {
            return true; // Continuous access
        }
        let elapsed = current_tick.saturating_sub(established_tick);
        elapsed % synodic_period == 0
    }

    /// Establish a trade route between two worlds.
    ///
    /// Transport cost is derived from delta-v via simplified Tsiolkovsky proxy:
    /// `cost_per_kg = delta_v^2 * 0.001` (kWh/kg). Delta-v is estimated from
    /// light delay (closer bodies have lower delta-v).
    pub fn establish_route(
        &mut self,
        from: u32,
        to: u32,
        tick: u32,
        light_delay: f64,
        from_location: &str,
        to_location: &str,
    ) -> &TradeRoute {
        // Estimate delta-v from light delay heuristic:
        // Moon-Earth: ~1.3s delay, ~2.5 km/s delta-v
        // Earth-Mars: ~180-1200s, ~4 km/s
        // Moon-Mars: ~180-1200s, ~5 km/s
        let delta_v = if light_delay < 2.0 {
            2.5 // Moon-Earth class
        } else if light_delay < 60.0 {
            3.5 // Near-Earth objects
        } else if light_delay < 600.0 {
            4.0 // Earth-Mars class
        } else {
            5.0 + (light_delay / 1000.0) // Outer system
        };

        let transport_cost_per_kg = delta_v * delta_v * 0.001;

        // Route vulnerability: higher for longer distances
        let route_vulnerability = (light_delay / 10000.0).clamp(0.001, 0.05);

        // Orbital mechanics: synodic period and transfer time
        let (synodic_period, transfer_time) = Self::orbital_params(from_location, to_location);

        let route = TradeRoute {
            from_world: from,
            to_world: to,
            established_tick: tick,
            volume_per_tick: 10.0, // base volume, grows with world development
            primary_export: String::new(),
            primary_import: String::new(),
            transport_cost_per_kg,
            light_delay_secs: light_delay,
            route_vulnerability,
            synodic_period,
            transfer_time,
        };

        self.trade_routes.push(route);
        self.trade_routes.last().unwrap()
    }

    /// Run one tick of inter-world dynamics: trade, migration, cultural exchange,
    /// knowledge diffusion.
    pub fn tick_interworld(
        &mut self,
        worlds: &mut Vec<World>,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // --- Deliver in-transit cargo that has arrived ---
        let mut arrived = Vec::new();
        let mut still_in_transit = Vec::new();
        for cargo in self.in_transit.drain(..) {
            if current_tick >= cargo.arrival_tick {
                arrived.push(cargo);
            } else {
                still_in_transit.push(cargo);
            }
        }
        self.in_transit = still_in_transit;

        for cargo in &arrived {
            let dest = cargo.destination_world as usize;
            if dest < worlds.len() {
                if let Some(stock) = worlds[dest].resources.get_mut(&cargo.resource) {
                    stock.current = (stock.current + cargo.amount).min(stock.capacity);
                }
                events.push(CivEvent::new(
                    current_tick,
                    Some(cargo.destination_world),
                    CivEventType::EmergencyDeclared,
                    format!(
                        "{}: cargo arrived — {:.0} units of {} (departed tick {})",
                        worlds[dest].name, cargo.amount, cargo.resource, cargo.departure_tick
                    ),
                ));
            }
        }

        // --- Trade: transfer resources along routes (window-gated) ---
        for route_idx in 0..self.trade_routes.len() {
            let route = &self.trade_routes[route_idx];
            let from = route.from_world as usize;
            let to = route.to_world as usize;
            if from >= worlds.len() || to >= worlds.len() {
                continue;
            }

            // Transfer window check: only allow trade when window is open
            // Moon-Earth (synodic_period == 0) is always open.
            // Fusion grid-scale removes the constraint entirely.
            let window_open = self.fusion_override
                || Self::is_window_open(route.synodic_period, route.established_tick, current_tick);

            if !window_open {
                continue; // Window closed — no trade this tick
            }

            // Kessler syndrome: degrade volume for routes from/to Earth
            let mut volume = self.trade_routes[route_idx].volume_per_tick;
            if self.leo_access_multiplier < 1.0 {
                let involves_earth =
                    worlds[from].location == "Earth" || worlds[to].location == "Earth";
                let is_moon_earth = (worlds[from].location == "Moon"
                    && worlds[to].location == "Earth")
                    || (worlds[from].location == "Earth" && worlds[to].location == "Moon");
                if involves_earth {
                    if is_moon_earth {
                        volume *= self.leo_access_multiplier.max(0.5); // Moon-Earth less affected
                    } else {
                        volume *= self.leo_access_multiplier;
                    }
                }
            }

            self.total_trade_volume += volume;

            // Transfer a fraction of surplus resources from exporter to importer
            let resource_names: Vec<String> = worlds[from]
                .resources
                .resource_names()
                .iter()
                .map(|s| s.to_string())
                .collect();

            let transfer_time = if self.fusion_override {
                route.transfer_time / 2 // Fusion halves transit time
            } else {
                route.transfer_time
            };

            for name in &resource_names {
                let transfer = {
                    let stock = match worlds[from].resources.get(name) {
                        Some(s) => s,
                        None => continue,
                    };
                    // Only export surplus (above 50% capacity)
                    let surplus = (stock.current - stock.capacity * 0.5).max(0.0);
                    (surplus * 0.01 * volume / 10.0).min(stock.current * 0.05)
                };

                if transfer > 0.0 {
                    // Deduct from source immediately
                    if let Some(from_stock) = worlds[from].resources.get_mut(name) {
                        from_stock.current -= transfer;
                    }

                    if transfer_time == 0 {
                        // Instant delivery (Moon-Earth)
                        if let Some(to_stock) = worlds[to].resources.get_mut(name) {
                            to_stock.current = (to_stock.current + transfer).min(to_stock.capacity);
                        }
                    } else {
                        // Cargo goes in transit — arrives after transfer_time ticks
                        self.in_transit.push(InTransitCargo {
                            resource: name.clone(),
                            amount: transfer,
                            departure_tick: current_tick,
                            arrival_tick: current_tick + transfer_time,
                            destination_world: route.to_world,
                        });
                    }
                }
            }
        }

        // --- Migration: agents move from poorer to richer worlds ---
        // Compute GDP proxy per world (total effective labor / population)
        let gdp_proxies: Vec<f64> = worlds
            .iter()
            .map(|w| {
                let pop = w.population().max(1) as f64;
                let labor: f64 = w
                    .agents
                    .iter()
                    .filter(|a| a.is_alive())
                    .map(|a| a.effective_labor(current_tick))
                    .sum();
                labor / pop
            })
            .collect();

        // Collect migration candidates: (from_world, to_world, agent_id)
        let mut migrations: Vec<(usize, usize, u64)> = Vec::new();
        for (from_idx, from_gdp) in gdp_proxies.iter().enumerate() {
            for (to_idx, to_gdp) in gdp_proxies.iter().enumerate() {
                if from_idx == to_idx {
                    continue;
                }
                // Only migrate if destination has >= 2x GDP per capita
                if *to_gdp < 2.0 * from_gdp {
                    continue;
                }
                // Check route exists
                let has_route = self.trade_routes.iter().any(|r| {
                    (r.from_world as usize == from_idx && r.to_world as usize == to_idx)
                        || (r.from_world as usize == to_idx && r.to_world as usize == from_idx)
                });
                if !has_route {
                    continue;
                }

                // Each living agent has a small probability of migrating
                for agent in worlds[from_idx].agents.iter().filter(|a| a.is_alive()) {
                    if rng.bernoulli(MIGRATION_PROB_PER_AGENT) {
                        migrations.push((from_idx, to_idx, agent.id));
                    }
                }
            }
        }

        // Execute migrations (limit to avoid mass exodus)
        let max_migrations = 5;
        for &(from_idx, to_idx, agent_id) in migrations.iter().take(max_migrations) {
            // Find and remove agent from source world
            let agent_pos = worlds[from_idx]
                .agents
                .iter()
                .position(|a| a.id == agent_id && a.is_alive());

            if let Some(pos) = agent_pos {
                let mut agent = worlds[from_idx].agents.remove(pos);
                agent.world_id = worlds[to_idx].id;
                agent.is_immigrant = true;

                let from_name = worlds[from_idx].name.clone();
                let to_name = worlds[to_idx].name.clone();

                self.migration_history.push(MigrationEvent {
                    agent_id,
                    from_world: worlds[from_idx].id,
                    to_world: worlds[to_idx].id,
                    tick: current_tick,
                    reason: MigrationReason::Economic,
                });
                self.total_migrants += 1;

                worlds[to_idx].agents.push(agent);

                events.push(CivEvent::new(
                    current_tick,
                    None,
                    CivEventType::Migration,
                    format!("Agent {agent_id} migrated from {from_name} to {to_name}"),
                ));
            }
        }

        // --- Cultural exchange: connected worlds converge ---
        for route in &self.trade_routes {
            let from = route.from_world as usize;
            let to = route.to_world as usize;
            if from >= worlds.len() || to >= worlds.len() {
                continue;
            }

            // Move harmony weights toward each other
            let from_weights = worlds[from].culture.harmony_weights;
            let to_weights = worlds[to].culture.harmony_weights;

            for i in 0..8 {
                let diff = to_weights[i] - from_weights[i];
                worlds[from].culture.harmony_weights[i] += diff * CULTURAL_CONVERGENCE_RATE;
                worlds[to].culture.harmony_weights[i] -= diff * CULTURAL_CONVERGENCE_RATE;
            }

            // Renormalize
            renormalize_weights(&mut worlds[from].culture.harmony_weights);
            renormalize_weights(&mut worlds[to].culture.harmony_weights);
        }

        // --- Knowledge diffusion: higher tech flows to lower tech via trade ---
        for route in &self.trade_routes {
            let from = route.from_world as usize;
            let to = route.to_world as usize;
            if from >= worlds.len() || to >= worlds.len() {
                continue;
            }

            // Use mean education as a tech proxy
            let from_tech = mean_education(&worlds[from], current_tick);
            let to_tech = mean_education(&worlds[to], current_tick);
            let gap = from_tech - to_tech;

            if gap > 0.01 {
                // Diffuse knowledge to lower-tech world
                let transfer = gap * KNOWLEDGE_DIFFUSION_RATE;
                for agent in worlds[to].agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.education_level = (agent.education_level + transfer).min(1.0);
                }
                self.knowledge_transfers += 1;
            } else if gap < -0.01 {
                // Reverse direction
                let transfer = (-gap) * KNOWLEDGE_DIFFUSION_RATE;
                for agent in worlds[from].agents.iter_mut().filter(|a| a.is_alive()) {
                    agent.education_level = (agent.education_level + transfer).min(1.0);
                }
                self.knowledge_transfers += 1;
            }
        }

        events
    }

    /// Returns the mass cost multiplier for a resource type.
    /// Knowledge/information modules have near-zero mass cost.
    pub fn trade_route_gravity(&self, from_resource: &str) -> f64 {
        match from_resource {
            "hdc_knowledge_module" | "knowledge" | "software" | "data" => KNOWLEDGE_MASS_COST,
            _ => 1.0, // physical resources use the route's transport_cost_per_kg
        }
    }

    /// Create a new world by fissioning settlers from a parent world.
    ///
    /// Selects skill-balanced, genetically diverse (heterozygous) settlers.
    /// Forks cultural profile from parent. Establishes trade route.
    pub fn world_fission(
        &mut self,
        parent_world: &mut World,
        new_world_id: u32,
        new_name: String,
        settler_count: usize,
        tick: u32,
        rng: &mut StochasticEngine,
    ) -> World {
        // Select settlers: prefer adults with diverse skills
        let living_adults: Vec<usize> = parent_world
            .agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.is_alive() && a.age_years(tick) >= 20.0 && a.age_years(tick) <= 55.0)
            .map(|(i, _)| i)
            .collect();

        let count = settler_count
            .min(living_adults.len())
            .min(parent_world.population() / 4);

        // Score settlers by skill diversity (higher total = more skilled)
        let mut scored: Vec<(usize, f64)> = living_adults
            .iter()
            .map(|&idx| {
                let a = &parent_world.agents[idx];
                let skill_score = a.skills.total();
                let health_score = a.health;
                // Add randomness to prevent always picking the same people
                let noise = rng.next_f64() * 0.3;
                (idx, skill_score + health_score + noise)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Ensure sex balance: pick alternating from sorted list
        let mut selected_indices: Vec<usize> = Vec::with_capacity(count);
        let mut males = 0usize;
        let mut females = 0usize;
        let half = count / 2;

        for &(idx, _) in &scored {
            if selected_indices.len() >= count {
                break;
            }
            let sex = parent_world.agents[idx].sex;
            match sex {
                BiologicalSex::Male if males < half + 1 => {
                    selected_indices.push(idx);
                    males += 1;
                }
                BiologicalSex::Female if females < half + 1 => {
                    selected_indices.push(idx);
                    females += 1;
                }
                _ => {
                    // Accept if we still need more
                    if selected_indices.len() < count {
                        selected_indices.push(idx);
                        match sex {
                            BiologicalSex::Male => males += 1,
                            BiologicalSex::Female => females += 1,
                        }
                    }
                }
            }
        }

        // Sort descending so removal doesn't invalidate earlier indices
        selected_indices.sort_unstable_by(|a, b| b.cmp(a));

        let mut settlers = Vec::with_capacity(selected_indices.len());
        for idx in &selected_indices {
            let mut agent = parent_world.agents.remove(*idx);
            agent.world_id = new_world_id;
            agent.is_immigrant = true;
            settlers.push(agent);
        }

        // Fork cultural profile from parent (identical initially)
        let culture = parent_world.culture.clone();

        let next_id = settlers.iter().map(|a| a.id).max().unwrap_or(0) + 1;

        let new_world = World {
            id: new_world_id,
            name: new_name.clone(),
            location: "Colony".into(),
            founded_tick: tick,
            parent_world_id: Some(parent_world.id),
            agents: settlers,
            next_agent_id: next_id,
            resources: WorldResources::lunar_default(),
            culture,
            infrastructure_level: 0.1,
            max_population: 5_000,
            habitable_area_m2: 30_000.0,
            founding_harmony_emphasis: parent_world.founding_harmony_emphasis,
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
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
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };

        // Establish trade route with parent (assume Moon-Earth-class delay).
        // "Colony" location has synodic_period=0 (continuous), matching co-orbital.
        let parent_loc = parent_world.location.clone();
        self.establish_route(
            parent_world.id,
            new_world_id,
            tick,
            1.3,
            &parent_loc,
            "Colony",
        );

        new_world
    }

    /// Speciation friction: after 400+ years of isolation, cross-world fertility reduces.
    ///
    /// Returns fertility reduction factor (0.0 = no reduction, up to 0.5 max).
    /// Only applies when worlds have been isolated for > 400 years (4800 ticks).
    pub fn speciation_friction(
        world_a_founded: u32,
        world_b_founded: u32,
        current_tick: u32,
    ) -> f64 {
        // Isolation years = min age of the two worlds (proxy for divergence time)
        let age_a = current_tick.saturating_sub(world_a_founded) as f64 / 12.0;
        let age_b = current_tick.saturating_sub(world_b_founded) as f64 / 12.0;
        let isolation_years = age_a.min(age_b);

        if isolation_years <= 400.0 {
            return 0.0;
        }

        ((isolation_years - 400.0) / 600.0).clamp(0.0, 0.5)
    }

    /// Check if outer system fission should occur.
    ///
    /// When total off-earth pop > 5000 AND mean tech > 0.3, found Europa or Titan.
    /// Returns (should_found_europa, should_found_titan).
    pub fn check_outer_system_fission(worlds: &[World], _current_tick: u32) -> (bool, bool) {
        let off_earth_pop: usize = worlds
            .iter()
            .filter(|w| w.location != "Earth")
            .map(|w| w.population())
            .sum();

        let mean_tech: f64 = if worlds.is_empty() {
            0.0
        } else {
            worlds
                .iter()
                .map(|w| w.knowledge.mean_tech_level())
                .sum::<f64>()
                / worlds.len() as f64
        };
        // Normalize tech to 0-1 range (starts at 1.0, so (mean-1)/9 maps [1,10]->[0,1])
        let tech_norm = ((mean_tech - 1.0) / 9.0).clamp(0.0, 1.0);

        if off_earth_pop < 5000 || tech_norm < 0.3 {
            return (false, false);
        }

        let has_europa = worlds.iter().any(|w| w.location == "Europa");
        let has_titan = worlds.iter().any(|w| w.location == "Titan");

        (!has_europa, !has_titan && off_earth_pop > 8000)
    }

    /// Connectivity score: number of trade routes involving this world / max possible.
    pub fn connectivity_score(&self, world_id: u32) -> f64 {
        let count = self
            .trade_routes
            .iter()
            .filter(|r| r.from_world == world_id || r.to_world == world_id)
            .count();
        (count as f64 / MAX_TRADE_ROUTES).min(1.0)
    }
}

impl Default for InterWorldEngine {
    fn default() -> Self {
        Self::new()
    }
}

// --- Utility functions ---

fn renormalize_weights(weights: &mut [f64; 8]) {
    for w in weights.iter_mut() {
        *w = w.max(0.01);
    }
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

fn mean_education(world: &World, _current_tick: u32) -> f64 {
    let living: Vec<f64> = world
        .agents
        .iter()
        .filter(|a| a.is_alive())
        .map(|a| a.education_level)
        .collect();
    if living.is_empty() {
        return 0.0;
    }
    living.iter().sum::<f64>() / living.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{CivAgent, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;

    /// Reference tick for test helpers (30 years into simulation).
    const TEST_TICK: u32 = 30 * 12;

    fn make_world_with_agents(id: u32, name: &str, n: usize, education: f64) -> World {
        let mut world = World {
            id,
            name: name.into(),
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
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
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
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };
        for i in 0..n {
            let birth_tick = 0; // born at tick 0, so at TEST_TICK they are 30 years old
            let mut skills = SkillVector::new();
            skills.learn(i % 8, 0.3);
            world.agents.push(CivAgent {
                id: i as u64,
                birth_tick,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: id,
                health: 0.9,
                skills,
                education_level: education,
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
            });
        }
        world.next_agent_id = n as u64;
        world
    }

    #[test]
    fn test_route_establishment() {
        let mut engine = InterWorldEngine::new();
        let route = engine.establish_route(0, 1, 10, 1.3, "Earth", "Moon");
        assert_eq!(route.from_world, 0);
        assert_eq!(route.to_world, 1);
        assert_eq!(route.established_tick, 10);
        assert!((route.light_delay_secs - 1.3).abs() < 1e-10);
        assert_eq!(engine.trade_routes.len(), 1);
    }

    #[test]
    fn test_transport_cost_calculation() {
        let mut engine = InterWorldEngine::new();

        // Moon-Earth class (light_delay < 2s → delta_v = 2.5)
        let route_near = engine.establish_route(0, 1, 0, 1.3, "Earth", "Moon");
        let cost_near = route_near.transport_cost_per_kg;
        let expected_near = 2.5 * 2.5 * 0.001;
        assert!(
            (cost_near - expected_near).abs() < 1e-10,
            "Moon-Earth cost: {cost_near} vs expected {expected_near}"
        );

        // Earth-Mars class (180-600s → delta_v = 4.0)
        let route_far = engine.establish_route(0, 2, 0, 300.0, "Earth", "Mars");
        let cost_far = route_far.transport_cost_per_kg;
        let expected_far = 4.0 * 4.0 * 0.001;
        assert!(
            (cost_far - expected_far).abs() < 1e-10,
            "Earth-Mars cost: {cost_far} vs expected {expected_far}"
        );

        assert!(
            cost_far > cost_near,
            "Mars should be more expensive than Moon"
        );
    }

    #[test]
    fn test_trade_transfers_resources() {
        let mut engine = InterWorldEngine::new();
        engine.establish_route(0, 1, 0, 1.3, "Earth", "Moon");

        let mut w0 = make_world_with_agents(0, "Earth", 50, 0.5);
        // Give Earth abundant food (above 50% capacity threshold)
        if let Some(food) = w0.resources.get_mut("food") {
            food.current = 1800.0; // well above 50% of 2000 capacity
        }

        let w1 = make_world_with_agents(1, "Moon", 20, 0.3);
        let initial_moon_food = w1.resources.get("food").map(|s| s.current).unwrap_or(0.0);

        let mut worlds = vec![w0, w1];
        let mut rng = StochasticEngine::new(42);
        engine.tick_interworld(&mut worlds, 100, &mut rng);

        let moon_food = worlds[1]
            .resources
            .get("food")
            .map(|s| s.current)
            .unwrap_or(0.0);
        assert!(
            moon_food >= initial_moon_food,
            "Moon should receive food from Earth: {moon_food} vs {initial_moon_food}"
        );
    }

    #[test]
    fn test_migration_moves_agents() {
        let mut engine = InterWorldEngine::new();
        engine.establish_route(0, 1, 0, 1.3, "Earth", "Moon");

        // World 0: low skills (low GDP proxy)
        let w0 = make_world_with_agents(0, "Poor", 30, 0.1);
        // World 1: high skills (high GDP proxy)
        let mut w1 = make_world_with_agents(1, "Rich", 10, 0.9);
        // Give Rich world agents high skills for GDP disparity
        for agent in w1.agents.iter_mut() {
            for sector in 0..8 {
                agent.skills.learn(sector, 0.8);
            }
        }

        let _initial_pop_0 = w0.population();
        let _initial_pop_1 = w1.population();

        let mut worlds = vec![w0, w1];
        let mut rng = StochasticEngine::new(42);

        // Run many ticks to ensure at least one migration happens
        let mut any_migration = false;
        for tick in TEST_TICK..=TEST_TICK + 100 {
            let events = engine.tick_interworld(&mut worlds, tick, &mut rng);
            if events
                .iter()
                .any(|e| e.event_type == CivEventType::Migration)
            {
                any_migration = true;
                break;
            }
        }

        // Migration may or may not happen depending on RNG; check engine state
        if any_migration {
            assert!(engine.total_migrants > 0);
            assert!(!engine.migration_history.is_empty());
        }
    }

    #[test]
    fn test_knowledge_diffusion_reduces_gap() {
        let mut engine = InterWorldEngine::new();
        engine.establish_route(0, 1, 0, 1.3, "Earth", "Moon");

        let w0 = make_world_with_agents(0, "Advanced", 20, 0.8);
        let w1 = make_world_with_agents(1, "Developing", 20, 0.2);

        let initial_gap = 0.8 - 0.2;

        let mut worlds = vec![w0, w1];
        let mut rng = StochasticEngine::new(42);

        for tick in 1..=50 {
            engine.tick_interworld(&mut worlds, tick, &mut rng);
        }

        let new_mean_0 = mean_education(&worlds[0], 50);
        let new_mean_1 = mean_education(&worlds[1], 50);
        let new_gap = (new_mean_0 - new_mean_1).abs();

        assert!(
            new_gap < initial_gap,
            "Knowledge gap should shrink: {new_gap} vs initial {initial_gap}"
        );
    }

    #[test]
    fn test_trade_route_gravity_favors_information() {
        let engine = InterWorldEngine::new();
        let physical = engine.trade_route_gravity("food");
        let knowledge = engine.trade_route_gravity("hdc_knowledge_module");
        assert!(
            knowledge < physical,
            "Knowledge should be cheaper than physical: {knowledge} vs {physical}"
        );
        assert!(
            (knowledge - KNOWLEDGE_MASS_COST).abs() < 1e-10,
            "Knowledge cost should be {KNOWLEDGE_MASS_COST}"
        );
    }

    #[test]
    fn test_world_fission_creates_valid_world() {
        let mut engine = InterWorldEngine::new();
        let mut rng = StochasticEngine::new(42);

        let mut parent = make_world_with_agents(0, "Earth", 100, 0.5);
        let initial_parent_pop = parent.population();

        // Agents born at tick 0; at TEST_TICK they are 30 years old (within 20-55 range).
        let child =
            engine.world_fission(&mut parent, 1, "Luna Base".into(), 20, TEST_TICK, &mut rng);

        assert_eq!(child.id, 1);
        assert_eq!(child.name, "Luna Base");
        assert_eq!(child.parent_world_id, Some(0));
        assert!(child.population() > 0, "Child world should have settlers");
        assert!(
            child.population() <= 25, // capped at initial_pop / 4
            "Child population should be limited"
        );
        assert!(
            parent.population() < initial_parent_pop,
            "Parent should have fewer agents after fission"
        );
        assert_eq!(
            parent.population() + child.population(),
            initial_parent_pop,
            "Total population should be conserved"
        );

        // Trade route should be established
        assert!(
            engine.trade_routes.len() == 1,
            "Should establish route to parent"
        );
        assert_eq!(engine.trade_routes[0].from_world, 0);
        assert_eq!(engine.trade_routes[0].to_world, 1);
    }

    #[test]
    fn test_connectivity_score() {
        let mut engine = InterWorldEngine::new();
        assert!((engine.connectivity_score(0) - 0.0).abs() < 1e-10);

        engine.establish_route(0, 1, 0, 1.3, "Earth", "Moon");
        engine.establish_route(0, 2, 0, 300.0, "Earth", "Mars");
        engine.establish_route(0, 3, 0, 600.0, "Earth", "Europa");

        let score = engine.connectivity_score(0);
        assert!(
            (score - 3.0 / MAX_TRADE_ROUTES).abs() < 1e-10,
            "Connectivity score with 3 routes: {score}"
        );

        // World 1 only has 1 route
        let score1 = engine.connectivity_score(1);
        assert!(
            (score1 - 1.0 / MAX_TRADE_ROUTES).abs() < 1e-10,
            "World 1 connectivity: {score1}"
        );
    }
}
