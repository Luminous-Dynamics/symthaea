// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Resontia: Earth-hardening scenario for the civilization simulator.
//!
//! Models a civilization that treats Earth's surface as a hostile environment
//! and builds protected infrastructure in the planetary crust.
//!
//! Key components:
//! - Subterranean vault network (radiation shielding, thermal stability)
//! - Maglev evacuation corridors (Mach 1+, vacuum tubes, blast doors)
//! - Equatorial spaceport (460 m/s rotational velocity bonus)
//! - Hybrid coastal-vault protocol (live on coast, evacuate to vault)
//! - FEP-driven early warning + TEND evacuation incentives

use serde::{Deserialize, Serialize};

use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Heat load per person in vault (kW). Metabolic + lighting + ECLSS.
const HEAT_PER_PERSON_KW: f64 = 0.1;

/// Equatorial rotational velocity bonus (km/s). Cape Canaveral gets ~0.41,
/// equatorial sites get ~0.46. Delta-v savings reduce launch cost by ~5%.
const EQUATORIAL_DELTA_V_BONUS_KMS: f64 = 0.46;

/// Launch cost reduction fraction when equatorial spaceport is operational.
const SPACEPORT_LAUNCH_COST_REDUCTION: f64 = 0.05;

/// Ticks per year (re-exported from config for clarity).
const TICKS_PER_YEAR: u32 = 12;

/// Sacred Stillness harmony index (8th harmony, 0-indexed).
const SACRED_STILLNESS_INDEX: usize = 7;

// ---------------------------------------------------------------------------
// Evacuation triggers
// ---------------------------------------------------------------------------

/// Events that can trigger a vault evacuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvacuationTrigger {
    /// Carrington-class solar event — geomagnetic storm fries surface electronics.
    CarringtonEvent,
    /// Mega-tsunami from submarine landslide or mega-quake.
    MegaTsunami,
    /// Asteroid impact with weeks-to-months warning from survey programs.
    AsteroidImpact,
    /// Supervolcanic eruption (VEI 7+) — pyroclastic flows, ash winter.
    SupervolcanoEruption,
    /// Orbital debris cascade makes surface dangerous from re-entry debris.
    KesslerSyndrome,
    /// Temporary weakening of Earth's magnetic field — surface radiation spikes.
    MagnetosphereExcursion,
    /// Nuclear winter scenario (included for completeness, not modeled as likely).
    NuclearWinter,
}

impl EvacuationTrigger {
    /// Typical warning time in hours for this threat class.
    pub fn warning_hours(&self) -> f64 {
        match self {
            EvacuationTrigger::CarringtonEvent => 18.0, // NOAA 15-30 min to hours; assume 18h from CME detection
            EvacuationTrigger::MegaTsunami => 12.0, // Seismic detection gives 6-24h for distant events
            EvacuationTrigger::AsteroidImpact => 720.0, // Weeks to months from survey detection
            EvacuationTrigger::SupervolcanoEruption => 168.0, // Days to weeks from seismic/deformation monitoring
            EvacuationTrigger::KesslerSyndrome => 48.0, // Tracked debris, hours to days warning
            EvacuationTrigger::MagnetosphereExcursion => 720.0, // Geomagnetic drift monitored over months
            EvacuationTrigger::NuclearWinter => 4.0, // Minutes to hours (ICBM flight time)
        }
    }

    /// Base threat level (0.0-1.0) — how severe the threat is without mitigation.
    pub fn base_threat_level(&self) -> f64 {
        match self {
            EvacuationTrigger::CarringtonEvent => 0.6,
            EvacuationTrigger::MegaTsunami => 0.7,
            EvacuationTrigger::AsteroidImpact => 0.9,
            EvacuationTrigger::SupervolcanoEruption => 0.95,
            EvacuationTrigger::KesslerSyndrome => 0.3,
            EvacuationTrigger::MagnetosphereExcursion => 0.5,
            EvacuationTrigger::NuclearWinter => 0.85,
        }
    }
}

// ---------------------------------------------------------------------------
// Evacuation event record
// ---------------------------------------------------------------------------

/// Record of a single evacuation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvacuationEvent {
    /// What triggered the evacuation.
    pub trigger: EvacuationTrigger,
    /// Threat severity (0.0-1.0).
    pub threat_level: f64,
    /// Hours of warning before impact.
    pub warning_hours: f64,
    /// People needing evacuation.
    pub population_to_evacuate: usize,
    /// Fraction of population successfully evacuated (0.0-1.0).
    pub evacuation_success_rate: f64,
    /// Casualties (people who could not be evacuated).
    pub casualties: usize,
    /// TEND cost of the evacuation operation.
    pub economic_cost: f64,
    /// Tick when the evacuation occurred.
    pub tick: u32,
}

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

/// Resontia subterranean infrastructure state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResontiaInfrastructure {
    /// Number of completed vault nodes (subterranean cities).
    pub vault_count: usize,
    /// Total vault capacity (people).
    pub vault_capacity: usize,
    /// Current vault occupancy.
    pub vault_occupancy: usize,
    /// Maglev network throughput (people per hour per corridor).
    pub maglev_throughput_per_hour: usize,
    /// Number of maglev corridors connecting coast to vaults.
    pub maglev_corridors: usize,
    /// Vault infrastructure level [0.0, 1.0] (affects shielding, ECLSS, self-sufficiency).
    pub vault_infrastructure: f64,
    /// Equatorial spaceport built? (delta-v bonus for launches).
    pub equatorial_spaceport: bool,
    /// Spaceport throughput (kg to orbit per month).
    pub spaceport_throughput_kg_per_month: f64,
    /// Heat rejection capacity (MW) — the thermodynamic constraint.
    pub heat_rejection_mw: f64,
    /// Maglev segmentation (blast door count — seismic resilience).
    pub maglev_blast_doors: usize,
    /// Vaults under construction (remaining ticks until completion).
    pub vaults_under_construction: Vec<u32>,
    /// Maglev corridors under construction (remaining ticks).
    pub corridors_under_construction: Vec<u32>,
    /// History of evacuation events.
    pub evacuation_history: Vec<EvacuationEvent>,
}

impl ResontiaInfrastructure {
    /// Create new infrastructure with no vaults or corridors.
    pub fn new() -> Self {
        Self {
            vault_count: 0,
            vault_capacity: 0,
            vault_occupancy: 0,
            maglev_throughput_per_hour: 5_000,
            maglev_corridors: 0,
            vault_infrastructure: 0.0,
            equatorial_spaceport: false,
            spaceport_throughput_kg_per_month: 0.0,
            heat_rejection_mw: 0.0,
            maglev_blast_doors: 0,
            vaults_under_construction: Vec::new(),
            corridors_under_construction: Vec::new(),
            evacuation_history: Vec::new(),
        }
    }

    /// Advance construction by one tick. Returns events for completed projects.
    pub fn tick_construction(&mut self, tick: u32) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // Advance vault construction
        for remaining in self.vaults_under_construction.iter_mut() {
            *remaining = remaining.saturating_sub(1);
        }
        let completed_vaults = self
            .vaults_under_construction
            .iter()
            .filter(|&&r| r == 0)
            .count();
        self.vaults_under_construction.retain(|&r| r > 0);

        for _ in 0..completed_vaults {
            self.vault_count += 1;
            // Each vault holds 50,000 people and rejects 10 MW of heat
            self.vault_capacity += 50_000;
            self.heat_rejection_mw += 10.0;
            self.maglev_blast_doors += 8; // 8 blast doors per vault
            self.vault_infrastructure = (self.vault_infrastructure + 0.08).min(1.0);
            events.push(CivEvent::new(
                tick,
                None,
                CivEventType::ResontiaInfrastructureBuilt,
                format!(
                    "Resontia vault #{} completed — capacity now {} people, {:.0} MW heat rejection",
                    self.vault_count, self.vault_capacity, self.heat_rejection_mw
                ),
            ));
        }

        // Advance corridor construction
        for remaining in self.corridors_under_construction.iter_mut() {
            *remaining = remaining.saturating_sub(1);
        }
        let completed_corridors = self
            .corridors_under_construction
            .iter()
            .filter(|&&r| r == 0)
            .count();
        self.corridors_under_construction.retain(|&r| r > 0);

        for _ in 0..completed_corridors {
            self.maglev_corridors += 1;
            events.push(CivEvent::new(
                tick,
                None,
                CivEventType::ResontiaInfrastructureBuilt,
                format!(
                    "Resontia maglev corridor #{} operational — {} corridors total",
                    self.maglev_corridors, self.maglev_corridors
                ),
            ));
        }

        // Equatorial spaceport: built automatically after 4 vaults and 2 corridors
        if !self.equatorial_spaceport && self.vault_count >= 4 && self.maglev_corridors >= 2 {
            self.equatorial_spaceport = true;
            self.spaceport_throughput_kg_per_month = 500_000.0; // 500 tonnes/month
            events.push(CivEvent::new(
                tick,
                None,
                CivEventType::ResontiaInfrastructureBuilt,
                format!(
                    "Equatorial spaceport operational — {:.2} km/s delta-v bonus, {:.0} kg/month throughput",
                    EQUATORIAL_DELTA_V_BONUS_KMS,
                    self.spaceport_throughput_kg_per_month
                ),
            ));
        }

        events
    }

    /// Whether any vault is available for evacuation.
    pub fn is_vault_available(&self) -> bool {
        self.vault_count > 0 && self.vault_capacity > 0
    }

    /// Total evacuation capacity (people per hour across all corridors).
    pub fn total_evacuation_capacity(&self) -> usize {
        self.maglev_corridors * self.maglev_throughput_per_hour
    }

    /// Maximum occupancy limited by heat rejection.
    /// Each person generates ~0.1 kW; heat_rejection_mw * 1000 kW / 0.1 kW/person.
    pub fn heat_limited_capacity(&self) -> usize {
        ((self.heat_rejection_mw * 1000.0) / HEAT_PER_PERSON_KW) as usize
    }

    /// Begin construction of a new vault.
    pub fn start_vault_construction(&mut self, construction_ticks: u32) {
        self.vaults_under_construction.push(construction_ticks);
    }

    /// Begin construction of a new maglev corridor.
    pub fn start_corridor_construction(&mut self, construction_ticks: u32) {
        self.corridors_under_construction.push(construction_ticks);
    }
}

impl Default for ResontiaInfrastructure {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Resontia Earth-hardening scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResontiaConfig {
    /// Enable Resontia infrastructure on Earth.
    pub enabled: bool,
    /// Simulation year when vault construction begins.
    pub construction_start_year: f64,
    /// Years to build each vault node.
    pub vault_construction_time_years: f64,
    /// Maximum vault nodes (limited by geology and resources).
    pub max_vault_nodes: usize,
    /// Maglev construction rate (corridors per decade).
    pub maglev_construction_rate: f64,
    /// Cost multiplier for maintaining dormant vaults (fraction of active cost).
    pub dormant_vault_cost_fraction: f64,
}

impl Default for ResontiaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            construction_start_year: 50.0, // Year 2076 (Epoch 3)
            vault_construction_time_years: 10.0,
            max_vault_nodes: 12,               // 12 vault cities worldwide
            maglev_construction_rate: 2.0,     // 2 corridors per decade
            dormant_vault_cost_fraction: 0.05, // 5% of active maintenance cost
        }
    }
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute evacuation success rate given infrastructure and conditions.
///
/// - `infra`: current Resontia infrastructure state
/// - `population`: number of people to evacuate
/// - `warning_hours`: hours of advance warning
/// - `stillness`: Sacred Stillness harmony score [0.0, 1.0]
///
/// Returns the fraction of population that can be evacuated (0.0-1.0).
pub fn compute_evacuation_success(
    infra: &ResontiaInfrastructure,
    population: usize,
    warning_hours: f64,
    stillness: f64,
) -> f64 {
    if population == 0 || !infra.is_vault_available() {
        return 0.0;
    }

    let max_evacuated =
        infra.maglev_corridors as f64 * infra.maglev_throughput_per_hour as f64 * warning_hours;

    // Calm populations don't panic-clog corridors.
    // At stillness=0.0: 50% efficiency (panicked crowd, corridor clogging).
    // At stillness=1.0: 100% efficiency (calm, orderly evacuation).
    let orderly_factor = 0.5 + stillness * 0.5;
    let effective_evacuated = max_evacuated * orderly_factor;

    // Can't evacuate more than vault capacity or heat-limited capacity
    let capacity_limit = infra.vault_capacity.min(infra.heat_limited_capacity()) as f64;
    let capped_evacuated = effective_evacuated.min(capacity_limit);

    (capped_evacuated / population as f64).min(1.0)
}

/// Delta-v savings from equatorial spaceport (km/s).
/// Returns 0.0 if spaceport not built.
pub fn equatorial_spaceport_bonus(infra: &ResontiaInfrastructure) -> f64 {
    if infra.equatorial_spaceport {
        EQUATORIAL_DELTA_V_BONUS_KMS
    } else {
        0.0
    }
}

/// Launch cost reduction fraction when equatorial spaceport is operational.
/// Returns 0.0 if spaceport not built, ~0.05 (5%) when operational.
pub fn launch_cost_reduction(infra: &ResontiaInfrastructure) -> f64 {
    if infra.equatorial_spaceport {
        SPACEPORT_LAUNCH_COST_REDUCTION
    } else {
        0.0
    }
}

/// Run one tick of the Resontia scenario.
///
/// Handles construction progress, starts new construction when appropriate,
/// and processes evacuations when disasters hit Earth.
///
/// `disaster_population_loss` is the number of casualties the disaster engine
/// would inflict on Earth this tick (before Resontia mitigation).
///
/// Returns (events, mitigated_casualties).
pub fn tick_resontia(
    infra: &mut ResontiaInfrastructure,
    config: &ResontiaConfig,
    earth_world: &World,
    current_tick: u32,
    disaster_population_loss: usize,
    disaster_trigger: Option<EvacuationTrigger>,
    _rng: &mut StochasticEngine,
) -> (Vec<CivEvent>, usize) {
    let mut events = Vec::new();
    let mut mitigated = 0usize;

    if !config.enabled {
        return (events, mitigated);
    }

    let current_year = current_tick as f64 / TICKS_PER_YEAR as f64;

    // --- Phase 1: Start construction if time has come ---
    let construction_ticks = (config.vault_construction_time_years * TICKS_PER_YEAR as f64) as u32;

    if current_year >= config.construction_start_year {
        // Start a new vault every construction_time_years if below max
        let total_planned = infra.vault_count + infra.vaults_under_construction.len();
        if total_planned < config.max_vault_nodes && infra.vaults_under_construction.is_empty() {
            infra.start_vault_construction(construction_ticks);
            events.push(CivEvent::new(
                current_tick,
                None,
                CivEventType::ResontiaInfrastructureBuilt,
                format!(
                    "Resontia vault construction begun — will complete in {:.0} years (tick {})",
                    config.vault_construction_time_years,
                    current_tick + construction_ticks
                ),
            ));
        }

        // Start maglev corridors: one per (120 / construction_rate) ticks
        let corridor_interval = (120.0 / config.maglev_construction_rate) as u32; // ticks between corridor starts
        let corridor_construction_ticks = corridor_interval / 2; // corridors build faster than vaults
        if current_tick % corridor_interval == 0
            && infra.corridors_under_construction.is_empty()
            && infra.vault_count > 0
        {
            infra.start_corridor_construction(corridor_construction_ticks);
        }
    }

    // --- Phase 2: Advance construction ---
    let construction_events = infra.tick_construction(current_tick);
    events.extend(construction_events);

    // --- Phase 3: Handle evacuation if disaster strikes Earth ---
    if disaster_population_loss > 0 && infra.is_vault_available() {
        if let Some(trigger) = disaster_trigger {
            let warning = trigger.warning_hours();
            let stillness = earth_world.harmony.current_scores[SACRED_STILLNESS_INDEX];

            let success_rate =
                compute_evacuation_success(infra, disaster_population_loss, warning, stillness);

            let evacuated = (disaster_population_loss as f64 * success_rate) as usize;
            mitigated = evacuated;
            let casualties = disaster_population_loss.saturating_sub(evacuated);

            // TEND cost: 10 per person evacuated (logistics, supplies, temporary shelter)
            let economic_cost = evacuated as f64 * 10.0;

            let evac_event = EvacuationEvent {
                trigger,
                threat_level: trigger.base_threat_level(),
                warning_hours: warning,
                population_to_evacuate: disaster_population_loss,
                evacuation_success_rate: success_rate,
                casualties,
                economic_cost,
                tick: current_tick,
            };
            infra.evacuation_history.push(evac_event);

            events.push(CivEvent::new(
                current_tick,
                None,
                CivEventType::ResontiaEvacuation,
                format!(
                    "Resontia evacuation: {:?} — {}/{} evacuated ({:.1}% success), {} casualties, {:.0} TEND cost",
                    trigger,
                    evacuated,
                    disaster_population_loss,
                    success_rate * 100.0,
                    casualties,
                    economic_cost
                ),
            ));
        }
    }

    // --- Phase 4: Heat rejection constraint ---
    // Vault occupancy limited by thermodynamics
    let heat_cap = infra.heat_limited_capacity();
    if infra.vault_occupancy > heat_cap {
        let excess = infra.vault_occupancy - heat_cap;
        infra.vault_occupancy = heat_cap;
        events.push(CivEvent::new(
            current_tick,
            None,
            CivEventType::EmergencyDeclared,
            format!(
                "Resontia heat rejection limit reached — {} people forced to surface, {:.0} MW capacity",
                excess, infra.heat_rejection_mw
            ),
        ));
    }

    (events, mitigated)
}

// ---------------------------------------------------------------------------
// Config preset
// ---------------------------------------------------------------------------

/// Create a Resontia-enabled configuration for the 777-year scenario.
///
/// Enables vault construction starting at year 50 (Epoch 3: Maturation).
pub fn default_resontia_config() -> ResontiaConfig {
    ResontiaConfig {
        enabled: true,
        construction_start_year: 50.0,
        vault_construction_time_years: 10.0,
        max_vault_nodes: 12,
        maglev_construction_rate: 2.0,
        dormant_vault_cost_fraction: 0.05,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::harmony::HarmonyTracker;
    use crate::needs::PsychologicalNeeds;
    use crate::stochastic::StochasticEngine;
    use crate::world::{CulturalProfile, World, WorldResources};

    /// Build a minimal Earth world for testing.
    fn make_earth(pop: usize, stillness: f64) -> World {
        let mut agents = Vec::new();
        for i in 0..pop {
            agents.push(CivAgent {
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
            });
        }
        let mut harmony = HarmonyTracker::new();
        harmony.current_scores[SACRED_STILLNESS_INDEX] = stillness;

        World {
            id: 0,
            name: "Earth".into(),
            location: "Earth".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents,
            next_agent_id: pop as u64,
            resources: WorldResources::earth_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.8,
            max_population: 1_000_000,
            habitable_area_m2: 1_000_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony,
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
        }
    }

    fn make_infra_with_vaults(vaults: usize, corridors: usize) -> ResontiaInfrastructure {
        ResontiaInfrastructure {
            vault_count: vaults,
            vault_capacity: vaults * 50_000,
            vault_occupancy: 0,
            maglev_throughput_per_hour: 5_000,
            maglev_corridors: corridors,
            vault_infrastructure: (vaults as f64 * 0.08).min(1.0),
            equatorial_spaceport: vaults >= 4 && corridors >= 2,
            spaceport_throughput_kg_per_month: if vaults >= 4 && corridors >= 2 {
                500_000.0
            } else {
                0.0
            },
            heat_rejection_mw: vaults as f64 * 10.0,
            maglev_blast_doors: vaults * 8,
            vaults_under_construction: Vec::new(),
            corridors_under_construction: Vec::new(),
            evacuation_history: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Vault construction timeline correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_vault_construction_timeline() {
        let mut infra = ResontiaInfrastructure::new();
        let construction_ticks = 120; // 10 years
        infra.start_vault_construction(construction_ticks);

        assert_eq!(infra.vaults_under_construction.len(), 1);
        assert_eq!(infra.vault_count, 0);

        // Advance 119 ticks — not yet complete
        for tick in 0..119 {
            infra.tick_construction(tick);
        }
        assert_eq!(
            infra.vault_count, 0,
            "Vault should not complete before 120 ticks"
        );

        // Tick 120 — completes
        let events = infra.tick_construction(119);
        assert_eq!(infra.vault_count, 1, "Vault should complete at tick 120");
        assert_eq!(infra.vault_capacity, 50_000);
        assert!(!events.is_empty(), "Should generate completion event");
        assert!(events[0].description.contains("vault #1"));
    }

    // -----------------------------------------------------------------------
    // Test 2: Maglev throughput computation
    // -----------------------------------------------------------------------
    #[test]
    fn test_maglev_throughput_computation() {
        let infra = make_infra_with_vaults(2, 3);

        // 3 corridors * 5000 people/hour = 15,000 people/hour
        assert_eq!(infra.total_evacuation_capacity(), 15_000);

        // In 12 hours: 180,000 people
        let hours = 12.0;
        let max_evac = infra.total_evacuation_capacity() as f64 * hours;
        assert!((max_evac - 180_000.0).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // Test 3: Evacuation success rate scales with infrastructure
    // -----------------------------------------------------------------------
    #[test]
    fn test_evacuation_success_scales_with_infrastructure() {
        let small_infra = make_infra_with_vaults(1, 1);
        let large_infra = make_infra_with_vaults(6, 6);
        let population = 100_000;
        let warning = 12.0; // 12 hours
        let stillness = 0.5;

        let small_rate = compute_evacuation_success(&small_infra, population, warning, stillness);
        let large_rate = compute_evacuation_success(&large_infra, population, warning, stillness);

        assert!(
            large_rate > small_rate,
            "More infrastructure should yield higher evacuation rate: {large_rate} vs {small_rate}"
        );
        assert!(
            small_rate > 0.0,
            "Even small infra should evacuate some people"
        );
        assert!(large_rate <= 1.0, "Rate should be capped at 1.0");
    }

    // -----------------------------------------------------------------------
    // Test 4: Heat rejection limits vault occupancy
    // -----------------------------------------------------------------------
    #[test]
    fn test_heat_rejection_limits_occupancy() {
        let mut infra = make_infra_with_vaults(1, 1);
        // 1 vault = 10 MW heat rejection
        // At 0.1 kW/person: max = 10,000 / 0.1 = 100,000 people
        let heat_cap = infra.heat_limited_capacity();
        assert_eq!(heat_cap, 100_000);

        // Exceed it
        infra.vault_occupancy = 150_000;
        let config = ResontiaConfig {
            enabled: true,
            ..Default::default()
        };
        let earth = make_earth(500, 0.5);
        let mut rng = StochasticEngine::new(42);

        let (events, _) = tick_resontia(&mut infra, &config, &earth, 1000, 0, None, &mut rng);

        assert_eq!(
            infra.vault_occupancy, heat_cap,
            "Occupancy should be clamped to heat-limited capacity"
        );
        assert!(
            events
                .iter()
                .any(|e| e.description.contains("heat rejection")),
            "Should generate heat rejection event"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Stillness improves evacuation efficiency
    // -----------------------------------------------------------------------
    #[test]
    fn test_stillness_improves_evacuation() {
        // Use 1 vault + 1 corridor with large population so we don't saturate at 1.0
        let infra = make_infra_with_vaults(1, 1);
        let population = 200_000; // Much larger than evacuation capacity
        let warning = 2.0; // Short warning to avoid saturation

        let calm_rate = compute_evacuation_success(&infra, population, warning, 1.0);
        let panicked_rate = compute_evacuation_success(&infra, population, warning, 0.0);

        assert!(
            calm_rate > panicked_rate,
            "Calm population (stillness=1.0) should evacuate better: {calm_rate} vs {panicked_rate}"
        );
        // orderly_factor at stillness=1.0 is 1.0, at stillness=0.0 is 0.5
        // So calm populations achieve full throughput, panicked crowds lose 50%.
        assert!(
            (calm_rate / panicked_rate - 2.0).abs() < 0.1,
            "Calm should evacuate ~2x more than panicked: ratio {:.2}",
            calm_rate / panicked_rate
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Equatorial spaceport provides delta-v bonus
    // -----------------------------------------------------------------------
    #[test]
    fn test_equatorial_spaceport_bonus() {
        let infra_no_port = make_infra_with_vaults(2, 1);
        let infra_with_port = make_infra_with_vaults(5, 3);

        assert!(!infra_no_port.equatorial_spaceport);
        assert!(infra_with_port.equatorial_spaceport);

        assert_eq!(equatorial_spaceport_bonus(&infra_no_port), 0.0);
        assert!(
            (equatorial_spaceport_bonus(&infra_with_port) - 0.46).abs() < 0.01,
            "Should provide ~0.46 km/s delta-v bonus"
        );

        assert_eq!(launch_cost_reduction(&infra_no_port), 0.0);
        assert!(
            (launch_cost_reduction(&infra_with_port) - 0.05).abs() < 0.001,
            "Should provide ~5% launch cost reduction"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: No evacuation without vaults (baseline)
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_evacuation_without_vaults() {
        let infra = ResontiaInfrastructure::new();
        let success = compute_evacuation_success(&infra, 10_000, 24.0, 0.8);
        assert_eq!(success, 0.0, "No vaults means zero evacuation capability");
    }

    // -----------------------------------------------------------------------
    // Test 8: Carrington event with Resontia vs without
    // -----------------------------------------------------------------------
    #[test]
    fn test_carrington_with_vs_without_resontia() {
        let earth = make_earth(100_000, 0.6);
        let mut rng = StochasticEngine::new(42);
        let disaster_loss = 5_000; // 5% of population from Carrington

        // Without Resontia
        let mut infra_none = ResontiaInfrastructure::new();
        let config_off = ResontiaConfig::default(); // enabled: false
        let (_, mitigated_none) = tick_resontia(
            &mut infra_none,
            &config_off,
            &earth,
            1000,
            disaster_loss,
            Some(EvacuationTrigger::CarringtonEvent),
            &mut rng,
        );
        assert_eq!(mitigated_none, 0, "No mitigation without Resontia");

        // With Resontia (4 vaults, 4 corridors)
        let mut infra_full = make_infra_with_vaults(4, 4);
        let config_on = ResontiaConfig {
            enabled: true,
            ..Default::default()
        };
        let (events, mitigated_full) = tick_resontia(
            &mut infra_full,
            &config_on,
            &earth,
            1000,
            disaster_loss,
            Some(EvacuationTrigger::CarringtonEvent),
            &mut rng,
        );

        assert!(
            mitigated_full > mitigated_none,
            "Resontia should mitigate casualties: {mitigated_full} vs {mitigated_none}"
        );
        assert!(!events.is_empty(), "Should generate evacuation events");
        assert!(
            events
                .iter()
                .any(|e| e.description.contains("Resontia evacuation")),
            "Should contain Resontia evacuation event description"
        );
        assert!(
            !infra_full.evacuation_history.is_empty(),
            "Should record evacuation in history"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Default config preset
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_resontia_config() {
        let config = default_resontia_config();
        assert!(config.enabled);
        assert_eq!(config.max_vault_nodes, 12);
        assert!((config.construction_start_year - 50.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Test 10: Warning hours vary by trigger type
    // -----------------------------------------------------------------------
    #[test]
    fn test_evacuation_trigger_warning_hours() {
        // Asteroid impact should give much more warning than Carrington
        assert!(
            EvacuationTrigger::AsteroidImpact.warning_hours()
                > EvacuationTrigger::CarringtonEvent.warning_hours()
        );
        // Nuclear winter has least warning
        assert!(EvacuationTrigger::NuclearWinter.warning_hours() < 10.0);
    }
}
