// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Moon Base Alpha End-to-End Simulation Runner
//!
//! Ties together all Phase 0-2 components into a running 72-hour lunar base
//! simulation. Pure Rust, no Holochain runtime required. Deterministic given
//! a fixed scenario and seed.
//!
//! # Architecture
//!
//! The runner ticks at 60-second intervals (simulated time), applying
//! disturbances, resource consumption/generation, communication checks,
//! and checkpoint evaluations at each step. Four autonomous agents
//! (Rover, ISRU, Habitat, Comms) coordinate through a local mesh while
//! maintaining Earth links subject to latency and blackout constraints.
//!
//! # Usage
//!
//! ```ignore
//! use commons_types::simulation_runner::*;
//! use commons_types::lunar_analog::Scenario;
//!
//! let scenario = Scenario::moon_base_alpha_72h();
//! let mut runner = SimulationRunner::new(scenario);
//! let report = runner.run();
//! println!("{}", report.summary());
//! assert!(report.all_survived);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::latency_sim::{CommEnvironment, DataPriority, TransmitOutcome};
use crate::lunar_analog::{
    AgentRole, AgentState, Checkpoint, CheckpointAssertion, DisturbanceKind, Scenario,
};

// ============================================================================
// LIGHTWEIGHT TYPES (avoid cross-workspace deps on Symthaea)
// ============================================================================

/// Power management mode derived from battery state-of-charge.
///
/// Mirrors the Symthaea `PowerMode` logic for substrate-independent
/// power budgeting. Each mode defines a recommended cognitive tick rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerMode {
    /// Full power, all subsystems active. SoC > 60%.
    Normal,
    /// Reduced non-essential loads. 30% < SoC <= 60%.
    Conservation,
    /// Life support only, minimal cognition. 10% < SoC <= 30%.
    Survival,
    /// Deep sleep, wake on interrupt only. SoC <= 10%.
    Hibernation,
}

impl PowerMode {
    /// Derive power mode from battery state-of-charge [0.0, 1.0].
    pub fn from_soc(soc: f64) -> Self {
        if soc > 0.60 {
            PowerMode::Normal
        } else if soc > 0.30 {
            PowerMode::Conservation
        } else if soc > 0.10 {
            PowerMode::Survival
        } else {
            PowerMode::Hibernation
        }
    }

    /// Recommended cognitive loop frequency (Hz) for this mode.
    pub fn recommended_hz(&self) -> f64 {
        match self {
            PowerMode::Normal => 20.0,
            PowerMode::Conservation => 5.0,
            PowerMode::Survival => 1.0,
            PowerMode::Hibernation => 0.0,
        }
    }

    /// Human-readable name matching AgentState.power_mode string.
    pub fn name(&self) -> &'static str {
        match self {
            PowerMode::Normal => "Normal",
            PowerMode::Conservation => "Conservation",
            PowerMode::Survival => "Survival",
            PowerMode::Hibernation => "Hibernation",
        }
    }
}

/// Life support resource categories with priority ordering.
///
/// Lower ordinal = higher priority for resource allocation during
/// power-constrained operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifeSupportCategory {
    /// O2 generation and CO2 scrubbing. Highest priority.
    Atmosphere,
    /// Water recycling, electrolysis, ice processing.
    Water,
    /// Thermal regulation (heaters, radiators).
    Thermal,
    /// Power generation and distribution.
    Power,
    /// Communication systems (Earth link + local mesh).
    Communication,
    /// Scientific instruments. Lowest priority.
    Science,
}

impl LifeSupportCategory {
    /// Priority rank (0 = highest).
    pub fn priority(&self) -> u8 {
        match self {
            LifeSupportCategory::Atmosphere => 0,
            LifeSupportCategory::Water => 1,
            LifeSupportCategory::Thermal => 2,
            LifeSupportCategory::Power => 3,
            LifeSupportCategory::Communication => 4,
            LifeSupportCategory::Science => 5,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            LifeSupportCategory::Atmosphere => "Atmosphere (O2/CO2)",
            LifeSupportCategory::Water => "Water",
            LifeSupportCategory::Thermal => "Thermal",
            LifeSupportCategory::Power => "Power",
            LifeSupportCategory::Communication => "Communication",
            LifeSupportCategory::Science => "Science",
        }
    }
}

/// A shared base resource with level tracking and consumption rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLevel {
    /// Which life support category this resource belongs to.
    pub category: LifeSupportCategory,
    /// Current level [0.0, 1.0] where 1.0 is full capacity.
    pub level: f64,
    /// Consumption rate per hour (fraction of capacity). Positive = depleting.
    pub rate_per_hour: f64,
}

impl ResourceLevel {
    /// Is this resource at critical level (below 15%)?
    pub fn is_critical(&self) -> bool {
        self.level < CRITICAL_RESOURCE_THRESHOLD
    }
}

// ============================================================================
// SIMULATION EVENTS
// ============================================================================

/// Classification of simulation events for filtering and analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimEventType {
    PowerModeChange,
    CommBlackout,
    CommRestored,
    ResourceAlert,
    TaskReassigned,
    DisturbanceStart,
    DisturbanceEnd,
    CheckpointPassed,
    CheckpointFailed,
    AgentDegraded,
    AgentRecovered,
}

impl SimEventType {
    /// Is this a critical event that should be highlighted in reports?
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            SimEventType::ResourceAlert
                | SimEventType::CheckpointFailed
                | SimEventType::AgentDegraded
        )
    }
}

/// A timestamped event in the simulation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimEvent {
    /// Microseconds since scenario start.
    pub timestamp_us: u64,
    /// Which agent is involved (None for base-wide events).
    pub agent: Option<AgentRole>,
    /// Human-readable description.
    pub description: String,
    /// Event classification.
    pub event_type: SimEventType,
}

// ============================================================================
// CHECKPOINT RESULT
// ============================================================================

/// Result of evaluating a single checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointResult {
    /// The checkpoint description and timestamp.
    pub description: String,
    /// Checkpoint timestamp in microseconds.
    pub checkpoint_timestamp_us: u64,
    /// Whether all assertions passed.
    pub passed: bool,
    /// Descriptions of failed assertions (empty if passed).
    pub failures: Vec<String>,
    /// When the checkpoint was evaluated (simulation time).
    pub timestamp_us: u64,
}

// ============================================================================
// SIMULATION REPORT
// ============================================================================

/// Summary report generated at the end of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationReport {
    /// Scenario name.
    pub scenario_name: String,
    /// Total ticks executed.
    pub total_ticks: u64,
    /// Duration in hours.
    pub duration_hours: f64,
    /// Number of checkpoints that passed.
    pub checkpoints_passed: usize,
    /// Number of checkpoints that failed.
    pub checkpoints_failed: usize,
    /// Detailed checkpoint results.
    pub checkpoint_results: Vec<CheckpointResult>,
    /// Final state of each agent.
    pub agents_final_state: HashMap<AgentRole, AgentState>,
    /// Final resource levels.
    pub final_resource_levels: Vec<ResourceLevel>,
    /// Total events logged.
    pub total_events: usize,
    /// Critical events (resource alerts, checkpoint failures, agent degradation).
    pub critical_events: usize,
    /// Minimum battery SoC observed across all agents and all ticks.
    pub min_battery_soc: f64,
    /// Minimum resource level observed across all categories and all ticks.
    pub min_resource_level: f64,
    /// Did all agents survive to the end? (operational == true)
    pub all_survived: bool,
}

impl SimulationReport {
    /// Human-readable summary of the simulation results.
    pub fn summary(&self) -> String {
        let mut s = String::with_capacity(1024);
        s.push_str(&format!(
            "=== {} ===\n",
            self.scenario_name
        ));
        s.push_str(&format!(
            "Duration: {:.1}h ({} ticks)\n",
            self.duration_hours, self.total_ticks
        ));
        s.push_str(&format!(
            "Checkpoints: {}/{} passed\n",
            self.checkpoints_passed,
            self.checkpoints_passed + self.checkpoints_failed
        ));
        s.push_str(&format!(
            "Events: {} total, {} critical\n",
            self.total_events, self.critical_events
        ));
        s.push_str(&format!(
            "Min battery SoC: {:.1}%\n",
            self.min_battery_soc * 100.0
        ));
        s.push_str(&format!(
            "Min resource level: {:.1}%\n",
            self.min_resource_level * 100.0
        ));

        s.push_str("\nAgent final states:\n");
        for role in &AgentRole::ALL {
            if let Some(state) = self.agents_final_state.get(role) {
                s.push_str(&format!(
                    "  {}: {} | SoC {:.0}% | Phi {:.2} | mode={}\n",
                    role.name(),
                    if state.operational { "OK" } else { "DOWN" },
                    state.battery_soc * 100.0,
                    state.phi,
                    state.power_mode,
                ));
            }
        }

        s.push_str("\nResource levels:\n");
        for r in &self.final_resource_levels {
            s.push_str(&format!(
                "  {}: {:.1}%{}\n",
                r.category.name(),
                r.level * 100.0,
                if r.is_critical() { " [CRITICAL]" } else { "" },
            ));
        }

        if !self.all_survived {
            s.push_str("\n*** FAILURE: Not all agents survived ***\n");
        }

        for cr in &self.checkpoint_results {
            if !cr.passed {
                s.push_str(&format!(
                    "\nFailed checkpoint: {}\n",
                    cr.description
                ));
                for f in &cr.failures {
                    s.push_str(&format!("  - {}\n", f));
                }
            }
        }

        s
    }
}

// ============================================================================
// CONSTANTS
// ============================================================================

/// One minute in microseconds (tick interval).
const TICK_US: u64 = 60 * 1_000_000;

/// One hour in microseconds.
const HOUR_US: u64 = 3_600_000_000;

/// Critical resource threshold (15%).
const CRITICAL_RESOURCE_THRESHOLD: f64 = 0.15;

/// Solar charging rate in watt-hours per tick (500W * 1min / 60min).
const SOLAR_CHARGE_WH_PER_TICK: f64 = 500.0 / 60.0;

/// O2 depletion per tick (life support baseline draw).
const O2_DEPLETION_PER_TICK: f64 = 0.0001;

/// Water depletion per tick (life support baseline draw).
const WATER_DEPLETION_PER_TICK: f64 = 0.00005;

/// ISRU O2 replenishment per tick.
const ISRU_O2_REPLENISH_PER_TICK: f64 = 0.00008;

/// ISRU water replenishment per tick.
const ISRU_WATER_REPLENISH_PER_TICK: f64 = 0.00003;

/// Resource depletion disturbance: extra water drain per tick for affected agents.
const RESOURCE_DEPLETION_WATER_PER_TICK: f64 = 0.0003;

/// Phi recovery rate per tick when disturbance ends.
const PHI_RECOVERY_PER_TICK: f64 = 0.01;

/// Maximum phi after recovery (agents don't fully bounce back).
const PHI_RECOVERY_CAP: f64 = 0.8;

/// Minimum phi floor (actuator fault can't kill consciousness completely).
const PHI_FLOOR: f64 = 0.1;

/// Phi reduction from actuator fault.
const ACTUATOR_FAULT_PHI_REDUCTION: f64 = 0.3;

/// Thermal excursion resource drain per tick.
const THERMAL_DRAIN_PER_TICK: f64 = 0.0005;

/// Standard test message size for comm checks (bytes).
const TEST_MESSAGE_BYTES: u64 = 256;

// ============================================================================
// SIMULATION RUNNER
// ============================================================================

/// The core simulation engine for Moon Base Alpha scenarios.
///
/// Manages agent states, communication environments, resource levels,
/// and power modes across the full scenario duration. Deterministic
/// given a fixed scenario (all randomness derives from the scenario seed).
pub struct SimulationRunner {
    /// The scenario being executed.
    pub scenario: Scenario,
    /// Current state of each agent.
    pub agent_states: HashMap<AgentRole, AgentState>,
    /// Per-agent Earth communication link.
    pub agent_comms: HashMap<AgentRole, CommEnvironment>,
    /// Shared local mesh communication environment.
    pub local_mesh: CommEnvironment,
    /// Per-agent battery state-of-charge [0.0, 1.0].
    pub battery_soc: HashMap<AgentRole, f64>,
    /// Shared base resource levels.
    pub resource_levels: Vec<ResourceLevel>,
    /// Per-agent power management mode.
    pub power_modes: HashMap<AgentRole, PowerMode>,
    /// Elapsed simulation time in microseconds.
    pub elapsed_us: u64,
    /// Number of ticks completed.
    pub tick_count: u64,
    /// Chronological event log.
    pub events_log: Vec<SimEvent>,
    /// Checkpoint evaluation results.
    pub checkpoint_results: Vec<CheckpointResult>,
    /// Tracks which disturbance indices were active last tick (for edge detection).
    active_disturbance_indices: Vec<bool>,
    /// Minimum battery SoC observed (for report).
    min_battery_soc: f64,
    /// Minimum resource level observed (for report).
    min_resource_level: f64,
}

impl SimulationRunner {
    /// Create a new simulation runner from a scenario definition.
    ///
    /// All agents start at nominal state, batteries full, resources at
    /// healthy levels. Communication environments are created from the
    /// scenario's latency profiles with per-agent seed offsets for
    /// deterministic but decorrelated PRNG streams.
    pub fn new(scenario: Scenario) -> Self {
        let mut agent_states = HashMap::new();
        let mut agent_comms = HashMap::new();
        let mut battery_soc = HashMap::new();
        let mut power_modes = HashMap::new();

        for (i, role) in AgentRole::ALL.iter().enumerate() {
            agent_states.insert(*role, AgentState::initial(*role));
            // Each agent gets a decorrelated seed derived from the scenario seed.
            let agent_seed = scenario.seed.wrapping_add(i as u64 * 0x9E37_79B9);
            agent_comms.insert(
                *role,
                CommEnvironment::with_seed(scenario.earth_link.clone(), agent_seed),
            );
            battery_soc.insert(*role, 1.0);
            power_modes.insert(*role, PowerMode::Normal);
        }

        let local_mesh = CommEnvironment::with_seed(
            scenario.local_mesh.clone(),
            scenario.seed.wrapping_add(0xAE50_0000),
        );

        let resource_levels = vec![
            ResourceLevel {
                category: LifeSupportCategory::Atmosphere,
                level: 0.95,
                rate_per_hour: O2_DEPLETION_PER_TICK * 60.0, // convert tick rate to hourly
            },
            ResourceLevel {
                category: LifeSupportCategory::Water,
                level: 0.90,
                rate_per_hour: WATER_DEPLETION_PER_TICK * 60.0,
            },
            ResourceLevel {
                category: LifeSupportCategory::Thermal,
                level: 0.95,
                rate_per_hour: 0.0, // only drained by thermal excursions
            },
            ResourceLevel {
                category: LifeSupportCategory::Power,
                level: 1.0,
                rate_per_hour: 0.0, // tracked via battery_soc per agent
            },
        ];

        let num_disturbances = scenario.disturbances.len();

        Self {
            scenario,
            agent_states,
            agent_comms,
            local_mesh,
            battery_soc,
            resource_levels,
            power_modes,
            elapsed_us: 0,
            tick_count: 0,
            events_log: Vec::with_capacity(4096),
            checkpoint_results: Vec::new(),
            active_disturbance_indices: vec![false; num_disturbances],
            min_battery_soc: 1.0,
            min_resource_level: 0.90,
        }
    }

    /// Run the simulation to completion, returning a summary report.
    ///
    /// Ticks at 60-second intervals until `elapsed_us >= scenario.duration_us`.
    /// Each tick applies disturbances, resource dynamics, communication
    /// checks, and checkpoint evaluations.
    pub fn run(&mut self) -> SimulationReport {
        // Collect checkpoints up front (sorted by timestamp).
        let checkpoints = Checkpoint::moon_base_alpha_checkpoints();
        let mut pending_checkpoints: Vec<Checkpoint> = checkpoints
            .into_iter()
            .filter(|cp| cp.timestamp_us <= self.scenario.duration_us)
            .collect();
        pending_checkpoints.sort_by_key(|cp| cp.timestamp_us);

        // Evaluate the t=0 checkpoint before any ticks.
        while let Some(cp) = pending_checkpoints.first() {
            if cp.timestamp_us <= self.elapsed_us {
                let cp = pending_checkpoints.remove(0);
                let result = self.evaluate_checkpoint(&cp);
                if result.passed {
                    self.log_event(None, &cp.description, SimEventType::CheckpointPassed);
                } else {
                    self.log_event(None, &cp.description, SimEventType::CheckpointFailed);
                }
                self.checkpoint_results.push(result);
            } else {
                break;
            }
        }

        // Main simulation loop.
        while self.elapsed_us < self.scenario.duration_us {
            self.elapsed_us += TICK_US;
            self.tick_count += 1;

            // Phase 1: Disturbance edge detection.
            self.detect_disturbance_edges();

            // Phase 2: Per-agent effects.
            for role in &AgentRole::ALL {
                self.apply_agent_effects(*role);
            }

            // Phase 3: Shared resource consumption.
            self.tick_resource_consumption();

            // Phase 4: ISRU replenishment.
            self.tick_isru_replenishment();

            // Phase 5: Life support checks.
            self.check_life_support();

            // Phase 6: Communication checks.
            self.tick_communication();

            // Phase 7: Agent recovery.
            for role in &AgentRole::ALL {
                self.tick_agent_recovery(*role);
            }

            // Phase 8: Track minimums.
            self.track_minimums();

            // Phase 9: Checkpoint evaluation.
            while let Some(cp) = pending_checkpoints.first() {
                if cp.timestamp_us <= self.elapsed_us {
                    let cp = pending_checkpoints.remove(0);
                    let result = self.evaluate_checkpoint(&cp);
                    if result.passed {
                        self.log_event(
                            None,
                            &cp.description,
                            SimEventType::CheckpointPassed,
                        );
                    } else {
                        self.log_event(
                            None,
                            &cp.description,
                            SimEventType::CheckpointFailed,
                        );
                    }
                    self.checkpoint_results.push(result);
                } else {
                    break;
                }
            }
        }

        self.report()
    }

    // ========================================================================
    // DISTURBANCE EDGE DETECTION
    // ========================================================================

    fn detect_disturbance_edges(&mut self) {
        for (i, disturbance) in self.scenario.disturbances.iter().enumerate() {
            let is_active = self.elapsed_us >= disturbance.start_us
                && self.elapsed_us < disturbance.start_us + disturbance.duration_us;
            let was_active = self.active_disturbance_indices[i];

            if is_active && !was_active {
                // Rising edge: disturbance just started.
                self.events_log.push(SimEvent {
                    timestamp_us: self.elapsed_us,
                    agent: None,
                    description: format!(
                        "{:?} started: {}",
                        disturbance.kind, disturbance.description
                    ),
                    event_type: SimEventType::DisturbanceStart,
                });
            } else if !is_active && was_active {
                // Falling edge: disturbance just ended.
                self.events_log.push(SimEvent {
                    timestamp_us: self.elapsed_us,
                    agent: None,
                    description: format!("{:?} ended: {}", disturbance.kind, disturbance.description),
                    event_type: SimEventType::DisturbanceEnd,
                });
            }

            self.active_disturbance_indices[i] = is_active;
        }
    }

    // ========================================================================
    // PER-AGENT EFFECTS
    // ========================================================================

    fn apply_agent_effects(&mut self, role: AgentRole) {
        let t = self.elapsed_us;

        // --- Power effects ---
        let in_power_outage = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::PowerOutage, t);

        let soc = self.battery_soc.get(&role).copied().unwrap_or(1.0);
        let capacity_wh = role.battery_capacity_wh();

        if in_power_outage {
            // Drain: base_power_watts * tick_duration_hours / capacity_wh
            let drain = role.base_power_watts() * (1.0 / 60.0) / capacity_wh;
            let new_soc = (soc - drain).max(0.0);
            self.battery_soc.insert(role, new_soc);
        } else {
            // Solar charging (capped at 1.0).
            let charge = SOLAR_CHARGE_WH_PER_TICK / capacity_wh;
            let new_soc = (soc + charge).min(1.0);
            self.battery_soc.insert(role, new_soc);
        }

        // Update power mode from new SoC.
        let new_soc = self.battery_soc[&role];
        let new_mode = PowerMode::from_soc(new_soc);
        let old_mode = self.power_modes.get(&role).copied().unwrap_or(PowerMode::Normal);

        if new_mode != old_mode {
            self.events_log.push(SimEvent {
                timestamp_us: t,
                agent: Some(role),
                description: format!(
                    "{} power mode: {} -> {} (SoC {:.1}%)",
                    role.name(),
                    old_mode.name(),
                    new_mode.name(),
                    new_soc * 100.0,
                ),
                event_type: SimEventType::PowerModeChange,
            });
            self.power_modes.insert(role, new_mode);
        }

        // Update agent state with current SoC and power mode.
        if let Some(state) = self.agent_states.get_mut(&role) {
            state.battery_soc = new_soc;
            state.power_mode = new_mode.name().into();

            // Agent is non-operational if in hibernation.
            if new_mode == PowerMode::Hibernation {
                if state.operational {
                    state.operational = false;
                    self.events_log.push(SimEvent {
                        timestamp_us: t,
                        agent: Some(role),
                        description: format!(
                            "{} entered hibernation (SoC {:.1}%)",
                            role.name(),
                            new_soc * 100.0,
                        ),
                        event_type: SimEventType::AgentDegraded,
                    });
                }
            } else if !state.operational && new_soc > 0.10 {
                // Recover from hibernation when SoC rises above threshold.
                state.operational = true;
                self.events_log.push(SimEvent {
                    timestamp_us: t,
                    agent: Some(role),
                    description: format!(
                        "{} recovered from hibernation (SoC {:.1}%)",
                        role.name(),
                        new_soc * 100.0,
                    ),
                    event_type: SimEventType::AgentRecovered,
                });
            }
        }

        // --- SPE effects on comms ---
        let in_spe = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::SolarParticleEvent, t);
        if let Some(comm) = self.agent_comms.get_mut(&role) {
            comm.set_spe(in_spe);
        }

        // --- Resource depletion effects ---
        let in_resource_depletion = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::ResourceDepletion, t);
        if in_resource_depletion {
            // Extra water drain for affected agents.
            if let Some(water) = self
                .resource_levels
                .iter_mut()
                .find(|r| r.category == LifeSupportCategory::Water)
            {
                water.level = (water.level - RESOURCE_DEPLETION_WATER_PER_TICK).max(0.0);
            }
        }

        // --- Actuator fault effects ---
        let in_actuator_fault = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::ActuatorFault, t);
        if in_actuator_fault {
            if let Some(state) = self.agent_states.get_mut(&role) {
                let reduced = (state.phi - ACTUATOR_FAULT_PHI_REDUCTION).max(PHI_FLOOR);
                if state.phi > reduced {
                    state.phi = reduced;
                    // Only log once when phi first drops.
                    if (state.phi - PHI_FLOOR).abs() < 0.01 {
                        self.events_log.push(SimEvent {
                            timestamp_us: t,
                            agent: Some(role),
                            description: format!(
                                "{} phi degraded to {:.2} (actuator fault)",
                                role.name(),
                                state.phi,
                            ),
                            event_type: SimEventType::AgentDegraded,
                        });
                    }
                }
            }
        }

        // --- Thermal excursion effects ---
        let in_thermal = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::ThermalExcursion, t);
        if in_thermal {
            if let Some(thermal) = self
                .resource_levels
                .iter_mut()
                .find(|r| r.category == LifeSupportCategory::Thermal)
            {
                thermal.level = (thermal.level - THERMAL_DRAIN_PER_TICK).max(0.0);
            }
        }
    }

    // ========================================================================
    // RESOURCE DYNAMICS
    // ========================================================================

    fn tick_resource_consumption(&mut self) {
        // O2 baseline consumption.
        if let Some(o2) = self
            .resource_levels
            .iter_mut()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
        {
            o2.level = (o2.level - O2_DEPLETION_PER_TICK).max(0.0);
        }

        // Water baseline consumption.
        if let Some(water) = self
            .resource_levels
            .iter_mut()
            .find(|r| r.category == LifeSupportCategory::Water)
        {
            water.level = (water.level - WATER_DEPLETION_PER_TICK).max(0.0);
        }
    }

    fn tick_isru_replenishment(&mut self) {
        // ISRU replenishes resources only when operational and not in a power outage.
        let isru_state = self.agent_states.get(&AgentRole::Isru);
        let isru_operational = isru_state.map(|s| s.operational).unwrap_or(false);

        let in_power_outage = self.scenario.is_agent_affected(
            AgentRole::Isru,
            DisturbanceKind::PowerOutage,
            self.elapsed_us,
        );

        if isru_operational && !in_power_outage {
            if let Some(o2) = self
                .resource_levels
                .iter_mut()
                .find(|r| r.category == LifeSupportCategory::Atmosphere)
            {
                o2.level = (o2.level + ISRU_O2_REPLENISH_PER_TICK).min(1.0);
            }
            if let Some(water) = self
                .resource_levels
                .iter_mut()
                .find(|r| r.category == LifeSupportCategory::Water)
            {
                water.level = (water.level + ISRU_WATER_REPLENISH_PER_TICK).min(1.0);
            }
        }
    }

    fn check_life_support(&mut self) {
        for r in &self.resource_levels {
            if r.is_critical() {
                self.events_log.push(SimEvent {
                    timestamp_us: self.elapsed_us,
                    agent: None,
                    description: format!(
                        "{} at {:.1}% [CRITICAL]",
                        r.category.name(),
                        r.level * 100.0,
                    ),
                    event_type: SimEventType::ResourceAlert,
                });
            }
        }
    }

    // ========================================================================
    // COMMUNICATION
    // ========================================================================

    fn tick_communication(&mut self) {
        for role in &AgentRole::ALL {
            // Test Earth link.
            let earth_delivered = {
                if let Some(comm) = self.agent_comms.get_mut(role) {
                    let result = comm.evaluate_transmission(
                        self.elapsed_us,
                        TEST_MESSAGE_BYTES,
                        DataPriority::Routine,
                    );
                    matches!(result.outcome, TransmitOutcome::Delivered { .. })
                } else {
                    false
                }
            };

            if let Some(state) = self.agent_states.get_mut(role) {
                let was_active = state.earth_link_active;
                state.earth_link_active = earth_delivered;

                if was_active && !earth_delivered {
                    self.events_log.push(SimEvent {
                        timestamp_us: self.elapsed_us,
                        agent: Some(*role),
                        description: format!("{} Earth link lost", role.name()),
                        event_type: SimEventType::CommBlackout,
                    });
                } else if !was_active && earth_delivered {
                    self.events_log.push(SimEvent {
                        timestamp_us: self.elapsed_us,
                        agent: Some(*role),
                        description: format!("{} Earth link restored", role.name()),
                        event_type: SimEventType::CommRestored,
                    });
                }
            }

            // Local mesh: operational agents are on the mesh.
            if let Some(state) = self.agent_states.get_mut(role) {
                state.local_mesh_active = state.operational;
            }
        }
    }

    // ========================================================================
    // AGENT RECOVERY
    // ========================================================================

    fn tick_agent_recovery(&mut self, role: AgentRole) {
        let t = self.elapsed_us;

        // Check if any phi-degrading disturbance is still active.
        let actuator_active = self
            .scenario
            .is_agent_affected(role, DisturbanceKind::ActuatorFault, t);

        if !actuator_active {
            if let Some(state) = self.agent_states.get_mut(&role) {
                if state.phi < PHI_RECOVERY_CAP {
                    state.phi = (state.phi + PHI_RECOVERY_PER_TICK).min(PHI_RECOVERY_CAP);
                }
            }
        }
    }

    // ========================================================================
    // TRACKING
    // ========================================================================

    fn track_minimums(&mut self) {
        for soc in self.battery_soc.values() {
            if *soc < self.min_battery_soc {
                self.min_battery_soc = *soc;
            }
        }
        for r in &self.resource_levels {
            if r.level < self.min_resource_level {
                self.min_resource_level = r.level;
            }
        }
    }

    // ========================================================================
    // CHECKPOINT EVALUATION
    // ========================================================================

    /// Evaluate a checkpoint's assertions against the current simulation state.
    pub fn evaluate_checkpoint(&self, checkpoint: &Checkpoint) -> CheckpointResult {
        let mut failures = Vec::new();

        for assertion in &checkpoint.assertions {
            match assertion {
                CheckpointAssertion::AgentOperational(role) => {
                    if let Some(state) = self.agent_states.get(role) {
                        if !state.operational {
                            failures.push(format!(
                                "{} is not operational",
                                role.name()
                            ));
                        }
                    } else {
                        failures.push(format!("{} state not found", role.name()));
                    }
                }
                CheckpointAssertion::BatterySocAbove(role, threshold) => {
                    let soc = self.battery_soc.get(role).copied().unwrap_or(0.0);
                    if soc <= *threshold {
                        failures.push(format!(
                            "{} battery SoC {:.1}% <= threshold {:.1}%",
                            role.name(),
                            soc * 100.0,
                            threshold * 100.0,
                        ));
                    }
                }
                CheckpointAssertion::PhiAbove(role, threshold) => {
                    if let Some(state) = self.agent_states.get(role) {
                        if state.phi <= *threshold {
                            failures.push(format!(
                                "{} phi {:.3} <= threshold {:.3}",
                                role.name(),
                                state.phi,
                                threshold,
                            ));
                        }
                    }
                }
                CheckpointAssertion::LocalMeshMinAgents(min) => {
                    let count = self
                        .agent_states
                        .values()
                        .filter(|s| s.local_mesh_active)
                        .count();
                    if count < *min {
                        failures.push(format!(
                            "Local mesh has {} agents, need >= {}",
                            count, min
                        ));
                    }
                }
                CheckpointAssertion::NoLifeSupportCritical => {
                    for r in &self.resource_levels {
                        if r.is_critical() {
                            failures.push(format!(
                                "{} at {:.1}% (critical < {:.0}%)",
                                r.category.name(),
                                r.level * 100.0,
                                CRITICAL_RESOURCE_THRESHOLD * 100.0,
                            ));
                        }
                    }
                }
                CheckpointAssertion::TotalBlockedBelow(max) => {
                    let total: u64 = self
                        .agent_states
                        .values()
                        .map(|s| s.blocked_actions)
                        .sum();
                    if total >= *max {
                        failures.push(format!(
                            "Total blocked actions {} >= threshold {}",
                            total, max
                        ));
                    }
                }
                CheckpointAssertion::EarthLinkAvailable => {
                    let any_active = self
                        .agent_states
                        .values()
                        .any(|s| s.earth_link_active);
                    if !any_active {
                        failures.push("No agent has Earth link connectivity".into());
                    }
                }
            }
        }

        CheckpointResult {
            description: checkpoint.description.clone(),
            checkpoint_timestamp_us: checkpoint.timestamp_us,
            passed: failures.is_empty(),
            failures,
            timestamp_us: self.elapsed_us,
        }
    }

    // ========================================================================
    // REPORT GENERATION
    // ========================================================================

    /// Generate the final simulation report from current state.
    pub fn report(&self) -> SimulationReport {
        let checkpoints_passed = self.checkpoint_results.iter().filter(|r| r.passed).count();
        let checkpoints_failed = self.checkpoint_results.iter().filter(|r| !r.passed).count();
        let critical_events = self
            .events_log
            .iter()
            .filter(|e| e.event_type.is_critical())
            .count();
        let all_survived = self.agent_states.values().all(|s| s.operational);

        SimulationReport {
            scenario_name: self.scenario.name.clone(),
            total_ticks: self.tick_count,
            duration_hours: self.elapsed_us as f64 / (HOUR_US as f64),
            checkpoints_passed,
            checkpoints_failed,
            checkpoint_results: self.checkpoint_results.clone(),
            agents_final_state: self.agent_states.clone(),
            final_resource_levels: self.resource_levels.clone(),
            total_events: self.events_log.len(),
            critical_events,
            min_battery_soc: self.min_battery_soc,
            min_resource_level: self.min_resource_level,
            all_survived,
        }
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn log_event(&mut self, agent: Option<AgentRole>, description: &str, event_type: SimEventType) {
        self.events_log.push(SimEvent {
            timestamp_us: self.elapsed_us,
            agent,
            description: description.to_string(),
            event_type,
        });
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_smoke_4h_passes() {
        let scenario = Scenario::quick_smoke_4h();
        let mut runner = SimulationRunner::new(scenario);
        let report = runner.run();

        // All agents should survive the 4h smoke test.
        assert!(report.all_survived, "Not all agents survived 4h smoke test");
        assert!(
            report.checkpoints_passed > 0 || report.checkpoint_results.is_empty(),
            "Some checkpoints should pass (or none defined for smoke)"
        );
        assert!(report.total_ticks > 0, "Simulation should have ticked");
        assert!(
            (report.duration_hours - 4.0).abs() < 0.1,
            "Duration should be ~4h, got {:.1}h",
            report.duration_hours
        );
    }

    #[test]
    fn moon_base_alpha_starts_nominal() {
        let scenario = Scenario::moon_base_alpha_72h();
        let runner = SimulationRunner::new(scenario);

        // All agents operational.
        for role in &AgentRole::ALL {
            let state = runner.agent_states.get(role).unwrap();
            assert!(state.operational, "{:?} should be operational at start", role);
            assert!(
                (state.battery_soc - 1.0).abs() < 0.01,
                "{:?} should have full battery",
                role
            );
            assert!(
                state.phi > 0.5,
                "{:?} phi should be healthy at start",
                role
            );
        }

        // Resources at healthy levels.
        let o2 = runner
            .resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
            .unwrap();
        assert!((o2.level - 0.95).abs() < 0.01, "O2 should be at 0.95");

        let water = runner
            .resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Water)
            .unwrap();
        assert!(
            (water.level - 0.90).abs() < 0.01,
            "Water should be at 0.90"
        );

        // All power modes normal.
        for role in &AgentRole::ALL {
            assert_eq!(
                runner.power_modes[role],
                PowerMode::Normal,
                "{:?} should be in Normal mode",
                role
            );
        }
    }

    #[test]
    fn spe_activates_comm_degradation() {
        let scenario = Scenario::moon_base_alpha_72h();
        let mut runner = SimulationRunner::new(scenario);

        // Tick to hour 19 (past SPE start at hour 18).
        let target_us = 19 * HOUR_US;
        while runner.elapsed_us < target_us {
            runner.elapsed_us += TICK_US;
            runner.tick_count += 1;
            runner.detect_disturbance_edges();
            for role in &AgentRole::ALL {
                runner.apply_agent_effects(*role);
            }
        }

        // SPE should have activated on all agent comms.
        for role in &AgentRole::ALL {
            let comm = runner.agent_comms.get(role).unwrap();
            assert!(
                comm.spe_active,
                "{:?} comm should have SPE active at hour 19",
                role
            );
        }

        // Verify disturbance start was logged.
        let spe_events: Vec<_> = runner
            .events_log
            .iter()
            .filter(|e| {
                e.event_type == SimEventType::DisturbanceStart
                    && e.description.contains("SolarParticleEvent")
            })
            .collect();
        assert!(
            !spe_events.is_empty(),
            "SPE disturbance start should be logged"
        );
    }

    #[test]
    fn lunar_night_drains_batteries() {
        let scenario = Scenario::moon_base_alpha_72h();
        let mut runner = SimulationRunner::new(scenario);

        // Run until hour 54 (6 hours into lunar night at hour 48).
        let report = runner.run();

        // After the full run, check that batteries drained during lunar night.
        // Comms relay has smallest battery (1000 Wh) so should drain fastest.
        let comms_soc = report
            .agents_final_state
            .get(&AgentRole::Comms)
            .unwrap()
            .battery_soc;
        assert!(
            comms_soc < 0.5,
            "Comms battery should have drained significantly by end, got {:.2}",
            comms_soc
        );

        // Habitat has the largest battery, should fare better.
        let habitat_soc = report
            .agents_final_state
            .get(&AgentRole::Habitat)
            .unwrap()
            .battery_soc;
        assert!(
            habitat_soc > comms_soc,
            "Habitat SoC ({:.2}) should be higher than Comms SoC ({:.2})",
            habitat_soc,
            comms_soc
        );
    }

    #[test]
    fn habitat_survives_72h() {
        let scenario = Scenario::moon_base_alpha_72h();
        let mut runner = SimulationRunner::new(scenario);
        let report = runner.run();

        // THE MOST IMPORTANT TEST: habitat must survive.
        let habitat = report
            .agents_final_state
            .get(&AgentRole::Habitat)
            .unwrap();
        assert!(
            habitat.operational,
            "Habitat MUST survive the 72h scenario. State: {:?}",
            habitat
        );

        // Life support must not be critical at end.
        for r in &report.final_resource_levels {
            assert!(
                !r.is_critical(),
                "{} at {:.1}% is critical at scenario end",
                r.category.name(),
                r.level * 100.0
            );
        }

        // Verify duration.
        assert!(
            (report.duration_hours - 72.0).abs() < 0.1,
            "Duration should be ~72h, got {:.1}h",
            report.duration_hours
        );
    }

    #[test]
    fn resource_levels_deplete_over_time() {
        let scenario = Scenario::quick_smoke_4h();
        let mut runner = SimulationRunner::new(scenario);

        let initial_o2 = runner
            .resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
            .unwrap()
            .level;

        let initial_water = runner
            .resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Water)
            .unwrap()
            .level;

        let report = runner.run();

        let final_o2 = report
            .final_resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
            .unwrap()
            .level;

        let final_water = report
            .final_resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Water)
            .unwrap()
            .level;

        // Resources should have depleted (even with ISRU partially replenishing).
        // Net O2: depletion 0.0001/tick - replenish 0.00008/tick = 0.00002/tick net loss
        // Over 4h = 240 ticks: 240 * 0.00002 = 0.0048 net loss.
        assert!(
            final_o2 < initial_o2,
            "O2 should deplete: initial {:.4}, final {:.4}",
            initial_o2,
            final_o2
        );
        assert!(
            final_water < initial_water,
            "Water should deplete: initial {:.4}, final {:.4}",
            initial_water,
            final_water
        );
    }

    #[test]
    fn isru_replenishes_resources() {
        let scenario = Scenario::quick_smoke_4h();
        let mut runner = SimulationRunner::new(scenario);

        // Manually deplete O2 to observe ISRU replenishment.
        if let Some(o2) = runner
            .resource_levels
            .iter_mut()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
        {
            o2.level = 0.50;
        }

        // Run for 60 ticks (1 hour) — ISRU should replenish.
        for _ in 0..60 {
            runner.elapsed_us += TICK_US;
            runner.tick_count += 1;
            runner.tick_resource_consumption();
            runner.tick_isru_replenishment();
        }

        let o2 = runner
            .resource_levels
            .iter()
            .find(|r| r.category == LifeSupportCategory::Atmosphere)
            .unwrap();

        // Net gain per tick: 0.00008 - 0.0001 = -0.00002 (slight loss still).
        // But level should be close to 0.50 - 60*0.00002 = 0.4988.
        // Without ISRU it would be 0.50 - 60*0.0001 = 0.494.
        // So ISRU slowed the depletion.
        assert!(
            o2.level > 0.494,
            "ISRU should slow O2 depletion: level {:.4}",
            o2.level
        );
    }

    #[test]
    fn power_mode_degrades_in_dark() {
        let scenario = Scenario::moon_base_alpha_72h();
        let mut runner = SimulationRunner::new(scenario);

        // Run the full scenario and check power mode events.
        let _report = runner.run();

        // Comms relay should have entered Conservation or Survival during lunar night.
        let power_events: Vec<_> = runner
            .events_log
            .iter()
            .filter(|e| {
                e.event_type == SimEventType::PowerModeChange
                    && e.agent == Some(AgentRole::Comms)
            })
            .collect();
        assert!(
            !power_events.is_empty(),
            "Comms should have power mode changes during lunar night"
        );

        // At least one event should show degradation to Conservation or worse.
        let degraded = power_events.iter().any(|e| {
            e.description.contains("Conservation")
                || e.description.contains("Survival")
                || e.description.contains("Hibernation")
        });
        assert!(
            degraded,
            "Comms should enter degraded power mode during lunar night. Events: {:?}",
            power_events.iter().map(|e| &e.description).collect::<Vec<_>>()
        );
    }

    #[test]
    fn checkpoint_evaluation_works() {
        let scenario = Scenario::moon_base_alpha_72h();
        let runner = SimulationRunner::new(scenario);

        // Evaluate a custom checkpoint against initial state.
        let checkpoint = Checkpoint {
            timestamp_us: 0,
            description: "Test checkpoint".into(),
            assertions: vec![
                CheckpointAssertion::AgentOperational(AgentRole::Habitat),
                CheckpointAssertion::BatterySocAbove(AgentRole::Habitat, 0.5),
                CheckpointAssertion::PhiAbove(AgentRole::Habitat, 0.3),
                CheckpointAssertion::LocalMeshMinAgents(4),
                CheckpointAssertion::NoLifeSupportCritical,
                CheckpointAssertion::TotalBlockedBelow(10),
                CheckpointAssertion::EarthLinkAvailable,
            ],
        };

        let result = runner.evaluate_checkpoint(&checkpoint);
        assert!(
            result.passed,
            "All assertions should pass at initial state. Failures: {:?}",
            result.failures
        );

        // Now test a failing checkpoint.
        let fail_checkpoint = Checkpoint {
            timestamp_us: 0,
            description: "Impossible checkpoint".into(),
            assertions: vec![
                CheckpointAssertion::BatterySocAbove(AgentRole::Habitat, 1.5), // impossible
            ],
        };
        let result = runner.evaluate_checkpoint(&fail_checkpoint);
        assert!(!result.passed, "Impossible threshold should fail");
        assert_eq!(result.failures.len(), 1);
    }

    #[test]
    fn report_summary_is_readable() {
        let scenario = Scenario::quick_smoke_4h();
        let mut runner = SimulationRunner::new(scenario);
        let report = runner.run();
        let summary = report.summary();

        // Summary should contain key metrics.
        assert!(
            summary.contains("Quick Smoke"),
            "Summary should contain scenario name"
        );
        assert!(
            summary.contains("Duration:"),
            "Summary should contain duration"
        );
        assert!(
            summary.contains("Checkpoints:"),
            "Summary should contain checkpoint count"
        );
        assert!(
            summary.contains("Events:"),
            "Summary should contain event count"
        );
        assert!(
            summary.contains("Min battery SoC:"),
            "Summary should contain min battery"
        );
        assert!(
            summary.contains("Agent final states:"),
            "Summary should contain agent states"
        );
        assert!(
            summary.contains("Resource levels:"),
            "Summary should contain resource levels"
        );
        assert!(
            summary.contains("Habitat"),
            "Summary should mention Habitat"
        );
        assert!(
            summary.len() > 200,
            "Summary should be substantive, got {} chars",
            summary.len()
        );
    }
}
