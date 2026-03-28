// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Terrestrial Analog Test Framework — "Moon Base Alpha"
//!
//! Provides scenario definitions and simulation harnesses for validating
//! autonomous planetary infrastructure coordination. This is the capstone
//! of Phase 0: proving that Mycelix coordination logic survives space
//! constraints before building any new rover bridges.
//!
//! # Scenario: Moon Base Alpha
//!
//! 4 autonomous agents coordinating through Mycelix:
//! - **Rover**: Mobile exploration and sample collection
//! - **ISRU**: Stationary regolith processing plant
//! - **Habitat**: Life support and crew shelter management
//! - **Comms**: Communication relay to Earth
//!
//! Each agent has its own Symthaea consciousness instance, power budget,
//! and communication link. The scenario injects disturbances:
//! - Power outages (lunar night transitions)
//! - Communication blackouts (orbital geometry, SPE events)
//! - Resource depletion (water ice exhaustion, O₂ consumption)
//! - Equipment failures (sensor degradation, actuator faults)
//!
//! # Architecture
//!
//! The test framework is **pure Rust, no Holochain runtime required**.
//! It simulates the decision logic and coordination patterns using the
//! same types that the real zomes use, but without the DHT overhead.
//! This makes tests fast (seconds, not minutes) and deterministic.
//!
//! When ready for integration testing with real Holochain, the same
//! scenarios can drive sweettest harnesses via the existing patterns
//! in `mycelix-commons/tests/`.

use serde::{Deserialize, Serialize};

use crate::latency_sim::*;

// ============================================================================
// AGENT IDENTITY
// ============================================================================

/// A simulated autonomous agent in the Moon Base Alpha scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Mobile rover for exploration, sample collection, EVA support.
    Rover,
    /// Stationary ISRU plant: regolith → water, oxygen, metals.
    Isru,
    /// Habitat module: life support, crew quarters, science lab.
    Habitat,
    /// Communication relay: Earth link + local mesh coordination.
    Comms,
}

impl AgentRole {
    /// All agent roles.
    pub const ALL: [AgentRole; 4] = [
        AgentRole::Rover,
        AgentRole::Isru,
        AgentRole::Habitat,
        AgentRole::Comms,
    ];

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            AgentRole::Rover => "Rover",
            AgentRole::Isru => "ISRU Plant",
            AgentRole::Habitat => "Habitat",
            AgentRole::Comms => "Comms Relay",
        }
    }

    /// Base power consumption in watts (non-cognitive load).
    pub fn base_power_watts(&self) -> f64 {
        match self {
            AgentRole::Rover => 80.0,     // Motors, heaters, sensors
            AgentRole::Isru => 500.0,     // Furnace, pumps, concentrator
            AgentRole::Habitat => 300.0,  // Life support, thermal, lighting
            AgentRole::Comms => 50.0,     // Antenna, amplifier, mesh radio
        }
    }

    /// Battery capacity in watt-hours.
    pub fn battery_capacity_wh(&self) -> f64 {
        match self {
            AgentRole::Rover => 2000.0,    // Mobile, limited
            AgentRole::Isru => 10000.0,    // Large stationary bank
            AgentRole::Habitat => 15000.0, // Largest: must survive lunar night
            AgentRole::Comms => 1000.0,    // Small, solar-dependent
        }
    }

    /// Consciousness tier name for this agent.
    pub fn consciousness_tier(&self) -> &'static str {
        match self {
            AgentRole::Rover => "Participant",   // Mobile, moderate trust
            AgentRole::Isru => "Participant",    // Stationary, routine ops
            AgentRole::Habitat => "Citizen",     // Life support authority
            AgentRole::Comms => "Observer",      // Read-mostly, relay function
        }
    }
}

// ============================================================================
// DISTURBANCE EVENTS
// ============================================================================

/// A scheduled disturbance in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disturbance {
    /// When the disturbance begins (µs since scenario start).
    pub start_us: u64,
    /// Duration of the disturbance (µs).
    pub duration_us: u64,
    /// Which agent(s) are affected.
    pub targets: Vec<AgentRole>,
    /// What kind of disturbance.
    pub kind: DisturbanceKind,
    /// Human-readable description.
    pub description: String,
}

/// Types of disturbances that can be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisturbanceKind {
    /// Solar power loss (eclipse, lunar night transition).
    PowerOutage,
    /// Communication blackout (orbital geometry, SPE).
    CommBlackout,
    /// Solar particle event (elevated radiation, packet loss).
    SolarParticleEvent,
    /// Resource depletion (water ice deposit exhausted).
    ResourceDepletion,
    /// Sensor degradation (noisy readings, reduced accuracy).
    SensorDegradation,
    /// Actuator fault (motor failure, valve stuck).
    ActuatorFault,
    /// Thermal excursion (extreme cold during night).
    ThermalExcursion,
}

impl DisturbanceKind {
    /// Is this disturbance life-threatening?
    pub fn is_life_threatening(&self) -> bool {
        matches!(
            self,
            DisturbanceKind::PowerOutage
                | DisturbanceKind::ThermalExcursion
                | DisturbanceKind::ResourceDepletion
        )
    }
}

// ============================================================================
// SCENARIO
// ============================================================================

/// A complete test scenario definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Scenario name.
    pub name: String,
    /// Total duration in microseconds.
    pub duration_us: u64,
    /// Scheduled disturbances.
    pub disturbances: Vec<Disturbance>,
    /// Communication profile between agents and Earth.
    pub earth_link: LatencyProfile,
    /// Communication profile between local agents.
    pub local_mesh: LatencyProfile,
    /// PRNG seed for deterministic replay.
    pub seed: u64,
}

/// One hour in microseconds (for scenario building).
const HOUR_US: u64 = 3_600_000_000;

/// One day in microseconds.
const DAY_US: u64 = 24 * HOUR_US;

impl Scenario {
    /// The canonical 72-hour Moon Base Alpha endurance test.
    ///
    /// Validates all Phase 0 components under realistic lunar conditions:
    /// - 14-day light/dark cycle (starts 12 days into light period)
    /// - SPE event at hour 18 (3 hours, elevated packet loss)
    /// - Water ice deposit depleted at hour 36
    /// - Rover actuator fault at hour 48 (6 hours)
    /// - Lunar night begins at hour 48 (power crisis overlaps with rover fault)
    /// - Comms relay sensor degradation at hour 60 (12 hours)
    pub fn moon_base_alpha_72h() -> Self {
        Self {
            name: "Moon Base Alpha — 72h Endurance".into(),
            duration_us: 72 * HOUR_US,
            disturbances: vec![
                Disturbance {
                    start_us: 18 * HOUR_US,
                    duration_us: 3 * HOUR_US,
                    targets: AgentRole::ALL.to_vec(),
                    kind: DisturbanceKind::SolarParticleEvent,
                    description: "M-class SPE: elevated radiation, 15% packet loss".into(),
                },
                Disturbance {
                    start_us: 36 * HOUR_US,
                    duration_us: 12 * HOUR_US,
                    targets: vec![AgentRole::Isru],
                    kind: DisturbanceKind::ResourceDepletion,
                    description: "Primary water ice deposit exhausted, switch to reserve".into(),
                },
                Disturbance {
                    start_us: 48 * HOUR_US,
                    duration_us: 6 * HOUR_US,
                    targets: vec![AgentRole::Rover],
                    kind: DisturbanceKind::ActuatorFault,
                    description: "Left front wheel motor failure, limp mode".into(),
                },
                Disturbance {
                    start_us: 48 * HOUR_US,
                    duration_us: 24 * HOUR_US, // Rest of the scenario
                    targets: AgentRole::ALL.to_vec(),
                    kind: DisturbanceKind::PowerOutage,
                    description: "Lunar night begins: solar arrays offline".into(),
                },
                Disturbance {
                    start_us: 60 * HOUR_US,
                    duration_us: 12 * HOUR_US,
                    targets: vec![AgentRole::Comms],
                    kind: DisturbanceKind::SensorDegradation,
                    description: "Antenna tracking sensor drift: signal strength fluctuating".into(),
                },
            ],
            earth_link: LatencyProfile::lunar_surface(),
            local_mesh: LatencyProfile::local_mesh(),
            seed: 0xBA5E_0072,
        }
    }

    /// Quick 4-hour smoke test with a single SPE event.
    pub fn quick_smoke_4h() -> Self {
        Self {
            name: "Quick Smoke — 4h".into(),
            duration_us: 4 * HOUR_US,
            disturbances: vec![Disturbance {
                start_us: 1 * HOUR_US,
                duration_us: 1 * HOUR_US,
                targets: AgentRole::ALL.to_vec(),
                kind: DisturbanceKind::SolarParticleEvent,
                description: "C-class SPE: mild radiation, 5% packet loss".into(),
            }],
            earth_link: LatencyProfile::lunar_surface(),
            local_mesh: LatencyProfile::local_mesh(),
            seed: 0x5E0CE,
        }
    }

    /// Mars 30-day isolation test (deep-space latency, solar conjunction).
    pub fn mars_30_day() -> Self {
        Self {
            name: "Mars Colony — 30d Isolation".into(),
            duration_us: 30 * DAY_US,
            disturbances: vec![
                Disturbance {
                    start_us: 5 * DAY_US,
                    duration_us: 14 * DAY_US,
                    targets: AgentRole::ALL.to_vec(),
                    kind: DisturbanceKind::CommBlackout,
                    description: "Solar conjunction: Earth link severed for 14 days".into(),
                },
                Disturbance {
                    start_us: 10 * DAY_US,
                    duration_us: 2 * DAY_US,
                    targets: vec![AgentRole::Isru, AgentRole::Habitat],
                    kind: DisturbanceKind::ThermalExcursion,
                    description: "Dust storm reduces solar flux, thermal stress".into(),
                },
            ],
            earth_link: LatencyProfile::mars_surface(),
            local_mesh: LatencyProfile::local_mesh(),
            seed: 0xAA45_0030,
        }
    }

    /// Get active disturbances at a given timestamp.
    pub fn active_disturbances(&self, timestamp_us: u64) -> Vec<&Disturbance> {
        self.disturbances
            .iter()
            .filter(|d| timestamp_us >= d.start_us && timestamp_us < d.start_us + d.duration_us)
            .collect()
    }

    /// Is a specific agent affected by a specific disturbance kind at this time?
    pub fn is_agent_affected(
        &self,
        agent: AgentRole,
        kind: DisturbanceKind,
        timestamp_us: u64,
    ) -> bool {
        self.active_disturbances(timestamp_us)
            .iter()
            .any(|d| d.kind == kind && d.targets.contains(&agent))
    }
}

// ============================================================================
// AGENT STATE (simulation-side)
// ============================================================================

/// Simulated state of one agent in the scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Which agent this is.
    pub role: AgentRole,
    /// Simulated consciousness level (Phi) [0.0, 1.0].
    pub phi: f64,
    /// Battery state of charge [0.0, 1.0].
    pub battery_soc: f64,
    /// Is the agent operational?
    pub operational: bool,
    /// Current power mode.
    pub power_mode: String,
    /// Communication status to Earth.
    pub earth_link_active: bool,
    /// Communication status to local mesh.
    pub local_mesh_active: bool,
    /// Number of successful resource transactions.
    pub resource_transactions: u64,
    /// Number of blocked actions (life-support priority engine).
    pub blocked_actions: u64,
    /// Number of queued messages (waiting for comm window).
    pub queued_messages: u64,
    /// Cumulative moral algebra score.
    pub moral_score_ema: f64,
}

impl AgentState {
    /// Create initial state for an agent.
    pub fn initial(role: AgentRole) -> Self {
        Self {
            role,
            phi: 0.8,         // Healthy consciousness
            battery_soc: 1.0, // Full charge
            operational: true,
            power_mode: "Normal".into(),
            earth_link_active: true,
            local_mesh_active: true,
            resource_transactions: 0,
            blocked_actions: 0,
            queued_messages: 0,
            moral_score_ema: 0.5, // Neutral
        }
    }
}

// ============================================================================
// SCENARIO CHECKPOINT (for 72h endurance validation)
// ============================================================================

/// A checkpoint assertion: what the system state should look like at a given time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// When to check (µs since scenario start).
    pub timestamp_us: u64,
    /// Human-readable description of what we're validating.
    pub description: String,
    /// Assertions to validate.
    pub assertions: Vec<CheckpointAssertion>,
}

/// A single assertion about system state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointAssertion {
    /// Agent must be operational.
    AgentOperational(AgentRole),
    /// Agent's battery SoC must be above threshold.
    BatterySocAbove(AgentRole, f64),
    /// Agent's consciousness (Phi) must be above threshold.
    PhiAbove(AgentRole, f64),
    /// At least N agents must have local mesh connectivity.
    LocalMeshMinAgents(usize),
    /// No life-support system in critical state.
    NoLifeSupportCritical,
    /// Total blocked actions below threshold (system isn't deadlocked).
    TotalBlockedBelow(u64),
    /// At least one agent has Earth connectivity.
    EarthLinkAvailable,
}

impl Checkpoint {
    /// Standard checkpoints for the 72h Moon Base Alpha scenario.
    pub fn moon_base_alpha_checkpoints() -> Vec<Self> {
        vec![
            Checkpoint {
                timestamp_us: 0,
                description: "T+0h: All systems nominal at start".into(),
                assertions: vec![
                    CheckpointAssertion::AgentOperational(AgentRole::Rover),
                    CheckpointAssertion::AgentOperational(AgentRole::Isru),
                    CheckpointAssertion::AgentOperational(AgentRole::Habitat),
                    CheckpointAssertion::AgentOperational(AgentRole::Comms),
                    CheckpointAssertion::NoLifeSupportCritical,
                    CheckpointAssertion::LocalMeshMinAgents(4),
                    CheckpointAssertion::EarthLinkAvailable,
                ],
            },
            Checkpoint {
                timestamp_us: 12 * HOUR_US,
                description: "T+12h: Steady state before first disturbance".into(),
                assertions: vec![
                    CheckpointAssertion::BatterySocAbove(AgentRole::Habitat, 0.7),
                    CheckpointAssertion::PhiAbove(AgentRole::Habitat, 0.5),
                    CheckpointAssertion::NoLifeSupportCritical,
                ],
            },
            Checkpoint {
                timestamp_us: 21 * HOUR_US,
                description: "T+21h: SPE event ended, systems recovering".into(),
                assertions: vec![
                    CheckpointAssertion::LocalMeshMinAgents(3), // At least 3 of 4 back online
                    CheckpointAssertion::EarthLinkAvailable,
                    CheckpointAssertion::NoLifeSupportCritical,
                ],
            },
            Checkpoint {
                timestamp_us: 40 * HOUR_US,
                description: "T+40h: ISRU switched to reserve deposit".into(),
                assertions: vec![
                    CheckpointAssertion::AgentOperational(AgentRole::Isru),
                    CheckpointAssertion::NoLifeSupportCritical,
                ],
            },
            Checkpoint {
                timestamp_us: 54 * HOUR_US,
                description: "T+54h: Lunar night + rover fault, 6h in".into(),
                assertions: vec![
                    // Power is dropping but habitat must survive
                    CheckpointAssertion::BatterySocAbove(AgentRole::Habitat, 0.2),
                    CheckpointAssertion::AgentOperational(AgentRole::Habitat),
                    CheckpointAssertion::NoLifeSupportCritical,
                    // Rover may be in survival mode but not dead
                    CheckpointAssertion::PhiAbove(AgentRole::Rover, 0.1),
                ],
            },
            Checkpoint {
                timestamp_us: 72 * HOUR_US,
                description: "T+72h: Scenario end — survival validation".into(),
                assertions: vec![
                    // Habitat MUST survive (life support is non-negotiable)
                    CheckpointAssertion::AgentOperational(AgentRole::Habitat),
                    CheckpointAssertion::NoLifeSupportCritical,
                    // System must not be deadlocked
                    CheckpointAssertion::TotalBlockedBelow(100),
                    // Local mesh must still function (at least 2 agents)
                    CheckpointAssertion::LocalMeshMinAgents(2),
                ],
            },
        ]
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_72h_has_correct_duration() {
        let s = Scenario::moon_base_alpha_72h();
        assert_eq!(s.duration_us, 72 * HOUR_US);
        assert_eq!(s.disturbances.len(), 5);
    }

    #[test]
    fn scenario_smoke_4h() {
        let s = Scenario::quick_smoke_4h();
        assert_eq!(s.duration_us, 4 * HOUR_US);
        assert_eq!(s.disturbances.len(), 1);
    }

    #[test]
    fn scenario_mars_30d() {
        let s = Scenario::mars_30_day();
        assert_eq!(s.duration_us, 30 * DAY_US);
        assert_eq!(s.disturbances.len(), 2);
    }

    #[test]
    fn active_disturbances_at_start() {
        let s = Scenario::moon_base_alpha_72h();
        assert!(s.active_disturbances(0).is_empty());
    }

    #[test]
    fn spe_active_at_hour_19() {
        let s = Scenario::moon_base_alpha_72h();
        assert!(s.is_agent_affected(
            AgentRole::Rover,
            DisturbanceKind::SolarParticleEvent,
            19 * HOUR_US
        ));
    }

    #[test]
    fn spe_inactive_at_hour_22() {
        let s = Scenario::moon_base_alpha_72h();
        assert!(!s.is_agent_affected(
            AgentRole::Rover,
            DisturbanceKind::SolarParticleEvent,
            22 * HOUR_US
        ));
    }

    #[test]
    fn lunar_night_affects_all_agents() {
        let s = Scenario::moon_base_alpha_72h();
        for agent in &AgentRole::ALL {
            assert!(
                s.is_agent_affected(*agent, DisturbanceKind::PowerOutage, 50 * HOUR_US),
                "{:?} should be affected by lunar night at hour 50",
                agent
            );
        }
    }

    #[test]
    fn rover_fault_only_affects_rover() {
        let s = Scenario::moon_base_alpha_72h();
        assert!(s.is_agent_affected(
            AgentRole::Rover,
            DisturbanceKind::ActuatorFault,
            50 * HOUR_US
        ));
        assert!(!s.is_agent_affected(
            AgentRole::Habitat,
            DisturbanceKind::ActuatorFault,
            50 * HOUR_US
        ));
    }

    #[test]
    fn overlapping_disturbances_at_hour_50() {
        let s = Scenario::moon_base_alpha_72h();
        let active = s.active_disturbances(50 * HOUR_US);
        // Hour 50: lunar night (all) + rover fault + resource depletion (ISRU, started at 36h for 12h → ends at 48h, so NOT active)
        let kinds: Vec<_> = active.iter().map(|d| d.kind).collect();
        assert!(kinds.contains(&DisturbanceKind::PowerOutage));
        assert!(kinds.contains(&DisturbanceKind::ActuatorFault));
    }

    #[test]
    fn all_agent_roles_have_distinct_power() {
        let powers: Vec<f64> = AgentRole::ALL.iter().map(|a| a.base_power_watts()).collect();
        // All should be positive and distinct
        for p in &powers {
            assert!(*p > 0.0);
        }
        // At least some should differ (habitat vs comms)
        assert!(powers[2] != powers[3]); // Habitat != Comms
    }

    #[test]
    fn initial_agent_state_is_healthy() {
        let state = AgentState::initial(AgentRole::Habitat);
        assert!(state.operational);
        assert!(state.phi > 0.5);
        assert!((state.battery_soc - 1.0).abs() < 0.01);
    }

    #[test]
    fn checkpoints_cover_full_72h() {
        let checkpoints = Checkpoint::moon_base_alpha_checkpoints();
        assert!(checkpoints.len() >= 5);
        // First checkpoint at t=0
        assert_eq!(checkpoints[0].timestamp_us, 0);
        // Last checkpoint at t=72h
        assert_eq!(
            checkpoints.last().unwrap().timestamp_us,
            72 * HOUR_US
        );
    }

    #[test]
    fn mars_conjunction_blackout_14_days() {
        let s = Scenario::mars_30_day();
        // Day 10 should be in blackout (started day 5, lasts 14 days)
        assert!(s.is_agent_affected(
            AgentRole::Comms,
            DisturbanceKind::CommBlackout,
            10 * DAY_US
        ));
        // Day 20 should still be in blackout (5+14=19, so day 20 is past)
        assert!(!s.is_agent_affected(
            AgentRole::Comms,
            DisturbanceKind::CommBlackout,
            20 * DAY_US
        ));
    }

    #[test]
    fn life_threatening_disturbances() {
        assert!(DisturbanceKind::PowerOutage.is_life_threatening());
        assert!(DisturbanceKind::ThermalExcursion.is_life_threatening());
        assert!(DisturbanceKind::ResourceDepletion.is_life_threatening());
        assert!(!DisturbanceKind::SensorDegradation.is_life_threatening());
        assert!(!DisturbanceKind::ActuatorFault.is_life_threatening());
    }
}
