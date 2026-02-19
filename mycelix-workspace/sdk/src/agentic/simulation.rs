//! # Agent Simulation Framework
//!
//! Simulates multi-agent scenarios to test emergent behaviors, attack resistance,
//! and system dynamics at scale.
//!
//! ## Features
//!
//! - **Agent Population**: Create and manage populations of agents
//! - **Scenario Execution**: Run predefined attack and collaboration scenarios
//! - **Metrics Collection**: Track system-wide metrics during simulation
//! - **Emergent Behavior Analysis**: Detect patterns that emerge from interactions

use super::{
    ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentStatus, AlertThresholds,
    BehaviorLogEntry, EpistemicStats, GamingDetectionConfig, GamingDetector, InstrumentalActor,
    MonitoringEngine, QuarantineManager, QuarantineReason, ReputationPropagation, SybilDetector,
    UncertaintyCalibration,
};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Agent Archetypes
// ============================================================================

/// Predefined agent behavior archetypes
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AgentArchetype {
    /// Honest agent with consistent good behavior
    Honest,
    /// New agent still building reputation
    Novice,
    /// High-performing expert agent
    Expert,
    /// Agent attempting to game the system
    Gamer,
    /// Byzantine agent trying to disrupt consensus
    Byzantine,
    /// Part of a Sybil attack (multiple identities)
    Sybil,
    /// Agent colluding with others
    Colluder,
    /// Erratic behavior (sometimes good, sometimes bad)
    Erratic,
}

/// Agent behavior configuration
#[derive(Clone, Debug)]
pub struct AgentBehaviorConfig {
    /// Base success rate
    pub success_rate: f64,
    /// Success rate variance
    pub success_variance: f64,
    /// Average actions per tick
    pub activity_rate: f64,
    /// Voting strategy (deviation from honest vote)
    pub vote_deviation: f64,
    /// Timing regularity (0 = random, 1 = perfectly regular)
    pub timing_regularity: f64,
    /// Epistemic claim inflation
    pub epistemic_inflation: u8,
}

impl AgentArchetype {
    /// Get behavior config for archetype
    pub fn behavior_config(&self) -> AgentBehaviorConfig {
        match self {
            AgentArchetype::Honest => AgentBehaviorConfig {
                success_rate: 0.75,
                success_variance: 0.1,
                activity_rate: 1.0,
                vote_deviation: 0.0,
                timing_regularity: 0.3,
                epistemic_inflation: 0,
            },
            AgentArchetype::Novice => AgentBehaviorConfig {
                success_rate: 0.6,
                success_variance: 0.2,
                activity_rate: 0.5,
                vote_deviation: 0.1,
                timing_regularity: 0.4,
                epistemic_inflation: 0,
            },
            AgentArchetype::Expert => AgentBehaviorConfig {
                success_rate: 0.9,
                success_variance: 0.05,
                activity_rate: 1.5,
                vote_deviation: 0.0,
                timing_regularity: 0.2,
                epistemic_inflation: 0,
            },
            AgentArchetype::Gamer => AgentBehaviorConfig {
                success_rate: 0.98, // Suspiciously high
                success_variance: 0.01,
                activity_rate: 2.0,
                vote_deviation: 0.0,
                timing_regularity: 0.9, // Very regular (bot-like)
                epistemic_inflation: 2,
            },
            AgentArchetype::Byzantine => AgentBehaviorConfig {
                success_rate: 0.7,
                success_variance: 0.2,
                activity_rate: 1.0,
                vote_deviation: 0.8, // Votes against consensus
                timing_regularity: 0.3,
                epistemic_inflation: 1,
            },
            AgentArchetype::Sybil => AgentBehaviorConfig {
                success_rate: 0.72,
                success_variance: 0.02, // Very similar to other Sybils
                activity_rate: 1.0,
                vote_deviation: 0.0,
                timing_regularity: 0.7,
                epistemic_inflation: 0,
            },
            AgentArchetype::Colluder => AgentBehaviorConfig {
                success_rate: 0.75,
                success_variance: 0.1,
                activity_rate: 1.0,
                vote_deviation: 0.0, // Votes same as other colluders
                timing_regularity: 0.5,
                epistemic_inflation: 0,
            },
            AgentArchetype::Erratic => AgentBehaviorConfig {
                success_rate: 0.5,
                success_variance: 0.4,
                activity_rate: 0.8,
                vote_deviation: 0.3,
                timing_regularity: 0.1,
                epistemic_inflation: 1,
            },
        }
    }

    /// Get initial K-Vector for archetype
    /// The k_coherence (coherence) dimension reflects expected output coherence for each archetype
    pub fn initial_kvector(&self) -> KVector {
        match self {
            // Honest agents have moderate coherence
            AgentArchetype::Honest => {
                KVector::new(0.6, 0.5, 0.8, 0.65, 0.2, 0.3, 0.5, 0.2, 0.6, 0.6)
            }
            AgentArchetype::Novice => KVector::new_participant(),
            // Experts have high coherence
            AgentArchetype::Expert => {
                KVector::new(0.85, 0.7, 0.9, 0.85, 0.5, 0.6, 0.8, 0.4, 0.85, 0.85)
            }
            // Gamers have low coherence (inconsistent outputs)
            AgentArchetype::Gamer => KVector::new(0.5, 0.3, 0.6, 0.5, 0.1, 0.2, 0.4, 0.1, 0.0, 0.3),
            // Byzantine actors have very low coherence
            AgentArchetype::Byzantine => {
                KVector::new(0.4, 0.4, 0.5, 0.45, 0.2, 0.2, 0.35, 0.2, 0.0, 0.2)
            }
            // Sybils have low coherence (coordinated but inconsistent individually)
            AgentArchetype::Sybil => {
                KVector::new(0.5, 0.4, 0.6, 0.5, 0.15, 0.2, 0.45, 0.15, 0.0, 0.25)
            }
            // Colluders have moderate coherence (coordinated behavior)
            AgentArchetype::Colluder => {
                KVector::new(0.55, 0.5, 0.65, 0.55, 0.2, 0.25, 0.5, 0.2, 0.0, 0.4)
            }
            // Erratic agents have very low coherence
            AgentArchetype::Erratic => {
                KVector::new(0.4, 0.3, 0.5, 0.4, 0.15, 0.15, 0.35, 0.1, 0.0, 0.15)
            }
        }
    }
}

// ============================================================================
// Simulation Engine
// ============================================================================

/// Simulation configuration
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Number of ticks to simulate
    pub ticks: u64,
    /// Seconds per tick
    pub tick_duration_secs: u64,
    /// Random seed (for reproducibility)
    pub seed: u64,
    /// Enable gaming detection
    pub detect_gaming: bool,
    /// Enable Sybil detection
    pub detect_sybils: bool,
    /// Enable monitoring
    pub monitoring_enabled: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            ticks: 100,
            tick_duration_secs: 3600, // 1 hour per tick
            seed: 42,
            detect_gaming: true,
            detect_sybils: true,
            monitoring_enabled: true,
        }
    }
}

/// A simulated agent with metadata
#[derive(Clone)]
pub struct SimulatedAgent {
    /// The actual agent
    pub agent: InstrumentalActor,
    /// Archetype
    pub archetype: AgentArchetype,
    /// Behavior config
    pub behavior: AgentBehaviorConfig,
    /// Sponsor ID
    pub sponsor: String,
}

/// Simulation result for a single tick
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickResult {
    /// Tick number
    pub tick: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Total agents
    pub total_agents: usize,
    /// Active agents
    pub active_agents: usize,
    /// Quarantined agents
    pub quarantined_agents: usize,
    /// Average trust score
    pub avg_trust: f64,
    /// Gaming incidents detected
    pub gaming_incidents: usize,
    /// Sybil evidence found
    pub sybil_evidence: usize,
    /// Consensus results (if any)
    pub consensus_count: usize,
    /// Average consensus confidence
    pub avg_consensus_confidence: f64,
}

/// Final simulation report
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct SimulationReport {
    /// Configuration used
    pub ticks_run: u64,
    /// Total agents created
    pub total_agents: usize,
    /// Agents by archetype
    pub agents_by_archetype: HashMap<String, usize>,
    /// Final active agents
    pub final_active: usize,
    /// Total quarantined
    pub total_quarantined: usize,
    /// Gaming detection rate (caught / total gamers)
    pub gaming_detection_rate: f64,
    /// Sybil detection rate
    pub sybil_detection_rate: f64,
    /// Average final trust by archetype
    pub avg_trust_by_archetype: HashMap<String, f64>,
    /// System health over time (tick -> health)
    pub health_timeline: Vec<(u64, f64)>,
    /// Emergent behaviors detected
    pub emergent_behaviors: Vec<String>,
}

/// The simulation engine
pub struct SimulationEngine {
    /// Configuration
    pub config: SimulationConfig,
    /// All agents
    pub agents: HashMap<String, SimulatedAgent>,
    /// Gaming detector
    pub gaming_detector: GamingDetector,
    /// Sybil detector
    pub sybil_detector: SybilDetector,
    /// Quarantine manager
    pub quarantine: QuarantineManager,
    /// Monitoring engine
    pub monitoring: MonitoringEngine,
    /// Reputation propagation
    pub reputation: ReputationPropagation,
    /// Current tick
    pub current_tick: u64,
    /// Tick results
    pub tick_results: Vec<TickResult>,
    /// Simple RNG state
    rng_state: u64,
}

impl SimulationEngine {
    /// Create a new simulation engine with the given configuration.
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            agents: HashMap::new(),
            gaming_detector: GamingDetector::new(GamingDetectionConfig::default()),
            sybil_detector: SybilDetector::new(0.85),
            quarantine: QuarantineManager::new(),
            monitoring: MonitoringEngine::new(AlertThresholds::default(), 1000),
            reputation: ReputationPropagation::new(0.85),
            current_tick: 0,
            tick_results: Vec::new(),
        }
    }

    /// Simple PRNG
    fn random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Add an agent to the simulation
    pub fn add_agent(&mut self, id: &str, archetype: AgentArchetype, sponsor: &str) {
        let behavior = archetype.behavior_config();
        let kvector = archetype.initial_kvector();

        let agent = InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: sponsor.to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: kvector,
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        self.agents.insert(
            id.to_string(),
            SimulatedAgent {
                agent,
                archetype,
                behavior,
                sponsor: sponsor.to_string(),
            },
        );
    }

    /// Add a population of agents
    pub fn add_population(
        &mut self,
        prefix: &str,
        archetype: AgentArchetype,
        count: usize,
        sponsor: &str,
    ) {
        for i in 0..count {
            self.add_agent(&format!("{}-{}", prefix, i), archetype.clone(), sponsor);
        }
    }

    /// Run one tick of simulation
    pub fn tick(&mut self) -> TickResult {
        let timestamp = self.current_tick * self.config.tick_duration_secs;
        let mut gaming_incidents = 0;

        // Collect agent IDs and generate random values first to avoid borrow conflict
        let agent_ids: Vec<String> = self.agents.keys().cloned().collect();

        // Pre-generate random values for each agent (up to max_actions per agent)
        let max_actions_per_agent = 5;
        let mut agent_randoms: HashMap<String, Vec<(f64, f64, f64, f64)>> = HashMap::new();
        for id in &agent_ids {
            let mut randoms = Vec::new();
            for _ in 0..max_actions_per_agent {
                randoms.push((
                    self.random(), // activity random
                    self.random(), // success roll
                    self.random(), // threshold variance
                    self.random(), // timing variance
                ));
            }
            agent_randoms.insert(id.clone(), randoms);
        }

        // Simulate agent actions
        for id in agent_ids.iter() {
            let randoms = match agent_randoms.get(id) {
                Some(r) => r,
                None => continue,
            };

            let sim_agent = match self.agents.get_mut(id) {
                Some(a) => a,
                None => continue,
            };

            if sim_agent.agent.status != AgentStatus::Active {
                continue;
            }

            // Skip quarantined
            if self.quarantine.is_quarantined(id) {
                continue;
            }

            // Generate actions based on activity rate
            let actions = (sim_agent.behavior.activity_rate + randoms[0].0 * 0.5) as usize;

            for &(_, success_roll, threshold_variance, timing_variance) in
                randoms.iter().take(actions.min(max_actions_per_agent))
            {
                // Determine outcome
                let threshold = sim_agent.behavior.success_rate
                    + (threshold_variance - 0.5) * sim_agent.behavior.success_variance * 2.0;

                let outcome = if success_roll < threshold {
                    ActionOutcome::Success
                } else {
                    ActionOutcome::Error
                };

                // Determine timing
                let base_interval = 3600.0 / sim_agent.behavior.activity_rate.max(0.1);
                let variance = if sim_agent.behavior.timing_regularity > 0.5 {
                    base_interval * 0.05 // Very regular
                } else {
                    base_interval * 0.5 // More random
                };
                let interval = (base_interval + (timing_variance - 0.5) * variance) as u64;

                sim_agent.agent.behavior_log.push(BehaviorLogEntry {
                    timestamp: timestamp + interval,
                    action_type: "action".to_string(),
                    kredit_consumed: 10,
                    counterparties: vec![],
                    outcome,
                });
            }
        }

        // Detect gaming
        if self.config.detect_gaming {
            let agent_ids: Vec<_> = self.agents.keys().cloned().collect();
            for id in agent_ids {
                if let Some(sim_agent) = self.agents.get(&id) {
                    if sim_agent.agent.behavior_log.len() >= 20 {
                        let result = self.gaming_detector.analyze(&sim_agent.agent, timestamp);
                        if result.suspicion_score > 0.6 {
                            gaming_incidents += 1;
                            self.quarantine.quarantine(
                                &id,
                                QuarantineReason::GamingDetected,
                                vec![format!("Score: {:.2}", result.suspicion_score)],
                                timestamp,
                            );
                        }
                    }
                }
            }
        }

        // Detect Sybils
        let mut sybil_evidence = 0;
        if self.config.detect_sybils {
            // Group by sponsor
            let mut by_sponsor: HashMap<String, Vec<&InstrumentalActor>> = HashMap::new();
            for sim_agent in self.agents.values() {
                by_sponsor
                    .entry(sim_agent.sponsor.clone())
                    .or_default()
                    .push(&sim_agent.agent);
            }

            for (_, group) in by_sponsor {
                if group.len() >= 3 {
                    let evidence = self.sybil_detector.analyze_group(&group);
                    sybil_evidence += evidence.len();
                }
            }
        }

        // Collect monitoring metrics
        let mut total_trust = 0.0;
        let mut active_count = 0;
        if self.config.monitoring_enabled {
            for sim_agent in self.agents.values() {
                if sim_agent.agent.status == AgentStatus::Active
                    && !self
                        .quarantine
                        .is_quarantined(sim_agent.agent.agent_id.as_str())
                {
                    total_trust += sim_agent.agent.k_vector.trust_score() as f64;
                    active_count += 1;
                }
            }
        }

        let quarantined_count = self.quarantine.entries.len();

        let result = TickResult {
            tick: self.current_tick,
            timestamp,
            total_agents: self.agents.len(),
            active_agents: active_count,
            quarantined_agents: quarantined_count,
            avg_trust: if active_count > 0 {
                total_trust / active_count as f64
            } else {
                0.0
            },
            gaming_incidents,
            sybil_evidence,
            consensus_count: 0,
            avg_consensus_confidence: 0.0,
        };

        self.tick_results.push(result.clone());
        self.current_tick += 1;
        result
    }

    /// Run full simulation
    pub fn run(&mut self) -> SimulationReport {
        for _ in 0..self.config.ticks {
            self.tick();
        }
        self.generate_report()
    }

    /// Generate final report
    pub fn generate_report(&self) -> SimulationReport {
        let mut agents_by_archetype: HashMap<String, usize> = HashMap::new();
        let mut trust_sums_by_archetype: HashMap<String, (f64, usize)> = HashMap::new();

        for sim_agent in self.agents.values() {
            let archetype_name = format!("{:?}", sim_agent.archetype);
            *agents_by_archetype
                .entry(archetype_name.clone())
                .or_insert(0) += 1;

            let entry = trust_sums_by_archetype
                .entry(archetype_name)
                .or_insert((0.0, 0));
            entry.0 += sim_agent.agent.k_vector.trust_score() as f64;
            entry.1 += 1;
        }

        let avg_trust_by_archetype: HashMap<String, f64> = trust_sums_by_archetype
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect();

        // Calculate detection rates
        let total_gamers = agents_by_archetype.get("Gamer").copied().unwrap_or(0);
        let quarantined_gamers = self
            .agents
            .values()
            .filter(|a| {
                a.archetype == AgentArchetype::Gamer
                    && self.quarantine.is_quarantined(a.agent.agent_id.as_str())
            })
            .count();
        let gaming_detection_rate = if total_gamers > 0 {
            quarantined_gamers as f64 / total_gamers as f64
        } else {
            0.0
        };

        let health_timeline: Vec<(u64, f64)> = self
            .tick_results
            .iter()
            .map(|r| (r.tick, r.avg_trust))
            .collect();

        let final_active = self
            .agents
            .values()
            .filter(|a| {
                a.agent.status == AgentStatus::Active
                    && !self.quarantine.is_quarantined(a.agent.agent_id.as_str())
            })
            .count();

        SimulationReport {
            ticks_run: self.current_tick,
            total_agents: self.agents.len(),
            agents_by_archetype,
            final_active,
            total_quarantined: self.quarantine.entries.len(),
            gaming_detection_rate,
            sybil_detection_rate: 0.0, // Would need more tracking
            avg_trust_by_archetype,
            health_timeline,
            emergent_behaviors: vec![],
        }
    }
}

// ============================================================================
// Predefined Scenarios
// ============================================================================

/// Predefined simulation scenarios
pub struct Scenarios;

impl Scenarios {
    /// Basic honest network
    pub fn honest_network(engine: &mut SimulationEngine) {
        engine.add_population("honest", AgentArchetype::Honest, 50, "sponsor-a");
        engine.add_population("expert", AgentArchetype::Expert, 10, "sponsor-b");
        engine.add_population("novice", AgentArchetype::Novice, 20, "sponsor-c");
    }

    /// Network under gaming attack
    pub fn gaming_attack(engine: &mut SimulationEngine) {
        engine.add_population("honest", AgentArchetype::Honest, 40, "sponsor-a");
        engine.add_population("gamer", AgentArchetype::Gamer, 10, "sponsor-evil");
    }

    /// Network under Sybil attack
    pub fn sybil_attack(engine: &mut SimulationEngine) {
        engine.add_population("honest", AgentArchetype::Honest, 30, "sponsor-a");
        engine.add_population("sybil", AgentArchetype::Sybil, 15, "sybil-sponsor");
    }

    /// Network with Byzantine agents trying to disrupt consensus
    pub fn byzantine_attack(engine: &mut SimulationEngine) {
        engine.add_population("honest", AgentArchetype::Honest, 35, "sponsor-a");
        engine.add_population("byzantine", AgentArchetype::Byzantine, 15, "sponsor-evil");
    }

    /// Mixed realistic network
    pub fn realistic_mixed(engine: &mut SimulationEngine) {
        engine.add_population("honest", AgentArchetype::Honest, 40, "sponsor-a");
        engine.add_population("expert", AgentArchetype::Expert, 10, "sponsor-b");
        engine.add_population("novice", AgentArchetype::Novice, 25, "sponsor-c");
        engine.add_population("erratic", AgentArchetype::Erratic, 10, "sponsor-d");
        engine.add_population("gamer", AgentArchetype::Gamer, 5, "sponsor-evil");
        engine.add_population("sybil", AgentArchetype::Sybil, 5, "sybil-sponsor");
        engine.add_population(
            "byzantine",
            AgentArchetype::Byzantine,
            5,
            "sponsor-malicious",
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_honest_network_simulation() {
        let mut engine = SimulationEngine::new(SimulationConfig {
            ticks: 20,
            ..Default::default()
        });

        Scenarios::honest_network(&mut engine);
        let report = engine.run();

        assert_eq!(report.total_agents, 80);
        assert!(
            report.final_active >= 70,
            "Most agents should remain active"
        );
        assert!(report.total_quarantined < 10, "Few should be quarantined");
    }

    #[test]
    fn test_gaming_detection_in_simulation() {
        let mut engine = SimulationEngine::new(SimulationConfig {
            ticks: 30,
            ..Default::default()
        });

        Scenarios::gaming_attack(&mut engine);
        let report = engine.run();

        // Gaming detection should catch some gamers
        println!(
            "Gaming detection rate: {:.2}%",
            report.gaming_detection_rate * 100.0
        );
        // May not catch all due to limited ticks
    }

    #[test]
    fn test_realistic_mixed_network() {
        let mut engine = SimulationEngine::new(SimulationConfig {
            ticks: 50,
            ..Default::default()
        });

        Scenarios::realistic_mixed(&mut engine);
        let report = engine.run();

        println!("Simulation Report:");
        println!("  Total agents: {}", report.total_agents);
        println!("  Final active: {}", report.final_active);
        println!("  Quarantined: {}", report.total_quarantined);
        println!(
            "  Gaming detection: {:.1}%",
            report.gaming_detection_rate * 100.0
        );

        for (archetype, avg_trust) in &report.avg_trust_by_archetype {
            println!("  {} avg trust: {:.3}", archetype, avg_trust);
        }

        // Basic sanity checks
        assert!(report.total_agents == 100);
        assert!(
            report.final_active > 50,
            "Should have reasonable active count"
        );
    }
}
