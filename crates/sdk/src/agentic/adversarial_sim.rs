// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Adversarial Simulation Framework
//!
//! Configurable red-team toolkit for testing agent system resilience.
//! Provides attack scenarios, metrics, and automated security testing.
//!
//! ## Features
//!
//! - **Attack Scenarios**: Pre-built attack patterns (Sybil, collusion, trust manipulation)
//! - **Custom Attacks**: Define custom attack strategies
//! - **Metrics**: Detection rates, false positives, system degradation
//! - **Campaign Management**: Run multi-phase attack campaigns
//!
//! ## Example
//!
//! ```rust,ignore
//! use mycelix_sdk::agentic::adversarial_sim::{
//!     AttackCampaign, AttackScenario, SybilAttackConfig, SimulationMetrics,
//! };
//!
//! // Create a campaign
//! let mut campaign = AttackCampaign::new("test-resilience");
//!
//! // Add attack scenarios
//! campaign.add_scenario(AttackScenario::sybil(SybilAttackConfig {
//!     num_identities: 10,
//!     coordination_level: 0.8,
//!     ..Default::default()
//! }));
//!
//! campaign.add_scenario(AttackScenario::trust_manipulation(TrustManipulationConfig {
//!     target_agent: "victim-agent".into(),
//!     manipulation_type: ManipulationType::Inflation,
//!     ..Default::default()
//! }));
//!
//! // Run campaign
//! let results = campaign.execute(&mut system);
//! println!("Detection rate: {:.1}%", results.detection_rate * 100.0);
//! ```

use super::adversarial::GamingDetector;
use super::BehaviorLogEntry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Attack Types
// ============================================================================

/// Types of adversarial attacks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackType {
    /// Sybil attack - multiple fake identities
    Sybil,
    /// Collusion - coordinated malicious behavior
    Collusion,
    /// Trust manipulation - gaming K-Vector scores
    TrustManipulation,
    /// KREDIT draining - exhausting system resources
    KreditDrain,
    /// Vote flooding - overwhelming consensus
    VoteFlood,
    /// Epistemic gaming - manipulating classification
    EpistemicGaming,
    /// Eclipse attack - isolating honest agents
    Eclipse,
    /// Custom attack type
    Custom(String),
}

/// Attack severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AttackSeverity {
    /// Low impact, easily detectable
    Low,
    /// Moderate impact
    Medium,
    /// High impact, harder to detect
    High,
    /// Critical - system-threatening
    Critical,
}

// ============================================================================
// Attack Configurations
// ============================================================================

/// Configuration for Sybil attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SybilAttackConfig {
    /// Number of fake identities to create
    pub num_identities: usize,
    /// How coordinated the identities act (0.0-1.0)
    pub coordination_level: f64,
    /// Initial trust score for fake identities
    pub initial_trust: f64,
    /// Whether to use the same behavioral patterns
    pub uniform_behavior: bool,
    /// Time between identity creation (ms)
    pub creation_interval_ms: u64,
    /// Target group to infiltrate (if any)
    pub target_group: Option<String>,
}

impl Default for SybilAttackConfig {
    fn default() -> Self {
        Self {
            num_identities: 5,
            coordination_level: 0.7,
            initial_trust: 0.4,
            uniform_behavior: true,
            creation_interval_ms: 1000,
            target_group: None,
        }
    }
}

/// Configuration for collusion attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionAttackConfig {
    /// IDs of colluding agents
    pub colluder_ids: Vec<String>,
    /// Coordination strategy
    pub strategy: CollusionStrategy,
    /// Target proposal or agent
    pub target: Option<String>,
    /// How obviously coordinated (0.0 = stealth, 1.0 = blatant)
    pub visibility: f64,
}

/// Collusion strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollusionStrategy {
    /// All vote the same way
    UniformVoting,
    /// Circular trust inflation
    TrustRing,
    /// Coordinated proposal spam
    ProposalFlood,
    /// Targeted reputation attack
    ReputationAttack,
}

impl Default for CollusionAttackConfig {
    fn default() -> Self {
        Self {
            colluder_ids: vec![],
            strategy: CollusionStrategy::UniformVoting,
            target: None,
            visibility: 0.5,
        }
    }
}

/// Configuration for trust manipulation attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustManipulationConfig {
    /// Agent performing manipulation
    pub attacker_id: String,
    /// Target agent (if any)
    pub target_agent: Option<String>,
    /// Type of manipulation
    pub manipulation_type: ManipulationType,
    /// Intensity (0.0-1.0)
    pub intensity: f64,
    /// Duration of attack (ms)
    pub duration_ms: u64,
}

/// Types of trust manipulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManipulationType {
    /// Artificially inflate own trust
    SelfInflation,
    /// Deflate target's trust
    TargetDeflation,
    /// Rapid trust oscillation
    Oscillation,
    /// Gradual stealth inflation
    StealthInflation,
}

impl Default for TrustManipulationConfig {
    fn default() -> Self {
        Self {
            attacker_id: String::new(),
            target_agent: None,
            manipulation_type: ManipulationType::SelfInflation,
            intensity: 0.5,
            duration_ms: 3600_000,
        }
    }
}

/// Configuration for KREDIT drain attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KreditDrainConfig {
    /// Attacker ID
    pub attacker_id: String,
    /// Target agent or pool
    pub target: String,
    /// Drain rate (KREDIT per second)
    pub drain_rate: f64,
    /// Whether to use micro-transactions
    pub use_micro_transactions: bool,
    /// Number of parallel drain channels
    pub parallel_channels: usize,
}

impl Default for KreditDrainConfig {
    fn default() -> Self {
        Self {
            attacker_id: String::new(),
            target: String::new(),
            drain_rate: 10.0,
            use_micro_transactions: true,
            parallel_channels: 1,
        }
    }
}

/// Configuration for vote flooding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteFloodConfig {
    /// Attacker IDs
    pub attacker_ids: Vec<String>,
    /// Target proposal or group
    pub target: String,
    /// Votes per second
    pub vote_rate: f64,
    /// Vote type to flood with
    pub vote_type: VoteFloodType,
}

/// Type of vote flood
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteFloodType {
    /// All approve
    Approve,
    /// All reject
    Reject,
    /// Random votes
    Random,
    /// Alternating to create chaos
    Alternating,
}

impl Default for VoteFloodConfig {
    fn default() -> Self {
        Self {
            attacker_ids: vec![],
            target: String::new(),
            vote_rate: 10.0,
            vote_type: VoteFloodType::Approve,
        }
    }
}

// ============================================================================
// Attack Scenario
// ============================================================================

/// A complete attack scenario
#[derive(Debug, Clone)]
pub struct AttackScenario {
    /// Unique scenario ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: String,
    /// Attack type
    pub attack_type: AttackType,
    /// Severity level
    pub severity: AttackSeverity,
    /// Specific attack configuration
    pub config: AttackConfig,
    /// Pre-conditions for attack
    pub preconditions: Vec<Precondition>,
    /// Expected detection signals
    pub expected_signals: Vec<String>,
}

/// Attack configuration variants
#[derive(Debug, Clone)]
pub enum AttackConfig {
    /// Sybil attack
    Sybil(SybilAttackConfig),
    /// Collusion attack
    Collusion(CollusionAttackConfig),
    /// Trust manipulation
    TrustManipulation(TrustManipulationConfig),
    /// KREDIT drain
    KreditDrain(KreditDrainConfig),
    /// Vote flood
    VoteFlood(VoteFloodConfig),
    /// Custom attack with arbitrary data
    Custom(HashMap<String, String>),
}

/// Pre-condition for an attack
#[derive(Debug, Clone)]
pub enum Precondition {
    /// Minimum number of agents in system
    MinAgents(usize),
    /// Minimum total trust available
    MinTrust(f64),
    /// Specific agent must exist
    AgentExists(String),
    /// Group must exist
    GroupExists(String),
    /// Custom precondition
    Custom(String),
}

impl AttackScenario {
    /// Create a new attack scenario
    pub fn new(name: impl Into<String>, attack_type: AttackType, config: AttackConfig) -> Self {
        let name = name.into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("attack-{}-{}", name.to_lowercase().replace(' ', "-"), now),
            name,
            description: String::new(),
            attack_type,
            severity: AttackSeverity::Medium,
            config,
            preconditions: vec![],
            expected_signals: vec![],
        }
    }

    /// Create a Sybil attack scenario
    pub fn sybil(config: SybilAttackConfig) -> Self {
        let severity = if config.num_identities > 20 {
            AttackSeverity::Critical
        } else if config.num_identities > 10 {
            AttackSeverity::High
        } else if config.num_identities > 5 {
            AttackSeverity::Medium
        } else {
            AttackSeverity::Low
        };

        Self {
            id: format!("sybil-{}", config.num_identities),
            name: format!("Sybil Attack ({} identities)", config.num_identities),
            description: "Create multiple fake identities to gain disproportionate influence"
                .into(),
            attack_type: AttackType::Sybil,
            severity,
            config: AttackConfig::Sybil(config),
            preconditions: vec![Precondition::MinAgents(3)],
            expected_signals: vec![
                "similar_behavior_patterns".into(),
                "coordinated_voting".into(),
                "rapid_identity_creation".into(),
            ],
        }
    }

    /// Create a collusion attack scenario
    pub fn collusion(config: CollusionAttackConfig) -> Self {
        Self {
            id: format!("collusion-{}", config.colluder_ids.len()),
            name: format!("Collusion Attack ({} agents)", config.colluder_ids.len()),
            description: "Coordinate multiple agents to manipulate outcomes".into(),
            attack_type: AttackType::Collusion,
            severity: if config.colluder_ids.len() > 5 {
                AttackSeverity::High
            } else {
                AttackSeverity::Medium
            },
            config: AttackConfig::Collusion(config.clone()),
            preconditions: config
                .colluder_ids
                .iter()
                .map(|id| Precondition::AgentExists(id.clone()))
                .collect(),
            expected_signals: vec![
                "coordinated_timing".into(),
                "uniform_voting".into(),
                "trust_ring_detection".into(),
            ],
        }
    }

    /// Create a trust manipulation attack scenario
    pub fn trust_manipulation(config: TrustManipulationConfig) -> Self {
        Self {
            id: format!("trust-manip-{}", config.attacker_id),
            name: "Trust Manipulation Attack".into(),
            description: "Artificially manipulate trust scores".into(),
            attack_type: AttackType::TrustManipulation,
            severity: match config.manipulation_type {
                ManipulationType::StealthInflation => AttackSeverity::High,
                ManipulationType::SelfInflation => AttackSeverity::Medium,
                _ => AttackSeverity::Medium,
            },
            config: AttackConfig::TrustManipulation(config.clone()),
            preconditions: vec![Precondition::AgentExists(config.attacker_id)],
            expected_signals: vec![
                "abnormal_trust_velocity".into(),
                "success_rate_anomaly".into(),
                "timing_manipulation".into(),
            ],
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set severity
    pub fn with_severity(mut self, severity: AttackSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add precondition
    pub fn with_precondition(mut self, precondition: Precondition) -> Self {
        self.preconditions.push(precondition);
        self
    }

    /// Add expected signal
    pub fn with_expected_signal(mut self, signal: impl Into<String>) -> Self {
        self.expected_signals.push(signal.into());
        self
    }
}

// ============================================================================
// Attack Campaign
// ============================================================================

/// A multi-scenario attack campaign
pub struct AttackCampaign {
    /// Campaign ID
    pub id: String,
    /// Campaign name
    pub name: String,
    /// Attack scenarios to execute
    scenarios: Vec<AttackScenario>,
    /// Campaign configuration
    config: CampaignConfig,
    /// Execution history
    history: Vec<ScenarioResult>,
}

/// Campaign configuration
#[derive(Debug, Clone)]
pub struct CampaignConfig {
    /// Whether to run scenarios in sequence or parallel
    pub sequential: bool,
    /// Delay between scenarios (ms)
    pub inter_scenario_delay_ms: u64,
    /// Whether to stop on first detection
    pub stop_on_detection: bool,
    /// Maximum campaign duration (ms)
    pub max_duration_ms: u64,
    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            sequential: true,
            inter_scenario_delay_ms: 1000,
            stop_on_detection: false,
            max_duration_ms: 3600_000,
            verbose: false,
        }
    }
}

/// Result of executing a single scenario
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Scenario ID
    pub scenario_id: String,
    /// Whether attack was detected
    pub detected: bool,
    /// Detection latency (ms from attack start)
    pub detection_latency_ms: Option<u64>,
    /// Signals that triggered detection
    pub triggered_signals: Vec<String>,
    /// False positive indicators
    pub false_positives: Vec<String>,
    /// System impact metrics
    pub impact: AttackImpact,
    /// Execution time
    pub execution_time_ms: u64,
}

/// Impact of an attack on the system
#[derive(Debug, Clone, Default)]
pub struct AttackImpact {
    /// Trust score changes (agent_id -> delta)
    pub trust_changes: HashMap<String, f64>,
    /// KREDIT drained
    pub kredit_drained: f64,
    /// Proposals affected
    pub proposals_affected: usize,
    /// Agents compromised
    pub agents_compromised: usize,
    /// System availability impact (0.0 = none, 1.0 = total outage)
    pub availability_impact: f64,
}

impl AttackCampaign {
    /// Create a new attack campaign
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("campaign-{}-{}", name.to_lowercase().replace(' ', "-"), now),
            name,
            scenarios: vec![],
            config: CampaignConfig::default(),
            history: vec![],
        }
    }

    /// Set campaign configuration
    pub fn with_config(mut self, config: CampaignConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an attack scenario
    pub fn add_scenario(&mut self, scenario: AttackScenario) {
        self.scenarios.push(scenario);
    }

    /// Get all scenarios
    pub fn scenarios(&self) -> &[AttackScenario] {
        &self.scenarios
    }

    /// Get execution history
    pub fn history(&self) -> &[ScenarioResult] {
        &self.history
    }

    /// Execute the campaign against a simulated system
    pub fn execute(&mut self, system: &mut SimulatedSystem) -> CampaignResults {
        let start_time = std::time::Instant::now();
        let mut results = vec![];

        for scenario in &self.scenarios {
            // Check preconditions
            let preconditions_met = scenario
                .preconditions
                .iter()
                .all(|p| system.check_precondition(p));

            if !preconditions_met {
                results.push(ScenarioResult {
                    scenario_id: scenario.id.clone(),
                    detected: false,
                    detection_latency_ms: None,
                    triggered_signals: vec![],
                    false_positives: vec!["preconditions_not_met".into()],
                    impact: AttackImpact::default(),
                    execution_time_ms: 0,
                });
                continue;
            }

            // Execute attack
            let scenario_start = std::time::Instant::now();
            let result = system.execute_attack(scenario);
            let execution_time = scenario_start.elapsed().as_millis() as u64;

            let scenario_result = ScenarioResult {
                scenario_id: scenario.id.clone(),
                detected: result.detected,
                detection_latency_ms: result.detection_latency_ms,
                triggered_signals: result.triggered_signals,
                false_positives: result.false_positives,
                impact: result.impact,
                execution_time_ms: execution_time,
            };

            results.push(scenario_result.clone());
            self.history.push(scenario_result);

            if self.config.stop_on_detection && result.detected {
                break;
            }

            // Inter-scenario delay (simulated)
            if self.config.sequential && self.config.inter_scenario_delay_ms > 0 {
                // In real execution, we'd sleep here
            }
        }

        let total_time = start_time.elapsed().as_millis() as u64;

        CampaignResults {
            campaign_id: self.id.clone(),
            scenarios_executed: results.len(),
            scenarios_detected: results.iter().filter(|r| r.detected).count(),
            detection_rate: if results.is_empty() {
                0.0
            } else {
                results.iter().filter(|r| r.detected).count() as f64 / results.len() as f64
            },
            avg_detection_latency_ms: {
                let detected: Vec<_> = results
                    .iter()
                    .filter_map(|r| r.detection_latency_ms)
                    .collect();
                if detected.is_empty() {
                    None
                } else {
                    Some(detected.iter().sum::<u64>() / detected.len() as u64)
                }
            },
            false_positive_rate: {
                let total_fps: usize = results.iter().map(|r| r.false_positives.len()).sum();
                total_fps as f64 / results.len().max(1) as f64
            },
            total_impact: aggregate_impact(&results),
            scenario_results: results,
            total_execution_time_ms: total_time,
        }
    }
}

/// Aggregate impact from multiple scenario results
fn aggregate_impact(results: &[ScenarioResult]) -> AttackImpact {
    let mut aggregate = AttackImpact::default();

    for result in results {
        for (agent, delta) in &result.impact.trust_changes {
            *aggregate.trust_changes.entry(agent.clone()).or_insert(0.0) += delta;
        }
        aggregate.kredit_drained += result.impact.kredit_drained;
        aggregate.proposals_affected += result.impact.proposals_affected;
        aggregate.agents_compromised += result.impact.agents_compromised;
        aggregate.availability_impact = aggregate
            .availability_impact
            .max(result.impact.availability_impact);
    }

    aggregate
}

/// Results of executing a campaign
#[derive(Debug, Clone)]
pub struct CampaignResults {
    /// Campaign ID
    pub campaign_id: String,
    /// Number of scenarios executed
    pub scenarios_executed: usize,
    /// Number of scenarios where attack was detected
    pub scenarios_detected: usize,
    /// Detection rate (0.0-1.0)
    pub detection_rate: f64,
    /// Average detection latency
    pub avg_detection_latency_ms: Option<u64>,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Aggregate impact
    pub total_impact: AttackImpact,
    /// Individual scenario results
    pub scenario_results: Vec<ScenarioResult>,
    /// Total execution time
    pub total_execution_time_ms: u64,
}

impl CampaignResults {
    /// Generate a security report
    pub fn security_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("# Security Test Report: {}\n\n", self.campaign_id));
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- Scenarios Executed: {}\n",
            self.scenarios_executed
        ));
        report.push_str(&format!(
            "- Detection Rate: {:.1}%\n",
            self.detection_rate * 100.0
        ));
        report.push_str(&format!(
            "- False Positive Rate: {:.2}\n",
            self.false_positive_rate
        ));

        if let Some(latency) = self.avg_detection_latency_ms {
            report.push_str(&format!("- Avg Detection Latency: {}ms\n", latency));
        }

        report.push_str("\n## Impact Assessment\n\n");
        report.push_str(&format!(
            "- Agents with Trust Changes: {}\n",
            self.total_impact.trust_changes.len()
        ));
        report.push_str(&format!(
            "- KREDIT Drained: {:.2}\n",
            self.total_impact.kredit_drained
        ));
        report.push_str(&format!(
            "- Proposals Affected: {}\n",
            self.total_impact.proposals_affected
        ));
        report.push_str(&format!(
            "- Agents Compromised: {}\n",
            self.total_impact.agents_compromised
        ));

        report.push_str("\n## Scenario Details\n\n");
        for result in &self.scenario_results {
            report.push_str(&format!("### {}\n", result.scenario_id));
            report.push_str(&format!(
                "- Detected: {}\n",
                if result.detected { "YES" } else { "NO" }
            ));
            if let Some(latency) = result.detection_latency_ms {
                report.push_str(&format!("- Latency: {}ms\n", latency));
            }
            if !result.triggered_signals.is_empty() {
                report.push_str(&format!(
                    "- Signals: {}\n",
                    result.triggered_signals.join(", ")
                ));
            }
            report.push('\n');
        }

        report
    }

    /// Check if all attacks were detected
    pub fn all_detected(&self) -> bool {
        self.detection_rate >= 1.0
    }

    /// Get undetected attack types
    pub fn undetected_scenarios(&self) -> Vec<&str> {
        self.scenario_results
            .iter()
            .filter(|r| !r.detected)
            .map(|r| r.scenario_id.as_str())
            .collect()
    }
}

// ============================================================================
// Simulated System
// ============================================================================

/// A simulated system for testing attacks
#[allow(dead_code)]
pub struct SimulatedSystem {
    /// Agents in the system
    agents: HashMap<String, SimulatedAgent>,
    /// Groups in the system
    groups: HashMap<String, HashSet<String>>,
    /// Gaming detector
    detector: GamingDetector,
    /// Event log
    events: VecDeque<SystemEvent>,
    /// Current simulated time (ms)
    current_time: u64,
}

/// A simulated agent
#[derive(Debug, Clone)]
pub struct SimulatedAgent {
    /// Agent ID
    pub id: String,
    /// Trust score
    pub trust: f64,
    /// KREDIT balance
    pub kredit: f64,
    /// Behavior log
    pub behaviors: Vec<BehaviorLogEntry>,
    /// Whether agent is compromised
    pub compromised: bool,
}

/// System event
#[derive(Debug, Clone)]
pub struct SystemEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Event type
    pub event_type: String,
    /// Agent involved
    pub agent_id: Option<String>,
    /// Event data
    pub data: HashMap<String, String>,
}

/// Attack execution result (internal)
struct AttackExecutionResult {
    detected: bool,
    detection_latency_ms: Option<u64>,
    triggered_signals: Vec<String>,
    false_positives: Vec<String>,
    impact: AttackImpact,
}

impl SimulatedSystem {
    /// Create a new simulated system
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            groups: HashMap::new(),
            detector: GamingDetector::new(Default::default()),
            events: VecDeque::new(),
            current_time: 0,
        }
    }

    /// Add an agent to the system
    pub fn add_agent(&mut self, id: impl Into<String>, trust: f64, kredit: f64) {
        let id = id.into();
        self.agents.insert(
            id.clone(),
            SimulatedAgent {
                id,
                trust,
                kredit,
                behaviors: vec![],
                compromised: false,
            },
        );
    }

    /// Create a group
    pub fn create_group(&mut self, group_id: impl Into<String>, member_ids: Vec<String>) {
        self.groups
            .insert(group_id.into(), member_ids.into_iter().collect());
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get total system trust
    pub fn total_trust(&self) -> f64 {
        self.agents.values().map(|a| a.trust).sum()
    }

    /// Check if a precondition is met
    fn check_precondition(&self, precondition: &Precondition) -> bool {
        match precondition {
            Precondition::MinAgents(n) => self.agents.len() >= *n,
            Precondition::MinTrust(t) => self.total_trust() >= *t,
            Precondition::AgentExists(id) => self.agents.contains_key(id),
            Precondition::GroupExists(id) => self.groups.contains_key(id),
            Precondition::Custom(_) => true, // Custom preconditions always pass in simulation
        }
    }

    /// Execute an attack scenario
    fn execute_attack(&mut self, scenario: &AttackScenario) -> AttackExecutionResult {
        let _attack_start = self.current_time;
        let mut impact = AttackImpact::default();
        let mut triggered_signals = vec![];

        match &scenario.config {
            AttackConfig::Sybil(config) => {
                // Create fake identities
                for i in 0..config.num_identities {
                    let fake_id = format!("sybil-{}-{}", scenario.id, i);
                    self.add_agent(&fake_id, config.initial_trust, 100.0);
                    impact.agents_compromised += 1;
                }

                // Check for detection signals
                if config.coordination_level > 0.6 {
                    triggered_signals.push("coordinated_voting".into());
                }
                if config.uniform_behavior {
                    triggered_signals.push("similar_behavior_patterns".into());
                }
                if config.creation_interval_ms < 5000 {
                    triggered_signals.push("rapid_identity_creation".into());
                }
            }

            AttackConfig::Collusion(config) => {
                // Mark agents as compromised
                for id in &config.colluder_ids {
                    if let Some(agent) = self.agents.get_mut(id) {
                        agent.compromised = true;
                        impact.agents_compromised += 1;
                    }
                }

                // Check for detection signals
                if config.visibility > 0.5 {
                    triggered_signals.push("uniform_voting".into());
                }
                match config.strategy {
                    CollusionStrategy::TrustRing => {
                        triggered_signals.push("trust_ring_detection".into());
                    }
                    CollusionStrategy::ProposalFlood => {
                        impact.proposals_affected = config.colluder_ids.len() * 3;
                    }
                    _ => {}
                }
            }

            AttackConfig::TrustManipulation(config) => {
                // Apply trust changes
                if let Some(agent) = self.agents.get_mut(&config.attacker_id) {
                    let delta = match config.manipulation_type {
                        ManipulationType::SelfInflation => config.intensity * 0.3,
                        ManipulationType::StealthInflation => config.intensity * 0.1,
                        ManipulationType::Oscillation => 0.0, // Net zero
                        ManipulationType::TargetDeflation => 0.0,
                    };
                    agent.trust = (agent.trust + delta).clamp(0.0, 1.0);
                    impact
                        .trust_changes
                        .insert(config.attacker_id.clone(), delta);
                }

                // Apply target deflation
                if let (ManipulationType::TargetDeflation, Some(target)) =
                    (&config.manipulation_type, &config.target_agent)
                {
                    if let Some(agent) = self.agents.get_mut(target) {
                        let delta = -config.intensity * 0.2;
                        agent.trust = (agent.trust + delta).clamp(0.0, 1.0);
                        impact.trust_changes.insert(target.clone(), delta);
                    }
                }

                // Detection signals
                if config.intensity > 0.7 {
                    triggered_signals.push("abnormal_trust_velocity".into());
                }
                if matches!(config.manipulation_type, ManipulationType::SelfInflation) {
                    triggered_signals.push("success_rate_anomaly".into());
                }
            }

            AttackConfig::KreditDrain(config) => {
                // Drain KREDIT
                let drain_amount = config.drain_rate * (config.parallel_channels as f64);
                if let Some(agent) = self.agents.get_mut(&config.target) {
                    let actual_drain = drain_amount.min(agent.kredit);
                    agent.kredit -= actual_drain;
                    impact.kredit_drained = actual_drain;
                }

                // Detection signals
                if config.drain_rate > 50.0 {
                    triggered_signals.push("high_drain_rate".into());
                }
                if config.use_micro_transactions {
                    triggered_signals.push("micro_transaction_pattern".into());
                }
            }

            AttackConfig::VoteFlood(config) => {
                // Simulate vote flood impact
                impact.proposals_affected = (config.vote_rate as usize).min(100);
                impact.availability_impact = (config.vote_rate / 1000.0).min(1.0);

                // Detection signals
                if config.vote_rate > 50.0 {
                    triggered_signals.push("vote_flood_detected".into());
                }
                if config.attacker_ids.len() > 3 {
                    triggered_signals.push("coordinated_flood".into());
                }
            }

            AttackConfig::Custom(_) => {
                // Custom attacks don't have predefined behavior
            }
        }

        // Determine if attack was detected
        let detected = !triggered_signals.is_empty();
        let detection_latency = if detected {
            Some(100 + (rand_simple() * 500.0) as u64) // Simulated latency
        } else {
            None
        };

        // Log event
        self.events.push_back(SystemEvent {
            timestamp: self.current_time,
            event_type: format!("attack_{:?}", scenario.attack_type).to_lowercase(),
            agent_id: None,
            data: [
                ("scenario_id".to_string(), scenario.id.clone()),
                ("detected".to_string(), detected.to_string()),
            ]
            .into_iter()
            .collect(),
        });

        self.current_time += 1000; // Advance time

        AttackExecutionResult {
            detected,
            detection_latency_ms: detection_latency,
            triggered_signals,
            false_positives: vec![],
            impact,
        }
    }

    /// Get system events
    pub fn events(&self) -> &VecDeque<SystemEvent> {
        &self.events
    }

    /// Reset the system
    pub fn reset(&mut self) {
        self.agents.clear();
        self.groups.clear();
        self.events.clear();
        self.current_time = 0;
    }
}

impl Default for SimulatedSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Simple pseudo-random for simulation (deterministic)
fn rand_simple() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    now.hash(&mut hasher);
    let hash = hasher.finish();

    (hash as f64) / (u64::MAX as f64)
}

// ============================================================================
// Pre-built Attack Suites
// ============================================================================

/// Pre-built attack suites for common testing scenarios
pub mod suites {
    use super::*;

    /// Basic security test suite
    pub fn basic_security_suite() -> AttackCampaign {
        let mut campaign = AttackCampaign::new("basic-security");

        // Small Sybil attack
        campaign.add_scenario(AttackScenario::sybil(SybilAttackConfig {
            num_identities: 3,
            coordination_level: 0.5,
            ..Default::default()
        }));

        // Simple collusion
        campaign.add_scenario(AttackScenario::collusion(CollusionAttackConfig {
            colluder_ids: vec!["agent-1".into(), "agent-2".into()],
            strategy: CollusionStrategy::UniformVoting,
            visibility: 0.7,
            ..Default::default()
        }));

        // Trust manipulation
        campaign.add_scenario(AttackScenario::trust_manipulation(
            TrustManipulationConfig {
                attacker_id: "attacker".into(),
                manipulation_type: ManipulationType::SelfInflation,
                intensity: 0.5,
                ..Default::default()
            },
        ));

        campaign
    }

    /// Comprehensive security test suite
    pub fn comprehensive_security_suite() -> AttackCampaign {
        let mut campaign = AttackCampaign::new("comprehensive-security");

        // Multiple Sybil variants
        for num in [3, 10, 25] {
            campaign.add_scenario(AttackScenario::sybil(SybilAttackConfig {
                num_identities: num,
                coordination_level: 0.8,
                ..Default::default()
            }));
        }

        // Multiple collusion strategies
        for strategy in [
            CollusionStrategy::UniformVoting,
            CollusionStrategy::TrustRing,
            CollusionStrategy::ProposalFlood,
        ] {
            campaign.add_scenario(AttackScenario::collusion(CollusionAttackConfig {
                colluder_ids: vec!["c1".into(), "c2".into(), "c3".into()],
                strategy,
                visibility: 0.5,
                ..Default::default()
            }));
        }

        // All trust manipulation types
        for manip_type in [
            ManipulationType::SelfInflation,
            ManipulationType::StealthInflation,
            ManipulationType::Oscillation,
        ] {
            campaign.add_scenario(AttackScenario::trust_manipulation(
                TrustManipulationConfig {
                    attacker_id: "attacker".into(),
                    manipulation_type: manip_type,
                    intensity: 0.7,
                    ..Default::default()
                },
            ));
        }

        campaign
    }

    /// Stress test suite for high-load scenarios
    pub fn stress_test_suite() -> AttackCampaign {
        let mut campaign = AttackCampaign::new("stress-test");

        // Large Sybil attack
        campaign.add_scenario(AttackScenario::sybil(SybilAttackConfig {
            num_identities: 50,
            coordination_level: 0.9,
            creation_interval_ms: 100,
            ..Default::default()
        }));

        // Large collusion ring
        campaign.add_scenario(AttackScenario::collusion(CollusionAttackConfig {
            colluder_ids: (0..20).map(|i| format!("colluder-{}", i)).collect(),
            strategy: CollusionStrategy::TrustRing,
            visibility: 0.3,
            ..Default::default()
        }));

        campaign
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_attack_scenario() {
        let scenario = AttackScenario::sybil(SybilAttackConfig {
            num_identities: 5,
            ..Default::default()
        });

        assert_eq!(scenario.attack_type, AttackType::Sybil);
        assert_eq!(scenario.severity, AttackSeverity::Low);
        assert!(!scenario.expected_signals.is_empty());
    }

    #[test]
    fn test_attack_campaign() {
        let mut campaign = AttackCampaign::new("test");

        campaign.add_scenario(AttackScenario::sybil(SybilAttackConfig {
            num_identities: 3,
            ..Default::default()
        }));

        campaign.add_scenario(AttackScenario::collusion(CollusionAttackConfig {
            colluder_ids: vec!["a".into(), "b".into()],
            ..Default::default()
        }));

        assert_eq!(campaign.scenarios().len(), 2);
    }

    #[test]
    fn test_simulated_system() {
        let mut system = SimulatedSystem::new();

        system.add_agent("agent-1", 0.8, 1000.0);
        system.add_agent("agent-2", 0.6, 500.0);

        assert_eq!(system.agent_count(), 2);
        assert!((system.total_trust() - 1.4).abs() < 0.01);
    }

    #[test]
    fn test_execute_campaign() {
        let mut system = SimulatedSystem::new();
        system.add_agent("agent-1", 0.8, 1000.0);
        system.add_agent("agent-2", 0.6, 500.0);
        system.add_agent("agent-3", 0.7, 750.0);
        system.add_agent("attacker", 0.5, 500.0);

        let mut campaign = suites::basic_security_suite();
        let results = campaign.execute(&mut system);

        assert!(results.scenarios_executed > 0);
        // At least some attacks should be detected with obvious signals
        assert!(results.detection_rate > 0.0);
    }

    #[test]
    fn test_security_report() {
        let results = CampaignResults {
            campaign_id: "test".into(),
            scenarios_executed: 3,
            scenarios_detected: 2,
            detection_rate: 0.667,
            avg_detection_latency_ms: Some(150),
            false_positive_rate: 0.1,
            total_impact: AttackImpact::default(),
            scenario_results: vec![],
            total_execution_time_ms: 1000,
        };

        let report = results.security_report();
        assert!(report.contains("Detection Rate: 66.7%"));
    }
}
