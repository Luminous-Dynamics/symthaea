//! # Integration Flows
//!
//! End-to-end integration of all ecosystem modules demonstrating how
//! components work together in production scenarios.
//!
//! ## Flows Implemented
//!
//! 1. **Trust Pipeline** - K-Vector → Attestation → Consensus → Update
//! 2. **Attack Response** - Detection → Thresholds → Quarantine → Slash
//! 3. **Privacy Analytics** - DP Aggregation → Dashboard → Alerts
//! 4. **Agent Lifecycle** - Create → Output → Classify → Measure Phi → Update Trust

use crate::matl::KVector;
use crate::agentic::{
    AgentId, InstrumentalActor, AgentClass, AgentConstraints, AgentStatus,
    EpistemicStats, UncertaintyCalibration, ActionOutcome,
    // Trust pipeline
    kvector_bridge::{compute_trust_score, calculate_kredit_from_trust, BehaviorAnalysis},
    // Attack detection
    attack_detection::{StreamingAnalyzer, AttackDetectionConfig},
    // Adaptive thresholds
    adaptive_thresholds::{AdaptiveThresholdEngine, AdaptiveConfig, ThresholdType, ThresholdFeedback, FeedbackOutcome, FeedbackContext},
    // Quarantine
    adversarial::{QuarantineManager, QuarantineReason, GamingDetector, GamingDetectionConfig},
    // Economics
    economics::{SlashingEngine, SlashingConfig, RewardEngine, RewardConfig, ViolationType, ViolationSeverity, SlashResult},
    // Differential privacy
    differential_privacy::{PrivateTrustAnalytics, DPConfig, TrustDistribution},
    // Dashboard
    dashboard::{Dashboard, DashboardConfig, MetricsInput, LiveMetrics, AlertSeverity},
    // Cascade analysis
    cascade_analysis::{CascadeEngine, CascadeConfig, NetworkAgent, NetworkEdge, EdgeType, CascadeResult},
    // Verification
    verification::{VerificationEngine, SystemState, InvariantCheckResult},
    // Epistemic
    epistemic_classifier::{AgentOutput, AgentOutputBuilder, OutputContent, calculate_epistemic_weight},
    // Phi measurement
    phi_bridge::{measure_phi_simple, check_coherence_for_action, PhiMeasurementConfig},
};
use crate::epistemic::{EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Flow 1: Trust Pipeline Integration
// ============================================================================

/// Configuration for the integrated trust pipeline
#[derive(Debug, Clone)]
pub struct TrustPipelineConfig {
    /// Minimum trust to participate
    pub min_trust_threshold: f64,
    /// Consensus approval threshold
    pub approval_threshold: f64,
    /// Enable quadratic voting
    pub quadratic_voting: bool,
    /// KREDIT base multiplier
    pub kredit_base: u64,
}

impl Default for TrustPipelineConfig {
    fn default() -> Self {
        Self {
            min_trust_threshold: 0.3,
            approval_threshold: 0.5,
            quadratic_voting: true,
            kredit_base: 10000,
        }
    }
}

/// Integrated trust pipeline connecting all trust-related modules
pub struct IntegratedTrustPipeline {
    config: TrustPipelineConfig,
    /// Agents in the pipeline
    agents: HashMap<String, InstrumentalActor>,
    /// Cascade engine for trust propagation
    cascade_engine: CascadeEngine,
    /// Verification engine for invariants
    verification: VerificationEngine,
}

impl IntegratedTrustPipeline {
    /// Create new integrated pipeline
    pub fn new(config: TrustPipelineConfig) -> Self {
        let cascade_config = CascadeConfig::default();
        Self {
            config,
            agents: HashMap::new(),
            cascade_engine: CascadeEngine::new(cascade_config),
            verification: VerificationEngine::with_defaults(),
        }
    }

    /// Register an agent in the pipeline
    pub fn register_agent(&mut self, agent: InstrumentalActor) {
        let agent_id = agent.agent_id.as_str().to_string();
        let trust = agent.k_vector.trust_score();

        // Add to cascade network
        self.cascade_engine.network_mut().add_agent(NetworkAgent::new(
            agent_id.clone(),
            trust as f64,
        ));

        self.agents.insert(agent_id, agent);
    }

    /// Process a trust attestation between agents
    pub fn process_attestation(
        &mut self,
        from_agent: &str,
        to_agent: &str,
        attestation_weight: f64,
    ) -> Result<AttestationResult, IntegrationError> {
        // Verify both agents exist
        let from = self.agents.get(from_agent)
            .ok_or_else(|| IntegrationError::AgentNotFound(from_agent.to_string()))?;
        let _to = self.agents.get(to_agent)
            .ok_or_else(|| IntegrationError::AgentNotFound(to_agent.to_string()))?;

        // Check from_agent has sufficient trust
        let from_trust = from.k_vector.trust_score();
        if from_trust < self.config.min_trust_threshold as f32 {
            return Err(IntegrationError::InsufficientTrust {
                agent: from_agent.to_string(),
                required: self.config.min_trust_threshold,
                actual: from_trust as f64,
            });
        }

        // Add edge to cascade network
        self.cascade_engine.network_mut().add_edge(NetworkEdge {
            from: from_agent.to_string(),
            to: to_agent.to_string(),
            weight: attestation_weight,
            edge_type: EdgeType::Attestation,
        });

        // Calculate trust impact using quadratic weighting if enabled
        let weight = if self.config.quadratic_voting {
            (from_trust as f64).sqrt() * attestation_weight
        } else {
            from_trust as f64 * attestation_weight
        };

        // Update to_agent's trust
        let to_agent_mut = self.agents.get_mut(to_agent).unwrap();
        let old_trust = to_agent_mut.k_vector.trust_score();

        // Apply bounded update to k_r dimension
        let new_k_r = (to_agent_mut.k_vector.k_r + weight as f32 * 0.1).clamp(0.0, 1.0);
        to_agent_mut.k_vector.k_r = new_k_r;

        let new_trust = to_agent_mut.k_vector.trust_score();

        // Recalculate KREDIT cap
        let new_kredit_cap = calculate_kredit_from_trust(new_trust as f64);
        to_agent_mut.kredit_cap = (new_kredit_cap * self.config.kredit_base as f64) as u64;

        Ok(AttestationResult {
            from_agent: from_agent.to_string(),
            to_agent: to_agent.to_string(),
            weight,
            old_trust: old_trust as f64,
            new_trust: new_trust as f64,
            new_kredit_cap: to_agent_mut.kredit_cap,
        })
    }

    /// Simulate trust cascade from a shock
    pub fn simulate_cascade(&mut self, agent_id: &str, shock_magnitude: f64) -> CascadeResult {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.cascade_engine.apply_shock(agent_id, shock_magnitude, timestamp)
    }

    /// Verify all invariants hold
    pub fn verify_invariants(&mut self) -> Vec<InvariantCheckResult> {
        let state = self.create_system_state();
        self.verification.check_invariants(&state)
    }

    fn create_system_state(&self) -> SystemState {
        let trust_scores: HashMap<String, f64> = self.agents
            .iter()
            .map(|(id, a)| (id.clone(), a.k_vector.trust_score() as f64))
            .collect();

        SystemState {
            index: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            trust_scores,
            byzantine_count: 0,
            network_health: 1.0,
            variables: HashMap::new(),
        }
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&InstrumentalActor> {
        self.agents.get(agent_id)
    }

    /// Get all agents
    pub fn agents(&self) -> &HashMap<String, InstrumentalActor> {
        &self.agents
    }
}

/// Result of processing an attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    pub from_agent: String,
    pub to_agent: String,
    pub weight: f64,
    pub old_trust: f64,
    pub new_trust: f64,
    pub new_kredit_cap: u64,
}

// ============================================================================
// Flow 2: Attack Response Integration
// ============================================================================

/// Configuration for integrated attack response
#[derive(Debug, Clone)]
pub struct AttackResponseConfig {
    /// Detection config
    pub detection: AttackDetectionConfig,
    /// Adaptive threshold config
    pub thresholds: AdaptiveConfig,
    /// Slashing config
    pub slashing: SlashingConfig,
    /// Gaming detection config
    pub gaming: GamingDetectionConfig,
}

impl Default for AttackResponseConfig {
    fn default() -> Self {
        Self {
            detection: AttackDetectionConfig::default(),
            thresholds: AdaptiveConfig::default(),
            slashing: SlashingConfig::default(),
            gaming: GamingDetectionConfig::default(),
        }
    }
}

/// Integrated attack response system
pub struct IntegratedAttackResponse {
    #[allow(dead_code)]
    config: AttackResponseConfig,
    /// Streaming attack analyzer
    #[allow(dead_code)]
    detector: StreamingAnalyzer,
    /// Adaptive thresholds
    thresholds: AdaptiveThresholdEngine,
    /// Quarantine manager
    quarantine: QuarantineManager,
    /// Slashing engine
    slashing: SlashingEngine,
    /// Gaming detector
    gaming: GamingDetector,
    /// Response history
    responses: Vec<AttackResponse>,
}

impl IntegratedAttackResponse {
    /// Create new attack response system
    pub fn new(config: AttackResponseConfig) -> Self {
        Self {
            detector: StreamingAnalyzer::new(config.detection.clone()),
            thresholds: AdaptiveThresholdEngine::new(config.thresholds.clone()),
            quarantine: QuarantineManager::new(),
            slashing: SlashingEngine::new(config.slashing.clone()),
            gaming: GamingDetector::new(config.gaming.clone()),
            config,
            responses: vec![],
        }
    }

    /// Process an agent's behavior and respond to any detected attacks
    pub fn process_behavior(
        &mut self,
        agent: &mut InstrumentalActor,
        _behavior: &BehaviorAnalysis,
    ) -> Option<AttackResponse> {
        let agent_id = agent.agent_id.as_str();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Check for gaming behavior using analyze method
        let gaming_result = self.gaming.analyze(agent, timestamp);

        if gaming_result.is_gaming {
            // Quarantine immediately with evidence
            self.quarantine.quarantine(
                agent_id,
                QuarantineReason::GamingDetected,
                gaming_result.indicators.iter().map(|i| i.description.clone()).collect(),
                timestamp,
            );

            // Slash based on severity
            let severity = if gaming_result.confidence > 0.8 {
                ViolationSeverity::Critical
            } else if gaming_result.confidence > 0.5 {
                ViolationSeverity::Major
            } else {
                ViolationSeverity::Minor
            };

            let slash_result = self.slashing.slash(
                agent_id,
                ViolationType::TrustGaming,
                severity,
                agent.kredit_balance.max(0) as u64,
                "Gaming behavior detected",
            );

            // Apply slash to agent
            if let SlashResult::Slashed { event, .. } = &slash_result {
                agent.kredit_balance -= event.amount_slashed as i64;
            }

            // Update adaptive thresholds with feedback
            self.thresholds.process_feedback(ThresholdFeedback {
                threshold_type: ThresholdType::Quarantine,
                threshold_value: self.thresholds.get_threshold(ThresholdType::Quarantine),
                outcome: FeedbackOutcome::TruePositive,
                context: FeedbackContext {
                    network_health: 0.8,
                    active_agents: 100,
                    threat_level: gaming_result.confidence,
                    ..Default::default()
                },
                timestamp,
            });

            let slashed_amount = if let SlashResult::Slashed { event, .. } = &slash_result {
                event.amount_slashed
            } else {
                0
            };

            let response = AttackResponse {
                agent_id: agent_id.to_string(),
                attack_type: "Gaming".to_string(),
                confidence: gaming_result.confidence,
                actions_taken: vec![
                    ResponseAction::Quarantined { duration_secs: 86400 },
                    ResponseAction::Slashed { amount: slashed_amount },
                ],
                timestamp,
            };

            self.responses.push(response.clone());
            return Some(response);
        }

        None
    }

    /// Check if agent is quarantined
    pub fn is_quarantined(&self, agent_id: &str) -> bool {
        self.quarantine.is_quarantined(agent_id)
    }

    /// Get current detection threshold
    pub fn get_threshold(&self, threshold_type: ThresholdType) -> f64 {
        self.thresholds.get_threshold(threshold_type)
    }

    /// Get response history
    pub fn responses(&self) -> &[AttackResponse] {
        &self.responses
    }
}

/// Response to a detected attack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackResponse {
    pub agent_id: String,
    pub attack_type: String,
    pub confidence: f64,
    pub actions_taken: Vec<ResponseAction>,
    pub timestamp: u64,
}

/// Actions taken in response to attack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseAction {
    Quarantined { duration_secs: u64 },
    Slashed { amount: u64 },
    TrustReduced { amount: f64 },
    AlertRaised { severity: String },
}

// ============================================================================
// Flow 3: Privacy-Preserving Analytics Integration
// ============================================================================

/// Configuration for privacy-preserving analytics
#[derive(Debug, Clone)]
pub struct PrivacyAnalyticsConfig {
    /// DP configuration
    pub dp: DPConfig,
    /// Dashboard configuration
    pub dashboard: DashboardConfig,
}

impl Default for PrivacyAnalyticsConfig {
    fn default() -> Self {
        Self {
            dp: DPConfig {
                epsilon: 1.0,
                delta: 1e-6,
                ..Default::default()
            },
            dashboard: DashboardConfig::default(),
        }
    }
}

/// Integrated privacy-preserving analytics system
pub struct IntegratedPrivacyAnalytics {
    #[allow(dead_code)]
    config: PrivacyAnalyticsConfig,
    /// Private trust analytics
    analytics: PrivateTrustAnalytics,
    /// Dashboard
    dashboard: Dashboard,
    /// Queries this epoch
    queries_this_epoch: usize,
}

impl IntegratedPrivacyAnalytics {
    /// Create new privacy analytics system
    pub fn new(config: PrivacyAnalyticsConfig) -> Self {
        Self {
            analytics: PrivateTrustAnalytics::new(config.dp.clone()),
            dashboard: Dashboard::new(config.dashboard.clone()),
            queries_this_epoch: 0,
            config,
        }
    }

    /// Compute private trust distribution and update dashboard
    pub fn analyze_and_display(
        &mut self,
        trust_scores: &[f64],
        phi_values: &[f64],
        threat_level: f64,
    ) -> Result<PrivateAnalyticsResult, IntegrationError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Compute private trust distribution
        let epsilon_per_query = 0.25; // Budget 4 queries
        let distribution = self.analytics
            .analyze_trust_distribution(trust_scores, epsilon_per_query)
            .map_err(|e| IntegrationError::PrivacyError(e.to_string()))?;

        self.queries_this_epoch += 1;

        // Update dashboard with metrics
        let metrics_input = MetricsInput {
            trust_scores: trust_scores.to_vec(),
            transaction_count: 0,
            alerts: vec![],
            phi_values: phi_values.to_vec(),
            threats: vec![threat_level],
        };

        let live_metrics = self.dashboard.update(metrics_input, timestamp);

        // Create alerts if needed
        if distribution.mean < 0.3 {
            self.dashboard.alerts.create_alert(
                AlertSeverity::High,
                "Low Average Trust",
                &format!("Private mean trust is {:.2}", distribution.mean),
                "privacy_analytics",
            );
        }

        Ok(PrivateAnalyticsResult {
            distribution,
            live_metrics,
            remaining_budget: self.analytics.remaining_budget(),
        })
    }

    /// Get dashboard reference
    pub fn dashboard(&self) -> &Dashboard {
        &self.dashboard
    }

    /// Reset privacy budget for new epoch
    pub fn reset_epoch(&mut self) {
        self.analytics.reset_budget();
        self.queries_this_epoch = 0;
    }
}

/// Result of privacy-preserving analytics
#[derive(Debug, Clone)]
pub struct PrivateAnalyticsResult {
    pub distribution: TrustDistribution,
    pub live_metrics: LiveMetrics,
    pub remaining_budget: (f64, f64),
}

// ============================================================================
// Flow 4: Epistemic Agent Lifecycle Integration
// ============================================================================

/// Configuration for epistemic agent lifecycle
#[derive(Debug, Clone)]
pub struct EpistemicLifecycleConfig {
    /// Phi threshold for high-stakes actions
    pub phi_threshold: f64,
    /// Minimum epistemic weight for trust updates
    pub min_epistemic_weight: f32,
    /// KREDIT base
    pub kredit_base: u64,
}

impl Default for EpistemicLifecycleConfig {
    fn default() -> Self {
        Self {
            phi_threshold: 0.6,
            min_epistemic_weight: 0.1,
            kredit_base: 10000,
        }
    }
}

/// Integrated epistemic agent lifecycle manager
pub struct IntegratedEpistemicLifecycle {
    config: EpistemicLifecycleConfig,
    /// Phi measurement config
    phi_config: PhiMeasurementConfig,
    /// Reward engine
    rewards: RewardEngine,
    /// Output history for Phi measurement
    output_vectors: Vec<Vec<f64>>,
}

impl IntegratedEpistemicLifecycle {
    /// Create new lifecycle manager
    pub fn new(config: EpistemicLifecycleConfig) -> Self {
        Self {
            config,
            phi_config: PhiMeasurementConfig::default(),
            rewards: RewardEngine::new(RewardConfig::default()),
            output_vectors: vec![],
        }
    }

    /// Create a new agent with initial trust profile
    pub fn create_agent(&self, sponsor_did: &str, agent_class: AgentClass) -> InstrumentalActor {
        let agent_id = AgentId::generate();
        let k_vector = KVector::new_participant();
        let trust = k_vector.trust_score();
        let kredit_cap = (calculate_kredit_from_trust(trust as f64) * self.config.kredit_base as f64) as u64;

        InstrumentalActor {
            agent_id,
            sponsor_did: sponsor_did.to_string(),
            agent_class,
            kredit_balance: (kredit_cap / 2) as i64, // Start with half cap
            kredit_cap,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_activity: 0,
            actions_this_hour: 0,
            k_vector,
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    /// Process an agent output through the full epistemic pipeline
    pub fn process_output(
        &mut self,
        agent: &mut InstrumentalActor,
        content: OutputContent,
    ) -> Result<OutputProcessingResult, IntegrationError> {
        let agent_id = agent.agent_id.as_str();

        // Build and classify output
        let output = AgentOutputBuilder::new(agent_id)
            .content(content)
            .classification(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            )
            .confidence(0.8)
            .build()
            .map_err(|e| IntegrationError::ClassificationError(e))?;

        // Calculate epistemic weight
        let weight = calculate_epistemic_weight(&output.classification);

        // Record output on agent
        agent.record_output(&output);

        // Convert output to vector for Phi measurement
        let output_vector = self.output_to_vector(&output);
        self.output_vectors.push(output_vector);

        // Measure Phi if we have enough outputs
        let phi_result = if self.output_vectors.len() >= 3 {
            let phi = measure_phi_simple(&self.output_vectors);
            Some(phi)
        } else {
            None
        };

        // Check coherence for high-stakes actions
        let coherence_ok = if let Some(phi) = phi_result {
            let check = check_coherence_for_action(
                phi,
                &self.phi_config,
                true, // high stakes
            );
            check.can_proceed
        } else {
            true // Allow if we don't have enough data yet
        };

        // Update K-Vector based on epistemic weight
        if weight >= self.config.min_epistemic_weight {
            // Increase k_r (reputation) based on output quality
            let delta = weight * 0.01;
            agent.k_vector.k_r = (agent.k_vector.k_r + delta).clamp(0.0, 1.0);

            // Update k_phi if we have Phi measurement
            if let Some(phi) = phi_result {
                agent.k_vector.k_phi = phi as f32;
            }
        }

        // Recalculate KREDIT cap
        let trust = agent.k_vector.trust_score();
        agent.kredit_cap = (calculate_kredit_from_trust(trust as f64) * self.config.kredit_base as f64) as u64;

        // Calculate reward
        let reward = self.rewards.calculate_participation_reward(agent_id, trust as f64);

        Ok(OutputProcessingResult {
            output_id: output.output_id,
            epistemic_weight: weight,
            phi: phi_result,
            coherence_ok,
            trust_delta: weight * 0.01,
            new_trust: trust as f64,
            new_kredit_cap: agent.kredit_cap,
            reward,
        })
    }

    fn output_to_vector(&self, output: &AgentOutput) -> Vec<f64> {
        // Convert epistemic classification to feature vector
        vec![
            output.classification.empirical as u8 as f64 / 4.0,
            output.classification.normative as u8 as f64 / 3.0,
            output.classification.materiality as u8 as f64 / 3.0,
            output.classification.harmonic.map(|h| h as u8 as f64 / 4.0).unwrap_or(0.0),
            output.classification_confidence as f64,
        ]
    }

    /// Verify an agent's output and update trust accordingly
    pub fn verify_output(
        &mut self,
        agent: &mut InstrumentalActor,
        output_id: &str,
        correct: bool,
    ) {
        use crate::agentic::VerificationOutcome;

        let outcome = if correct {
            VerificationOutcome::Correct
        } else {
            VerificationOutcome::Incorrect
        };

        agent.verify_output(output_id, outcome);

        // Adjust trust based on verification
        if correct {
            agent.k_vector.k_r = (agent.k_vector.k_r + 0.02).clamp(0.0, 1.0);
        } else {
            agent.k_vector.k_r = (agent.k_vector.k_r - 0.05).clamp(0.0, 1.0);
        }

        // Update KREDIT cap
        let trust = agent.k_vector.trust_score();
        agent.kredit_cap = (calculate_kredit_from_trust(trust as f64) * self.config.kredit_base as f64) as u64;
    }
}

/// Result of processing an agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputProcessingResult {
    pub output_id: String,
    pub epistemic_weight: f32,
    pub phi: Option<f64>,
    pub coherence_ok: bool,
    pub trust_delta: f32,
    pub new_trust: f64,
    pub new_kredit_cap: u64,
    pub reward: u64,
}

// ============================================================================
// Errors
// ============================================================================

/// Integration errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum IntegrationError {
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Insufficient trust: {agent} has {actual}, needs {required}")]
    InsufficientTrust {
        agent: String,
        required: f64,
        actual: f64,
    },

    #[error("Privacy error: {0}")]
    PrivacyError(String),

    #[error("Classification error: {0}")]
    ClassificationError(String),

    #[error("Verification error: {0}")]
    VerificationError(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent(id: &str) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(0.6, 0.5, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.5),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_trust_pipeline_attestation() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        let agent1 = create_test_agent("agent-1");
        let agent2 = create_test_agent("agent-2");

        pipeline.register_agent(agent1);
        pipeline.register_agent(agent2);

        let result = pipeline.process_attestation("agent-1", "agent-2", 0.8).unwrap();

        assert!(result.new_trust >= result.old_trust, "Trust should increase after positive attestation");
        assert!(result.new_kredit_cap > 0, "KREDIT cap should be positive");
    }

    #[test]
    fn test_trust_pipeline_cascade() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        // Create chain of agents
        for i in 0..5 {
            let agent = create_test_agent(&format!("agent-{}", i));
            pipeline.register_agent(agent);
        }

        // Create attestation chain
        for i in 0..4 {
            let _ = pipeline.process_attestation(
                &format!("agent-{}", i),
                &format!("agent-{}", i + 1),
                0.8,
            );
        }

        // Apply shock and verify cascade
        let result = pipeline.simulate_cascade("agent-0", 0.5);
        assert!(result.agents_affected > 0, "Cascade should affect agents");
    }

    #[test]
    fn test_trust_pipeline_verification() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        let agent = create_test_agent("agent-1");
        pipeline.register_agent(agent);

        let results = pipeline.verify_invariants();

        // All invariants should hold for valid agent
        for result in &results {
            assert!(result.holds, "Invariant {} should hold", result.invariant_id);
        }
    }

    #[test]
    fn test_privacy_analytics() {
        let config = PrivacyAnalyticsConfig::default();
        let mut analytics = IntegratedPrivacyAnalytics::new(config);

        let trust_scores = vec![0.5, 0.6, 0.7, 0.4, 0.55, 0.65];
        let phi_values = vec![0.7, 0.65];
        let threat_level = 0.1;

        let result = analytics.analyze_and_display(&trust_scores, &phi_values, threat_level);

        assert!(result.is_ok(), "Analysis should succeed");
        let result = result.unwrap();

        // Private mean should be in reasonable range
        assert!(result.distribution.mean > 0.0 && result.distribution.mean < 1.0);

        // Should have remaining budget
        assert!(result.remaining_budget.0 > 0.0);
    }

    #[test]
    fn test_epistemic_lifecycle_create() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        let agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

        assert!(agent.kredit_cap > 0, "New agent should have KREDIT cap");
        assert!(agent.kredit_balance > 0, "New agent should have initial KREDIT");
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_epistemic_lifecycle_output() {
        let config = EpistemicLifecycleConfig::default();
        let mut lifecycle = IntegratedEpistemicLifecycle::new(config);

        let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);
        let initial_trust = agent.k_vector.trust_score();

        // Process several outputs
        for i in 0..5 {
            let result = lifecycle.process_output(
                &mut agent,
                OutputContent::Text(format!("Output {}", i)),
            );
            assert!(result.is_ok());
        }

        // Trust should have increased
        let final_trust = agent.k_vector.trust_score();
        assert!(final_trust >= initial_trust, "Trust should not decrease for valid outputs");

        // Should have outputs recorded
        assert!(agent.epistemic_stats.total_outputs >= 5);
    }

    #[test]
    fn test_epistemic_lifecycle_verification() {
        let config = EpistemicLifecycleConfig::default();
        let mut lifecycle = IntegratedEpistemicLifecycle::new(config);

        let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

        // Process output
        let result = lifecycle.process_output(
            &mut agent,
            OutputContent::Text("Test output".to_string()),
        ).unwrap();

        let initial_trust = agent.k_vector.trust_score();

        // Verify as correct
        lifecycle.verify_output(&mut agent, &result.output_id, true);
        let trust_after_correct = agent.k_vector.trust_score();
        assert!(trust_after_correct > initial_trust, "Trust should increase after correct verification");

        // Verify another as incorrect
        let result2 = lifecycle.process_output(
            &mut agent,
            OutputContent::Text("Another output".to_string()),
        ).unwrap();

        lifecycle.verify_output(&mut agent, &result2.output_id, false);
        let trust_after_incorrect = agent.k_vector.trust_score();
        assert!(trust_after_incorrect < trust_after_correct, "Trust should decrease after incorrect verification");
    }

    #[test]
    fn test_full_integration_flow() {
        // Create all systems
        let mut trust_pipeline = IntegratedTrustPipeline::new(TrustPipelineConfig::default());
        let _attack_response = IntegratedAttackResponse::new(AttackResponseConfig::default());
        let mut privacy_analytics = IntegratedPrivacyAnalytics::new(PrivacyAnalyticsConfig::default());
        let epistemic_lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());

        // Create agents
        let agent1 = epistemic_lifecycle.create_agent("did:sponsor:1", AgentClass::Supervised);
        let agent2 = epistemic_lifecycle.create_agent("did:sponsor:2", AgentClass::Supervised);

        // Register in trust pipeline
        trust_pipeline.register_agent(agent1.clone());
        trust_pipeline.register_agent(agent2.clone());

        // Process attestation
        let attest_result = trust_pipeline.process_attestation(
            agent1.agent_id.as_str(),
            agent2.agent_id.as_str(),
            0.9,
        ).unwrap();

        // Collect trust scores for privacy analytics
        let trust_scores: Vec<f64> = trust_pipeline.agents()
            .values()
            .map(|a| a.k_vector.trust_score() as f64)
            .collect();

        // Analyze with privacy
        let analytics_result = privacy_analytics.analyze_and_display(
            &trust_scores,
            &[0.7],
            0.1,
        ).unwrap();

        // Verify invariants
        let invariant_results = trust_pipeline.verify_invariants();

        // All systems worked together
        assert!(attest_result.new_trust > 0.0);
        assert!(analytics_result.distribution.mean > 0.0);
        assert!(invariant_results.iter().all(|r| r.holds));
    }
}
