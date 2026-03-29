// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! 4. **Agent Lifecycle** - Create → Output → Classify → Update Trust

use crate::agentic::{
    // Adaptive thresholds
    adaptive_thresholds::{
        AdaptiveConfig, AdaptiveThresholdEngine, FeedbackContext, FeedbackOutcome,
        ThresholdFeedback, ThresholdType,
    },
    // Quarantine
    adversarial::{
        GamingDetectionConfig, GamingDetector, GamingResponse, QuarantineManager, QuarantineReason,
    },
    // Cascade analysis
    cascade_analysis::{
        CascadeConfig, CascadeEngine, CascadeResult, EdgeType, NetworkAgent, NetworkEdge,
    },
    // Coherence
    coherence_bridge::{
        check_zk_operation_coherence, coherence_weighted_attestation, export_coherence_metrics,
        update_agent_coherence, CoherenceExport, CoherenceHistory, CoherenceMeasurementConfig,
        CoherenceState, ZKCoherenceGatingResult, ZKOperationType,
    },
    // Dashboard
    dashboard::{AlertSeverity, Dashboard, DashboardConfig, LiveMetrics, MetricsInput},
    // Differential privacy
    differential_privacy::{DPConfig, PrivateTrustAnalytics, TrustDistribution},
    // Economics
    economics::{
        RewardConfig, RewardEngine, SlashResult, SlashingConfig, SlashingEngine, ViolationSeverity,
        ViolationType,
    },
    // Epistemic
    epistemic_classifier::{
        calculate_epistemic_weight, AgentOutput, AgentOutputBuilder, OutputContent,
    },
    // Trust pipeline
    kvector_bridge::calculate_kredit_from_trust,
    lifecycle::record_uncertainty_outcome,
    // Uncertainty (GIS)
    uncertainty::{EscalationRequest, MoralActionGuidance, MoralUncertainty},
    // Verification
    verification::{InvariantCheckResult, SystemState, VerificationEngine},
    AgentClass,
    AgentConstraints,
    AgentId,
    AgentStatus,
    EpistemicStats,
    InstrumentalActor,
    UncertaintyCalibration,
};
use crate::epistemic::{EmpiricalLevel, HarmonicLevel, MaterialityLevel, NormativeLevel};
use crate::matl::KVector;
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
        self.cascade_engine
            .network_mut()
            .add_agent(NetworkAgent::new(agent_id.clone(), trust as f64));

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
        let from = self
            .agents
            .get(from_agent)
            .ok_or_else(|| IntegrationError::AgentNotFound(from_agent.to_string()))?;
        let _to = self
            .agents
            .get(to_agent)
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

        // Update to_agent's trust (unwrap safe: existence checked above via _to)
        #[allow(clippy::unwrap_used)]
        let to_agent_mut = self.agents.get_mut(to_agent).unwrap();
        let old_trust = to_agent_mut.k_vector.trust_score();

        // Apply bounded update to k_r dimension
        let new_k_r = (to_agent_mut.k_vector.k_r + weight as f32 * 0.1).clamp(0.0, 1.0);
        to_agent_mut.k_vector.k_r = new_k_r;

        let new_trust = to_agent_mut.k_vector.trust_score();

        // Recalculate KREDIT cap
        let kredit_multiplier = calculate_kredit_from_trust(new_trust);
        to_agent_mut.kredit_cap = kredit_multiplier * self.config.kredit_base;

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

        self.cascade_engine
            .apply_shock(agent_id, shock_magnitude, timestamp)
    }

    /// Verify all invariants hold
    pub fn verify_invariants(&mut self) -> Vec<InvariantCheckResult> {
        let state = self.create_system_state();
        self.verification.check_invariants(&state)
    }

    fn create_system_state(&self) -> SystemState {
        let trust_scores: HashMap<String, f64> = self
            .agents
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
    /// Agent that created the attestation.
    pub from_agent: String,
    /// Agent that received the attestation.
    pub to_agent: String,
    /// Effective attestation weight.
    pub weight: f64,
    /// Trust score before the attestation.
    pub old_trust: f64,
    /// Trust score after the attestation.
    pub new_trust: f64,
    /// Updated KREDIT cap after trust change.
    pub new_kredit_cap: u64,
}

// ============================================================================
// Flow 2: Attack Response Integration
// ============================================================================

/// Configuration for integrated attack response
#[derive(Debug, Clone, Default)]
pub struct AttackResponseConfig {
    /// Adaptive threshold config
    pub thresholds: AdaptiveConfig,
    /// Slashing config
    pub slashing: SlashingConfig,
    /// Gaming detection config
    pub gaming: GamingDetectionConfig,
}

/// Integrated attack response system
pub struct IntegratedAttackResponse {
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
            thresholds: AdaptiveThresholdEngine::new(config.thresholds),
            quarantine: QuarantineManager::new(),
            slashing: SlashingEngine::new(config.slashing),
            gaming: GamingDetector::new(config.gaming),
            responses: vec![],
        }
    }

    /// Process an agent's behavior and respond to any detected attacks
    pub fn process_behavior(&mut self, agent: &mut InstrumentalActor) -> Option<AttackResponse> {
        let agent_id = agent.agent_id.as_str();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Check for gaming behavior using analyze method
        let gaming_result = self.gaming.analyze(agent, timestamp);

        // Check if gaming detected (suspicion score above threshold)
        let is_gaming = gaming_result.suspicion_score > 0.5
            || matches!(
                gaming_result.recommended_action,
                GamingResponse::Quarantine | GamingResponse::EscalateToSponsor
            );

        if is_gaming {
            // Quarantine with evidence
            self.quarantine.quarantine(
                agent_id,
                QuarantineReason::GamingDetected,
                gaming_result
                    .indicators
                    .iter()
                    .map(|i| i.description.clone())
                    .collect(),
                timestamp,
            );

            // Slash based on severity
            let severity = if gaming_result.suspicion_score > 0.8 {
                ViolationSeverity::Critical
            } else if gaming_result.suspicion_score > 0.5 {
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
                    threat_level: gaming_result.suspicion_score,
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
                confidence: gaming_result.suspicion_score,
                actions_taken: vec![
                    ResponseAction::Quarantined {
                        duration_secs: 86400,
                    },
                    ResponseAction::Slashed {
                        amount: slashed_amount,
                    },
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
    /// ID of the offending agent.
    pub agent_id: String,
    /// Type of attack detected.
    pub attack_type: String,
    /// Detection confidence (0.0-1.0).
    pub confidence: f64,
    /// Actions taken in response.
    pub actions_taken: Vec<ResponseAction>,
    /// When the response was generated.
    pub timestamp: u64,
}

/// Actions taken in response to attack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseAction {
    /// Agent was quarantined.
    Quarantined {
        /// Quarantine duration in seconds.
        duration_secs: u64,
    },
    /// Agent was slashed.
    Slashed {
        /// Amount of KREDIT slashed.
        amount: u64,
    },
    /// Agent trust was reduced.
    TrustReduced {
        /// Amount of trust reduction.
        amount: f64,
    },
    /// An alert was raised.
    AlertRaised {
        /// Alert severity level.
        severity: String,
    },
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
    /// Private trust analytics
    analytics: PrivateTrustAnalytics,
    /// Dashboard
    dashboard: Dashboard,
}

impl IntegratedPrivacyAnalytics {
    /// Create new privacy analytics system
    pub fn new(config: PrivacyAnalyticsConfig) -> Self {
        Self {
            analytics: PrivateTrustAnalytics::new(config.dp),
            dashboard: Dashboard::new(config.dashboard),
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
        let epsilon_per_query = 0.25;
        let distribution = self
            .analytics
            .analyze_trust_distribution(trust_scores, epsilon_per_query)
            .map_err(|e| IntegrationError::PrivacyError(e.to_string()))?;

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
    }
}

/// Result of privacy-preserving analytics
#[derive(Debug, Clone)]
pub struct PrivateAnalyticsResult {
    /// Differentially-private trust distribution.
    pub distribution: TrustDistribution,
    /// Current dashboard metrics.
    pub live_metrics: LiveMetrics,
    /// Remaining privacy budget (epsilon, delta).
    pub remaining_budget: (f64, f64),
}

// ============================================================================
// Flow 4: Epistemic Agent Lifecycle Integration
// ============================================================================

/// Configuration for epistemic agent lifecycle
#[derive(Debug, Clone)]
pub struct EpistemicLifecycleConfig {
    /// Minimum epistemic weight for trust updates
    pub min_epistemic_weight: f32,
    /// KREDIT base
    pub kredit_base: u64,
}

impl Default for EpistemicLifecycleConfig {
    fn default() -> Self {
        Self {
            min_epistemic_weight: 0.1,
            kredit_base: 10000,
        }
    }
}

/// Integrated epistemic agent lifecycle manager
pub struct IntegratedEpistemicLifecycle {
    config: EpistemicLifecycleConfig,
    /// Reward engine
    rewards: RewardEngine,
}

impl IntegratedEpistemicLifecycle {
    /// Create new lifecycle manager
    pub fn new(config: EpistemicLifecycleConfig) -> Self {
        Self {
            config,
            rewards: RewardEngine::new(RewardConfig::default()),
        }
    }

    /// Create a new agent with initial trust profile
    pub fn create_agent(&self, sponsor_did: &str, agent_class: AgentClass) -> InstrumentalActor {
        let agent_id = AgentId::generate();
        let k_vector = KVector::new_participant();
        let trust = k_vector.trust_score();
        let kredit_multiplier = calculate_kredit_from_trust(trust);
        let kredit_cap = kredit_multiplier * self.config.kredit_base;

        InstrumentalActor {
            agent_id,
            sponsor_did: sponsor_did.to_string(),
            agent_class,
            kredit_balance: (kredit_cap / 2) as i64,
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
        &self,
        agent: &mut InstrumentalActor,
        content: OutputContent,
    ) -> Result<OutputProcessingResult, IntegrationError> {
        let agent_id = agent.agent_id.as_str().to_string();

        // Build and classify output
        let output = AgentOutputBuilder::new(&agent_id)
            .content(content)
            .classification(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            )
            .confidence(0.8)
            .build()
            .map_err(|e: &str| IntegrationError::ClassificationError(e.to_string()))?;

        // Calculate epistemic weight
        let weight = calculate_epistemic_weight(&output.classification);

        // Record output on agent
        agent.record_output(&output);

        // Update K-Vector based on epistemic weight
        if weight >= self.config.min_epistemic_weight {
            let delta = weight * 0.01;
            agent.k_vector.k_r = (agent.k_vector.k_r + delta).clamp(0.0, 1.0);
        }

        // Recalculate KREDIT cap
        let trust = agent.k_vector.trust_score();
        let kredit_multiplier = calculate_kredit_from_trust(trust);
        agent.kredit_cap = kredit_multiplier * self.config.kredit_base;

        // Calculate reward
        let reward = self
            .rewards
            .calculate_participation_reward(&agent_id, trust as f64);

        Ok(OutputProcessingResult {
            output_id: output.output_id,
            epistemic_weight: weight,
            trust_delta: weight * 0.01,
            new_trust: trust as f64,
            new_kredit_cap: agent.kredit_cap,
            reward,
        })
    }

    /// Verify an agent's output and update trust accordingly
    pub fn verify_output(&self, agent: &mut InstrumentalActor, output_id: &str, correct: bool) {
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
        let kredit_multiplier = calculate_kredit_from_trust(trust);
        agent.kredit_cap = kredit_multiplier * self.config.kredit_base;
    }
}

/// Result of processing an agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputProcessingResult {
    /// Unique output identifier.
    pub output_id: String,
    /// Epistemic weight assigned to the output.
    pub epistemic_weight: f32,
    /// Trust change from this output.
    pub trust_delta: f32,
    /// Agent's trust score after processing.
    pub new_trust: f64,
    /// Updated KREDIT cap.
    pub new_kredit_cap: u64,
    /// KREDIT reward earned.
    pub reward: u64,
}

// ============================================================================
// Flow 5: MATL-Integrated Trust Pipeline
// ============================================================================

use crate::matl::{MatlEngine, ProofOfGradientQuality};

/// Configuration for MATL-integrated pipeline
#[derive(Debug, Clone)]
pub struct MatlPipelineConfig {
    /// MATL engine window size
    pub matl_window_size: usize,
    /// MATL hierarchy levels
    pub hierarchy_levels: usize,
    /// Minimum cluster size for detection
    pub min_cluster_size: usize,
    /// Trust pipeline config
    pub trust_config: TrustPipelineConfig,
}

impl Default for MatlPipelineConfig {
    fn default() -> Self {
        Self {
            matl_window_size: 100,
            hierarchy_levels: 3,
            min_cluster_size: 3,
            trust_config: TrustPipelineConfig::default(),
        }
    }
}

/// MATL-integrated trust pipeline with Byzantine detection
pub struct MatlIntegratedPipeline {
    /// Core trust pipeline
    trust_pipeline: IntegratedTrustPipeline,
    /// MATL engine for Byzantine detection
    matl_engine: MatlEngine,
    /// Network status history
    status_history: Vec<MatlNetworkSnapshot>,
}

/// Snapshot of MATL network status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlNetworkSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Estimated Byzantine fraction
    pub byzantine_fraction: f64,
    /// Adaptive threshold
    pub adaptive_threshold: f64,
    /// Network status
    pub status: String,
    /// Anomalous nodes
    pub anomalous_nodes: Vec<String>,
}

impl MatlIntegratedPipeline {
    /// Create new MATL-integrated pipeline
    pub fn new(config: MatlPipelineConfig) -> Self {
        Self {
            trust_pipeline: IntegratedTrustPipeline::new(config.trust_config),
            matl_engine: MatlEngine::new(
                config.matl_window_size,
                config.hierarchy_levels,
                config.min_cluster_size,
            ),
            status_history: Vec::new(),
        }
    }

    /// Register an agent with MATL integration
    pub fn register_agent(&mut self, agent: InstrumentalActor) {
        self.trust_pipeline.register_agent(agent);
    }

    /// Evaluate agent contribution through MATL
    pub fn evaluate_contribution(
        &mut self,
        agent_id: &str,
        gradient_quality: f64,
        consistency: f64,
    ) -> Result<MatlEvaluationResult, IntegrationError> {
        let agent = self
            .trust_pipeline
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        // Create PoGQ from gradient metrics
        let pogq = ProofOfGradientQuality::new(
            gradient_quality,
            consistency,
            0.5, // entropy
        );

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let reputation = agent.k_vector.trust_score() as f64;

        // Evaluate through MATL engine
        let (node_eval, network_eval) = self.matl_engine.evaluate_node(agent_id, &pogq, reputation);

        // Record network snapshot
        let snapshot = MatlNetworkSnapshot {
            timestamp,
            byzantine_fraction: network_eval.estimated_byzantine_fraction,
            adaptive_threshold: network_eval.adaptive_byzantine_threshold,
            status: format!("{:?}", network_eval.status),
            anomalous_nodes: if node_eval.is_node_anomalous {
                vec![agent_id.to_string()]
            } else {
                vec![]
            },
        };
        self.status_history.push(snapshot);

        // Update agent K-Vector based on MATL evaluation
        if node_eval.is_node_anomalous || node_eval.is_in_suspicious_cluster {
            // Penalize anomalous behavior
            if let Some(agent_mut) = self.trust_pipeline.agents.get_mut(agent_id) {
                agent_mut.k_vector.k_r = (agent_mut.k_vector.k_r - 0.05).clamp(0.0, 1.0);
                agent_mut.k_vector.k_i = (agent_mut.k_vector.k_i - 0.03).clamp(0.0, 1.0);
            }
        }

        Ok(MatlEvaluationResult {
            agent_id: agent_id.to_string(),
            composite_score: node_eval.composite_score,
            node_threshold: node_eval.node_threshold,
            is_anomalous: node_eval.is_node_anomalous,
            is_in_suspicious_cluster: node_eval.is_in_suspicious_cluster,
            byzantine_fraction: network_eval.estimated_byzantine_fraction,
            network_status: format!("{:?}", network_eval.status),
        })
    }

    /// Get network health summary
    pub fn network_health(&self) -> MatlNetworkHealth {
        let recent_snapshots: Vec<_> = self.status_history.iter().rev().take(10).collect();

        let avg_byzantine = if !recent_snapshots.is_empty() {
            recent_snapshots
                .iter()
                .map(|s| s.byzantine_fraction)
                .sum::<f64>()
                / recent_snapshots.len() as f64
        } else {
            0.0
        };

        let anomalous_count = recent_snapshots
            .iter()
            .flat_map(|s| &s.anomalous_nodes)
            .collect::<std::collections::HashSet<_>>()
            .len();

        MatlNetworkHealth {
            avg_byzantine_fraction: avg_byzantine,
            unique_anomalous_agents: anomalous_count,
            is_under_attack: avg_byzantine > 0.3,
            total_evaluations: self.status_history.len(),
        }
    }

    /// Access inner trust pipeline
    pub fn trust_pipeline(&self) -> &IntegratedTrustPipeline {
        &self.trust_pipeline
    }

    /// Access agents
    pub fn agents(&self) -> &HashMap<String, InstrumentalActor> {
        self.trust_pipeline.agents()
    }
}

/// Result of MATL evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlEvaluationResult {
    /// Agent ID
    pub agent_id: String,
    /// Composite score from MATL
    pub composite_score: f64,
    /// Per-node threshold
    pub node_threshold: f64,
    /// Whether agent is anomalous
    pub is_anomalous: bool,
    /// Whether in suspicious cluster
    pub is_in_suspicious_cluster: bool,
    /// Current Byzantine fraction
    pub byzantine_fraction: f64,
    /// Network status as string
    pub network_status: String,
}

/// MATL network health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlNetworkHealth {
    /// Average Byzantine fraction (recent)
    pub avg_byzantine_fraction: f64,
    /// Unique anomalous agents detected
    pub unique_anomalous_agents: usize,
    /// Whether network is under attack
    pub is_under_attack: bool,
    /// Total evaluations performed
    pub total_evaluations: usize,
}

// ============================================================================
// Flow 6: ML-Enhanced Attack Response
// ============================================================================

use crate::agentic::ml_anomaly::{AgentFeatures, IsolationForest, ReconstructionDetector};

/// Configuration for ML-enhanced attack response
#[derive(Debug, Clone)]
pub struct MLAttackConfig {
    /// Base attack response config
    pub base: AttackResponseConfig,
    /// Isolation forest estimators
    pub forest_estimators: usize,
    /// Isolation forest sample size
    pub forest_sample_size: usize,
    /// Anomaly score threshold
    pub anomaly_threshold: f64,
    /// Ensemble voting threshold (fraction of detectors agreeing)
    pub ensemble_threshold: f64,
}

impl Default for MLAttackConfig {
    fn default() -> Self {
        Self {
            base: AttackResponseConfig::default(),
            forest_estimators: 100,
            forest_sample_size: 256,
            anomaly_threshold: 0.6,
            ensemble_threshold: 0.5,
        }
    }
}

/// ML-enhanced attack response with ensemble detection
pub struct MLEnhancedAttackResponse {
    /// Base attack response
    base_response: IntegratedAttackResponse,
    /// Isolation forest detector
    isolation_forest: IsolationForest,
    /// Reconstruction detector
    reconstruction: ReconstructionDetector,
    /// Training data for online learning
    training_data: Vec<Vec<f64>>,
    /// K-Vector history per agent
    kvector_history: HashMap<String, Vec<KVector>>,
    /// Configuration
    config: MLAttackConfig,
    /// Is model trained
    is_trained: bool,
    /// ML detection results
    ml_detections: Vec<MLDetectionResult>,
}

/// Result of ML detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLDetectionResult {
    /// Agent ID
    pub agent_id: String,
    /// Isolation forest score (0-1, higher = more anomalous)
    pub isolation_score: f64,
    /// Reconstruction error score
    pub reconstruction_score: f64,
    /// Rule-based gaming score
    pub rule_based_score: f64,
    /// Ensemble decision (true = anomaly)
    pub is_anomaly: bool,
    /// Confidence (fraction of detectors agreeing)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl MLEnhancedAttackResponse {
    /// Create new ML-enhanced attack response
    pub fn new(config: MLAttackConfig) -> Self {
        Self {
            base_response: IntegratedAttackResponse::new(config.base.clone()),
            isolation_forest: IsolationForest::new(
                config.forest_estimators,
                config.forest_sample_size,
            ),
            reconstruction: ReconstructionDetector::new(),
            training_data: Vec::new(),
            kvector_history: HashMap::new(),
            config,
            is_trained: false,
            ml_detections: Vec::new(),
        }
    }

    /// Add training sample from agent
    pub fn add_training_sample(&mut self, agent: &InstrumentalActor) {
        let agent_id = agent.agent_id.as_str().to_string();

        // Track K-Vector history
        let history = self.kvector_history.entry(agent_id.clone()).or_default();
        history.push(agent.k_vector);

        // Extract features
        let features = AgentFeatures::extract(agent, history);
        self.training_data.push(features.to_vector());
    }

    /// Train the ML models
    pub fn train(&mut self, seed: u64) {
        if self.training_data.len() < 10 {
            return; // Not enough data
        }

        // Train isolation forest
        self.isolation_forest.fit(&self.training_data, seed);

        // Train reconstruction detector
        self.reconstruction.fit(&self.training_data);

        self.is_trained = true;
    }

    /// Process agent behavior with ML enhancement
    pub fn process_behavior_ml(&mut self, agent: &mut InstrumentalActor) -> MLDetectionResult {
        let agent_id = agent.agent_id.as_str().to_string();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Track K-Vector history
        let history = self.kvector_history.entry(agent_id.clone()).or_default();
        history.push(agent.k_vector);

        // Extract features
        let features = AgentFeatures::extract(agent, history);
        let feature_vector = features.to_vector();

        // Get rule-based score from base detector
        let rule_result = self.base_response.gaming.analyze(agent, timestamp);
        let rule_based_score = rule_result.suspicion_score;

        // Get ML scores (if trained)
        let (isolation_score, reconstruction_score) = if self.is_trained {
            let iso = self.isolation_forest.score(&feature_vector);
            let recon = self.reconstruction.score(&feature_vector);
            (iso, recon)
        } else {
            (0.5, 0.5)
        };

        // Ensemble voting
        let mut votes = 0;
        let total_detectors = 3;

        if isolation_score > self.config.anomaly_threshold {
            votes += 1;
        }
        if reconstruction_score > self.config.anomaly_threshold {
            votes += 1;
        }
        if rule_based_score > 0.5 {
            votes += 1;
        }

        let confidence = votes as f64 / total_detectors as f64;
        let is_anomaly = confidence >= self.config.ensemble_threshold;

        // Take action if anomaly detected
        if is_anomaly {
            // Process through base response
            let _ = self.base_response.process_behavior(agent);
        }

        let result = MLDetectionResult {
            agent_id,
            isolation_score,
            reconstruction_score,
            rule_based_score,
            is_anomaly,
            confidence,
            timestamp,
        };

        self.ml_detections.push(result.clone());
        result
    }

    /// Get ML detection history
    pub fn ml_detections(&self) -> &[MLDetectionResult] {
        &self.ml_detections
    }

    /// Get training data size
    pub fn training_size(&self) -> usize {
        self.training_data.len()
    }

    /// Is the model trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

// ============================================================================
// Flow 7: Observability Exports (Prometheus/OpenTelemetry)
// ============================================================================

/// Prometheus-style metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusMetric {
    /// Metric name
    pub name: String,
    /// Metric help text
    pub help: String,
    /// Metric type (gauge, counter, histogram)
    pub metric_type: MetricType,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Value
    pub value: f64,
    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

/// Metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// Gauge (can go up or down)
    Gauge,
    /// Counter (only increases)
    Counter,
    /// Histogram (distribution)
    Histogram,
    /// Summary (quantiles)
    Summary,
}

/// Observability exports for monitoring systems
pub struct ObservabilityExports {
    /// Prefix for all metrics
    prefix: String,
    /// Collected metrics
    metrics: Vec<PrometheusMetric>,
    /// Counter values (for increments)
    counters: HashMap<String, f64>,
}

impl ObservabilityExports {
    /// Create new observability exports
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            metrics: Vec::new(),
            counters: HashMap::new(),
        }
    }

    /// Record a gauge metric
    pub fn gauge(&mut self, name: &str, value: f64, labels: HashMap<String, String>, help: &str) {
        let metric_name = format!("{}_{}", self.prefix, name);
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.metrics.push(PrometheusMetric {
            name: metric_name,
            help: help.to_string(),
            metric_type: MetricType::Gauge,
            labels,
            value,
            timestamp_ms,
        });
    }

    /// Increment a counter
    pub fn counter_inc(
        &mut self,
        name: &str,
        delta: f64,
        labels: HashMap<String, String>,
        help: &str,
    ) {
        let key = format!("{}_{}", self.prefix, name);
        let current = self.counters.entry(key.clone()).or_insert(0.0);
        *current += delta;

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.metrics.push(PrometheusMetric {
            name: key,
            help: help.to_string(),
            metric_type: MetricType::Counter,
            labels,
            value: *current,
            timestamp_ms,
        });
    }

    /// Export from dashboard metrics
    pub fn export_dashboard(&mut self, metrics: &LiveMetrics) {
        let mut labels = HashMap::new();
        labels.insert("source".to_string(), "dashboard".to_string());

        self.gauge(
            "active_agents",
            metrics.active_agents as f64,
            labels.clone(),
            "Number of active agents",
        );
        self.gauge(
            "average_trust",
            metrics.average_trust,
            labels.clone(),
            "Average trust score across all agents",
        );
        self.gauge(
            "network_health",
            metrics.network_health,
            labels.clone(),
            "Overall network health score",
        );
        self.gauge(
            "tps",
            metrics.tps,
            labels.clone(),
            "Transactions per second",
        );
        self.gauge(
            "collective_phi",
            metrics.collective_phi,
            labels.clone(),
            "Collective Phi coherence metric",
        );
        self.gauge(
            "byzantine_threat",
            metrics.byzantine_threat,
            labels.clone(),
            "Current Byzantine threat level",
        );

        // Alert counts
        let mut alert_labels = labels.clone();
        alert_labels.insert("severity".to_string(), "critical".to_string());
        self.gauge(
            "alerts",
            metrics.alerts.critical as f64,
            alert_labels.clone(),
            "Active alert count by severity",
        );

        alert_labels.insert("severity".to_string(), "high".to_string());
        self.gauge(
            "alerts",
            metrics.alerts.high as f64,
            alert_labels.clone(),
            "Active alert count by severity",
        );

        alert_labels.insert("severity".to_string(), "medium".to_string());
        self.gauge(
            "alerts",
            metrics.alerts.medium as f64,
            alert_labels.clone(),
            "Active alert count by severity",
        );

        alert_labels.insert("severity".to_string(), "low".to_string());
        self.gauge(
            "alerts",
            metrics.alerts.low as f64,
            alert_labels,
            "Active alert count by severity",
        );
    }

    /// Export from MATL network health
    pub fn export_matl(&mut self, health: &MatlNetworkHealth) {
        let mut labels = HashMap::new();
        labels.insert("source".to_string(), "matl".to_string());

        self.gauge(
            "byzantine_fraction",
            health.avg_byzantine_fraction,
            labels.clone(),
            "Estimated Byzantine fraction in network",
        );
        self.gauge(
            "anomalous_agents",
            health.unique_anomalous_agents as f64,
            labels.clone(),
            "Number of unique anomalous agents detected",
        );
        self.gauge(
            "under_attack",
            if health.is_under_attack { 1.0 } else { 0.0 },
            labels.clone(),
            "Whether network is under attack",
        );
        self.counter_inc(
            "matl_evaluations_total",
            health.total_evaluations as f64,
            labels,
            "Total MATL evaluations performed",
        );
    }

    /// Export from ML detection
    pub fn export_ml_detection(&mut self, result: &MLDetectionResult) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), result.agent_id.clone());

        self.gauge(
            "ml_isolation_score",
            result.isolation_score,
            labels.clone(),
            "Isolation forest anomaly score",
        );
        self.gauge(
            "ml_reconstruction_score",
            result.reconstruction_score,
            labels.clone(),
            "Reconstruction error anomaly score",
        );
        self.gauge(
            "ml_rule_based_score",
            result.rule_based_score,
            labels.clone(),
            "Rule-based gaming detection score",
        );
        self.gauge(
            "ml_ensemble_confidence",
            result.confidence,
            labels.clone(),
            "Ensemble detection confidence",
        );

        if result.is_anomaly {
            self.counter_inc(
                "ml_anomalies_detected_total",
                1.0,
                labels,
                "Total anomalies detected by ML ensemble",
            );
        }
    }

    /// Format as Prometheus text format
    pub fn to_prometheus_text(&self) -> String {
        let mut output = String::new();

        for metric in &self.metrics {
            // HELP line
            output.push_str(&format!("# HELP {} {}\n", metric.name, metric.help));

            // TYPE line
            let type_str = match metric.metric_type {
                MetricType::Gauge => "gauge",
                MetricType::Counter => "counter",
                MetricType::Histogram => "histogram",
                MetricType::Summary => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", metric.name, type_str));

            // Value line with labels
            if metric.labels.is_empty() {
                output.push_str(&format!(
                    "{} {} {}\n",
                    metric.name, metric.value, metric.timestamp_ms
                ));
            } else {
                let labels_str: String = metric
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                output.push_str(&format!(
                    "{}{{{}}} {} {}\n",
                    metric.name, labels_str, metric.value, metric.timestamp_ms
                ));
            }
        }

        output
    }

    /// Format as OpenTelemetry JSON
    pub fn to_otel_json(&self) -> String {
        let otel_metrics: Vec<_> = self
            .metrics
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "description": m.help,
                    "unit": "",
                    "data": {
                        "dataPoints": [{
                            "attributes": m.labels.iter().map(|(k, v)| {
                                serde_json::json!({"key": k, "value": {"stringValue": v}})
                            }).collect::<Vec<_>>(),
                            "timeUnixNano": m.timestamp_ms * 1_000_000,
                            "asDouble": m.value,
                        }]
                    }
                })
            })
            .collect();

        serde_json::json!({
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "mycelix-agentic"}}
                    ]
                },
                "scopeMetrics": [{
                    "scope": {"name": "mycelix.agentic"},
                    "metrics": otel_metrics
                }]
            }]
        })
        .to_string()
    }

    /// Get all metrics
    pub fn metrics(&self) -> &[PrometheusMetric] {
        &self.metrics
    }

    /// Clear collected metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
}

// ============================================================================
// Flow 8: ZK Trust Integration
// ============================================================================

use crate::agentic::epistemic_classifier::{
    compute_kvector_delta_from_epistemic, KVectorDelta, ZKEpistemicClassifier,
};
use crate::agentic::zk_trust::{
    aggregate_proofs, AggregateStatement, AggregatedTrustProof, KVectorCommitment, ProofStatement,
    ProverConfig, TrustProof, TrustProver, TrustVerifier, VerificationResult,
};

/// Configuration for ZK-integrated trust pipeline
#[derive(Debug, Clone)]
pub struct ZKTrustConfig {
    /// MATL pipeline configuration
    pub matl_config: MatlPipelineConfig,
    /// Use simulation mode for proofs (WARNING: not secure for production!)
    pub simulation_mode: bool,
    /// Minimum trust threshold for attestation proofs
    pub min_attestation_trust: f32,
    /// Required Byzantine threshold for aggregated proofs
    pub byzantine_proof_threshold: f64,
}

impl Default for ZKTrustConfig {
    fn default() -> Self {
        Self {
            matl_config: MatlPipelineConfig::default(),
            simulation_mode: true, // Safe default for testing
            min_attestation_trust: 0.3,
            byzantine_proof_threshold: 0.67, // 2/3 majority
        }
    }
}

/// ZK-integrated trust pipeline with verifiable trust proofs
///
/// Extends `MatlIntegratedPipeline` with zero-knowledge proof capabilities:
/// - Agents can generate verifiable proofs of trust properties
/// - Attestations can include ZK proofs (proving attester's trust without revealing it)
/// - Byzantine detection benefits from ZK-verified trust aggregation
pub struct ZKIntegratedPipeline {
    /// Core MATL-integrated pipeline
    matl_pipeline: MatlIntegratedPipeline,
    /// ZK trust prover (reserved for future proof generation)
    #[allow(dead_code)]
    prover: TrustProver,
    /// ZK trust verifier
    verifier: TrustVerifier,
    /// Configuration
    config: ZKTrustConfig,
    /// Agent commitments (agent_id -> most recent commitment)
    commitments: HashMap<String, KVectorCommitment>,
    /// Blinding factors (agent_id -> blinding) - in production, these would be securely stored
    blindings: HashMap<String, [u8; 32]>,
    /// Proof history
    proof_history: Vec<ZKProofRecord>,
    /// Counter for generating blinding factors
    blinding_counter: u64,
    /// Phi coherence history per agent (for tracking over time)
    coherence_histories: HashMap<String, CoherenceHistory>,
    /// Phi measurement configuration
    coherence_config: CoherenceMeasurementConfig,
}

/// Record of a ZK proof generation/verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKProofRecord {
    /// Agent who generated the proof
    pub agent_id: String,
    /// Statement that was proven
    pub statement_type: String,
    /// Whether the proof was valid
    pub valid: bool,
    /// Whether the statement was true
    pub result: bool,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of a ZK-verified attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKAttestationResult {
    /// Base attestation result
    pub attestation: AttestationResult,
    /// Proof from the attesting agent
    pub attester_proof: Option<ZKProofSummary>,
    /// Whether the attester's trust was ZK-verified
    pub trust_verified: bool,
}

/// Summary of a ZK proof (without the full proof data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKProofSummary {
    /// Commitment hash
    pub commitment: [u8; 32],
    /// Statement type
    pub statement_type: String,
    /// Result of the proof
    pub result: bool,
    /// Verification status
    pub verified: bool,
    /// Is simulation mode
    pub is_simulation: bool,
}

/// Result of generating a trust proof for an agent
#[derive(Debug, Clone)]
pub struct AgentTrustProofResult {
    /// The generated proof
    pub proof: TrustProof,
    /// Verification result
    pub verification: VerificationResult,
    /// Summary for export
    pub summary: ZKProofSummary,
}

impl ZKIntegratedPipeline {
    /// Create a new ZK-integrated pipeline
    pub fn new(config: ZKTrustConfig) -> Self {
        let prover_config = ProverConfig {
            simulation_mode: config.simulation_mode,
            agent_id: String::new(), // Set per-proof
            timestamp: 0,
        };

        Self {
            matl_pipeline: MatlIntegratedPipeline::new(config.matl_config.clone()),
            prover: TrustProver::new(prover_config),
            verifier: TrustVerifier::new(),
            config,
            commitments: HashMap::new(),
            blindings: HashMap::new(),
            proof_history: Vec::new(),
            blinding_counter: 0,
            coherence_histories: HashMap::new(),
            coherence_config: CoherenceMeasurementConfig::default(),
        }
    }

    /// Register an agent and create initial commitment
    pub fn register_agent_with_commitment(
        &mut self,
        agent: InstrumentalActor,
    ) -> KVectorCommitment {
        let agent_id = agent.agent_id.as_str().to_string();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate blinding factor
        let blinding = self.generate_blinding(&agent_id);
        self.blindings.insert(agent_id.clone(), blinding);

        // Create commitment
        let commitment =
            KVectorCommitment::create(&agent.k_vector, &agent_id, timestamp, &blinding);
        self.commitments
            .insert(agent_id.clone(), commitment.clone());

        // Register with verifier
        self.verifier.register_commitment(commitment.clone());

        // Register with MATL pipeline
        self.matl_pipeline.register_agent(agent);

        commitment
    }

    /// Generate a deterministic blinding factor for an agent
    fn generate_blinding(&mut self, agent_id: &str) -> [u8; 32] {
        self.blinding_counter += 1;
        let mut hasher = sha3::Sha3_256::new();
        use sha3::Digest;
        hasher.update(b"mycelix-blinding-v1");
        hasher.update(agent_id.as_bytes());
        hasher.update(self.blinding_counter.to_le_bytes());
        hasher.finalize().into()
    }

    /// Generate a trust proof for an agent
    pub fn generate_trust_proof(
        &mut self,
        agent_id: &str,
        statement: ProofStatement,
    ) -> Result<AgentTrustProofResult, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let blinding = self
            .blindings
            .get(agent_id)
            .ok_or_else(|| IntegrationError::ZKError("No blinding factor for agent".to_string()))?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create prover with agent context
        let prover_config = ProverConfig {
            simulation_mode: self.config.simulation_mode,
            agent_id: agent_id.to_string(),
            timestamp,
        };
        let prover = TrustProver::new(prover_config);

        // Generate proof
        let proof = prover
            .prove(&agent.k_vector, &statement, blinding)
            .map_err(|e| IntegrationError::ZKError(format!("{:?}", e)))?;

        // Verify proof
        let verification = self.verifier.verify(&proof);

        // Update commitment
        let commitment = KVectorCommitment::create(&agent.k_vector, agent_id, timestamp, blinding);
        self.commitments
            .insert(agent_id.to_string(), commitment.clone());
        self.verifier.register_commitment(commitment);

        // Create summary
        let (result, verified) = match &verification {
            VerificationResult::Valid {
                statement_result, ..
            } => (*statement_result, true),
            VerificationResult::Invalid(_) => (false, false),
        };

        let summary = ZKProofSummary {
            commitment: proof.commitment.commitment,
            statement_type: format!("{:?}", statement),
            result,
            verified,
            is_simulation: proof.is_simulation(),
        };

        // Record
        self.proof_history.push(ZKProofRecord {
            agent_id: agent_id.to_string(),
            statement_type: format!("{:?}", statement),
            valid: verified,
            result,
            timestamp,
        });

        Ok(AgentTrustProofResult {
            proof,
            verification,
            summary,
        })
    }

    /// Process an attestation with ZK proof of attester's trust
    pub fn process_zk_attestation(
        &mut self,
        from_agent: &str,
        to_agent: &str,
        attestation_weight: f64,
    ) -> Result<ZKAttestationResult, IntegrationError> {
        // Generate proof that attester meets minimum trust threshold
        let statement = ProofStatement::TrustExceedsThreshold {
            threshold: self.config.min_attestation_trust,
        };

        let proof_result = self.generate_trust_proof(from_agent, statement)?;

        // Only process attestation if proof is valid and true
        let trust_verified = proof_result.verification.is_valid_and_true();

        if !trust_verified {
            return Err(IntegrationError::ZKError(
                "Attester failed to prove minimum trust threshold".to_string(),
            ));
        }

        // Process the attestation through the MATL pipeline
        let attestation = self.matl_pipeline.trust_pipeline.process_attestation(
            from_agent,
            to_agent,
            attestation_weight,
        )?;

        Ok(ZKAttestationResult {
            attestation,
            attester_proof: Some(proof_result.summary),
            trust_verified,
        })
    }

    /// Generate an improvement proof between two epochs
    pub fn generate_improvement_proof(
        &mut self,
        agent_id: &str,
        previous_kvector: &KVector,
        previous_timestamp: u64,
    ) -> Result<AgentTrustProofResult, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let blinding = self
            .blindings
            .get(agent_id)
            .ok_or_else(|| IntegrationError::ZKError("No blinding factor for agent".to_string()))?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create previous blinding (deterministic from timestamp)
        let mut prev_blinding = *blinding;
        prev_blinding[0] ^= (previous_timestamp & 0xFF) as u8;

        // Create prover
        let prover_config = ProverConfig {
            simulation_mode: self.config.simulation_mode,
            agent_id: agent_id.to_string(),
            timestamp,
        };
        let prover = TrustProver::new(prover_config);

        // Generate improvement proof
        let proof = prover
            .prove_improvement(
                previous_kvector,
                &agent.k_vector,
                None, // Overall trust improvement
                &prev_blinding,
                blinding,
            )
            .map_err(|e| IntegrationError::ZKError(format!("{:?}", e)))?;

        // Verify
        let verification = self.verifier.verify(&proof);

        let (result, verified) = match &verification {
            VerificationResult::Valid {
                statement_result, ..
            } => (*statement_result, true),
            VerificationResult::Invalid(_) => (false, false),
        };

        let summary = ZKProofSummary {
            commitment: proof.commitment.commitment,
            statement_type: "TrustImproved".to_string(),
            result,
            verified,
            is_simulation: proof.is_simulation(),
        };

        self.proof_history.push(ZKProofRecord {
            agent_id: agent_id.to_string(),
            statement_type: "TrustImproved".to_string(),
            valid: verified,
            result,
            timestamp,
        });

        Ok(AgentTrustProofResult {
            proof,
            verification,
            summary,
        })
    }

    /// Aggregate proofs from multiple agents for Byzantine-resistant verification
    pub fn aggregate_trust_proofs(
        &self,
        proofs: Vec<TrustProof>,
        statement: ProofStatement,
    ) -> AggregatedTrustProof {
        // Get trust scores as weights
        let weights: Vec<f64> = proofs
            .iter()
            .filter_map(|p| {
                self.matl_pipeline
                    .trust_pipeline()
                    .get_agent(&p.commitment.agent_id)
                    .map(|a| a.k_vector.trust_score() as f64)
            })
            .collect();

        aggregate_proofs(
            proofs,
            AggregateStatement::WeightedThreshold {
                statement,
                threshold: self.config.byzantine_proof_threshold,
            },
            Some(&weights),
        )
    }

    /// Check if aggregate proof satisfies Byzantine threshold
    pub fn verify_byzantine_consensus(&self, aggregate: &AggregatedTrustProof) -> bool {
        aggregate.is_satisfied()
    }

    /// Get network health including ZK proof metrics
    pub fn zk_network_health(&self) -> ZKNetworkHealth {
        let matl_health = self.matl_pipeline.network_health();

        let recent_proofs: Vec<_> = self.proof_history.iter().rev().take(100).collect();

        let valid_proofs = recent_proofs.iter().filter(|p| p.valid).count();
        let true_proofs = recent_proofs.iter().filter(|p| p.result).count();

        ZKNetworkHealth {
            matl_health,
            total_proofs_generated: self.proof_history.len(),
            recent_valid_rate: if !recent_proofs.is_empty() {
                valid_proofs as f64 / recent_proofs.len() as f64
            } else {
                1.0
            },
            recent_true_rate: if !recent_proofs.is_empty() {
                true_proofs as f64 / recent_proofs.len() as f64
            } else {
                1.0
            },
            agents_with_commitments: self.commitments.len(),
            simulation_mode: self.config.simulation_mode,
        }
    }

    /// Access inner MATL pipeline
    pub fn matl_pipeline(&self) -> &MatlIntegratedPipeline {
        &self.matl_pipeline
    }

    /// Access mutable MATL pipeline
    pub fn matl_pipeline_mut(&mut self) -> &mut MatlIntegratedPipeline {
        &mut self.matl_pipeline
    }

    /// Get proof history
    pub fn proof_history(&self) -> &[ZKProofRecord] {
        &self.proof_history
    }

    /// Get agent's most recent commitment
    pub fn get_commitment(&self, agent_id: &str) -> Option<&KVectorCommitment> {
        self.commitments.get(agent_id)
    }

    /// Process agent output with ZK proof and epistemic classification
    ///
    /// This combines ZK trust proofs with epistemic classification to:
    /// 1. Auto-classify the output content
    /// 2. Elevate to E3+ if ZK proof is attached
    /// 3. Update agent's K-Vector based on epistemic weight
    pub fn process_zk_output(
        &mut self,
        agent_id: &str,
        content: OutputContent,
        statement: ProofStatement,
    ) -> Result<ZKOutputResult, IntegrationError> {
        // Generate ZK proof for this output
        let proof_result = self.generate_trust_proof(agent_id, statement)?;

        // Create ZK-classified output
        let classifier = ZKEpistemicClassifier::new();
        let output = classifier.create_zk_output(
            agent_id,
            content,
            proof_result.proof.clone(),
            false, // Not publicly verifiable by default
        );

        // Compute K-Vector delta from epistemic classification
        let delta = compute_kvector_delta_from_epistemic(&output.classification, 0.02);

        // Apply delta to agent's K-Vector
        if let Some(agent) = self.matl_pipeline.trust_pipeline.agents.get_mut(agent_id) {
            agent.k_vector = delta.apply(&agent.k_vector);

            // Record output in agent's history
            agent.record_output(&AgentOutput {
                output_id: output.output_id.clone(),
                agent_id: output.agent_id.clone(),
                content: output.content.clone(),
                classification: output.classification,
                classification_confidence: output.classification_confidence,
                timestamp: output.timestamp,
                has_proof: output.has_proof,
                proof_data: output.proof_data.clone(),
                context_references: output.context_references.clone(),
            });
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(ZKOutputResult {
            output,
            proof_summary: proof_result.summary,
            kvector_delta: delta,
            timestamp,
        })
    }

    /// Process multiple outputs with batch ZK proofs
    pub fn process_zk_output_batch(
        &mut self,
        agent_id: &str,
        outputs: Vec<(OutputContent, ProofStatement)>,
    ) -> Result<Vec<ZKOutputResult>, IntegrationError> {
        let mut results = Vec::new();

        for (content, statement) in outputs {
            let result = self.process_zk_output(agent_id, content, statement)?;
            results.push(result);
        }

        Ok(results)
    }

    // =========================================================================
    // Phase 3: Phi-Gated ZK Operations
    // =========================================================================

    /// Check if an agent has sufficient Phi coherence for a ZK operation
    ///
    /// Returns the gating result with recommendations if the operation is not permitted.
    pub fn check_coherence_for_zk_operation(
        &self,
        agent_id: &str,
        operation: ZKOperationType,
    ) -> Result<ZKCoherenceGatingResult, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        Ok(check_zk_operation_coherence(agent, operation))
    }

    /// Generate a trust proof with Phi coherence gating
    ///
    /// Only allows proof generation if the agent has sufficient coherence.
    /// This prevents incoherent agents from polluting the proof network.
    pub fn generate_trust_proof_phi_gated(
        &mut self,
        agent_id: &str,
        statement: ProofStatement,
    ) -> Result<AgentTrustProofResult, IntegrationError> {
        // Check Phi coherence first
        let gating =
            self.check_coherence_for_zk_operation(agent_id, ZKOperationType::GenerateProof)?;

        if !gating.permitted {
            return Err(IntegrationError::PhiCoherenceError(format!(
                "Agent {} has insufficient coherence ({:?}) for proof generation. Required: {:?}. Recommendation: {:?}",
                agent_id, gating.current_state, gating.required_state, gating.recommendation
            )));
        }

        // Proceed with proof generation
        self.generate_trust_proof(agent_id, statement)
    }

    /// Process an attestation with Phi coherence weighting
    ///
    /// Attestations from higher-coherence agents carry more weight.
    /// Incoherent agents cannot attest at all.
    pub fn process_zk_attestation_phi_weighted(
        &mut self,
        from_agent: &str,
        to_agent: &str,
        base_attestation_weight: f64,
    ) -> Result<ZKAttestationResult, IntegrationError> {
        // Check if attester has sufficient coherence
        let gating =
            self.check_coherence_for_zk_operation(from_agent, ZKOperationType::GenerateProof)?;

        if !gating.permitted {
            return Err(IntegrationError::PhiCoherenceError(format!(
                "Agent {} cannot attest: coherence {:?} is below required {:?}",
                from_agent, gating.current_state, gating.required_state
            )));
        }

        // Apply Phi weighting to attestation
        let weighted_attestation = coherence_weighted_attestation(
            base_attestation_weight as f32,
            gating.current_coherence,
        );

        // Process with weighted attestation
        self.process_zk_attestation(from_agent, to_agent, weighted_attestation as f64)
    }

    /// Aggregate proofs with Phi coherence filtering
    ///
    /// Only proofs from agents with sufficient coherence are included.
    /// Coherence weights are used in the aggregation.
    pub fn aggregate_trust_proofs_phi_filtered(
        &self,
        proofs: Vec<TrustProof>,
        statement: ProofStatement,
    ) -> AggregatedTrustProof {
        // Filter proofs from agents with sufficient coherence
        let filtered_proofs: Vec<TrustProof> = proofs
            .into_iter()
            .filter(|p| {
                self.matl_pipeline
                    .trust_pipeline()
                    .get_agent(&p.commitment.agent_id)
                    .map(|a| {
                        let state = CoherenceState::from_phi(a.k_vector.k_coherence as f64);
                        state.allows_any_action()
                    })
                    .unwrap_or(false)
            })
            .collect();

        // Get Phi-weighted trust scores
        let weights: Vec<f64> = filtered_proofs
            .iter()
            .filter_map(|p| {
                self.matl_pipeline
                    .trust_pipeline()
                    .get_agent(&p.commitment.agent_id)
                    .map(|a| {
                        let trust = a.k_vector.trust_score() as f64;
                        let phi = a.k_vector.k_coherence as f64;
                        // Combined weight: trust * phi_weight
                        trust * phi.sqrt()
                    })
            })
            .collect();

        aggregate_proofs(
            filtered_proofs,
            AggregateStatement::WeightedThreshold {
                statement,
                threshold: self.config.byzantine_proof_threshold,
            },
            Some(&weights),
        )
    }

    /// Check if an agent can participate in Byzantine consensus based on Phi
    pub fn can_participate_in_byzantine_consensus(
        &self,
        agent_id: &str,
    ) -> Result<bool, IntegrationError> {
        let gating =
            self.check_coherence_for_zk_operation(agent_id, ZKOperationType::ByzantineConsensus)?;
        Ok(gating.permitted)
    }

    /// Update agent's Phi coherence measurement and k_coherence dimension
    ///
    /// Call this periodically or after significant agent activity to update coherence.
    pub fn update_agent_phi(&mut self, agent_id: &str) -> Result<PhiUpdateInfo, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let update_result = update_agent_coherence(agent, &self.coherence_config);

        // Update coherence history
        let history = self
            .coherence_histories
            .entry(agent_id.to_string())
            .or_default();
        history.add_measurement(update_result.measurement.clone());

        Ok(PhiUpdateInfo {
            agent_id: agent_id.to_string(),
            previous_k_coherence: update_result.previous_k_coherence,
            new_k_coherence: update_result.new_k_coherence,
            delta: update_result.delta,
            previous_state: update_result.previous_state,
            new_state: update_result.new_state,
            state_changed: update_result.state_changed,
            phi_measurement: update_result.measurement.coherence,
            trend: history.trend,
        })
    }

    /// Get agent's coherence state
    pub fn get_agent_coherence_state(
        &self,
        agent_id: &str,
    ) -> Result<CoherenceState, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        Ok(CoherenceState::from_phi(agent.k_vector.k_coherence as f64))
    }

    /// Get agent's coherence history
    pub fn get_coherence_history(&self, agent_id: &str) -> Option<&CoherenceHistory> {
        self.coherence_histories.get(agent_id)
    }

    /// Export Phi coherence metrics for an agent
    pub fn export_agent_phi_metrics(
        &self,
        agent_id: &str,
    ) -> Result<CoherenceExport, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let history = self
            .coherence_histories
            .get(agent_id)
            .cloned()
            .unwrap_or_default();

        Ok(export_coherence_metrics(agent_id, agent, &history))
    }

    /// Get network-wide Phi health summary
    pub fn phi_network_health(&self) -> PhiNetworkHealth {
        let agents = self.matl_pipeline.trust_pipeline().agents();

        let mut total_phi = 0.0;
        let mut coherent_count = 0;
        let mut degraded_count = 0;
        let mut critical_count = 0;

        for agent in agents.values() {
            let phi = agent.k_vector.k_coherence as f64;
            total_phi += phi;

            let state = CoherenceState::from_phi(phi);
            match state {
                CoherenceState::Coherent | CoherenceState::Stable => coherent_count += 1,
                CoherenceState::Unstable | CoherenceState::Degraded => degraded_count += 1,
                CoherenceState::Critical => critical_count += 1,
            }
        }

        let agent_count = agents.len();
        let avg_phi = if agent_count > 0 {
            total_phi / agent_count as f64
        } else {
            0.5
        };

        PhiNetworkHealth {
            agent_count,
            average_phi: avg_phi,
            coherent_agents: coherent_count,
            degraded_agents: degraded_count,
            critical_agents: critical_count,
            network_coherence_level: CoherenceState::from_phi(avg_phi),
        }
    }

    // =========================================================================
    // Phase 4: GIS (Graceful Ignorance System) Integration
    // =========================================================================

    /// Process output with moral uncertainty assessment
    ///
    /// Combines ZK proof, epistemic classification, and moral uncertainty.
    /// High uncertainty outputs are automatically escalated to sponsor.
    pub fn process_zk_output_with_uncertainty(
        &mut self,
        agent_id: &str,
        content: OutputContent,
        statement: ProofStatement,
        uncertainty: MoralUncertainty,
    ) -> Result<GISOutputResult, IntegrationError> {
        // Check if uncertainty allows proceeding
        let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

        if guidance.requires_human() {
            // Create escalation request
            let escalation = EscalationRequest::new(
                agent_id,
                uncertainty,
                "zk_output_generation",
                Some(format!("ZK proof statement: {:?}", statement)),
            );

            // Add to agent's pending escalations
            if let Some(agent) = self.matl_pipeline.trust_pipeline.agents.get_mut(agent_id) {
                agent.pending_escalations.push(escalation.clone());
            }

            return Ok(GISOutputResult {
                output_result: None,
                uncertainty,
                guidance,
                escalation: Some(escalation),
                calibration_updated: false,
            });
        }

        // Proceed with ZK output processing
        let output_result = self.process_zk_output(agent_id, content, statement)?;

        Ok(GISOutputResult {
            output_result: Some(output_result),
            uncertainty,
            guidance,
            escalation: None,
            calibration_updated: false,
        })
    }

    /// Infer moral uncertainty from output content characteristics
    ///
    /// Auto-estimates uncertainty based on:
    /// - Epistemic level (lower E → higher epistemic uncertainty)
    /// - Harmonic level (higher H → higher axiological uncertainty)
    /// - Materiality level (higher M → higher deontic uncertainty)
    pub fn infer_uncertainty_from_content(&self, content: &OutputContent) -> MoralUncertainty {
        use crate::agentic::epistemic_classifier::auto_classify;

        // Auto-classify the content
        let classification = auto_classify(content);

        // Infer epistemic uncertainty from E-level
        // Lower E = less verifiable = higher uncertainty
        let epistemic = match classification.empirical {
            EmpiricalLevel::E4PublicRepro => 0.1,
            EmpiricalLevel::E3Cryptographic => 0.2,
            EmpiricalLevel::E2PrivateVerify => 0.4,
            EmpiricalLevel::E1Testimonial => 0.6,
            EmpiricalLevel::E0Null => 0.8,
        };

        // Infer axiological uncertainty from H-level
        // Higher H = more values at stake = higher uncertainty
        let axiological = match classification.harmonic {
            HarmonicLevel::H0None => 0.1,
            HarmonicLevel::H1Local => 0.25,
            HarmonicLevel::H2Network => 0.4,
            HarmonicLevel::H3Civilizational => 0.6,
            HarmonicLevel::H4Kosmic => 0.8,
        };

        // Infer deontic uncertainty from M-level
        // Higher M = more lasting impact = higher uncertainty
        let deontic = match classification.materiality {
            MaterialityLevel::M0Ephemeral => 0.1,
            MaterialityLevel::M1Temporal => 0.25,
            MaterialityLevel::M2Persistent => 0.45,
            MaterialityLevel::M3Foundational => 0.7,
        };

        MoralUncertainty::new(epistemic, axiological, deontic)
    }

    /// Process output with auto-inferred uncertainty
    ///
    /// Combines content analysis, uncertainty inference, and ZK proof generation.
    pub fn process_zk_output_auto_uncertainty(
        &mut self,
        agent_id: &str,
        content: OutputContent,
        statement: ProofStatement,
    ) -> Result<GISOutputResult, IntegrationError> {
        let uncertainty = self.infer_uncertainty_from_content(&content);
        self.process_zk_output_with_uncertainty(agent_id, content, statement, uncertainty)
    }

    /// Record the outcome of an uncertainty-gated action
    ///
    /// Updates calibration tracking to improve future uncertainty estimates.
    pub fn record_gis_outcome(
        &mut self,
        agent_id: &str,
        was_uncertain: bool,
        was_good_outcome: bool,
    ) -> Result<CalibrationUpdateResult, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let old_score = agent.uncertainty_calibration.calibration_score();

        // Record the outcome
        record_uncertainty_outcome(agent, was_uncertain, was_good_outcome);

        let new_score = agent.uncertainty_calibration.calibration_score();

        // Update K-Vector k_s (social/stability dimension) based on calibration
        // Well-calibrated agents are more socially reliable
        let calibration_delta = (new_score - old_score) * 0.1;
        agent.k_vector.k_s = (agent.k_vector.k_s + calibration_delta).clamp(0.0, 1.0);

        Ok(CalibrationUpdateResult {
            agent_id: agent_id.to_string(),
            old_calibration_score: old_score,
            new_calibration_score: new_score,
            k_s_delta: calibration_delta,
            is_overconfident: agent.uncertainty_calibration.is_overconfident(),
            is_overcautious: agent.uncertainty_calibration.is_overcautious(),
            total_calibration_events: agent.uncertainty_calibration.total_events,
        })
    }

    /// Process a resolved escalation from sponsor
    ///
    /// When a sponsor resolves an escalation, this updates calibration
    /// and potentially allows the blocked action to proceed.
    pub fn resolve_escalation(
        &mut self,
        agent_id: &str,
        blocked_action: &str,
        sponsor_approved: bool,
    ) -> Result<EscalationResolutionResult, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        // Find and remove the escalation
        let escalation_idx = agent
            .pending_escalations
            .iter()
            .position(|e| e.blocked_action == blocked_action)
            .ok_or_else(|| {
                IntegrationError::ZKError(format!(
                    "No pending escalation for action: {}",
                    blocked_action
                ))
            })?;

        let escalation = agent.pending_escalations.remove(escalation_idx);

        // Update calibration: agent was uncertain, outcome determined by sponsor
        // If sponsor approved, the uncertainty may have been overcautious
        // If sponsor rejected, the uncertainty was appropriate
        record_uncertainty_outcome(agent, true, sponsor_approved);

        Ok(EscalationResolutionResult {
            agent_id: agent_id.to_string(),
            blocked_action: blocked_action.to_string(),
            sponsor_approved,
            escalation,
            calibration_updated: true,
            new_calibration_score: agent.uncertainty_calibration.calibration_score(),
        })
    }

    /// Check action gating with combined Phi coherence and moral uncertainty
    ///
    /// Both coherence and uncertainty must be acceptable for action to proceed.
    pub fn check_combined_gating(
        &self,
        agent_id: &str,
        uncertainty: &MoralUncertainty,
        zk_operation: ZKOperationType,
    ) -> Result<CombinedGatingResult, IntegrationError> {
        // Check Phi coherence
        let phi_gating = self.check_coherence_for_zk_operation(agent_id, zk_operation)?;

        // Check moral uncertainty
        let guidance = MoralActionGuidance::from_uncertainty(uncertainty);

        let permitted = phi_gating.permitted && guidance.can_proceed();

        let recommendation = if !phi_gating.permitted {
            CombinedGatingRecommendation::WaitForCoherence
        } else if guidance.requires_human() {
            CombinedGatingRecommendation::EscalateForUncertainty
        } else if !guidance.can_proceed() {
            CombinedGatingRecommendation::PauseForReflection
        } else {
            CombinedGatingRecommendation::Proceed
        };

        Ok(CombinedGatingResult {
            permitted,
            phi_gating,
            uncertainty_guidance: guidance,
            recommendation,
            requires_escalation: guidance.requires_human(),
        })
    }

    /// Get agent's uncertainty calibration summary
    pub fn get_calibration_summary(
        &self,
        agent_id: &str,
    ) -> Result<CalibrationSummary, IntegrationError> {
        let agent = self
            .matl_pipeline
            .trust_pipeline()
            .get_agent(agent_id)
            .ok_or_else(|| IntegrationError::AgentNotFound(agent_id.to_string()))?;

        let cal = &agent.uncertainty_calibration;

        Ok(CalibrationSummary {
            agent_id: agent_id.to_string(),
            calibration_score: cal.calibration_score(),
            total_events: cal.total_events,
            appropriate_uncertainty: cal.appropriate_uncertainty,
            appropriate_confidence: cal.appropriate_confidence,
            overconfident: cal.overconfident,
            overcautious: cal.overcautious,
            tendency: if cal.is_overconfident() {
                CalibrationTendency::Overconfident
            } else if cal.is_overcautious() {
                CalibrationTendency::Overcautious
            } else {
                CalibrationTendency::WellCalibrated
            },
        })
    }

    /// Get network-wide GIS health summary
    pub fn gis_network_health(&self) -> GISNetworkHealth {
        let agents = self.matl_pipeline.trust_pipeline().agents();

        let mut total_calibration = 0.0;
        let mut pending_escalations = 0;
        let mut overconfident_count = 0;
        let mut overcautious_count = 0;

        for agent in agents.values() {
            total_calibration += agent.uncertainty_calibration.calibration_score() as f64;
            pending_escalations += agent.pending_escalations.len();

            if agent.uncertainty_calibration.is_overconfident() {
                overconfident_count += 1;
            }
            if agent.uncertainty_calibration.is_overcautious() {
                overcautious_count += 1;
            }
        }

        let agent_count = agents.len();
        let avg_calibration = if agent_count > 0 {
            total_calibration / agent_count as f64
        } else {
            0.5
        };

        GISNetworkHealth {
            agent_count,
            average_calibration_score: avg_calibration,
            pending_escalations_total: pending_escalations,
            overconfident_agents: overconfident_count,
            overcautious_agents: overcautious_count,
            well_calibrated_agents: agent_count - overconfident_count - overcautious_count,
        }
    }
}

/// Result of processing a GIS-aware output
#[derive(Debug, Clone)]
pub struct GISOutputResult {
    /// The ZK output result (if not escalated)
    pub output_result: Option<ZKOutputResult>,
    /// Moral uncertainty assessment
    pub uncertainty: MoralUncertainty,
    /// Action guidance
    pub guidance: MoralActionGuidance,
    /// Escalation request (if uncertainty triggered escalation)
    pub escalation: Option<EscalationRequest>,
    /// Whether calibration was updated
    pub calibration_updated: bool,
}

/// Result of updating uncertainty calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationUpdateResult {
    /// Agent ID
    pub agent_id: String,
    /// Previous calibration score
    pub old_calibration_score: f32,
    /// New calibration score
    pub new_calibration_score: f32,
    /// Change in k_s (stability dimension)
    pub k_s_delta: f32,
    /// Whether agent tends to be overconfident
    pub is_overconfident: bool,
    /// Whether agent tends to be overcautious
    pub is_overcautious: bool,
    /// Total calibration events recorded
    pub total_calibration_events: u32,
}

/// Result of resolving an escalation
#[derive(Debug, Clone)]
pub struct EscalationResolutionResult {
    /// Agent ID
    pub agent_id: String,
    /// Action that was blocked
    pub blocked_action: String,
    /// Whether sponsor approved
    pub sponsor_approved: bool,
    /// The resolved escalation
    pub escalation: EscalationRequest,
    /// Whether calibration was updated
    pub calibration_updated: bool,
    /// New calibration score
    pub new_calibration_score: f32,
}

/// Combined gating result (Phi + uncertainty)
#[derive(Debug, Clone)]
pub struct CombinedGatingResult {
    /// Whether action is permitted
    pub permitted: bool,
    /// Phi gating result
    pub phi_gating: ZKCoherenceGatingResult,
    /// Uncertainty guidance
    pub uncertainty_guidance: MoralActionGuidance,
    /// Combined recommendation
    pub recommendation: CombinedGatingRecommendation,
    /// Whether escalation is required
    pub requires_escalation: bool,
}

/// Combined gating recommendation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinedGatingRecommendation {
    /// Proceed with action
    Proceed,
    /// Wait for Phi coherence to improve
    WaitForCoherence,
    /// Pause for reflection (uncertainty)
    PauseForReflection,
    /// Escalate to sponsor for uncertainty
    EscalateForUncertainty,
}

/// Summary of agent's uncertainty calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSummary {
    /// Agent ID
    pub agent_id: String,
    /// Overall calibration score (0-1)
    pub calibration_score: f32,
    /// Total calibration events
    pub total_events: u32,
    /// Times appropriately uncertain
    pub appropriate_uncertainty: u32,
    /// Times appropriately confident
    pub appropriate_confidence: u32,
    /// Times overconfident
    pub overconfident: u32,
    /// Times overcautious
    pub overcautious: u32,
    /// Calibration tendency
    pub tendency: CalibrationTendency,
}

/// Agent's calibration tendency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationTendency {
    /// Agent is well-calibrated
    WellCalibrated,
    /// Agent tends to be overconfident
    Overconfident,
    /// Agent tends to be overcautious
    Overcautious,
}

/// Network-wide GIS health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GISNetworkHealth {
    /// Total number of agents
    pub agent_count: usize,
    /// Average calibration score across agents
    pub average_calibration_score: f64,
    /// Total pending escalations
    pub pending_escalations_total: usize,
    /// Number of overconfident agents
    pub overconfident_agents: usize,
    /// Number of overcautious agents
    pub overcautious_agents: usize,
    /// Number of well-calibrated agents
    pub well_calibrated_agents: usize,
}

/// Information about a Phi update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiUpdateInfo {
    /// Agent ID
    pub agent_id: String,
    /// Previous k_coherence value
    pub previous_k_coherence: f32,
    /// New k_coherence value
    pub new_k_coherence: f32,
    /// Change in k_coherence
    pub delta: f32,
    /// Previous coherence state
    pub previous_state: CoherenceState,
    /// New coherence state
    pub new_state: CoherenceState,
    /// Whether state changed
    pub state_changed: bool,
    /// Measured Phi value
    pub phi_measurement: f64,
    /// Trend direction
    pub trend: i8,
}

/// Network-wide Phi health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiNetworkHealth {
    /// Total number of agents
    pub agent_count: usize,
    /// Average Phi across all agents
    pub average_phi: f64,
    /// Number of coherent/stable agents
    pub coherent_agents: usize,
    /// Number of unstable/degraded agents
    pub degraded_agents: usize,
    /// Number of critical agents
    pub critical_agents: usize,
    /// Network-wide coherence level
    pub network_coherence_level: CoherenceState,
}

/// Result of processing a ZK-classified output
#[derive(Debug, Clone)]
pub struct ZKOutputResult {
    /// The classified output
    pub output: AgentOutput,
    /// ZK proof summary
    pub proof_summary: ZKProofSummary,
    /// K-Vector delta applied
    pub kvector_delta: KVectorDelta,
    /// Processing timestamp
    pub timestamp: u64,
}

/// ZK-enhanced network health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKNetworkHealth {
    /// MATL network health
    pub matl_health: MatlNetworkHealth,
    /// Total proofs generated
    pub total_proofs_generated: usize,
    /// Recent proof validity rate (0-1)
    pub recent_valid_rate: f64,
    /// Recent proof true rate (0-1)
    pub recent_true_rate: f64,
    /// Number of agents with active commitments
    pub agents_with_commitments: usize,
    /// Whether running in simulation mode
    pub simulation_mode: bool,
}

// Add ZK export methods to ObservabilityExports
impl ObservabilityExports {
    /// Export from ZK network health
    pub fn export_zk_health(&mut self, health: &ZKNetworkHealth) {
        let mut labels = HashMap::new();
        labels.insert("source".to_string(), "zk_trust".to_string());

        self.gauge(
            "zk_total_proofs",
            health.total_proofs_generated as f64,
            labels.clone(),
            "Total ZK proofs generated",
        );
        self.gauge(
            "zk_valid_rate",
            health.recent_valid_rate,
            labels.clone(),
            "Recent ZK proof validity rate",
        );
        self.gauge(
            "zk_true_rate",
            health.recent_true_rate,
            labels.clone(),
            "Recent ZK proof true rate",
        );
        self.gauge(
            "zk_agents_with_commitments",
            health.agents_with_commitments as f64,
            labels.clone(),
            "Number of agents with active ZK commitments",
        );
        self.gauge(
            "zk_simulation_mode",
            if health.simulation_mode { 1.0 } else { 0.0 },
            labels,
            "Whether ZK is running in simulation mode",
        );

        // Also export MATL health
        self.export_matl(&health.matl_health);
    }

    /// Export ZK proof record
    pub fn export_zk_proof(&mut self, record: &ZKProofRecord) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), record.agent_id.clone());
        labels.insert("statement_type".to_string(), record.statement_type.clone());

        if record.valid {
            self.counter_inc(
                "zk_proofs_valid_total",
                1.0,
                labels.clone(),
                "Total valid ZK proofs",
            );
        } else {
            self.counter_inc(
                "zk_proofs_invalid_total",
                1.0,
                labels.clone(),
                "Total invalid ZK proofs",
            );
        }

        if record.result {
            self.counter_inc(
                "zk_proofs_true_total",
                1.0,
                labels,
                "Total ZK proofs with true result",
            );
        }
    }

    /// Export Phi coherence metrics for an agent
    pub fn export_phi_coherence(&mut self, metrics: &CoherenceExport) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), metrics.agent_id.clone());
        labels.insert(
            "coherence_state".to_string(),
            metrics.coherence_state.clone(),
        );

        self.gauge(
            "phi_current",
            metrics.coherence,
            labels.clone(),
            "Current Phi coherence value",
        );
        self.gauge(
            "k_coherence",
            metrics.k_coherence as f64,
            labels.clone(),
            "K-Vector k_coherence dimension value",
        );
        self.gauge(
            "phi_rolling",
            metrics.rolling_coherence,
            labels.clone(),
            "Rolling average Phi",
        );
        self.gauge(
            "phi_trend",
            metrics.trend as f64,
            labels.clone(),
            "Phi trend direction (-1=declining, 0=stable, 1=improving)",
        );
        self.gauge(
            "phi_sample_size",
            metrics.sample_size as f64,
            labels.clone(),
            "Number of outputs in Phi sample",
        );
        self.gauge(
            "phi_can_high_stakes",
            if metrics.can_high_stakes { 1.0 } else { 0.0 },
            labels.clone(),
            "Whether agent can perform high-stakes actions",
        );
        self.gauge(
            "phi_is_critical",
            if metrics.is_critical { 1.0 } else { 0.0 },
            labels,
            "Whether agent is in critical coherence state",
        );
    }

    /// Export network-wide Phi health metrics
    pub fn export_phi_network_health(&mut self, health: &PhiNetworkHealth) {
        let mut labels = HashMap::new();
        labels.insert("source".to_string(), "phi_network".to_string());
        labels.insert(
            "coherence_level".to_string(),
            format!("{:?}", health.network_coherence_level),
        );

        self.gauge(
            "phi_network_agent_count",
            health.agent_count as f64,
            labels.clone(),
            "Total number of agents in network",
        );
        self.gauge(
            "phi_network_average",
            health.average_phi,
            labels.clone(),
            "Network-wide average Phi",
        );
        self.gauge(
            "phi_network_coherent_agents",
            health.coherent_agents as f64,
            labels.clone(),
            "Number of coherent/stable agents",
        );
        self.gauge(
            "phi_network_degraded_agents",
            health.degraded_agents as f64,
            labels.clone(),
            "Number of degraded/unstable agents",
        );
        self.gauge(
            "phi_network_critical_agents",
            health.critical_agents as f64,
            labels,
            "Number of agents in critical coherence state",
        );
    }

    /// Export Phi update event
    pub fn export_phi_update(&mut self, update: &PhiUpdateInfo) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), update.agent_id.clone());
        labels.insert(
            "previous_state".to_string(),
            format!("{:?}", update.previous_state),
        );
        labels.insert("new_state".to_string(), format!("{:?}", update.new_state));

        self.gauge(
            "phi_update_delta",
            update.delta as f64,
            labels.clone(),
            "Change in k_coherence from update",
        );

        if update.state_changed {
            self.counter_inc(
                "phi_state_transitions_total",
                1.0,
                labels,
                "Total Phi state transitions",
            );
        }
    }

    /// Export GIS calibration metrics for an agent
    pub fn export_gis_calibration(&mut self, summary: &CalibrationSummary) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), summary.agent_id.clone());
        labels.insert("tendency".to_string(), format!("{:?}", summary.tendency));

        self.gauge(
            "gis_calibration_score",
            summary.calibration_score as f64,
            labels.clone(),
            "Agent uncertainty calibration score",
        );
        self.gauge(
            "gis_total_events",
            summary.total_events as f64,
            labels.clone(),
            "Total calibration events",
        );
        self.gauge(
            "gis_appropriate_uncertainty",
            summary.appropriate_uncertainty as f64,
            labels.clone(),
            "Times appropriately uncertain",
        );
        self.gauge(
            "gis_appropriate_confidence",
            summary.appropriate_confidence as f64,
            labels.clone(),
            "Times appropriately confident",
        );
        self.gauge(
            "gis_overconfident",
            summary.overconfident as f64,
            labels.clone(),
            "Times overconfident",
        );
        self.gauge(
            "gis_overcautious",
            summary.overcautious as f64,
            labels,
            "Times overcautious",
        );
    }

    /// Export GIS network health metrics
    pub fn export_gis_network_health(&mut self, health: &GISNetworkHealth) {
        let mut labels = HashMap::new();
        labels.insert("source".to_string(), "gis_network".to_string());

        self.gauge(
            "gis_network_agent_count",
            health.agent_count as f64,
            labels.clone(),
            "Total agents in network",
        );
        self.gauge(
            "gis_network_avg_calibration",
            health.average_calibration_score,
            labels.clone(),
            "Network-wide average calibration score",
        );
        self.gauge(
            "gis_network_pending_escalations",
            health.pending_escalations_total as f64,
            labels.clone(),
            "Total pending escalations",
        );
        self.gauge(
            "gis_network_overconfident",
            health.overconfident_agents as f64,
            labels.clone(),
            "Number of overconfident agents",
        );
        self.gauge(
            "gis_network_overcautious",
            health.overcautious_agents as f64,
            labels.clone(),
            "Number of overcautious agents",
        );
        self.gauge(
            "gis_network_well_calibrated",
            health.well_calibrated_agents as f64,
            labels,
            "Number of well-calibrated agents",
        );
    }

    /// Export escalation event
    pub fn export_gis_escalation(&mut self, escalation: &EscalationRequest) {
        let mut labels = HashMap::new();
        labels.insert("agent_id".to_string(), escalation.agent_id.clone());
        labels.insert("guidance".to_string(), format!("{:?}", escalation.guidance));
        labels.insert("action".to_string(), escalation.blocked_action.clone());

        self.counter_inc(
            "gis_escalations_total",
            1.0,
            labels.clone(),
            "Total escalations to sponsors",
        );

        self.gauge(
            "gis_escalation_epistemic",
            escalation.uncertainty.epistemic as f64,
            labels.clone(),
            "Epistemic uncertainty at escalation",
        );
        self.gauge(
            "gis_escalation_axiological",
            escalation.uncertainty.axiological as f64,
            labels.clone(),
            "Axiological uncertainty at escalation",
        );
        self.gauge(
            "gis_escalation_deontic",
            escalation.uncertainty.deontic as f64,
            labels,
            "Deontic uncertainty at escalation",
        );
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Integration errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum IntegrationError {
    /// Agent not found in the pipeline.
    #[error("Agent not found: {0}")]
    AgentNotFound(
        /// Agent ID.
        String,
    ),

    /// Agent's trust is below the required threshold.
    #[error("Insufficient trust: {agent} has {actual}, needs {required}")]
    InsufficientTrust {
        /// Agent identifier.
        agent: String,
        /// Required trust level.
        required: f64,
        /// Actual trust level.
        actual: f64,
    },

    /// Error from privacy-preserving analytics.
    #[error("Privacy error: {0}")]
    PrivacyError(
        /// Error description.
        String,
    ),

    /// Error during epistemic classification.
    #[error("Classification error: {0}")]
    ClassificationError(
        /// Error description.
        String,
    ),

    /// Error during ZK proof generation or verification.
    #[error("ZK error: {0}")]
    ZKError(
        /// Error description.
        String,
    ),

    /// Error due to insufficient Phi coherence.
    #[error("Phi coherence error: {0}")]
    PhiCoherenceError(
        /// Error description.
        String,
    ),
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

        let result = pipeline
            .process_attestation("agent-1", "agent-2", 0.8)
            .unwrap();

        assert!(result.new_trust >= result.old_trust);
        assert!(result.new_kredit_cap > 0);
    }

    #[test]
    fn test_trust_pipeline_cascade() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        for i in 0..5 {
            let agent = create_test_agent(&format!("agent-{}", i));
            pipeline.register_agent(agent);
        }

        for i in 0..4 {
            let _ = pipeline.process_attestation(
                &format!("agent-{}", i),
                &format!("agent-{}", i + 1),
                0.8,
            );
        }

        let result = pipeline.simulate_cascade("agent-0", 0.5);
        assert!(result.agents_affected > 0);
    }

    #[test]
    fn test_trust_pipeline_verification() {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        let agent = create_test_agent("agent-1");
        pipeline.register_agent(agent);

        let results = pipeline.verify_invariants();
        for result in &results {
            assert!(
                result.holds,
                "Invariant {} should hold",
                result.invariant_id
            );
        }
    }

    #[test]
    fn test_privacy_analytics() {
        // Use higher epsilon budget to ensure it's not exhausted by 4 queries
        let config = PrivacyAnalyticsConfig {
            dp: DPConfig {
                epsilon: 10.0,
                delta: 1e-6,
                ..Default::default()
            },
            dashboard: DashboardConfig::default(),
        };
        let mut analytics = IntegratedPrivacyAnalytics::new(config);

        let trust_scores = vec![0.5, 0.6, 0.7, 0.4, 0.55, 0.65];
        let phi_values = vec![0.7, 0.65];

        let result = analytics.analyze_and_display(&trust_scores, &phi_values, 0.1);
        assert!(result.is_ok());

        let result = result.unwrap();
        // DP noise (Laplace with epsilon_per_query=0.25) can significantly perturb
        // the private mean, so we only check that the result is finite.
        assert!(
            result.distribution.mean.is_finite(),
            "Private mean should be finite, got {}",
            result.distribution.mean
        );
        assert!(result.remaining_budget.0 > 0.0);
    }

    #[test]
    fn test_epistemic_lifecycle_create() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        let agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

        assert!(agent.kredit_cap > 0);
        assert!(agent.kredit_balance > 0);
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_epistemic_lifecycle_output() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);
        let initial_trust = agent.k_vector.trust_score();

        for i in 0..5 {
            let result =
                lifecycle.process_output(&mut agent, OutputContent::Text(format!("Output {}", i)));
            assert!(result.is_ok());
        }

        let final_trust = agent.k_vector.trust_score();
        assert!(final_trust >= initial_trust);
        assert!(agent.epistemic_stats.total_outputs >= 5);
    }

    #[test]
    fn test_epistemic_lifecycle_verification() {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

        let result = lifecycle
            .process_output(&mut agent, OutputContent::Text("Test output".to_string()))
            .unwrap();

        let initial_trust = agent.k_vector.trust_score();

        lifecycle.verify_output(&mut agent, &result.output_id, true);
        let trust_after_correct = agent.k_vector.trust_score();
        assert!(trust_after_correct > initial_trust);

        let result2 = lifecycle
            .process_output(
                &mut agent,
                OutputContent::Text("Another output".to_string()),
            )
            .unwrap();

        lifecycle.verify_output(&mut agent, &result2.output_id, false);
        let trust_after_incorrect = agent.k_vector.trust_score();
        assert!(trust_after_incorrect < trust_after_correct);
    }

    #[test]
    fn test_full_integration_flow() {
        let mut trust_pipeline = IntegratedTrustPipeline::new(TrustPipelineConfig::default());
        // Use higher epsilon for lower noise in test
        let privacy_config = PrivacyAnalyticsConfig {
            dp: crate::agentic::differential_privacy::DPConfig {
                epsilon: 10.0,
                delta: 1e-6,
                ..Default::default()
            },
            dashboard: crate::agentic::dashboard::DashboardConfig::default(),
        };
        let mut privacy_analytics = IntegratedPrivacyAnalytics::new(privacy_config);
        let epistemic_lifecycle =
            IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());

        let agent1 = epistemic_lifecycle.create_agent("did:sponsor:1", AgentClass::Supervised);
        let agent2 = epistemic_lifecycle.create_agent("did:sponsor:2", AgentClass::Supervised);

        trust_pipeline.register_agent(agent1.clone());
        trust_pipeline.register_agent(agent2.clone());

        let attest_result = trust_pipeline
            .process_attestation(agent1.agent_id.as_str(), agent2.agent_id.as_str(), 0.9)
            .unwrap();

        let trust_scores: Vec<f64> = trust_pipeline
            .agents()
            .values()
            .map(|a| a.k_vector.trust_score() as f64)
            .collect();

        let analytics_result = privacy_analytics
            .analyze_and_display(&trust_scores, &[0.7], 0.1)
            .unwrap();

        let invariant_results = trust_pipeline.verify_invariants();

        assert!(attest_result.new_trust > 0.0);
        // DP noise can make mean slightly negative, just check it's finite
        assert!(analytics_result.distribution.mean.is_finite());
        assert!(invariant_results.iter().all(|r| r.holds));
    }

    #[test]
    fn test_matl_integrated_pipeline() {
        let config = MatlPipelineConfig::default();
        let mut pipeline = MatlIntegratedPipeline::new(config);

        // Register agents
        for i in 0..5 {
            let agent = create_test_agent(&format!("matl-agent-{}", i));
            pipeline.register_agent(agent);
        }

        // Evaluate contributions
        for i in 0..5 {
            let result = pipeline.evaluate_contribution(
                &format!("matl-agent-{}", i),
                0.8 - (i as f64 * 0.1), // Varying quality
                1.0,
            );
            assert!(result.is_ok());
        }

        // Check network health
        let health = pipeline.network_health();
        assert!(health.avg_byzantine_fraction >= 0.0);
        assert!(health.total_evaluations >= 5);
    }

    #[test]
    fn test_ml_enhanced_attack_response() {
        let config = MLAttackConfig::default();
        let mut response = MLEnhancedAttackResponse::new(config);

        // Add training samples
        for i in 0..20 {
            let agent = create_test_agent(&format!("train-agent-{}", i));
            response.add_training_sample(&agent);
        }

        assert_eq!(response.training_size(), 20);
        assert!(!response.is_trained());

        // Train
        response.train(42);
        assert!(response.is_trained());

        // Process behavior
        let mut agent = create_test_agent("test-agent");
        let result = response.process_behavior_ml(&mut agent);

        assert!(result.isolation_score >= 0.0 && result.isolation_score <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_observability_exports() {
        let mut exports = ObservabilityExports::new("mycelix");

        // Test gauge
        let mut labels = HashMap::new();
        labels.insert("agent".to_string(), "test".to_string());
        exports.gauge("trust_score", 0.85, labels.clone(), "Agent trust score");

        // Test counter
        exports.counter_inc("actions_total", 5.0, labels, "Total actions");

        assert!(exports.metrics().len() >= 2);

        // Test Prometheus format
        let prom_text = exports.to_prometheus_text();
        assert!(prom_text.contains("mycelix_trust_score"));
        assert!(prom_text.contains("# HELP"));
        assert!(prom_text.contains("# TYPE"));

        // Test OTEL format
        let otel_json = exports.to_otel_json();
        assert!(otel_json.contains("resourceMetrics"));
        assert!(otel_json.contains("mycelix_trust_score"));
    }

    #[test]
    fn test_observability_dashboard_export() {
        let mut exports = ObservabilityExports::new("mycelix");

        let metrics = LiveMetrics {
            timestamp: 1000,
            active_agents: 10,
            total_trust: 8.5,
            average_trust: 0.85,
            network_health: 0.9,
            tps: 100.0,
            alerts: crate::agentic::dashboard::AlertCounts {
                critical: 1,
                high: 2,
                medium: 3,
                low: 5,
            },
            collective_phi: 0.7,
            byzantine_threat: 0.1,
        };

        exports.export_dashboard(&metrics);

        // Should have multiple metrics exported
        assert!(exports.metrics().len() >= 6);

        let prom_text = exports.to_prometheus_text();
        assert!(prom_text.contains("active_agents"));
        assert!(prom_text.contains("network_health"));
        assert!(prom_text.contains("byzantine_threat"));
    }

    // =========================================================================
    // ZK Trust Integration Tests
    // =========================================================================

    #[test]
    fn test_zk_pipeline_register_with_commitment() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("zk-agent-1");
        let commitment = pipeline.register_agent_with_commitment(agent);

        assert_eq!(commitment.agent_id, "zk-agent-1");
        assert!(pipeline.get_commitment("zk-agent-1").is_some());
    }

    #[test]
    fn test_zk_generate_trust_proof() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("zk-agent-1");
        pipeline.register_agent_with_commitment(agent);

        // Generate proof that trust exceeds 0.5
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
        let result = pipeline.generate_trust_proof("zk-agent-1", statement);

        assert!(result.is_ok());
        let proof_result = result.unwrap();
        assert!(proof_result.summary.verified);
        assert!(proof_result.summary.result); // Agent has trust > 0.5
    }

    #[test]
    fn test_zk_proof_failing_statement() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("zk-agent-1"); // Trust ~0.53
        pipeline.register_agent_with_commitment(agent);

        // Generate proof that trust exceeds 0.9 (should fail)
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.9 };
        let result = pipeline.generate_trust_proof("zk-agent-1", statement);

        assert!(result.is_ok());
        let proof_result = result.unwrap();
        assert!(proof_result.summary.verified); // Proof is valid
        assert!(!proof_result.summary.result); // But statement is false
    }

    #[test]
    fn test_zk_attestation() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent1 = create_test_agent("zk-attester");
        let agent2 = create_test_agent("zk-recipient");
        pipeline.register_agent_with_commitment(agent1);
        pipeline.register_agent_with_commitment(agent2);

        // Process ZK-verified attestation
        let result = pipeline.process_zk_attestation("zk-attester", "zk-recipient", 0.8);

        assert!(result.is_ok());
        let zk_result = result.unwrap();
        assert!(zk_result.trust_verified);
        assert!(zk_result.attester_proof.is_some());
    }

    #[test]
    fn test_zk_attestation_fails_low_trust() {
        let mut config = ZKTrustConfig::default();
        config.min_attestation_trust = 0.9; // Very high threshold
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent1 = create_test_agent("low-trust-attester"); // Trust ~0.53
        let agent2 = create_test_agent("recipient");
        pipeline.register_agent_with_commitment(agent1);
        pipeline.register_agent_with_commitment(agent2);

        // Should fail because attester trust is below threshold
        let result = pipeline.process_zk_attestation("low-trust-attester", "recipient", 0.8);

        assert!(result.is_err());
        match result {
            Err(IntegrationError::ZKError(msg)) => {
                assert!(msg.contains("minimum trust threshold"));
            }
            _ => panic!("Expected ZKError"),
        }
    }

    #[test]
    fn test_zk_improvement_proof() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("improving-agent");
        pipeline.register_agent_with_commitment(agent);

        // Simulate previous K-Vector with lower trust
        let previous_kv = KVector::new(0.3, 0.3, 0.4, 0.3, 0.2, 0.2, 0.3, 0.2, 0.3, 0.3);

        let result = pipeline.generate_improvement_proof(
            "improving-agent",
            &previous_kv,
            1000, // Previous timestamp
        );

        assert!(result.is_ok());
        let proof_result = result.unwrap();
        assert!(proof_result.summary.verified);
        assert!(proof_result.summary.result); // Trust improved
    }

    #[test]
    fn test_zk_aggregate_proofs() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register multiple agents
        for i in 0..5 {
            let agent = create_test_agent(&format!("agg-agent-{}", i));
            pipeline.register_agent_with_commitment(agent);
        }

        // Generate proofs from each
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
        let mut proofs = Vec::new();

        for i in 0..5 {
            let result = pipeline
                .generate_trust_proof(&format!("agg-agent-{}", i), statement.clone())
                .unwrap();
            proofs.push(result.proof);
        }

        // Aggregate proofs
        let aggregate = pipeline.aggregate_trust_proofs(proofs, statement);

        assert_eq!(aggregate.total_count, 5);
        assert!(aggregate.true_count > 0);
        assert!(aggregate.weighted_result.is_some());
    }

    #[test]
    fn test_zk_byzantine_consensus() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register agents with varying trust
        for i in 0..10 {
            let mut agent = create_test_agent(&format!("byz-agent-{}", i));
            // Make most agents high trust
            if i < 8 {
                agent.k_vector = KVector::new(0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.6, 0.5, 0.7, 0.6);
            } else {
                agent.k_vector = KVector::new(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2);
            }
            pipeline.register_agent_with_commitment(agent);
        }

        // Generate proofs
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
        let mut proofs = Vec::new();

        for i in 0..10 {
            let result = pipeline
                .generate_trust_proof(&format!("byz-agent-{}", i), statement.clone())
                .unwrap();
            proofs.push(result.proof);
        }

        // Aggregate and check Byzantine consensus
        let aggregate = pipeline.aggregate_trust_proofs(proofs, statement);

        // 8/10 agents should have true proofs, with higher weights
        // Weighted threshold is 0.67, and high-trust agents have more weight
        assert!(pipeline.verify_byzantine_consensus(&aggregate));
    }

    #[test]
    fn test_zk_network_health() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register agents and generate some proofs
        for i in 0..5 {
            let agent = create_test_agent(&format!("health-agent-{}", i));
            pipeline.register_agent_with_commitment(agent);

            let _ = pipeline
                .generate_trust_proof(&format!("health-agent-{}", i), ProofStatement::WellFormed);
        }

        let health = pipeline.zk_network_health();

        assert_eq!(health.agents_with_commitments, 5);
        assert_eq!(health.total_proofs_generated, 5);
        assert!(health.recent_valid_rate > 0.0);
        assert!(health.simulation_mode);
    }

    #[test]
    fn test_zk_observability_exports() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("obs-agent");
        pipeline.register_agent_with_commitment(agent);

        let _ = pipeline.generate_trust_proof("obs-agent", ProofStatement::IsVerified);

        let health = pipeline.zk_network_health();

        let mut exports = ObservabilityExports::new("mycelix");
        exports.export_zk_health(&health);

        let prom_text = exports.to_prometheus_text();
        assert!(prom_text.contains("zk_total_proofs"));
        assert!(prom_text.contains("zk_valid_rate"));
        assert!(prom_text.contains("zk_simulation_mode"));
    }

    #[test]
    fn test_zk_proof_history() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("history-agent");
        pipeline.register_agent_with_commitment(agent);

        // Generate multiple proofs
        let _ = pipeline.generate_trust_proof("history-agent", ProofStatement::WellFormed);
        let _ = pipeline.generate_trust_proof("history-agent", ProofStatement::IsVerified);
        let _ = pipeline.generate_trust_proof(
            "history-agent",
            ProofStatement::TrustExceedsThreshold { threshold: 0.3 },
        );

        let history = pipeline.proof_history();
        assert_eq!(history.len(), 3);

        // All should be valid
        assert!(history.iter().all(|r| r.valid));
    }

    #[test]
    fn test_zk_dimension_proof() {
        use crate::matl::KVectorDimension;

        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("dim-agent"); // k_i = 0.7
        pipeline.register_agent_with_commitment(agent);

        // Prove integrity dimension exceeds 0.6
        let statement = ProofStatement::DimensionExceedsThreshold {
            dimension: KVectorDimension::Integrity,
            threshold: 0.6,
        };

        let result = pipeline.generate_trust_proof("dim-agent", statement);
        assert!(result.is_ok());
        assert!(result.unwrap().summary.result);
    }

    #[test]
    fn test_zk_compound_proof() {
        use crate::matl::KVectorDimension;

        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("compound-agent");
        pipeline.register_agent_with_commitment(agent);

        // AND proof: trust > 0.4 AND integrity > 0.5
        let statement = ProofStatement::And(vec![
            ProofStatement::TrustExceedsThreshold { threshold: 0.4 },
            ProofStatement::DimensionExceedsThreshold {
                dimension: KVectorDimension::Integrity,
                threshold: 0.5,
            },
        ]);

        let result = pipeline.generate_trust_proof("compound-agent", statement);
        assert!(result.is_ok());
        assert!(result.unwrap().summary.result);
    }

    // =========================================================================
    // ZK Epistemic Output Tests
    // =========================================================================

    #[test]
    fn test_zk_output_processing() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("output-agent");
        let initial_trust = agent.k_vector.trust_score();
        pipeline.register_agent_with_commitment(agent);

        // Process an output with ZK proof
        let content = OutputContent::Text("This is a cryptographic proof of my claim".to_string());
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };

        let result = pipeline.process_zk_output("output-agent", content, statement);
        assert!(result.is_ok());

        let output_result = result.unwrap();
        // Output should have ZK proof attached
        assert!(output_result.output.has_proof);
        // Classification should be E3+ due to ZK proof
        assert!(
            output_result.output.classification.empirical
                >= crate::epistemic::EmpiricalLevel::E3Cryptographic
        );
        // Proof should be verified
        assert!(output_result.proof_summary.verified);

        // Agent's K-Vector should have been updated
        let updated_agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("output-agent")
            .unwrap();
        // Delta was applied (trust should have changed slightly)
        let final_trust = updated_agent.k_vector.trust_score();
        assert!(
            final_trust >= initial_trust,
            "Trust should increase with positive epistemic output"
        );
    }

    #[test]
    fn test_zk_output_batch_processing() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("batch-agent");
        pipeline.register_agent_with_commitment(agent);

        // Process multiple outputs
        let outputs = vec![
            (
                OutputContent::Text("First verified claim".to_string()),
                ProofStatement::WellFormed,
            ),
            (
                OutputContent::Text("Second cryptographic proof".to_string()),
                ProofStatement::IsVerified,
            ),
            (
                OutputContent::Json(r#"{"proof": true, "data": "verified"}"#.to_string()),
                ProofStatement::TrustExceedsThreshold { threshold: 0.4 },
            ),
        ];

        let results = pipeline.process_zk_output_batch("batch-agent", outputs);
        assert!(results.is_ok());

        let batch_results = results.unwrap();
        assert_eq!(batch_results.len(), 3);

        // All outputs should have proofs
        assert!(batch_results.iter().all(|r| r.output.has_proof));
        // All should be E3+
        assert!(batch_results
            .iter()
            .all(|r| r.output.classification.empirical
                >= crate::epistemic::EmpiricalLevel::E3Cryptographic));
    }

    #[test]
    fn test_zk_output_kvector_delta() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("delta-agent");
        pipeline.register_agent_with_commitment(agent);

        let content = OutputContent::Text(
            "This is a foundational network-wide cryptographic proof with global consensus"
                .to_string(),
        );
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.3 };

        let result = pipeline
            .process_zk_output("delta-agent", content, statement)
            .unwrap();

        // High epistemic content should produce meaningful deltas
        assert!(result.kvector_delta.k_r_delta > 0.0);
        assert!(result.kvector_delta.k_p_delta > 0.0);
        // E3+ should give k_v boost
        assert!(result.kvector_delta.k_v_delta > 0.0);
    }

    #[test]
    fn test_zk_output_records_history() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_test_agent("history-agent-2");
        pipeline.register_agent_with_commitment(agent);

        // Process several outputs
        for i in 0..3 {
            let content = OutputContent::Text(format!("Output number {}", i));
            let statement = ProofStatement::WellFormed;
            let _ = pipeline.process_zk_output("history-agent-2", content, statement);
        }

        // Check agent's output history
        let agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("history-agent-2")
            .unwrap();
        assert!(
            agent.output_history.len() >= 3,
            "Agent should have recorded outputs"
        );
    }

    // =========================================================================
    // Phase 3: Phi-Gated ZK Operation Tests
    // =========================================================================

    fn create_agent_with_phi(id: &str, phi: f32) -> InstrumentalActor {
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
            k_vector: KVector::new(0.6, 0.5, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, phi),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_phi_gating_coherent_agent_allowed() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // High-coherence agent (phi = 0.8)
        let agent = create_agent_with_phi("coherent-agent", 0.8);
        pipeline.register_agent_with_commitment(agent);

        // Should be able to generate proofs
        let result = pipeline
            .check_coherence_for_zk_operation("coherent-agent", ZKOperationType::GenerateProof);
        assert!(result.is_ok());
        assert!(result.unwrap().permitted);

        // Should be able to participate in Byzantine consensus
        let result = pipeline.check_coherence_for_zk_operation(
            "coherent-agent",
            ZKOperationType::ByzantineConsensus,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().permitted);
    }

    #[test]
    fn test_phi_gating_degraded_agent_restricted() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Low-coherence agent (phi = 0.2 = Degraded)
        let agent = create_agent_with_phi("degraded-agent", 0.2);
        pipeline.register_agent_with_commitment(agent);

        // Should NOT be able to generate proofs
        let result = pipeline
            .check_coherence_for_zk_operation("degraded-agent", ZKOperationType::GenerateProof);
        assert!(result.is_ok());
        let gating = result.unwrap();
        assert!(!gating.permitted);
        assert_eq!(gating.current_state, CoherenceState::Degraded);

        // Should NOT participate in Byzantine consensus
        let result = pipeline.check_coherence_for_zk_operation(
            "degraded-agent",
            ZKOperationType::ByzantineConsensus,
        );
        assert!(result.is_ok());
        assert!(!result.unwrap().permitted);

        // CAN verify proofs (low stakes)
        let result = pipeline
            .check_coherence_for_zk_operation("degraded-agent", ZKOperationType::VerifyProof);
        assert!(result.is_ok());
        assert!(result.unwrap().permitted);
    }

    #[test]
    fn test_phi_gated_proof_generation() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Coherent agent can generate proofs
        let coherent = create_agent_with_phi("coherent", 0.75);
        pipeline.register_agent_with_commitment(coherent);

        let result =
            pipeline.generate_trust_proof_phi_gated("coherent", ProofStatement::WellFormed);
        assert!(result.is_ok());

        // Critical agent cannot
        let critical = create_agent_with_phi("critical", 0.05);
        pipeline.register_agent_with_commitment(critical);

        let result =
            pipeline.generate_trust_proof_phi_gated("critical", ProofStatement::WellFormed);
        assert!(result.is_err());
        match result.unwrap_err() {
            IntegrationError::PhiCoherenceError(_) => (),
            e => panic!("Expected PhiCoherenceError, got {:?}", e),
        }
    }

    #[test]
    fn test_coherence_weighted_attestation() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // High-coherence attester
        let high_phi = create_agent_with_phi("high-phi", 0.9);
        let receiver = create_agent_with_phi("receiver", 0.5);
        pipeline.register_agent_with_commitment(high_phi);
        pipeline.register_agent_with_commitment(receiver);

        // Attestation should succeed
        let result = pipeline.process_zk_attestation_phi_weighted("high-phi", "receiver", 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_phi_filtered_aggregation() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Mix of coherence levels
        let coherent = create_agent_with_phi("coherent-agg", 0.8);
        let moderate = create_agent_with_phi("moderate-agg", 0.5);
        let critical = create_agent_with_phi("critical-agg", 0.05);

        pipeline.register_agent_with_commitment(coherent);
        pipeline.register_agent_with_commitment(moderate);
        pipeline.register_agent_with_commitment(critical);

        // Generate proofs from coherent agents
        let mut proofs = Vec::new();
        for agent_id in &["coherent-agg", "moderate-agg"] {
            if let Ok(result) = pipeline.generate_trust_proof(*agent_id, ProofStatement::WellFormed)
            {
                proofs.push(result.proof);
            }
        }

        // Aggregation should filter out critical agent's proofs
        let aggregate =
            pipeline.aggregate_trust_proofs_phi_filtered(proofs, ProofStatement::WellFormed);

        // Should have some proofs included
        assert!(aggregate.total_count > 0);
    }

    #[test]
    fn test_byzantine_participation_check() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let coherent = create_agent_with_phi("byz-coherent", 0.75);
        let unstable = create_agent_with_phi("byz-unstable", 0.4);

        pipeline.register_agent_with_commitment(coherent);
        pipeline.register_agent_with_commitment(unstable);

        // Coherent can participate
        assert!(pipeline
            .can_participate_in_byzantine_consensus("byz-coherent")
            .unwrap());

        // Unstable cannot
        assert!(!pipeline
            .can_participate_in_byzantine_consensus("byz-unstable")
            .unwrap());
    }

    #[test]
    fn test_phi_network_health() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register agents with varying coherence
        pipeline.register_agent_with_commitment(create_agent_with_phi("health-1", 0.8));
        pipeline.register_agent_with_commitment(create_agent_with_phi("health-2", 0.7));
        pipeline.register_agent_with_commitment(create_agent_with_phi("health-3", 0.3));
        pipeline.register_agent_with_commitment(create_agent_with_phi("health-4", 0.05));

        let health = pipeline.phi_network_health();

        assert_eq!(health.agent_count, 4);
        assert!(health.average_phi > 0.0 && health.average_phi < 1.0);
        assert!(health.coherent_agents >= 2); // At least the 0.8 and 0.7 agents
        assert!(health.critical_agents >= 1); // The 0.05 agent
    }

    #[test]
    fn test_agent_coherence_state() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        pipeline.register_agent_with_commitment(create_agent_with_phi("state-test", 0.65));

        let state = pipeline.get_agent_coherence_state("state-test").unwrap();
        assert_eq!(state, CoherenceState::Stable);
    }

    #[test]
    fn test_phi_observability_export() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        pipeline.register_agent_with_commitment(create_agent_with_phi("export-test", 0.7));

        let metrics = pipeline.export_agent_phi_metrics("export-test").unwrap();

        assert_eq!(metrics.agent_id, "export-test");
        assert!((metrics.coherence - 0.7).abs() < 0.01);
        assert!(metrics.can_high_stakes);
        assert!(!metrics.is_critical);
    }

    #[test]
    fn test_phi_update_tracking() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Create agent with some output history (needed for Phi measurement)
        let mut agent = create_agent_with_phi("update-test", 0.5);
        // Add some mock output history entries for Phi measurement
        for i in 0..5 {
            agent
                .output_history
                .push(crate::agentic::OutputHistoryEntry {
                    output_id: format!("out-{}", i),
                    classification: crate::epistemic::EpistemicClassificationExtended::new(
                        EmpiricalLevel::E2PrivateVerify,
                        NormativeLevel::N1Communal,
                        MaterialityLevel::M1Temporal,
                        HarmonicLevel::H1Local,
                    ),
                    confidence: 0.8,
                    epistemic_weight: 0.5,
                    timestamp: 1000 + i * 100,
                    verified: false,
                    verification_outcome: None,
                });
        }
        pipeline.register_agent_with_commitment(agent);

        // Update Phi
        let update_result = pipeline.update_agent_phi("update-test");
        assert!(update_result.is_ok());

        let info = update_result.unwrap();
        assert_eq!(info.agent_id, "update-test");
        // Phi was measured
        assert!(info.phi_measurement > 0.0);
    }

    // =========================================================================
    // Phase 4: GIS (Graceful Ignorance System) Integration Tests
    // =========================================================================

    fn create_agent_with_calibration(id: &str, phi: f32) -> InstrumentalActor {
        let mut agent = create_agent_with_phi(id, phi);
        // Initialize calibration
        agent.uncertainty_calibration = UncertaintyCalibration::default();
        agent
    }

    #[test]
    fn test_gis_output_low_uncertainty_proceeds() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("low-uncertainty-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // Low uncertainty should proceed without escalation
        let content = OutputContent::Text("A simple verified statement".to_string());
        let statement = ProofStatement::WellFormed;
        let uncertainty = MoralUncertainty::new(0.1, 0.1, 0.1); // Low uncertainty

        let result = pipeline.process_zk_output_with_uncertainty(
            "low-uncertainty-agent",
            content,
            statement,
            uncertainty,
        );

        assert!(result.is_ok());
        let gis_result = result.unwrap();

        // Should proceed with output (no escalation)
        assert!(gis_result.output_result.is_some());
        assert!(gis_result.escalation.is_none());
        assert!(gis_result.guidance.can_proceed());
    }

    #[test]
    fn test_gis_output_high_uncertainty_escalates() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("high-uncertainty-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // High uncertainty should trigger escalation
        let content = OutputContent::Text("A complex decision with global impact".to_string());
        let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
        let uncertainty = MoralUncertainty::new(0.8, 0.9, 0.85); // High uncertainty

        let result = pipeline.process_zk_output_with_uncertainty(
            "high-uncertainty-agent",
            content,
            statement,
            uncertainty,
        );

        assert!(result.is_ok());
        let gis_result = result.unwrap();

        // Should NOT proceed with output (escalation required)
        assert!(gis_result.output_result.is_none());
        assert!(gis_result.escalation.is_some());
        assert!(gis_result.guidance.requires_human());

        // Escalation should be pending on agent
        let agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("high-uncertainty-agent")
            .unwrap();
        assert!(!agent.pending_escalations.is_empty());
    }

    #[test]
    fn test_gis_infer_uncertainty_from_content() {
        let config = ZKTrustConfig::default();
        let pipeline = ZKIntegratedPipeline::new(config);

        // Simple text with low epistemic content
        let simple_content = OutputContent::Text("Hello world".to_string());
        let simple_uncertainty = pipeline.infer_uncertainty_from_content(&simple_content);

        // High epistemic content (contains "proof", "cryptographic")
        let crypto_content = OutputContent::Text(
            "This is a cryptographic proof with verifiable commitments".to_string(),
        );
        let crypto_uncertainty = pipeline.infer_uncertainty_from_content(&crypto_content);

        // Foundational content with global impact
        let global_content = OutputContent::Text(
            "This foundational change affects the entire network globally forever".to_string(),
        );
        let global_uncertainty = pipeline.infer_uncertainty_from_content(&global_content);

        // Simple content should have higher epistemic uncertainty (lower E)
        assert!(simple_uncertainty.epistemic > crypto_uncertainty.epistemic);

        // Global content should have higher axiological/deontic uncertainty
        assert!(global_uncertainty.axiological >= simple_uncertainty.axiological);
        assert!(global_uncertainty.deontic >= simple_uncertainty.deontic);
    }

    #[test]
    fn test_gis_auto_uncertainty_processing() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("auto-unc-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // Process with auto-inferred uncertainty
        let content =
            OutputContent::Text("A moderate statement with some verifiable claims".to_string());
        let statement = ProofStatement::WellFormed;

        let result =
            pipeline.process_zk_output_auto_uncertainty("auto-unc-agent", content, statement);

        assert!(result.is_ok());
        let gis_result = result.unwrap();

        // Uncertainty should be inferred (not all zeros)
        assert!(
            gis_result.uncertainty.epistemic > 0.0
                || gis_result.uncertainty.axiological > 0.0
                || gis_result.uncertainty.deontic > 0.0
        );
    }

    #[test]
    fn test_gis_record_outcome_updates_calibration() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("calibration-agent", 0.6);
        pipeline.register_agent_with_commitment(agent);

        // Record: Agent was uncertain, and outcome was good (appropriate uncertainty)
        let result1 = pipeline.record_gis_outcome("calibration-agent", true, true);
        assert!(result1.is_ok());
        let update1 = result1.unwrap();
        assert!(update1.total_calibration_events == 1);

        // Record: Agent was confident, and outcome was good (appropriate confidence)
        let result2 = pipeline.record_gis_outcome("calibration-agent", false, true);
        assert!(result2.is_ok());
        let update2 = result2.unwrap();
        assert!(update2.total_calibration_events == 2);

        // Calibration score should be improving with good behavior
        let agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("calibration-agent")
            .unwrap();
        assert!(agent.uncertainty_calibration.calibration_score() > 0.0);
    }

    #[test]
    fn test_gis_resolve_escalation() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("resolve-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // First, create an escalation
        let content = OutputContent::Text("Complex decision".to_string());
        let statement = ProofStatement::WellFormed;
        let uncertainty = MoralUncertainty::new(0.9, 0.9, 0.9);

        let _ = pipeline.process_zk_output_with_uncertainty(
            "resolve-agent",
            content,
            statement,
            uncertainty,
        );

        // Verify escalation exists
        let agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("resolve-agent")
            .unwrap();
        assert!(!agent.pending_escalations.is_empty());

        // Resolve the escalation (sponsor approves)
        let result = pipeline.resolve_escalation("resolve-agent", "zk_output_generation", true);
        assert!(result.is_ok());

        let resolution = result.unwrap();
        assert!(resolution.sponsor_approved);
        assert!(resolution.calibration_updated);

        // Escalation should be removed
        let agent = pipeline
            .matl_pipeline()
            .trust_pipeline()
            .get_agent("resolve-agent")
            .unwrap();
        assert!(agent.pending_escalations.is_empty());
    }

    #[test]
    fn test_gis_combined_gating() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // High coherence, low uncertainty -> proceed
        let good_agent = create_agent_with_calibration("good-gating", 0.8);
        pipeline.register_agent_with_commitment(good_agent);

        let low_uncertainty = MoralUncertainty::new(0.2, 0.2, 0.2);
        let result = pipeline.check_combined_gating(
            "good-gating",
            &low_uncertainty,
            ZKOperationType::GenerateProof,
        );

        assert!(result.is_ok());
        let combined = result.unwrap();
        assert!(combined.permitted);
        assert_eq!(
            combined.recommendation,
            CombinedGatingRecommendation::Proceed
        );
        assert!(!combined.requires_escalation);

        // Low coherence -> wait for coherence
        let low_phi_agent = create_agent_with_calibration("low-phi-gating", 0.1);
        pipeline.register_agent_with_commitment(low_phi_agent);

        let result = pipeline.check_combined_gating(
            "low-phi-gating",
            &low_uncertainty,
            ZKOperationType::GenerateProof,
        );

        assert!(result.is_ok());
        let combined = result.unwrap();
        assert!(!combined.permitted);
        assert_eq!(
            combined.recommendation,
            CombinedGatingRecommendation::WaitForCoherence
        );

        // High coherence, high uncertainty -> escalate
        let high_uncertainty = MoralUncertainty::new(0.9, 0.9, 0.9);
        let result = pipeline.check_combined_gating(
            "good-gating",
            &high_uncertainty,
            ZKOperationType::GenerateProof,
        );

        assert!(result.is_ok());
        let combined = result.unwrap();
        assert!(!combined.permitted);
        assert!(combined.requires_escalation);
        assert_eq!(
            combined.recommendation,
            CombinedGatingRecommendation::EscalateForUncertainty
        );
    }

    #[test]
    fn test_gis_calibration_summary() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("summary-agent", 0.6);
        pipeline.register_agent_with_commitment(agent);

        // Record some outcomes
        let _ = pipeline.record_gis_outcome("summary-agent", true, true); // Appropriate uncertainty
        let _ = pipeline.record_gis_outcome("summary-agent", false, true); // Appropriate confidence
        let _ = pipeline.record_gis_outcome("summary-agent", false, false); // Overconfident
        let _ = pipeline.record_gis_outcome("summary-agent", true, false); // Overcautious

        let summary = pipeline.get_calibration_summary("summary-agent");
        assert!(summary.is_ok());

        let cal = summary.unwrap();
        assert_eq!(cal.total_events, 4);
        assert!(cal.appropriate_uncertainty > 0);
        assert!(cal.appropriate_confidence > 0);
        assert!(cal.overconfident > 0);
        assert!(cal.overcautious > 0);
    }

    #[test]
    fn test_gis_network_health() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        // Register agents with varying calibration states
        for i in 0..5 {
            let agent = create_agent_with_calibration(&format!("health-agent-{}", i), 0.6);
            pipeline.register_agent_with_commitment(agent);

            // Make some agents overconfident, some overcautious
            // Note: is_overconfident() requires total_events >= 10, so record enough events
            if i < 2 {
                // Overconfident: confident but wrong (need > 10 events with more wrong than right)
                for _ in 0..8 {
                    let _ =
                        pipeline.record_gis_outcome(&format!("health-agent-{}", i), false, false);
                }
                for _ in 0..3 {
                    let _ =
                        pipeline.record_gis_outcome(&format!("health-agent-{}", i), false, true);
                }
            } else if i < 4 {
                // Well-calibrated: mix of appropriate outcomes
                for _ in 0..6 {
                    let _ = pipeline.record_gis_outcome(&format!("health-agent-{}", i), true, true);
                }
                for _ in 0..6 {
                    let _ =
                        pipeline.record_gis_outcome(&format!("health-agent-{}", i), false, true);
                }
            } else {
                // Overcautious: uncertain when shouldn't be (but threshold requires 10 events)
                for _ in 0..8 {
                    let _ =
                        pipeline.record_gis_outcome(&format!("health-agent-{}", i), true, false);
                }
                for _ in 0..3 {
                    let _ = pipeline.record_gis_outcome(&format!("health-agent-{}", i), true, true);
                }
            }
        }

        let health = pipeline.gis_network_health();

        assert_eq!(health.agent_count, 5);
        assert!(health.average_calibration_score > 0.0 && health.average_calibration_score <= 1.0);
        // Should detect some overconfident agents (the first 2)
        assert!(health.overconfident_agents > 0);
    }

    #[test]
    fn test_gis_observability_exports() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("obs-gis-agent", 0.6);
        pipeline.register_agent_with_commitment(agent);

        // Record some outcomes
        let _ = pipeline.record_gis_outcome("obs-gis-agent", true, true);
        let _ = pipeline.record_gis_outcome("obs-gis-agent", false, true);

        let calibration = pipeline.get_calibration_summary("obs-gis-agent").unwrap();
        let network_health = pipeline.gis_network_health();

        let mut exports = ObservabilityExports::new("mycelix");
        exports.export_gis_calibration(&calibration);
        exports.export_gis_network_health(&network_health);

        let prom_text = exports.to_prometheus_text();
        assert!(prom_text.contains("gis_calibration_score"));
        assert!(prom_text.contains("gis_total_events"));
        assert!(prom_text.contains("gis_network_avg_calibration"));
        assert!(prom_text.contains("gis_network_well_calibrated"));
    }

    #[test]
    fn test_gis_escalation_observability_export() {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent = create_agent_with_calibration("esc-obs-agent", 0.7);
        pipeline.register_agent_with_commitment(agent);

        // Create an escalation
        let content = OutputContent::Text("High stakes decision".to_string());
        let statement = ProofStatement::WellFormed;
        let uncertainty = MoralUncertainty::new(0.9, 0.85, 0.9);

        let result = pipeline
            .process_zk_output_with_uncertainty("esc-obs-agent", content, statement, uncertainty)
            .unwrap();

        // Export the escalation
        if let Some(ref escalation) = result.escalation {
            let mut exports = ObservabilityExports::new("mycelix");
            exports.export_gis_escalation(escalation);

            let prom_text = exports.to_prometheus_text();
            assert!(prom_text.contains("gis_escalations_total"));
            assert!(prom_text.contains("gis_escalation_epistemic"));
            assert!(prom_text.contains("gis_escalation_axiological"));
            assert!(prom_text.contains("gis_escalation_deontic"));
        } else {
            panic!("Expected escalation to be created");
        }
    }
}
