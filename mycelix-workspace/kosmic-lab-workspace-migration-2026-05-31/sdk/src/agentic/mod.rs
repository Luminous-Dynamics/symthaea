// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Agentic Economy Framework
//!
//! Implementation of MIP-E-004: Agentic Economy Framework
//!
//! Enables AI agents to participate in Mycelix economic activity as
//! Instrumental Actors with sponsor accountability.
//!
//! ## Epistemic-Aware AI Agency
//!
//! This module implements the Epistemic-Aware AI Agency framework where:
//! - Agents have verifiable K-Vector trust profiles that evolve based on behavior
//! - All outputs are classified by epistemic dimensions (E-N-M-H)
//! - Consciousness-adjacent metrics (Phi) serve as quality signals
//! - Constitutional constraints are enforced at the protocol level
//! - KREDIT allocation derives from trust scores

pub mod kredit;
pub mod constraints;
pub mod lifecycle;
pub mod kvector_bridge;
pub mod fl_bridge;
pub mod epistemic_classifier;
pub mod phi_bridge;
pub mod uncertainty;
pub mod calibration_engine;
pub mod metabolism_engine;
pub mod multi_agent;
pub mod adversarial;
pub mod adversarial_sim;
pub mod monitoring;
pub mod simulation;
pub mod ml_anomaly;
pub mod persistence;
pub mod phi_integration;
pub mod phi_consensus;
pub mod provenance;
pub mod zk_trust;
pub mod cross_domain;
pub mod trust_pipeline;
pub mod api;
pub mod coordination;
pub mod orchestration;
pub mod zk_coordination;
pub mod temporal_trust;
pub mod economics;
pub mod federation;
pub mod attack_detection;
pub mod differential_privacy;
pub mod cascade_analysis;
pub mod dashboard;
pub mod adaptive_thresholds;
pub mod game_theory;
pub mod verification;
pub mod trust_portability;
pub mod integration;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(test)]
mod e2e_tests;

#[cfg(test)]
mod property_tests;

pub use kredit::{KreditAllocation, SponsorCollateral, consume_kredit};
pub use constraints::{AgentConstraints, AgentClass, enforce_constraints};
pub use lifecycle::{create_agent, suspend_agent, revoke_agent};
pub use kvector_bridge::{
    KVectorBridgeConfig, BehaviorAnalysis, analyze_behavior,
    compute_kvector_update, update_agent_kvector, compute_trust_score,
    calculate_kredit_from_trust, record_and_maybe_update,
    // Epistemic-weighted K-Vector updates
    EpistemicOutputAnalysis, analyze_outputs,
    compute_epistemic_weighted_kvector_update, update_agent_kvector_epistemic,
};
pub use fl_bridge::{
    FLAgentBridge, FLAgentBridgeConfig, FLAgentUpdateResult,
    apply_fl_feedback_to_agent, apply_fl_feedback_to_agents,
    delta_from_gradient_quality, FLRoundAgentImpact,
};
pub use epistemic_classifier::{
    AgentOutput, OutputContent, ClassificationHints, AgentOutputBuilder,
    classify_output, calculate_epistemic_weight, create_classified_output,
    EpistemicStats, AgreementScope, RelevanceDuration,
};
pub use phi_bridge::{
    CoherenceState, CoherenceCheckResult, CoherenceHistory, AgentPhiResult,
    PhiMeasurementConfig, measure_phi_simple, check_coherence_for_action,
    output_to_vector, phi_to_kvector_dimension,
};
pub use uncertainty::{
    MoralUncertainty, MoralUncertaintyType, MoralActionGuidance,
    EscalationRequest, UncertaintyCalibration, UncertainOutput,
    get_recommendations, should_proceed, maybe_escalate,
};
pub use calibration_engine::{
    CalibrationBin, CalibrationCurve, CalibrationQuality,
    AgentCalibrationProfile, KVectorCalibrationAdjustment,
    CalibrationEngine, CalibrationEngineConfig, CalibrationStats,
    // Enhanced calibration with epistemic integration
    EpistemicCalibrationProfile, EpistemicCalibrationQuality,
    TemporalCalibrationCurve, TimestampedPrediction,
    EnhancedAgentCalibrationProfile, ComprehensiveCalibrationAdjustment,
    apply_calibration_to_agent, CalibratedAgent,
};
pub use metabolism_engine::{
    MetabolismEngine, MetabolismEngineConfig, MetabolismState,
    ResourceFlow, ResourceType, MetabolicProcess, MetabolicRate,
    MetabolismStats, FlowDirection, ResourceBalance,
};
pub use multi_agent::{
    // Trust-weighted consensus
    AgentVote, ConsensusResult, ConsensusConfig, compute_consensus,
    // Cross-agent calibration
    CalibrationKnowledge, CrossAgentCalibration,
    // Collaboration protocols
    CollaborativeTask, CollaborativeTaskType, CollaborativeTaskStatus,
    TaskContribution, CollaborativeResult, CollaborationManager, CollaborationError,
    // Reputation propagation
    AgentInteraction, InteractionType, ReputationPropagation,
};
pub use adversarial::{
    // Gaming detection
    GamingAttackType, GamingDetectionResult, GamingIndicator, GamingResponse,
    GamingDetectionConfig, GamingDetector,
    // Sybil resistance
    SybilEvidence, SybilEvidenceType, SybilDetector,
    // Collusion detection
    CollusionEvidence, CollusionType, CollusionDetector, AgentInteractionRecord,
    // Quarantine system
    QuarantineEntry, QuarantineReason, ReviewStatus, QuarantineManager,
};
pub use monitoring::{
    // Metrics
    AgentMetrics, MetricsHistory,
    // Trust evolution
    TrustEvent, TrustEventType, KVectorSnapshot,
    // Alerts
    AlertType, AlertSeverity, AgentAlert, AlertThresholds,
    // Engine
    MonitoringEngine, DashboardSummary,
};
pub use simulation::{
    // Agent archetypes
    AgentArchetype, AgentBehaviorConfig,
    // Simulation engine
    SimulationConfig, SimulatedAgent, SimulationEngine,
    // Results
    TickResult, SimulationReport,
    // Predefined scenarios
    Scenarios,
};
pub use ml_anomaly::{
    // Feature extraction
    AgentFeatures,
    // Isolation Forest
    IsolationForest, IsolationTree,
    // Reconstruction
    ReconstructionDetector,
    // Time-series
    TimeSeriesAnomalyDetector, TimeSeriesAnomalyResult,
    // ML Ensemble
    MLAnomalyDetector, MLAnomalyConfig, MLAnomalyResult,
    // Hybrid detector
    HybridAnomalyDetector, HybridAnomalyResult,
    AnomalyType, AnomalyRecommendation,
};
pub use persistence::{
    // Events
    AgentEvent, EventLogEntry,
    KVectorSnapshot as PersistedKVectorSnapshot,
    // Backend trait and errors
    AgentStorageBackend, PersistenceError, PersistenceResult,
    // Memory backend
    MemoryStorageBackend,
    // Repository
    AgentRepository, AgentQueryBuilder,
    // Statistics
    AgentStatistics,
};
pub use phi_integration::{
    // Collective Phi
    CollectivePhiResult, CollectiveCoherenceLevel, measure_collective_phi,
    // Emergent behavior detection
    EmergentBehaviorType, EmergentBehavior, EmergentBehaviorDetector,
    // Phi-gated actions
    PhiGatingConfig, StakesLevel, PhiGatingResult, PhiGatingRecommendation,
    check_phi_gating,
    // Clustering
    PhiClusterResult, PhiCluster, cluster_agents_by_phi,
    // Temporal analysis
    PhiEvolutionTracker, PhiEvolutionSummary,
};
pub use phi_consensus::{
    // Configuration
    PhiConsensusConfig,
    // Phi contribution analysis
    PhiContribution, compute_phi_contributions,
    // Phi-weighted consensus
    PhiConsensusResult, PhiConsensusStatus, PhiConsensusRecommendation,
    compute_phi_weighted_consensus,
    // Helpers
    should_proceed as phi_should_proceed,
    get_recommendation as get_phi_recommendation,
};
pub use provenance::{
    // Core types
    DerivationType, ProvenanceNode, ProvenanceChain,
    ChainVerificationResult, ChainError,
    // Builders
    ProvenanceBuilder, ChainBuilder,
    // Integration
    ProvenancedOutput,
    // Registry
    ProvenanceRegistry, RegistryError, RegistryStats,
};
pub use zk_trust::{
    // Proof statements
    ProofStatement,
    // Commitments
    KVectorCommitment,
    // Proofs
    TrustProof, ProofData,
    // Prover
    TrustProver, ProverConfig, ProofError,
    // Verifier
    TrustVerifier, VerificationResult, VerificationError,
    // Aggregation
    AggregatedTrustProof, AggregateStatement, aggregate_proofs,
};
pub use cross_domain::{
    // Domain types
    TrustDomain, DomainRelevance,
    // Domain templates
    DomainTemplates,
    // Translation
    TranslationResult, DimensionTranslation,
    translate_trust, compute_domain_trust,
    // Path translation
    TranslationPath, translate_path,
    // Registry
    DomainRegistry,
    // Compatibility analysis
    DomainCompatibility, analyze_domain_compatibility,
};
pub use trust_pipeline::{
    // Configuration
    PipelineConfig,
    // Pipeline stages
    RegisteredOutput, ConsensusOutcome, TrustUpdate,
    TrustDelta, TrustDirection,
    TrustAttestation as PipelineTrustAttestation, TranslatedAttestation,
    // Pipeline engine
    TrustPipeline,
    // Errors
    PipelineError,
};
pub use api::{
    // API types
    ApiError, ApiResult,
    CreateAgentRequest, CreateAgentResponse,
    UpdateAgentRequest,
    AgentSummary as ApiAgentSummary,
    ListAgentsResponse, KVectorHistoryResponse, KVectorHistoryEntry, KVectorValues,
    EventsResponse, EventSummary,
    // Escalation API types (GIS integration)
    EscalationSummary, EscalationResolutionResponse, CalibrationSummary,
    // Service
    AgentApiService,
};
pub use zk_coordination::{
    // Configuration
    ZKCoordinationConfig,
    // Proofs
    MembershipProof, VoteProof,
    // ZK-enabled group
    ZKAgentGroup,
    // Errors
    ZKCoordinationError,
};
pub use temporal_trust::{
    // Configuration
    TemporalTrustConfig, TrustDecayConfig, VelocityLimitConfig, ReputationMemoryConfig,
    // Decay curves
    DecayCurve, VelocityViolationAction,
    // Snapshots and events
    TrustSnapshot, SnapshotReason,
    // Manager
    TemporalTrustManager,
    // Results
    TrustUpdateResult, TemporalTrustError,
};
pub use economics::{
    // Slashing
    SlashingConfig, ViolationSeverity, SlashEvent, ViolationType,
    SlashingEngine, SlashResult,
    // Rewards
    RewardConfig, RewardEvent, RewardType, RewardEngine,
    // Bonding curves
    BondingCurve, BondingCurveType,
    // Commit-reveal voting
    CommitRevealVoting, VoteCommitment, VoteReveal, CommitRevealVote,
    CommitRevealError,
};
pub use federation::{
    // Configuration
    FederationConfig, SwarmId, SwarmProfile,
    // Attestations
    TrustAttestation, AttestationEvidence,
    // Federated consensus
    FederatedProposal, FederatedProposalType, FederatedProposalState,
    FederatedVote, FederatedVoteDecision,
    // Bridges
    TrustBridge, BridgeType, TrustTransfer, TransferStatus,
    // Engine
    FederationEngine, CrossSwarmTrust, FederationStats,
    FederationError,
};
pub use attack_detection::{
    // Configuration
    AttackDetectionConfig,
    // Events
    TrustEvent as AttackTrustEvent, TrustEventType as AttackEventType, EventSource,
    // Signatures
    AttackSignature, AttackPattern, AttackSeverity, RecommendedResponse,
    // Detection
    DetectionResult, DetectedAttackType, Evidence, EvidenceType,
    // Analyzer
    StreamingAnalyzer, AgentActivityProfile, DetectionStats,
    // Alerts
    Alert, AlertStatus, AlertPipeline,
};
pub use differential_privacy::{
    // Configuration
    DPConfig, ClippingBounds,
    // Noise mechanisms
    NoiseMechanism, NoiseGenerator, DPRng,
    // Budget
    PrivacyBudget, BudgetQuery,
    // Aggregations
    PrivateAggregator,
    // Local DP
    LocalDP,
    // Trust analytics
    PrivateTrustAnalytics, TrustDistribution,
    // Errors
    PrivacyError,
};
pub use cascade_analysis::{
    // Configuration
    CascadeConfig,
    // Network model
    NetworkAgent, NetworkEdge, EdgeType, TrustNetwork,
    NetworkSnapshot, AgentState,
    // Engine
    CascadeEngine, CascadeEvent, CascadeEventType,
    // Results
    CascadeResult, RecoveryResult, TickSnapshot,
    CriticalAgent, ResilienceScore, ContagionPath,
    TopologyAnalysis, TopologyRisk,
};
pub use dashboard::{
    // Configuration
    DashboardConfig,
    // Live metrics
    LiveMetrics, AlertCounts, MetricsAggregator, MetricsInput,
    // Events
    DashboardEvent, DashboardEventType, EventPriority, EventStream,
    // Charts
    DataPoint, TimeSeries, ChartType, ChartDataBuilder,
    // Alerts panel
    DashboardAlert, AlertSeverity as DashboardAlertSeverity,
    AlertStatus as DashboardAlertStatus, AlertAction, AlertActionType, AlertPanel,
    // Widgets
    Widget, WidgetType, WidgetPosition, WidgetSize, default_layout,
    // Dashboard state
    Dashboard,
};
pub use adaptive_thresholds::{
    // Configuration
    AdaptiveConfig,
    // Threshold types
    ThresholdType, ThresholdState,
    // Feedback
    ThresholdFeedback, FeedbackOutcome, FeedbackContext,
    // Bandit
    BanditArm, ThresholdBandit,
    // Gradient
    GradientEstimator,
    // Engine
    AdaptiveThresholdEngine, ThresholdRecommendation, RecommendationDirection,
};
pub use game_theory::{
    // Players and strategies
    Player, PlayerType, Strategy, StrategyAction, ActionType, ActionCondition,
    // Payoffs
    PayoffEntry, GameDefinition,
    // Equilibrium
    NashEquilibrium, EquilibriumFinder,
    // Incentive analysis
    IncentiveAnalysis, ProfitableDeviation, IncentiveAnalyzer,
    // Mechanism validation
    MechanismValidation, MechanismParams, validate_mechanism,
    // Pre-built games
    trust_attestation_game, voting_game,
};
pub use verification::{
    // Invariants
    Invariant, InvariantType, ViolationSeverity as VerificationSeverity,
    InvariantCheckResult, InvariantViolation,
    // Properties
    PropertySpec, PropertyFormula, AtomicPredicate, ProofStatus,
    // Proof obligations
    ProofObligation, ProofWitness, ProofTechnique, Counterexample,
    SystemState, Action,
    // Engine
    VerificationEngine, VerificationEvent, VerificationEventType, VerificationSummary,
};
pub use trust_portability::{
    // Configuration
    PortabilityConfig,
    // Chain identity
    ChainId, ChainProfile, ChainType, ProofType,
    // Portable trust
    PortableTrust, KVectorDimension, TrustProof as PortabilityTrustProof, ProofSignature,
    // Import/Export
    ExportResult, ExportStatus, ImportResult, ImportStatus,
    VerificationResult as PortabilityVerificationResult,
    // Bridge
    BridgeAdapter, BridgeError, MockBridgeAdapter,
    // Engine
    PortabilityEngine, PortabilityStats,
};

#[cfg(feature = "parallel")]
pub use parallel::{
    // Configuration
    ParallelSimConfig, SimAgentBehavior,
    // Engine
    ParallelSimEngine, SimAgent, ParallelTickResult,
    // Batch operations
    KVectorBatch, TickAggregators, RandomBuffer,
    // Benchmarking
    BenchmarkResult, benchmark_simulation,
};

pub use integration::{
    // Trust Pipeline
    TrustPipelineConfig, IntegratedTrustPipeline, AttestationResult,
    // Attack Response
    AttackResponseConfig, IntegratedAttackResponse, AttackResponse, ResponseAction,
    // Privacy Analytics
    PrivacyAnalyticsConfig, IntegratedPrivacyAnalytics, PrivateAnalyticsResult,
    // Epistemic Lifecycle
    EpistemicLifecycleConfig, IntegratedEpistemicLifecycle, OutputProcessingResult,
    // Errors
    IntegrationError,
};

use serde::{Deserialize, Serialize};
use crate::matl::KVector;
use crate::epistemic::EpistemicClassificationExtended;

/// Unique identifier for an Instrumental Actor
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    /// Generate new random agent ID
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self(format!("agent-{:x}", timestamp))
    }

    /// Create from string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Instrumental Actor (AI Agent)
///
/// An AI agent with a verifiable epistemic fingerprint - a trust profile
/// that proves reliability without revealing internal state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentalActor {
    /// Unique agent identifier
    pub agent_id: AgentId,
    /// DID of human sponsor
    pub sponsor_did: String,
    /// Agent classification
    pub agent_class: AgentClass,
    /// KREDIT balance (can go negative)
    pub kredit_balance: i64,
    /// Maximum KREDIT per epoch (derived from trust score)
    pub kredit_cap: u64,
    /// Operational constraints
    pub constraints: AgentConstraints,
    /// Behavior log entries
    pub behavior_log: Vec<BehaviorLogEntry>,
    /// Current status
    pub status: AgentStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Actions this hour (for rate limiting)
    pub actions_this_hour: u32,
    /// K-Vector trust profile (10 dimensions including k_phi coherence)
    /// Evolves based on behavioral outcomes
    pub k_vector: KVector,
    /// Epistemic statistics tracking agent output quality
    /// Used for epistemic-weighted K-Vector updates
    pub epistemic_stats: EpistemicStats,
    /// Recent output history for epistemic analysis
    #[serde(default)]
    pub output_history: Vec<OutputHistoryEntry>,
    /// Uncertainty calibration tracking (GIS v4.0)
    /// Tracks whether agent is appropriately uncertain
    #[serde(default)]
    pub uncertainty_calibration: UncertaintyCalibration,
    /// Pending escalations waiting for human sponsor response
    #[serde(default)]
    pub pending_escalations: Vec<EscalationRequest>,
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AgentStatus {
    /// Normal operation
    Active,
    /// Reduced capacity (KREDIT low or sponsor CIV dropped)
    Throttled,
    /// Manually suspended by sponsor
    Suspended,
    /// Permanently revoked
    Revoked,
}

/// Behavior log entry for audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorLogEntry {
    /// Timestamp
    pub timestamp: u64,
    /// Action type
    pub action_type: String,
    /// KREDIT consumed
    pub kredit_consumed: u64,
    /// Counterparties involved
    pub counterparties: Vec<String>,
    /// Action outcome
    pub outcome: ActionOutcome,
}

/// Action outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// Completed successfully
    Success,
    /// Failed due to constraints
    ConstraintViolation,
    /// Failed due to insufficient KREDIT
    InsufficientKredit,
    /// Failed for other reasons
    Error,
}

/// Output history entry for epistemic tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputHistoryEntry {
    /// Output identifier
    pub output_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Epistemic classification (E-N-M-H)
    pub classification: EpistemicClassificationExtended,
    /// Classification confidence
    pub confidence: f32,
    /// Epistemic weight for K-Vector updates
    pub epistemic_weight: f32,
    /// Whether outcome was verified
    pub verified: bool,
    /// Outcome after verification (if any)
    pub verification_outcome: Option<VerificationOutcome>,
}

/// Outcome of verifying an agent output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationOutcome {
    /// Output was correct/accurate
    Correct,
    /// Output was incorrect/inaccurate
    Incorrect,
    /// Output was partially correct
    Partial,
    /// Verification inconclusive
    Inconclusive,
}

/// Sponsorship requirements
pub mod sponsor_requirements {
    /// Minimum CIV to create agents
    pub const MIN_CIV_TO_CREATE: f64 = 0.5;
    /// Minimum CIV to keep agents active
    pub const MIN_CIV_TO_MAINTAIN: f64 = 0.4;
    /// Maximum agents per sponsor
    pub const MAX_AGENTS_PER_SPONSOR: u32 = 10;
}

impl InstrumentalActor {
    /// Check if agent can perform actions
    pub fn is_operational(&self) -> bool {
        matches!(self.status, AgentStatus::Active | AgentStatus::Throttled)
    }

    /// Get throttle factor (1.0 = normal, <1.0 = throttled)
    pub fn throttle_factor(&self) -> f64 {
        match self.status {
            AgentStatus::Active => 1.0,
            AgentStatus::Throttled => 0.5,
            _ => 0.0,
        }
    }

    /// Record action in behavior log
    pub fn record_action(&mut self, action_type: &str, kredit: u64, outcome: ActionOutcome) {
        self.behavior_log.push(BehaviorLogEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            action_type: action_type.to_string(),
            kredit_consumed: kredit,
            counterparties: vec![],
            outcome,
        });
        self.actions_this_hour += 1;
    }

    /// Get summary statistics
    pub fn summary_stats(&self) -> AgentSummary {
        let total_actions = self.behavior_log.len();
        let successful = self.behavior_log
            .iter()
            .filter(|e| e.outcome == ActionOutcome::Success)
            .count();
        let total_kredit: u64 = self.behavior_log.iter().map(|e| e.kredit_consumed).sum();

        AgentSummary {
            total_actions,
            successful_actions: successful,
            success_rate: if total_actions > 0 {
                successful as f64 / total_actions as f64
            } else {
                0.0
            },
            total_kredit_consumed: total_kredit,
            current_kredit: self.kredit_balance,
        }
    }

    /// Record an output with epistemic classification
    pub fn record_output(&mut self, output: &AgentOutput) {
        let weight = calculate_epistemic_weight(&output.classification);

        // Add to output history
        self.output_history.push(OutputHistoryEntry {
            output_id: output.output_id.clone(),
            timestamp: output.timestamp,
            classification: output.classification,
            confidence: output.classification_confidence,
            epistemic_weight: weight,
            verified: false,
            verification_outcome: None,
        });

        // Update epistemic stats
        self.epistemic_stats.add_output(&output.classification);

        // Keep history bounded (last 1000 outputs)
        if self.output_history.len() > 1000 {
            self.output_history.remove(0);
        }
    }

    /// Mark an output as verified with outcome
    pub fn verify_output(&mut self, output_id: &str, outcome: VerificationOutcome) {
        if let Some(entry) = self.output_history.iter_mut()
            .find(|e| e.output_id == output_id)
        {
            entry.verified = true;
            entry.verification_outcome = Some(outcome);
        }
    }

    /// Get epistemic quality score (0.0-1.0)
    pub fn epistemic_quality(&self) -> f32 {
        self.epistemic_stats.quality_score()
    }

    /// Get average epistemic weight
    pub fn average_epistemic_weight(&self) -> f32 {
        self.epistemic_stats.average_weight
    }

    /// Get verified output accuracy (among verified outputs)
    pub fn verified_accuracy(&self) -> f32 {
        let verified: Vec<_> = self.output_history.iter()
            .filter(|e| e.verified)
            .collect();

        if verified.is_empty() {
            return 0.5; // Neutral default
        }

        let correct = verified.iter()
            .filter(|e| matches!(e.verification_outcome, Some(VerificationOutcome::Correct)))
            .count();

        let partial = verified.iter()
            .filter(|e| matches!(e.verification_outcome, Some(VerificationOutcome::Partial)))
            .count();

        // Correct = 1.0, Partial = 0.5
        (correct as f32 + partial as f32 * 0.5) / verified.len() as f32
    }
}

/// Agent summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSummary {
    /// Total actions taken
    pub total_actions: usize,
    /// Successful actions
    pub successful_actions: usize,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Total KREDIT consumed
    pub total_kredit_consumed: u64,
    /// Current KREDIT balance
    pub current_kredit: i64,
}

/// Constitutional constraints that apply to all agents
pub const CONSTITUTIONAL_CONSTRAINTS: ConstContraints = ConstContraints {
    can_vote_governance: false,
    can_become_validator: false,
    can_govern_hearth: false,
    can_sponsor_agents: false,
    can_hold_civ: false,
    can_hold_sap: false,
    can_receive_cgc: true,
    can_send_cgc: false,
};

/// Constitutional constraints (compile-time constants)
pub struct ConstContraints {
    /// Can vote on governance proposals
    pub can_vote_governance: bool,
    /// Can become network validator
    pub can_become_validator: bool,
    /// Can participate in HEARTH governance
    pub can_govern_hearth: bool,
    /// Can sponsor other agents
    pub can_sponsor_agents: bool,
    /// Can accumulate CIV reputation
    pub can_hold_civ: bool,
    /// Can hold SAP directly (vs KREDIT)
    pub can_hold_sap: bool,
    /// Can receive CGC/SPORE gifts
    pub can_receive_cgc: bool,
    /// Can send CGC/SPORE gifts
    pub can_send_cgc: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id() {
        let id = AgentId::generate();
        assert!(id.as_str().starts_with("agent-"));
    }

    #[test]
    fn test_agent_operational() {
        let agent = InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        assert!(agent.is_operational());
        assert!((agent.throttle_factor() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_trust_derived_kredit() {
        let agent = InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector: KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.7, 0.65),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        // High trust K-Vector should produce high trust score
        let trust = agent.k_vector.trust_score();
        assert!(trust > 0.5);

        // Trust score derives KREDIT cap
        let derived_kredit = calculate_kredit_from_trust(trust);
        assert!(derived_kredit > 10000);
    }

    #[test]
    fn test_agent_epistemic_tracking() {
        use crate::epistemic::{EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel};

        let mut agent = InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 1000,
            last_activity: 1000,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        // Create and record a high-quality output
        let output = AgentOutputBuilder::new(agent.agent_id.as_str())
            .content(OutputContent::Text("Verified fact with proof".to_string()))
            .classification(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
                HarmonicLevel::H1Local,
            )
            .confidence(0.9)
            .build()
            .unwrap();

        agent.record_output(&output);

        // Check epistemic stats updated
        assert_eq!(agent.epistemic_stats.total_outputs, 1);
        assert!(agent.average_epistemic_weight() > 0.2);
        assert_eq!(agent.output_history.len(), 1);

        // Verify the output
        agent.verify_output(&output.output_id, VerificationOutcome::Correct);
        assert!((agent.verified_accuracy() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_constitutional_constraints() {
        assert!(!CONSTITUTIONAL_CONSTRAINTS.can_vote_governance);
        assert!(!CONSTITUTIONAL_CONSTRAINTS.can_become_validator);
        assert!(CONSTITUTIONAL_CONSTRAINTS.can_receive_cgc);
    }
}
