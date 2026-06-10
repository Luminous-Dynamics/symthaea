// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Agentic Economy Framework
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

pub mod adaptive_thresholds;
pub mod adversarial;
pub mod adversarial_sim;
pub mod api;
pub mod attack_detection;
pub mod calibration_engine;
pub mod cascade_analysis;
pub mod coherence_bridge;
pub mod coherence_integration;
pub mod constraints;
pub mod coordination;
pub mod cross_domain;
pub mod dashboard;
pub mod differential_privacy;
pub mod economics;
pub mod epistemic_classifier;
pub mod federation;
pub mod fl_bridge;
pub mod game_theory;
pub mod integration;
pub mod kredit;
pub mod kvector_bridge;
pub mod lifecycle;
pub mod metabolism_engine;
pub mod ml_anomaly;
pub mod monitoring;
pub mod multi_agent;
pub mod orchestration;
pub mod persistence;
pub mod phi_consensus;
pub mod provenance;
pub mod simulation;
pub mod temporal_trust;
pub mod trust_pipeline;
pub mod trust_portability;
pub mod uncertainty;
pub mod verification;
pub mod zk_coordination;
pub mod zk_trust;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(test)]
mod e2e_tests;

#[cfg(test)]
mod property_tests;

pub use adaptive_thresholds::{
    // Configuration
    AdaptiveConfig,
    // Engine
    AdaptiveThresholdEngine,
    // Bandit
    BanditArm,
    FeedbackContext,
    FeedbackOutcome,
    // Gradient
    GradientEstimator,
    RecommendationDirection,
    ThresholdBandit,
    // Feedback
    ThresholdFeedback,
    ThresholdRecommendation,
    ThresholdState,
    // Threshold types
    ThresholdType,
};
pub use adversarial::{
    AgentInteractionRecord,
    CollusionDetector,
    // Collusion detection
    CollusionEvidence,
    CollusionType,
    // Gaming detection
    GamingAttackType,
    GamingDetectionConfig,
    GamingDetectionResult,
    GamingDetector,
    GamingIndicator,
    GamingResponse,
    // Quarantine system
    QuarantineEntry,
    QuarantineManager,
    QuarantineReason,
    ReviewStatus,
    SybilDetector,
    // Sybil resistance
    SybilEvidence,
    SybilEvidenceType,
};
pub use api::{
    // Service
    AgentApiService,
    AgentSummary as ApiAgentSummary,
    // API types
    ApiError,
    ApiResult,
    CalibrationSummary,
    CreateAgentRequest,
    CreateAgentResponse,
    EscalationResolutionResponse,
    // Escalation API types (GIS integration)
    EscalationSummary,
    EventSummary,
    EventsResponse,
    KVectorHistoryEntry,
    KVectorHistoryResponse,
    KVectorValues,
    ListAgentsResponse,
    UpdateAgentRequest,
};
pub use attack_detection::{
    AgentActivityProfile,
    // Alerts
    Alert,
    AlertPipeline,
    AlertStatus,
    // Configuration
    AttackDetectionConfig,
    AttackPattern,
    AttackSeverity,
    // Signatures
    AttackSignature,
    DetectedAttackType,
    // Detection
    DetectionResult,
    DetectionStats,
    EventSource,
    Evidence,
    EvidenceType,
    RecommendedResponse,
    // Analyzer
    StreamingAnalyzer,
    // Events
    TrustEvent as AttackTrustEvent,
    TrustEventType as AttackEventType,
};
pub use calibration_engine::{
    apply_calibration_to_agent,
    AgentCalibrationProfile,
    CalibratedAgent,
    CalibrationBin,
    CalibrationCurve,
    CalibrationEngine,
    CalibrationEngineConfig,
    CalibrationQuality,
    CalibrationStats,
    ComprehensiveCalibrationAdjustment,
    EnhancedAgentCalibrationProfile,
    // Enhanced calibration with epistemic integration
    EpistemicCalibrationProfile,
    EpistemicCalibrationQuality,
    KVectorCalibrationAdjustment,
    TemporalCalibrationCurve,
    TimestampedPrediction,
};
pub use cascade_analysis::{
    AgentState,
    // Configuration
    CascadeConfig,
    // Engine
    CascadeEngine,
    CascadeEvent,
    CascadeEventType,
    // Results
    CascadeResult,
    ContagionPath,
    CriticalAgent,
    EdgeType,
    // Network model
    NetworkAgent,
    NetworkEdge,
    NetworkSnapshot,
    RecoveryResult,
    ResilienceScore,
    TickSnapshot,
    TopologyAnalysis,
    TopologyRisk,
    TrustNetwork,
};
pub use coherence_bridge::{
    check_coherence_for_action, coherence_to_kvector_dimension, measure_coherence,
    output_to_vector, AgentCoherenceResult, CoherenceCheckResult, CoherenceHistory,
    CoherenceMeasurementConfig, CoherenceState,
};
pub use coherence_integration::{
    check_coherence_gating,
    cluster_agents_by_coherence,
    measure_collective_coherence,
    CoherenceCluster,
    // Clustering
    CoherenceClusterResult,
    // Coherence-gated actions
    CoherenceGatingConfig,
    CoherenceGatingRecommendation,
    CoherenceGatingResult,
    CollectiveCoherenceLevel,
    // Collective Coherence
    CollectiveCoherenceResult,
    EmergentBehavior,
    // Emergent behavior detection
    EmergentBehaviorType,
    StakesLevel,
};
pub use constraints::{enforce_constraints, AgentClass, AgentConstraints};
pub use cross_domain::{
    analyze_domain_compatibility,
    compute_domain_trust,
    translate_path,
    translate_trust,
    DimensionTranslation,
    // Compatibility analysis
    DomainCompatibility,
    // Registry
    DomainRegistry,
    DomainRelevance,
    // Domain templates
    DomainTemplates,
    // Path translation
    TranslationPath,
    // Translation
    TranslationResult,
    // Domain types
    TrustDomain,
};
pub use dashboard::{
    default_layout,
    AlertAction,
    AlertActionType,
    AlertCounts,
    AlertPanel,
    AlertSeverity as DashboardAlertSeverity,
    AlertStatus as DashboardAlertStatus,
    ChartDataBuilder,
    ChartType,
    // Dashboard state
    Dashboard,
    // Alerts panel
    DashboardAlert,
    // Configuration
    DashboardConfig,
    // Events
    DashboardEvent,
    DashboardEventType,
    // Charts
    DataPoint,
    EventPriority,
    EventStream,
    // Live metrics
    LiveMetrics,
    MetricsAggregator,
    MetricsInput,
    TimeSeries,
    // Widgets
    Widget,
    WidgetPosition,
    WidgetSize,
    WidgetType,
};
pub use differential_privacy::{
    BudgetQuery,
    ClippingBounds,
    // Configuration
    DPConfig,
    DPRng,
    // Local DP
    LocalDP,
    NoiseGenerator,
    // Noise mechanisms
    NoiseMechanism,
    // Budget
    PrivacyBudget,
    // Errors
    PrivacyError,
    // Aggregations
    PrivateAggregator,
    // Trust analytics
    PrivateTrustAnalytics,
    TrustDistribution,
};
pub use economics::{
    // Bonding curves
    BondingCurve,
    BondingCurveType,
    CommitRevealError,
    CommitRevealVote,
    // Commit-reveal voting
    CommitRevealVoting,
    // Rewards
    RewardConfig,
    RewardEngine,
    RewardEvent,
    RewardType,
    SlashEvent,
    SlashResult,
    // Slashing
    SlashingConfig,
    SlashingEngine,
    ViolationSeverity,
    ViolationType,
    VoteCommitment,
    VoteReveal,
};
pub use epistemic_classifier::{
    calculate_epistemic_weight, classify_output, create_classified_output, AgentOutput,
    AgentOutputBuilder, AgreementScope, ClassificationHints, EpistemicStats, OutputContent,
    RelevanceDuration,
};
pub use federation::{
    AttestationEvidence,
    BridgeType,
    CrossSwarmTrust,
    // Federated consensus
    FederatedProposal,
    FederatedProposalState,
    FederatedProposalType,
    FederatedVote,
    FederatedVoteDecision,
    // Configuration
    FederationConfig,
    // Engine
    FederationEngine,
    FederationError,
    FederationStats,
    SwarmId,
    SwarmProfile,
    TransferStatus,
    // Attestations
    TrustAttestation,
    // Bridges
    TrustBridge,
    TrustTransfer,
};
pub use fl_bridge::{
    apply_fl_feedback_to_agent, apply_fl_feedback_to_agents, delta_from_gradient_quality,
    FLAgentBridge, FLAgentBridgeConfig, FLAgentUpdateResult, FLRoundAgentImpact,
};
pub use game_theory::{
    // Pre-built games
    trust_attestation_game,
    validate_mechanism,
    voting_game,
    ActionCondition,
    ActionType,
    EquilibriumFinder,
    GameDefinition,
    // Incentive analysis
    IncentiveAnalysis,
    IncentiveAnalyzer,
    MechanismParams,
    // Mechanism validation
    MechanismValidation,
    // Equilibrium
    NashEquilibrium,
    // Payoffs
    PayoffEntry,
    // Players and strategies
    Player,
    PlayerType,
    ProfitableDeviation,
    Strategy,
    StrategyAction,
};
pub use kredit::{consume_kredit, KreditAllocation, SponsorCollateral};
pub use kvector_bridge::{
    analyze_behavior,
    analyze_outputs,
    calculate_kredit_from_trust,
    compute_epistemic_weighted_kvector_update,
    compute_kvector_update,
    compute_trust_score,
    record_and_maybe_update,
    update_agent_kvector,
    update_agent_kvector_epistemic,
    BehaviorAnalysis,
    // Epistemic-weighted K-Vector updates
    EpistemicOutputAnalysis,
    KVectorBridgeConfig,
};
pub use lifecycle::{create_agent, revoke_agent, suspend_agent};
pub use metabolism_engine::{
    FlowDirection, MetabolicProcess, MetabolicRate, MetabolismEngine, MetabolismEngineConfig,
    MetabolismState, MetabolismStats, ResourceBalance, ResourceFlow, ResourceType,
};
pub use ml_anomaly::{
    // Feature extraction
    AgentFeatures,
    AnomalyRecommendation,
    AnomalyType,
    // Hybrid detector
    HybridAnomalyDetector,
    HybridAnomalyResult,
    // Isolation Forest
    IsolationForest,
    IsolationTree,
    MLAnomalyConfig,
    // ML Ensemble
    MLAnomalyDetector,
    MLAnomalyResult,
    // Reconstruction
    ReconstructionDetector,
    // Time-series
    TimeSeriesAnomalyDetector,
    TimeSeriesAnomalyResult,
};
pub use monitoring::{
    AgentAlert,
    // Metrics
    AgentMetrics,
    AlertSeverity,
    AlertThresholds,
    // Alerts
    AlertType,
    DashboardSummary,
    KVectorSnapshot,
    MetricsHistory,
    // Engine
    MonitoringEngine,
    // Trust evolution
    TrustEvent,
    TrustEventType,
};
pub use multi_agent::{
    compute_consensus,
    // Reputation propagation
    AgentInteraction,
    // Trust-weighted consensus
    AgentVote,
    // Cross-agent calibration
    CalibrationKnowledge,
    CollaborationError,
    CollaborationManager,
    CollaborativeResult,
    // Collaboration protocols
    CollaborativeTask,
    CollaborativeTaskStatus,
    CollaborativeTaskType,
    ConsensusConfig,
    ConsensusResult,
    CrossAgentCalibration,
    InteractionType,
    ReputationPropagation,
    TaskContribution,
};
pub use persistence::{
    // Events
    AgentEvent,
    AgentQueryBuilder,
    // Repository
    AgentRepository,
    // Statistics
    AgentStatistics,
    // Backend trait and errors
    AgentStorageBackend,
    EventLogEntry,
    KVectorSnapshot as PersistedKVectorSnapshot,
    // Memory backend
    MemoryStorageBackend,
    PersistenceError,
    PersistenceResult,
};
pub use phi_consensus::{
    compute_phi_contributions,
    compute_phi_weighted_consensus,
    get_recommendation as get_phi_recommendation,
    // Helpers
    should_proceed as phi_should_proceed,
    // Configuration
    PhiConsensusConfig,
    PhiConsensusRecommendation,
    // Phi-weighted consensus
    PhiConsensusResult,
    PhiConsensusStatus,
    // Phi contribution analysis
    PhiContribution,
};
pub use provenance::{
    ChainBuilder,
    ChainError,
    ChainVerificationResult,
    // Core types
    DerivationType,
    // Builders
    ProvenanceBuilder,
    ProvenanceChain,
    ProvenanceNode,
    // Registry
    ProvenanceRegistry,
    // Integration
    ProvenancedOutput,
    RegistryError,
    RegistryStats,
};
pub use simulation::{
    // Agent archetypes
    AgentArchetype,
    AgentBehaviorConfig,
    // Predefined scenarios
    Scenarios,
    SimulatedAgent,
    // Simulation engine
    SimulationConfig,
    SimulationEngine,
    SimulationReport,
    // Results
    TickResult,
};
pub use temporal_trust::{
    // Decay curves
    DecayCurve,
    ReputationMemoryConfig,
    SnapshotReason,
    // Configuration
    TemporalTrustConfig,
    TemporalTrustError,
    // Manager
    TemporalTrustManager,
    TrustDecayConfig,
    // Snapshots and events
    TrustSnapshot,
    // Results
    TrustUpdateResult,
    VelocityLimitConfig,
    VelocityViolationAction,
};
pub use trust_pipeline::{
    ConsensusOutcome,
    // Configuration
    PipelineConfig,
    // Errors
    PipelineError,
    // Pipeline stages
    RegisteredOutput,
    TranslatedAttestation,
    TrustAttestation as PipelineTrustAttestation,
    TrustDelta,
    TrustDirection,
    // Pipeline engine
    TrustPipeline,
    TrustUpdate,
};
pub use trust_portability::{
    // Bridge
    BridgeAdapter,
    BridgeError,
    // Chain identity
    ChainId,
    ChainProfile,
    ChainType,
    // Import/Export
    ExportResult,
    ExportStatus,
    ImportResult,
    ImportStatus,
    KVectorDimension,
    MockBridgeAdapter,
    // Configuration
    PortabilityConfig,
    // Engine
    PortabilityEngine,
    PortabilityStats,
    // Portable trust
    PortableTrust,
    ProofSignature,
    ProofType,
    TrustProof as PortabilityTrustProof,
    VerificationResult as PortabilityVerificationResult,
};
pub use uncertainty::{
    get_recommendations, maybe_escalate, should_proceed, EscalationRequest, MoralActionGuidance,
    MoralUncertainty, MoralUncertaintyType, UncertainOutput, UncertaintyCalibration,
};
pub use verification::{
    Action,
    AtomicPredicate,
    Counterexample,
    // Invariants
    Invariant,
    InvariantCheckResult,
    InvariantType,
    InvariantViolation,
    // Proof obligations
    ProofObligation,
    ProofStatus,
    ProofTechnique,
    ProofWitness,
    PropertyFormula,
    // Properties
    PropertySpec,
    SystemState,
    // Engine
    VerificationEngine,
    VerificationEvent,
    VerificationEventType,
    VerificationSummary,
    ViolationSeverity as VerificationSeverity,
};
pub use zk_coordination::{
    // Proofs
    MembershipProof,
    VoteProof,
    // ZK-enabled group
    ZKAgentGroup,
    // Configuration
    ZKCoordinationConfig,
    // Errors
    ZKCoordinationError,
};
pub use zk_trust::{
    aggregate_proofs,
    AggregateStatement,
    // Aggregation
    AggregatedTrustProof,
    // Commitments
    KVectorCommitment,
    ProofData,
    ProofError,
    // Proof statements
    ProofStatement,
    ProverConfig,
    // Proofs
    TrustProof,
    // Prover
    TrustProver,
    // Verifier
    TrustVerifier,
    VerificationError,
    VerificationResult,
};

#[cfg(feature = "parallel")]
pub use parallel::{
    benchmark_simulation,
    // Benchmarking
    BenchmarkResult,
    // Batch operations
    KVectorBatch,
    // Configuration
    ParallelSimConfig,
    // Engine
    ParallelSimEngine,
    ParallelTickResult,
    RandomBuffer,
    SimAgent,
    SimAgentBehavior,
    TickAggregators,
};

pub use integration::{
    AttackResponse,
    // Attack Response
    AttackResponseConfig,
    AttestationResult,
    // Epistemic Lifecycle
    EpistemicLifecycleConfig,
    IntegratedAttackResponse,
    IntegratedEpistemicLifecycle,
    IntegratedPrivacyAnalytics,
    IntegratedTrustPipeline,
    // Errors
    IntegrationError,
    OutputProcessingResult,
    // Privacy Analytics
    PrivacyAnalyticsConfig,
    PrivateAnalyticsResult,
    ResponseAction,
    // Trust Pipeline
    TrustPipelineConfig,
};

use crate::epistemic::EpistemicClassificationExtended;
use crate::matl::KVector;
use serde::{Deserialize, Serialize};

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
    /// K-Vector trust profile (10 dimensions including k_coherence coherence)
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
        let successful = self
            .behavior_log
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
        if let Some(entry) = self
            .output_history
            .iter_mut()
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
        let verified: Vec<_> = self.output_history.iter().filter(|e| e.verified).collect();

        if verified.is_empty() {
            return 0.5; // Neutral default
        }

        let correct = verified
            .iter()
            .filter(|e| matches!(e.verification_outcome, Some(VerificationOutcome::Correct)))
            .count();

        let partial = verified
            .iter()
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
        use crate::epistemic::{EmpiricalLevel, HarmonicLevel, MaterialityLevel, NormativeLevel};

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
