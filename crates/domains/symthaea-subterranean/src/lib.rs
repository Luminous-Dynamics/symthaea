// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-subterranean
//!
//! Consciousness-coupled subterranean scout / boring platform.
//!
//! Tier 1 platform goal:
//! - digging / spoil / thermal load as first-class embodied constraints
//! - intermittent communication and delayed surfacing
//! - geological exploration rather than open-air locomotion
//! - explicit field-survivability authority under partial sensor, actuator, power, and link failure

pub mod actuator_isolation;
pub mod adaptation_validation;
pub mod audit_chain;
pub mod authority_validation;
pub mod capability_profile;
pub mod causal_attribution;
pub mod certification_bundle;
pub mod certification_validation;
pub mod control_context;
pub mod controller;
pub mod counterfactual_explanation;
pub mod curriculum;
pub mod degradation_forecast;
pub mod degraded_operations;
pub mod delayed_observation;
pub mod embodiment;
pub mod encoder;
pub mod evidence;
pub mod fault_tree;
pub mod faults;
pub mod fep_agent;
pub mod field_envelope;
pub mod geology;
pub mod invariant_monitor;
pub mod lifecycle_validation;
pub mod logistics;
pub mod long_horizon_validation;
pub mod maintenance;
pub mod maintenance_window;
pub mod mission;
pub mod mission_executive;
pub mod observation_quality;
pub mod occupancy;
pub mod operational_checkpoint;
pub mod operational_validation;
pub mod operator_authority;
pub mod operator_challenge;
pub mod operator_protocol;
pub mod partition_recovery;
pub mod path_memory;
pub mod peer_trust;
pub mod plan_freshness;
pub mod plugin;
pub mod policy_ablation;
pub mod recovery_journal;
pub mod recovery_planner;
pub mod recovery_validation;
pub mod reflex;
pub mod relay_mesh;
pub mod release_signoff;
pub mod requirements;
pub mod rescue;
pub mod restoration_stewardship;
pub mod runtime_budget;
pub mod safety;
pub mod safety_case;
pub mod scenario_manifest;
pub mod scenario_runner;
pub mod sensor_redundancy;
pub mod shared_map;
pub mod simulator;
pub mod stewardship_validation;
pub mod survivability_validation;
pub mod team;
pub mod team_leadership;
pub mod team_operations;
pub mod team_validation;
pub mod temporal_assurance;
pub mod temporal_clock;
pub mod temporal_event;
pub mod temporal_runtime;
pub mod traceability;
pub mod training;
pub mod tunnel_graph;
pub mod types;
pub mod update_control;
pub mod work_orders;

pub use actuator_isolation::{
    ActuatorIsolationPolicy, ActuatorIsolationReport, ActuatorIsolationSupervisor,
    NUM_MONITORED_ACTUATORS, PhysicalActuator,
};
pub use adaptation_validation::{AdaptationGateFailure, AdaptationReport, AdaptationValidator};
pub use audit_chain::{
    AuditChainError, AuditDigestProvider, AuditEvent, AuditLedger, AuditRecord,
    DeterministicAuditDigest,
};
pub use authority_validation::{
    AuthorityContract, AuthorityGateFailure, AuthorityValidationReport, AuthorityValidator,
};
pub use capability_profile::{CapabilityDisposition, CapabilityProfile};
pub use causal_attribution::{
    AttributionDisposition, AttributionRecord, CAUSAL_ATTRIBUTION_SCHEMA_VERSION,
    CausalAttributionLedger, CommandCause, ExpectedResponseSign, MAX_ATTRIBUTION_RECORDS,
    MAX_PENDING_CAUSES, ResponseObservation,
};
pub use certification_bundle::{
    BuildIdentity, BundleDigestProvider, CERTIFICATION_BUNDLE_SCHEMA_VERSION, CertificationBundle,
    CertificationBundleError, DeterministicBundleDigest,
};
pub use certification_validation::{
    CertificationContract, CertificationGateFailure, CertificationValidationReport,
    CertificationValidator,
};
pub use controller::{CheckpointError, ControllerCheckpoint};
pub use counterfactual_explanation::{
    CounterfactualActuator, CounterfactualAnswer, CounterfactualQuestion, explain_counterfactual,
};
pub use degradation_forecast::{DegradationForecast, DegradationForecaster, ForecastDisposition};
pub use degraded_operations::{
    DegradedMode, DegradedObservation, DegradedOperationsSupervisor, DegradedPolicy,
    DegradedTransition,
};
pub use delayed_observation::{
    DELAYED_OBSERVATION_SCHEMA_VERSION, DelayedObservationSupervisor, MAX_TIMED_OBSERVATIONS,
    ObservationAgeDisposition, ObservationBatchAssessment, ObservationPurpose,
    ObservationTimingAssessment, ObservationTimingIssue, TimedObservation,
};
pub use embodiment::EmbodimentBuildError;
pub use encoder::EncoderError;
pub use evidence::{
    AuthorityEvidenceSnapshot, CertificationEvidenceSnapshot, ExecutiveEvidenceSnapshot,
    GeologyEvidenceSnapshot, ReturnPathEvidenceSnapshot, SafetyEvidenceRecord,
    SafetyEvidenceSummary, SensorQualityEvidenceSnapshot, SurvivabilityEvidenceSnapshot,
    TeamEvidenceSnapshot,
};
pub use fault_tree::{
    BasicFault, FaultTreeEvaluation, FaultTreeModel, FaultTreeNode, MAX_CUT_SETS,
    MAX_EVENTS_PER_CUT_SET, TopEvent,
};
pub use field_envelope::{
    FieldEnvelopeAssessment, FieldEnvelopeMode, FieldEnvelopePolicy, FieldEnvelopeSupervisor,
};
pub use geology::{
    GeologicalLookahead, GeologyError, GeologySample, GeotechnicalProfile, MaterialClass, Stratum,
};
pub use invariant_monitor::{
    InvariantAssessment, InvariantContext, RuntimeInvariant, RuntimeInvariantMonitor,
};
pub use lifecycle_validation::{
    LifecycleAssuranceReport, LifecycleAssuranceValidator, LifecycleGateFailure,
};
pub use logistics::{
    AdmissionRefusal, LogisticsError, LogisticsLedger, LogisticsPlanner, LogisticsPolicy,
    MissionResourceEnvelope, WorkAdmission,
};
pub use long_horizon_validation::{
    LongHorizonContract, LongHorizonGateFailure, LongHorizonValidationReport, LongHorizonValidator,
};
pub use maintenance::{
    ComponentKind, MaintenanceAssessment, MaintenanceError, MaintenanceMonitor,
    MaintenanceResources, NUM_COMPONENTS,
};
pub use maintenance_window::{
    MAINTENANCE_WINDOW_SCHEMA_VERSION, MaintenanceWindowAssessment, MaintenanceWindowDisposition,
    MaintenanceWindowPlanner, MaintenanceWindowPolicy,
};
pub use mission::SubterraneanMissionIntent;
pub use mission_executive::{
    ExecutiveAbortReason, ExecutiveAssessment, ExecutiveDirective,
    MISSION_CHECKPOINT_SCHEMA_VERSION, MissionCheckpointError, MissionExecutive,
    MissionExecutiveCheckpoint, MissionExecutiveError,
};
pub use operational_checkpoint::{
    MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION, OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
    OperationalCheckpointError, SubterraneanOperationalCheckpoint,
};

pub use observation_quality::{ChannelReliabilityMonitor, ObservationQualityReport};
pub use occupancy::{
    OccupancyAssessment, ReservationPriority, ReservationRejection, TunnelDirection,
    TunnelOccupancy, TunnelReservation,
};
pub use operational_validation::{
    OperationalContract, OperationalGateFailure, OperationalValidationReport,
    OperationalValidationSuite, OperationalValidator,
};
pub use operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorConstraint, OperatorDecision,
};
pub use operator_challenge::{
    ChallengeDisposition, ChallengeEnvelope, ChallengeKind, ChallengeRecord, ChallengeRejection,
    ChallengeResponse, ChallengeRole, MAX_CHALLENGE_TEXT, MAX_OPERATOR_CHALLENGES,
    OPERATOR_CHALLENGE_SCHEMA_VERSION, OperatorChallengeLedger,
};
pub use operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorCommandRejection,
    OperatorId, OperatorRole, OperatorTrustPolicy,
};
pub use partition_recovery::{
    PartitionObservation, PartitionRecoveryAssessment, PartitionRecoveryMode,
    PartitionRecoveryPolicy, PartitionRecoverySupervisor,
};
pub use path_memory::{ReturnPathAssessment, ReturnPathMemory, ReturnPathSegment};
pub use peer_trust::{
    PeerAuthenticationAssertion, PeerAuthenticationOutcome, PeerTrustPolicy, PeerTrustRejection,
    PeerTrustSupervisor,
};
pub use plan_freshness::{
    PLAN_FRESHNESS_SCHEMA_VERSION, PlanBasis, PlanFreshnessAssessment, PlanFreshnessSupervisor,
    PlanInvalidationReason, RuntimeRevisions,
};
pub use policy_ablation::{
    PolicyAblationReport, PolicyAblationRunner, PolicyAblationSuite, PolicyVariant,
};
pub use recovery_journal::{
    DeterministicJournalDigest, JournalDigestProvider, JournalSlot, RecoveryJournal,
    RecoveryJournalError,
};
pub use recovery_planner::{
    RecoveryAction, RecoveryPlan, RecoveryPlanner, VerifiedRecoveryPlanner,
};
pub use recovery_validation::{
    RecoveryContract, RecoveryValidationReport, RecoveryValidationSuite, RecoveryValidator,
};
pub use relay_mesh::{MeshAssessment, MeshLink, MeshLinkRejection, MeshNodeId, RelayMesh};
pub use release_signoff::{
    ReleaseBlocker, ReleaseDecision, ReleaseGateInput, ReleaseGateReport, ReleaseSignoffGate,
    RequirementWaiver, SignerId, SignerRole, VerifiedApproval,
};
pub use requirements::{
    RequirementCriticality, RequirementDefinition, RequirementId, RequirementRegistry,
    RequirementRegistryError, VerificationMethod,
};
pub use rescue::{
    RescueCapability, RescueCaseId, RescueFeasibility, RescueHandoff, RescueHandoffState,
    RescueOffer, RescueRequest, RescueTransitionError, evaluate_rescue,
};
pub use restoration_stewardship::{
    MAX_RESTORATION_OBLIGATIONS, RESTORATION_STEWARDSHIP_SCHEMA_VERSION, RestorationAssessment,
    RestorationDisposition, RestorationError, RestorationLedger, RestorationObligation,
    RestorationObligationKind, RestorationState,
};
pub use runtime_budget::{
    ControlLoopBudget, ControlLoopTimingReport, RuntimeBudgetError, benchmark_control_loop,
};
pub use safety_case::{
    ClaimDisposition, EvidenceReference, SAFETY_CASE_SCHEMA_VERSION, SafetyCase,
    SafetyCaseAssessment, SafetyClaim,
};
pub use scenario_manifest::{
    MAX_SCENARIO_STEPS, MAX_STATE_OVERRIDES, SCENARIO_MANIFEST_SCHEMA_VERSION, ScenarioFingerprint,
    ScenarioManifest, ScenarioManifestError, StateOverride,
};
pub use scenario_runner::{ScenarioFailure, ScenarioRunReport, ScenarioRunner};
pub use sensor_redundancy::{
    MAX_SENSOR_SOURCES, RedundantSensorFrame, SensorFusionPolicy, SensorFusionReport,
    SensorFusionSupervisor, SensorSourceId, SensorSourceObservation,
};
pub use shared_map::{
    SharedMapRejection, SharedRouteKnowledge, SharedTunnelBin, SharedTunnelMap,
    SharedTunnelObservation,
};
pub use stewardship_validation::{StewardshipGateFailure, StewardshipReport, StewardshipValidator};
pub use survivability_validation::{
    SurvivabilityContract, SurvivabilityGateFailure, SurvivabilityValidationReport,
    SurvivabilityValidator,
};
pub use team::{
    AgentId, HeartbeatRejection, PeerCondition, TeamDirectory, TeamHeartbeat, TeamRole, TeamStatus,
};
pub use team_leadership::{
    ByzantineContainmentAssessment, ByzantineContainmentAuthority, LeadershipLeaseVote,
    TeamLeadershipPolicy, TeamLeadershipSupervisor, VoteRejection,
};
pub use team_operations::{
    DISTRIBUTED_RECOVERY_CHECKPOINT_SCHEMA_VERSION, DistributedRecoveryCheckpoint, TeamCoordinator,
    TeamDirective, TeamOperationalAssessment,
};
pub use team_validation::{
    TeamOperationalContract, TeamOperationalGateFailure, TeamOperationalValidationReport,
    TeamOperationalValidator,
};
pub use temporal_assurance::{
    MAX_TEMPORAL_REASONS, TEMPORAL_ASSURANCE_SCHEMA_VERSION, TEMPORAL_REVIEW_CLEAN_DWELL_STEPS,
    TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION, TemporalAssuranceAssessment,
    TemporalAssuranceSupervisor, TemporalAuthority, TemporalRuntimeFrame,
};
pub use temporal_clock::{
    ClockAssessment, ClockDisposition, ClockDomain, ClockIssue, ClockPolicy, ClockSample,
    ClockSourceId, MAX_CLOCK_SOURCES, TEMPORAL_CLOCK_SCHEMA_VERSION, TemporalClockSupervisor,
};
pub use temporal_event::{
    CausalEvent, CausalEventId, CausalEventKind, CausalEventLedger, EventAppendAssessment,
    EventAppendError, EventOrdering, MAX_CAUSAL_EVENTS, MAX_EVENT_DEPENDENCIES,
    TEMPORAL_EVENT_SCHEMA_VERSION, TimeInterval,
};
pub use temporal_runtime::{TemporalRuntimeInputs, temporal_runtime_revisions};
pub use traceability::{
    TraceLink, TraceabilityMatrix, TraceabilityReport, VerificationArtifactKind,
};
pub use tunnel_graph::{
    BoundedTunnelGraph, MAX_TUNNEL_EDGES, MAX_TUNNEL_NODES, RouteCostPolicy, TunnelEdge,
    TunnelGraphError, TunnelNode, TunnelNodeId, TunnelNodeKind, TunnelRoute,
};
pub use types::*;
pub use update_control::{
    ArtifactDigest, UPDATE_MANIFEST_SCHEMA_VERSION, UpdateManager, UpdateManifest,
    UpdatePreconditions, UpdateRejection, UpdateState,
};
pub use work_orders::{
    MAX_WORK_ORDERS, MAX_WORK_PREREQUISITES, SchedulerSnapshot, WorkKind, WorkOrder,
    WorkOrderError, WorkOrderId, WorkPreemptionReason, WorkPriority, WorkResourceEstimate,
    WorkScheduler, WorkStatus,
};
