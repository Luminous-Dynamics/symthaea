// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-helicopter
//!
//! Consciousness-coupled SAR helicopter control using unified HDC-LTC + FEP
//! Active Inference. Part of the Symthaea robotics expansion (Phase 1).
//!
//! ## Architecture
//!
//! ```text
//! Mission reference → deterministic guidance backbone ───────────────┐
//! Sensors (18D) → HelicopterHdcEncoder → ContinuousHV(16,384D)       │
//!                                              ↓
//!                                   HdcLtcUnifiedNetwork (evolve_closed_form)
//!                                              ↓
//!                                   HelicopterController (output projection 16,384→6)
//!                                              ↓
//!                                   HelicopterCommand [collective, cyclic×2, pedal, thrust, tail]
//!                                              ↓
//!                                   SimpleHelicopterSimulator (rotor dynamics + body physics)
//! ```
//!
//! ## SAR Mission Profile
//!
//! Designed for search-and-rescue operations governed by the Mycelix emergency
//! cluster. Helicopter is registered as an `EmergencyResource` with 24-hour
//! authority expiry. Phi contributes to an explicit safety tier; scientific
//! benchmarks keep actuator authority fixed to avoid causal confounding:
//! - Green (Φ > 0.6): full flight authority
//! - Yellow (0.3–0.6): reduced maneuver envelope
//! - Orange (0.1–0.3): hover-only, no translation
//! - Red (Φ < 0.1): state-aware emergency landing controller
//!
//! ## Features
//!
//! - `mujoco` — reserved for a future high-fidelity backend; not implemented yet

#![allow(clippy::needless_range_loop)]

pub mod actuator_dynamics;
pub mod actuator_fault_model;
pub mod adaptive_update_guard;
pub mod aero_tables;
pub mod assurance_delta;
pub mod assurance_traceability;
pub mod atmosphere;
pub mod benchmarks;
pub mod build_provenance;
pub mod calibration;
pub mod campaign_design;
pub mod capability_envelope;
pub mod certification_dossier;
pub mod claim_ledger;
pub mod command_arbitration;
pub mod command_security;
pub mod common_cause;
pub mod control_allocation;
pub mod control_reconfiguration;
pub mod controllability_margin;
pub mod controller;
pub mod data_governance;
pub mod degraded_human_factors;
pub mod deployment_manifest;
pub mod digital_twin_divergence;
pub mod drivetrain_transients;
pub mod electrical_power_distribution;
pub mod embodiment;
pub mod emergency_landing;
pub mod encoder;
pub mod endurance_campaign;
pub mod energy_guidance;
pub mod envelope_conformance;
pub mod envelope_protection;
pub mod environmental_hazards;
pub mod estimator_health;
pub mod evidence_retention;
pub mod evidence_schema_migration;
pub mod evidence_signature;
pub mod fault_containment;
pub mod fault_monitor;
pub mod fault_recovery_campaign;
pub mod fep_agent;
pub mod fleet_anomaly;
pub mod fleet_drift;
pub mod fleet_rollout;
pub mod fleet_safety_action;
pub mod flight_recorder;
pub mod guidance;
pub mod hardware_interface;
pub mod hazard_closure;
pub mod incident_reconstruction;
pub mod independent_release_authorization;
pub mod independent_verification;
pub mod landing_zone;
pub mod maintenance_prognostics;
pub mod maintenance_trend;
pub mod mass_properties;
pub mod mission_abort_corridor;
pub mod mission_assurance;
pub mod mission_supervisor;
pub mod model_validation;
pub mod navigation_consistency;
pub mod navigation_estimator;
pub mod network_partition;
pub mod observability_assurance;
pub mod operational_limits;
pub mod operational_readiness;
pub mod partition_assurance;
pub mod perturbations;
pub mod plugin;
pub mod powertrain;
pub mod qualification;
pub mod qualification_evidence_bundle;
pub mod random_streams;
pub mod rare_event_campaign;
pub mod realtime_monitor;
pub mod release_closure;
pub mod resource_budget;
pub mod return_to_service;
pub mod rollback_drill;
pub mod rollback_manifest;
pub mod rotor_dynamics;
pub mod rotor_edge_regimes;
pub mod rotor_hub;
pub mod runtime_assurance;
pub mod safe_state_reachability;
pub mod safety_case_maintenance;
pub mod safety_envelope;
pub mod safety_monitor;
pub mod sar_mission;
pub mod scenario_manifest;
pub mod secure_recovery;
pub mod secure_update;
pub mod sensor_bus;
pub mod sensor_fault_model;
pub mod service_resilience;
pub mod simulator;
pub mod software_diversity;
pub mod structural_loads;
pub mod terrain_safety;
pub mod test_oracles;
pub mod timebase;
pub mod training;
pub mod transfer;
pub mod trusted_identity_time;
pub mod types;
pub mod uncertainty_budget;
pub mod wind_model;

pub use actuator_dynamics::{ActuatorDynamics, ActuatorDynamicsConfig};
pub use actuator_fault_model::{
    ActuatorChannel, ActuatorFaultError, ActuatorFaultEvidence, ActuatorFaultMode,
    ActuatorFaultModel, ActuatorFaultOutput, ScheduledActuatorFault,
};
pub use adaptive_update_guard::{
    AdaptiveParameterBound, AdaptiveUpdateDecision, AdaptiveUpdateDisposition, AdaptiveUpdateGuard,
    AdaptiveUpdateGuardConfig, AdaptiveUpdateGuardError, AdaptiveUpdateMode,
    AdaptiveUpdateProposal, AdaptiveUpdateRejection,
};
pub use aero_tables::{
    AeroCoefficientSurface2D, AeroCoefficientTable1D, AeroTableError, CoefficientLookup,
    ExtrapolationPolicy, LookupDisposition,
};
pub use assurance_delta::{
    AssuranceDeltaAnalyzer, AssuranceDeltaArtifact, AssuranceDeltaArtifactKind,
    AssuranceDeltaChange, AssuranceDeltaChangeKind, AssuranceDeltaError, AssuranceDeltaIssue,
    AssuranceDeltaPolicy, AssuranceDeltaReport, AssuranceDeltaSnapshot, AssuranceDeltaStatus,
    AssuranceImpactRule,
};
pub use assurance_traceability::{
    AssuranceTraceabilityGraph, TraceArtifact, TraceArtifactKind, TraceLink, TraceRelation,
    TraceabilityAssessment, TraceabilityError, TraceabilityIssue, TraceabilityStatus,
};
pub use atmosphere::{
    AtmosphereError, AtmosphereSample, StandardAtmosphere, StandardAtmosphereConfig,
};
pub use benchmarks::{
    BenchmarkManifest, BenchmarkSample, NegativeControlReport, fixed_authority_negative_control,
    pearson_correlation,
};
pub use build_provenance::{
    BuildOutputArtifact, BuildProvenanceError, BuildProvenanceIssue, BuildProvenanceManifest,
    BuildProvenancePolicy, BuildProvenanceReport, BuildProvenanceStatus, BuildProvenanceVerifier,
    BuildReproducibilityIssue, BuildReproducibilityReport,
};
pub use calibration::{
    CalibratedParameter, CalibrationAssessment, CalibrationError, CalibrationReadiness,
    FlightModelCalibration, ParameterSourceClass, REQUIRED_ROTOR_PARAMETERS,
};
pub use campaign_design::{
    AxisCoverage, CampaignAxis, CampaignAxisValue, CampaignCase, CampaignCaseClass,
    CampaignCoverageReport, CampaignDesignConfig, CampaignDesignError, CampaignPlan,
    PairwiseCoverage,
};
pub use capability_envelope::{
    CapabilityDerivedEnvelope, CapabilityEnvelopeAction, CapabilityEnvelopeConfig,
    CapabilityEnvelopeDeriver, CapabilityEnvelopeError,
};
pub use certification_dossier::{
    CertificationDossierAssembler, CertificationDossierError, CertificationDossierIssue,
    CertificationDossierPolicy, CertificationDossierReport, CertificationDossierStatus,
    DossierArtifact, DossierArtifactKind, DossierArtifactStatus, DossierReviewApproval,
    DossierReviewRole,
};
pub use claim_ledger::{
    AssuranceClaim, AssuranceLevel, ClaimAssessment, ClaimAssessmentStatus, ClaimEvidenceArtifact,
    ClaimEvidenceKind, ClaimEvidenceRequirement, ClaimLedger, ClaimLedgerError,
};
pub use command_arbitration::{
    CommandArbiterConfig, CommandArbitrationError, CommandArbitrationResult, CommandProposal,
    CommandSource, HelicopterCommandArbiter, ProposalRejectionReason, RejectedCommandProposal,
};
pub use command_security::{
    AuthenticatedCommandEnvelope, CommandAuthenticityVerifier, CommandSecurityDecision,
    CommandSecurityError, CommandSecurityIssue, CommandSecurityMonitor, CommandSecurityPolicy,
    CommandSecurityStatus, MissionCommandAuthority, UnavailableCommandVerifier,
};
pub use common_cause::{
    CommonCauseAnalyzer, CommonCauseAssessment, CommonCauseDomain, CommonCauseError,
    CommonCauseEvent, CommonCauseFunctionAssessment, CommonCauseFunctionStatus,
    CriticalFunctionRequirement, RedundancyLane, RedundantAsset, RedundantAssetRole,
};
pub use control_allocation::{
    ActuatorHealth, ControlAllocationConfig, ControlAllocationError, ControlAllocationResult,
    FaultAwareControlAllocator, VirtualControlDemand,
};
pub use control_reconfiguration::{
    ControlReconfigurationConfig, ControlReconfigurationError, ControlReconfigurationManager,
    ControlReconfigurationMode, ControlReconfigurationResult, ReconfigurationReason,
};
pub use controllability_margin::{
    AxisControllabilityMargin, ControlAxis, ControllabilityAssessment, ControllabilityMarginConfig,
    ControllabilityMarginError, ControllabilityMarginEvaluator, ControllabilityState,
};
pub use controller::{HelicopterController, pd_hover_baseline};
pub use data_governance::{
    FlightDataClass, FlightDataDestination, FlightDataExportDecision, FlightDataExportRule,
    FlightDataExportStatus, FlightDataGovernance, FlightDataGovernanceError,
    FlightDataGovernancePolicy, FlightDataGovernanceReason, FlightDataRecordDescriptor,
    FlightDataRedaction, FlightDataRetentionAction, FlightDataRetentionDecision,
    FlightDataRetentionRule,
};
pub use degraded_human_factors::{
    ActiveAlert, AlertDefinition, AlertSeverity, AnnunciationFrame, DegradedModeAnnunciator,
    DegradedModeKind, DegradedModeObservation, HumanFactorsConfig, HumanFactorsError,
    HumanFactorsFlightPhase, HumanFactorsIssue, HumanFactorsStatus,
};
pub use deployment_manifest::{
    DeploymentArtifactDigests, DeploymentAuthenticityReference, DeploymentBindingReport,
    DeploymentBindingStatus, DeploymentManifest, DeploymentManifestError, DeploymentMismatch,
    DeploymentRuntimeIdentity, ModuleVersionBinding,
};
pub use digital_twin_divergence::{
    DigitalTwinDivergenceError, DigitalTwinDivergenceIssue, DigitalTwinDivergenceMonitor,
    DigitalTwinDivergencePolicy, DigitalTwinDivergenceReport, DigitalTwinDivergenceStatus,
    TwinResidualSample, TwinSignal, TwinSignalDivergence, TwinSignalPolicy,
};
pub use drivetrain_transients::{
    DrivetrainInput, DrivetrainTransientConfig, DrivetrainTransientError, DrivetrainTransientModel,
    DrivetrainTransientState,
};
pub use electrical_power_distribution::{
    ElectricalBus, ElectricalDistributionError, ElectricalDistributionInput,
    ElectricalDistributionIssue, ElectricalDistributionPolicy, ElectricalDistributionReport,
    ElectricalDistributionStatus, ElectricalLoadDemand, ElectricalLoadDisposition,
    ElectricalLoadPriority, ElectricalLoadResult, ElectricalPowerDistributor,
    ElectricalSourceState,
};
pub use emergency_landing::{
    EmergencyLandingConfig, EmergencyLandingController, HelicopterFallbackStage,
};
pub use encoder::HelicopterHdcEncoder;
pub use endurance_campaign::{
    EnduranceCampaignError, EnduranceCampaignEvaluator, EnduranceCampaignIssue,
    EnduranceCampaignPolicy, EnduranceCampaignReport, EnduranceCampaignStatus, EndurancePhase,
    EnduranceRun, EnduranceRunMetrics, EnduranceSample,
};
pub use energy_guidance::{
    EnergyAwareGuidance, EnergyGuidanceAction, EnergyGuidanceConfig, EnergyGuidanceDecision,
    EnergyGuidanceError, EnergyRouteAssessment, EnergyRouteCandidate, EnergyRouteIssue,
    EnergyRouteKind, EnergyRouteSegment, EnergyRouteStatus,
};
pub use envelope_conformance::{
    DynamicEnvelopeLimit, EnvelopeConformanceAuditor, EnvelopeConformanceError,
    EnvelopeConformanceIssue, EnvelopeConformanceReport, EnvelopeConformanceStatus,
    EnvelopeQuantity, EnvelopeTraceSample, QuantityConformanceEvidence,
};
pub use envelope_protection::{
    RotorEnvelopeIntervention, RotorEnvelopeObservation, RotorEnvelopeProtectionConfig,
    RotorEnvelopeProtectionError, RotorEnvelopeProtectionResult, RotorEnvelopeProtector,
};
pub use environmental_hazards::{
    EnvironmentalHazardConfig, EnvironmentalHazardError, EnvironmentalHazardInput,
    EnvironmentalHazardLevel, EnvironmentalHazardModel, EnvironmentalHazardState,
};
pub use estimator_health::{
    EstimatorHealthAssessment, EstimatorHealthConfig, EstimatorHealthError, EstimatorHealthManager,
    EstimatorHealthReason, EstimatorHealthState,
};
pub use evidence_retention::{
    EvidencePriority, EvidenceRecordMetadata, EvidenceRetentionBuffer,
    EvidenceRetentionDisposition, EvidenceRetentionError, EvidenceRetentionEvidence,
    EvidenceRetentionPolicy, EvidenceRetentionResult,
};
pub use evidence_schema_migration::{
    EvidenceMigrationError, EvidenceMigrationGate, EvidenceMigrationIssue, EvidenceMigrationPolicy,
    EvidenceMigrationReport, EvidenceMigrationRun, EvidenceMigrationStatus, EvidenceMigrationStep,
    MigrationValidationEvidence, MigrationValidationKind, MigrationValidationStatus,
};
pub use evidence_signature::{
    EvidenceCryptoProvider, EvidenceSignatureError, SignedFlightSegmentSeal,
    UnavailableEvidenceCrypto,
};
pub use fault_containment::{
    ComponentBinding, ContainmentZone, FaultContainmentArchitecture, FaultContainmentAssessment,
    FaultContainmentError, FlightComponent, FlightService, RequiredDependency, ServiceRequirement,
    SinglePointFailureReport,
};
pub use fault_monitor::{
    FaultDiagnosis, FaultMonitorConfig, FaultMonitorError, FaultStatus, FlightHealthObservation,
    HelicopterFaultKind, HelicopterFaultMonitor,
};
pub use fault_recovery_campaign::{
    FaultRecoveryCampaign, FaultRecoveryCampaignError, FaultRecoveryCampaignReport,
    FaultRecoveryIssue, FaultRecoveryObservation, FaultRecoveryScenario,
    FaultRecoveryScenarioReport, FaultRecoveryStatus, RecoveryFaultClass,
};
pub use fep_agent::ActiveInferenceHelicopterAgent;
pub use fleet_anomaly::{
    FleetAircraftObservation, FleetAnomalyDetector, FleetAnomalyError, FleetAnomalyIssue,
    FleetAnomalyPolicy, FleetAnomalyReport, FleetAnomalyStatus, FleetMetricAssessment,
    FleetMetricBound,
};
pub use fleet_drift::{
    AircraftConfigurationSnapshot, AircraftDriftAssessment, AircraftDriftStatus,
    FleetConfigurationBaseline, FleetDriftError, FleetDriftIssue, FleetDriftMonitor,
    FleetDriftPolicy, FleetDriftReport,
};
pub use fleet_rollout::{
    FleetAircraftRolloutEvidence, FleetAircraftRolloutStatus, FleetRolloutAction,
    FleetRolloutDecision, FleetRolloutError, FleetRolloutGate, FleetRolloutIssue,
    FleetRolloutPhase, FleetRolloutPolicy,
};
pub use fleet_safety_action::{
    FleetAircraftIdentity, FleetSafetyAction, FleetSafetyActionCoordinator, FleetSafetyActionError,
    FleetSafetyActionIssue, FleetSafetyActionKind, FleetSafetyActionPolicy,
    FleetSafetyActionReport, FleetSafetyActionStatus, FleetSafetyAircraftAssessment,
    FleetSafetyComplianceEvidence, FleetSafetyComplianceState, FleetSafetyScope,
};
pub use flight_recorder::{
    FlightEvent, FlightEventKind, FlightFrame, FlightLogManifest, FlightRecordRef, FlightRecorder,
    FlightRecorderError, FlightSegmentSeal,
};
pub use guidance::{FlightReference, GuidanceConfig, position_hold_command};
pub use hardware_interface::{
    HardwareAuthorityToken, HardwareBackendKind, HardwareBridgeError, HardwareBridgeState,
    HardwareCommandFrame, HardwareIoError, HardwareSafetyConfig, HardwareSensorFrame,
    HelicopterHardwareBridge, HelicopterHardwareIo, NullHardwareIo,
};
pub use hazard_closure::{
    HazardClosureAssessment, HazardClosureError, HazardClosureGate, HazardClosureIssue,
    HazardClosurePolicy, HazardClosureRecord, HazardClosureStatus, HazardEvidenceKind,
    HazardSeverity, HazardVerificationEvidence, HazardVerificationStatus, ResidualRiskAcceptance,
};
pub use incident_reconstruction::{
    CandidateCausalLink, IncidentReconstructionError, IncidentReconstructionIssue,
    IncidentReconstructionReport, IncidentReconstructionStatus, IncidentReconstructor,
    IncidentRecord, IncidentRecordKind, IncidentTimelineEntry,
};
pub use independent_release_authorization::{
    IndependentReleaseError, IndependentReleaseGate, IndependentReleaseIssue,
    IndependentReleasePolicy, IndependentReleaseReport, IndependentReleaseStatus, ReleaseApproval,
    ReleaseApprovalDecision, ReleaseAuthorizationEvidence, ReleaseAuthorizationRole,
    ReleaseCandidate, ReleaseEvidenceKind, ReleaseEvidenceStatus,
};
pub use independent_verification::{
    IndependentVerificationError, IndependentVerificationGate, IndependentVerificationIssue,
    IndependentVerificationPolicy, IndependentVerificationReport, IndependentVerificationStatus,
    VerificationCriticality, VerificationImplementation, VerificationResult, VerificationVector,
};
pub use landing_zone::{
    LandingZoneAssessment, LandingZoneCandidate, LandingZoneConfig, LandingZoneError,
    LandingZoneEvaluator, LandingZoneRejection,
};
pub use maintenance_prognostics::{
    ComponentLifeLimit, ComponentMaintenanceReport, FleetMaintenanceReport, MaintenanceDisposition,
    MaintenanceError, MaintenanceIssue, MaintenanceLifeTracker, MaintenanceUsageObservation,
};
pub use maintenance_trend::{
    MaintenanceTrendDisposition, MaintenanceTrendError, MaintenanceTrendIssue,
    MaintenanceTrendMonitor, MaintenanceTrendObservation, MaintenanceTrendPolicy,
    MaintenanceTrendReport,
};
pub use mass_properties::{MassElement, MassProperties, MassPropertiesError, MassPropertiesModel};
pub use mission_abort_corridor::{
    AbortCorridorAssessment, AbortCorridorCandidate, AbortCorridorConfig, AbortCorridorError,
    AbortCorridorIssue, AbortCorridorPoint, AbortCorridorStatus, AbortDestinationKind,
    AbortSelection, MissionAbortCorridorEvaluator,
};
pub use mission_assurance::{
    AssuranceStatus, ExpectedContingency, MissionAssuranceInput, MissionAssuranceKernel,
    MissionDecisionAudit, MissionInvariantViolation,
};
pub use mission_supervisor::{
    ContingencyReason, MissionDecision, MissionDirective, MissionPhase, MissionSafetySnapshot,
    MissionSupervisor, MissionSupervisorConfig, MissionSupervisorError, MissionTransition,
};
pub use model_validation::{
    FlightModelSignal, FlightModelValidationConfig, FlightModelValidationReport,
    FlightModelValidationSample, FlightModelValidator, ModelValidationError,
    SignalValidationMetrics, ValidationMetricStatus, ValidationPartition, ValidationSignalGate,
};
pub use navigation_consistency::{
    NavigationConsistencyConfig, NavigationConsistencyError, NavigationConsistencyEvidence,
    NavigationConsistencyMonitor, NavigationConsistencySample, NavigationConsistencyState,
};
pub use navigation_estimator::{
    HelicopterNavigationEstimate, HelicopterNavigationEstimator, NavigationEstimateError,
    NavigationFusionConfig, NavigationFusionStats, NavigationHealth, NavigationHealthConfig,
    NavigationIntegrityConfig, NavigationObservability, NavigationSource, NavigationSourceStatus,
};
pub use network_partition::{
    LocalAutonomyState, NetworkLinkObservation, NetworkPartitionDecision, NetworkPartitionError,
    NetworkPartitionIssue, NetworkPartitionMode, NetworkPartitionPolicy,
    NetworkPartitionSupervisor, PartitionMissionCriticality, ReconnectionEvidence,
};
pub use observability_assurance::{
    EstimatedQuantity, ObservabilityAssessment, ObservabilityAssuranceModel, ObservabilityError,
    ObservationDomain, ObservationSource, QuantityObservabilityAssessment,
    QuantityObservabilityRequirement, QuantityObservabilityStatus, SensorAvailability,
    SensorCapability,
};
pub use operational_limits::{
    OperationPhase, OperationalGateReport, OperationalGateStatus, OperationalLimitIssue,
    OperationalLimitSet, OperationalLimitSeverity, OperationalLimitsError, OperationalLimitsGate,
    OperationalObservation,
};
pub use operational_readiness::{
    OperationalReadinessError, OperationalReadinessGate, OperationalReadinessIssue,
    OperationalReadinessPolicy, OperationalReadinessReport, OperationalReadinessStatus,
    ReadinessArtifact, ReadinessArtifactKind, ReadinessArtifactStatus,
};
pub use partition_assurance::{
    InterferenceTestEvidence, InterferenceTestStatus, PartitionAssuranceError,
    PartitionAssuranceEvaluator, PartitionAssuranceIssue, PartitionAssurancePolicy,
    PartitionAssuranceReport, PartitionAssuranceStatus, PartitionBudget, PartitionChannel,
    PartitionCriticality, PartitionResourceKind, SoftwarePartition,
};
pub use perturbations::{
    HelicopterPerturbation, PerturbationEffects, PerturbationError, PerturbationSchedule,
};
pub use powertrain::{
    FuelReserveAction, FuelReserveAssessment, PowertrainConfig, PowertrainModel, PowertrainState,
};
pub use qualification::{
    QualificationCampaign, QualificationDirection, QualificationError, QualificationGate,
    QualificationGateResult, QualificationMetric, QualificationMetricValue,
    QualificationObservation, QualificationReport, QualificationScenario,
    QualificationScenarioReport, QualificationStatus,
};
pub use qualification_evidence_bundle::{
    EvidenceBundleAssessment, EvidenceBundleError, EvidenceBundleIssue, EvidenceBundleStatus,
    EvidenceOperatingContext, QualificationArtifactKind, QualificationArtifactRef,
    QualificationEvidenceBundle,
};
pub use random_streams::{
    DerivedRandomStream, RandomStreamError, RandomStreamManifest, RandomStreamPurpose,
    RandomStreamRegistry, RandomStreamSpec,
};
pub use rare_event_campaign::{
    RareEventCampaignAssessor, RareEventCampaignError, RareEventCampaignIssue,
    RareEventCampaignPolicy, RareEventCampaignReport, RareEventCampaignStatus,
    RareEventFamilyReport, RareEventOutcome, RareEventSample,
};
pub use realtime_monitor::{
    ControlCycleAssessment, ControlCycleTiming, RealtimeControlMonitor, RealtimeEvidence,
    RealtimeHealth, RealtimeMonitorConfig, RealtimeMonitorError,
};
pub use release_closure::{
    ReleaseArtifact, ReleaseArtifactKind, ReleaseArtifactStatus, ReleaseClosureError,
    ReleaseClosureGate, ReleaseClosureIssue, ReleaseClosurePolicy, ReleaseClosureReport,
    ReleaseClosureStatus,
};
pub use resource_budget::{
    ResourceBudgetConfig, ResourceBudgetError, ResourceBudgetEvidence, ResourceBudgetMonitor,
    ResourceBudgetObservation, ResourceBudgetState, ResourceBudgetViolation,
};
pub use return_to_service::{
    ComponentInstallationEvidence, MaintenanceTaskCriticality, MaintenanceTaskEvidence,
    MaintenanceTaskStatus, ReturnToServiceError, ReturnToServiceGate, ReturnToServiceIssue,
    ReturnToServicePolicy, ReturnToServiceReport, ReturnToServiceStatus,
    ReturnToServiceTestEvidence, ReturnToServiceTestKind, ReturnToServiceWorkOrder,
};
pub use rollback_drill::{
    RollbackDrillError, RollbackDrillEvaluator, RollbackDrillIssue, RollbackDrillObservation,
    RollbackDrillPolicy, RollbackDrillReport, RollbackDrillStage, RollbackDrillStatus,
    RollbackStageEvidence,
};
pub use rollback_manifest::{
    RollbackArtifact, RollbackAssessment, RollbackCatalog, RollbackError, RollbackPolicy,
    RollbackRejection, RollbackStatus,
};
pub use rotor_dynamics::{
    RotorDynamics, RotorDynamicsConfig, RotorDynamicsState, RotorFlightCondition,
    RotorFlightRegime, RotorOutput,
};
pub use rotor_edge_regimes::{
    RotorEdgeAssessment, RotorEdgeObservation, RotorEdgeRegime, RotorEdgeRegimeConfig,
    RotorEdgeRegimeError, RotorEdgeRegimeProtector,
};
pub use rotor_hub::{
    RotorHubConfig, RotorHubDynamics, RotorHubError, RotorHubOutput, RotorHubState,
};
pub use runtime_assurance::{
    RuntimeAssuranceConfig, RuntimeAssuranceDecision, RuntimeAssuranceError, RuntimeAssuranceMode,
    RuntimeAssuranceMonitor, RuntimeAssuranceObservation, RuntimeAssuranceReason,
};
pub use safe_state_reachability::{
    AbstractFlightState, ReachabilityCase, ReachabilityStatus, SafeStateReachabilityError,
    SafeStateReachabilityModel, SafeStateReachabilityReport, SafeStateTransition, SafetyCapability,
};
pub use safety_case_maintenance::{
    SafetyCaseArtifact, SafetyCaseArtifactKind, SafetyCaseArtifactStatus, SafetyCaseChange,
    SafetyCaseLink, SafetyCaseMaintainer, SafetyCaseMaintenanceError, SafetyCaseMaintenanceIssue,
    SafetyCaseMaintenancePolicy, SafetyCaseMaintenanceReport, SafetyCaseMaintenanceStatus,
    SafetyCaseRelation,
};
pub use safety_envelope::{FlightAuthorityPolicy, FlightEnvelope};
pub use safety_monitor::{
    RuntimeSafetyMonitor, RuntimeSafetySnapshot, SafetyAssessment, SafetyAssessmentStatus,
    SafetyMonitorConfig, SafetyMonitorError, SafetyProperty, SafetyViolation,
};
pub use scenario_manifest::{
    CompiledFlightScenario, FlightScenarioManifest, ScenarioExpectedOutcome, ScenarioManifestError,
    TimedPerturbation,
};
pub use secure_recovery::{
    RecoveryAction, RecoveryApproval, RecoveryApprovalRole, SecureRecoveryDecision,
    SecureRecoveryError, SecureRecoveryGate, SecureRecoveryIssue, SecureRecoveryPolicy,
    SecureRecoveryRequest, SecureRecoveryStatus,
};
pub use secure_update::{
    DualBankUpdateManager, SecureUpdateError, SecureUpdateEvidence, SecureUpdateIssue,
    SecureUpdatePackage, SecureUpdatePolicy, SecureUpdateState, TrialBootReport,
    UnavailableUpdateVerifier, UpdateAuthenticityVerifier, UpdateBank,
};
pub use sensor_bus::{
    MultiRateSensorBus, SensorBusConfig, SensorBusError, SensorBusEvidence, SensorChannelPolicy,
    SensorKind, SensorSnapshot, SensorSnapshotChannel, SensorVector, TimedSensorMeasurement,
};
pub use sensor_fault_model::{
    SensorFaultConfig, SensorFaultError, SensorFaultEvidence, SensorFaultMode, SensorFaultModel,
};
pub use service_resilience::{
    ResilientService, ServiceAvailability, ServiceCriticality, ServiceObservation,
    ServiceResilienceAssessor, ServiceResilienceError, ServiceResilienceIssue,
    ServiceResiliencePolicy, ServiceResilienceReport, ServiceResilienceStatus,
};
pub use simulator::{
    HelicopterPhysicsSimulator, LandingContact, LandingOutcome, SimpleHelicopterSimulator,
};
pub use software_diversity::{
    DiversityDimension, PairwiseDiversityAssessment, SoftwareDiversityAssessor,
    SoftwareDiversityError, SoftwareDiversityIssue, SoftwareDiversityPolicy,
    SoftwareDiversityReport, SoftwareDiversityStatus, SoftwareLaneIdentity,
};
pub use structural_loads::{
    StructuralLoadConfig, StructuralLoadError, StructuralLoadEvidence, StructuralLoadMonitor,
    StructuralLoadObservation, StructuralLoadState,
};
pub use terrain_safety::{
    AxisAlignedGeofence, FlatTerrain, HeightGrid, TerrainProvider, TerrainSafetyAssessment,
    TerrainSafetyConfig, TerrainSafetyKernel, TerrainSafetyReason,
};
pub use test_oracles::{
    OracleIssue, OracleReport, OracleSample, OracleStatus, OracleTolerance, OracleViolation,
    ResponseDeadlineOracle, ResponseEvent, SignalOracle, TestOracleError, TestOracleSuite,
};
pub use timebase::{
    ClockDisciplineConfig, ClockDisciplineEvidence, ClockLockState, ClockObservation,
    CorrectedTimestamp, SensorClockDiscipline, TimebaseError,
};
pub use training::HelicopterTrainer;
pub use trusted_identity_time::{
    DeviceIdentityEvidence, TrustedIdentityTimeError, TrustedIdentityTimeIssue,
    TrustedIdentityTimePolicy, TrustedIdentityTimeReport, TrustedIdentityTimeState,
    TrustedIdentityTimeStatus, TrustedIdentityTimeVerifier, TrustedTimeObservation,
    TrustedTimeSourceKind,
};
pub use types::*;
pub use uncertainty_budget::{
    CorrelationGroupEvidence, UncertaintyBudgetConfig, UncertaintyBudgetError,
    UncertaintyBudgetEvaluator, UncertaintyBudgetIssue, UncertaintyBudgetReport,
    UncertaintyBudgetStatus, UncertaintyContribution, UncertaintySource,
};
pub use wind_model::{WindConfig, WindForce, WindModel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helicopter_command_hover() {
        let cmd = HelicopterCommand::hover();
        assert!(cmd.thrust > 0.0, "hover thrust should be positive");
        assert!(cmd.collective > 0.0, "hover collective should be positive");
    }

    #[test]
    fn channel_names_count() {
        assert_eq!(CHANNEL_NAMES.len(), NUM_STATE_CHANNELS);
    }
}
