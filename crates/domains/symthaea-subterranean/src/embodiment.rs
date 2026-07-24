// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::actuator_isolation::{
    ActuatorIsolationReport, ActuatorIsolationSupervisor, PhysicalActuator,
};
use crate::audit_chain::{AuditDigestProvider, AuditEvent, AuditLedger};
use crate::capability_profile::{CapabilityDisposition, CapabilityProfile};
use crate::control_context::SubterraneanControlContextEncoder;
use crate::controller::{CheckpointError, ControllerCheckpoint, SubterraneanController};
use crate::degraded_operations::{
    DegradedMode, DegradedObservation, DegradedOperationsSupervisor, DegradedTransition,
};
use crate::evidence::{
    AuthorityEvidenceSnapshot, CertificationEvidenceSnapshot, ExecutiveEvidenceSnapshot,
    GeologyEvidenceSnapshot, RecoveryResourceSnapshot, ReturnPathEvidenceSnapshot,
    SafetyEvidenceLedger, SafetyEvidenceRecord, SafetyEvidenceSummary,
    SensorQualityEvidenceSnapshot, SurvivabilityEvidenceSnapshot, TeamEvidenceSnapshot,
};
use crate::fep_agent::ActiveInferenceSubterraneanAgent;
use crate::field_envelope::{FieldEnvelopeAssessment, FieldEnvelopeMode, FieldEnvelopeSupervisor};
use crate::geology::{GeologicalLookahead, GeologySample, GeotechnicalProfile};
use crate::invariant_monitor::{InvariantAssessment, InvariantContext, RuntimeInvariantMonitor};
use crate::mission::{MissionManager, SubterraneanMissionIntent};
use crate::mission_executive::{ExecutiveAssessment, MissionExecutive, MissionExecutiveError};
use crate::observation_quality::{ChannelReliabilityMonitor, ObservationQualityReport};
use crate::occupancy::{ReservationRejection, TunnelReservation};
use crate::operational_checkpoint::{
    OPERATIONAL_CHECKPOINT_SCHEMA_VERSION, OperationalCheckpointError,
    SubterraneanOperationalCheckpoint,
};
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorConstraint, OperatorDecision,
};
use crate::operator_protocol::OperatorCommandEnvelope;
use crate::partition_recovery::{
    PartitionObservation, PartitionRecoveryAssessment, PartitionRecoveryMode,
    PartitionRecoverySupervisor,
};
use crate::path_memory::ReturnPathAssessment;
use crate::recovery_planner::RecoveryAction;
use crate::relay_mesh::{MeshLink, MeshLinkRejection};
use crate::rescue::{RescueFeasibility, RescueOffer, RescueRequest, RescueTransitionError};
use crate::safety::{
    HazardAssessment, HazardSupervisor, SubterraneanHazard,
    assess_hazard_portfolio_with_operational_context, plan_command_with_portfolio_resources,
};
use crate::sensor_redundancy::{RedundantSensorFrame, SensorFusionReport, SensorFusionSupervisor};
use crate::shared_map::{SharedMapRejection, SharedTunnelObservation};
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::team::{AgentId, HeartbeatRejection, TeamHeartbeat};
use crate::team_operations::{TeamCoordinator, TeamOperationalAssessment};
use crate::tunnel_graph::{TunnelEdge, TunnelNode, TunnelNodeId, TunnelNodeKind};
use crate::types::{ConfigError, NUM_PHYSICAL_ACTUATORS, SubterraneanCommand, SubterraneanConfig};
use crate::update_control::{
    ArtifactDigest, UpdateManager, UpdateManifest, UpdatePreconditions, UpdateRejection,
    UpdateState,
};
use crate::work_orders::{WorkOrder, WorkOrderId};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Observable fallback stage selected by the hazard-specific command arbiter.
///
/// A generic Red-tier zero command is unsafe underground: it can disable
/// cooling during thermal runaway, fail to withdraw from gas or collapse risk,
/// or waste the final battery reserve. The stage records the action actually
/// taken rather than reporting one generic fallback label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubterraneanFallbackStage {
    Nominal,
    GeologicalProbe,
    ThermalArrest,
    FloodIsolation,
    GasWithdrawal,
    RoofStabilization,
    SpoilClearing,
    ControlledWithdrawal,
    NavigationRecovery,
    SensorIsolation,
    EnergyConservation,
    ReserveProtectedReturn,
    TunnelYield,
    OperatorHold,
    OperatorReturn,
    DegradedReturn,
    DegradedHold,
    RecoveryLock,
    MaintenanceLock,
    CapabilityReturn,
    CapabilityHold,
    FieldDerating,
    PartitionReturn,
    PartitionHold,
    PartitionReconcile,
    InvariantStop,
    PolicyStop,
}

impl SubterraneanFallbackStage {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::GeologicalProbe => "geological_probe",
            Self::ThermalArrest => "thermal_arrest",
            Self::FloodIsolation => "flood_isolation",
            Self::GasWithdrawal => "gas_withdrawal",
            Self::RoofStabilization => "roof_stabilization",
            Self::SpoilClearing => "spoil_clearing",
            Self::ControlledWithdrawal => "controlled_withdrawal",
            Self::NavigationRecovery => "navigation_recovery",
            Self::SensorIsolation => "sensor_isolation",
            Self::EnergyConservation => "energy_conservation",
            Self::ReserveProtectedReturn => "reserve_protected_return",
            Self::TunnelYield => "tunnel_yield",
            Self::OperatorHold => "operator_hold",
            Self::OperatorReturn => "operator_return",
            Self::DegradedReturn => "degraded_return",
            Self::DegradedHold => "degraded_hold",
            Self::RecoveryLock => "recovery_lock",
            Self::MaintenanceLock => "maintenance_lock",
            Self::CapabilityReturn => "capability_return",
            Self::CapabilityHold => "capability_hold",
            Self::FieldDerating => "field_derating",
            Self::PartitionReturn => "partition_return",
            Self::PartitionHold => "partition_hold",
            Self::PartitionReconcile => "partition_reconcile",
            Self::InvariantStop => "invariant_stop",
            Self::PolicyStop => "policy_stop",
        }
    }
}

fn safety_level_label(level: MotorSafetyLevel) -> &'static str {
    match level {
        MotorSafetyLevel::Green => "green",
        MotorSafetyLevel::Yellow => "yellow",
        MotorSafetyLevel::Orange => "orange",
        MotorSafetyLevel::Red => "red",
    }
}

fn executive_directive_label(
    directive: crate::mission_executive::ExecutiveDirective,
) -> &'static str {
    use crate::mission_executive::ExecutiveDirective;
    match directive {
        ExecutiveDirective::Idle => "idle",
        ExecutiveDirective::Execute(_) => "execute",
        ExecutiveDirective::ReturnToBase(_) => "return_to_base",
        ExecutiveDirective::HoldPosition(_) => "hold_position",
        ExecutiveDirective::SafetyPreempted => "safety_preempted",
    }
}

fn admission_refusal_label(refusal: crate::logistics::AdmissionRefusal) -> &'static str {
    use crate::logistics::AdmissionRefusal;
    match refusal {
        AdmissionRefusal::InvalidEstimate => "invalid_estimate",
        AdmissionRefusal::NoOutboundRoute => "no_outbound_route",
        AdmissionRefusal::NoReturnRoute => "no_return_route",
        AdmissionRefusal::BatteryReserve => "battery_reserve",
        AdmissionRefusal::Sealant => "sealant",
        AdmissionRefusal::Relay => "relay",
        AdmissionRefusal::RoofSupport => "roof_support",
        AdmissionRefusal::SampleCapacity => "sample_capacity",
        AdmissionRefusal::SpoilCapacity => "spoil_capacity",
        AdmissionRefusal::CoolantUnavailable => "coolant_unavailable",
    }
}

#[derive(Debug)]
pub enum EmbodimentBuildError {
    Config(ConfigError),
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for EmbodimentBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid subterranean configuration: {error}"),
            Self::Checkpoint(error) => write!(f, "invalid subterranean checkpoint: {error}"),
        }
    }
}

impl std::error::Error for EmbodimentBuildError {}

impl From<ConfigError> for EmbodimentBuildError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<CheckpointError> for EmbodimentBuildError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

pub struct SubterraneanEmbodiment {
    controller: SubterraneanController,
    simulator: SimpleSubterraneanSimulator,
    context_encoder: SubterraneanControlContextEncoder,
    mission_manager: MissionManager,
    fep: ActiveInferenceSubterraneanAgent,
    observation_monitor: ChannelReliabilityMonitor,
    observation_quality: ObservationQualityReport,
    sensor_fusion: SensorFusionSupervisor,
    pending_sensor_frame: Option<RedundantSensorFrame>,
    last_sensor_fusion: SensorFusionReport,
    actuator_isolation: ActuatorIsolationSupervisor,
    last_actuator_isolation: ActuatorIsolationReport,
    invariant_monitor: RuntimeInvariantMonitor,
    last_invariant_assessment: InvariantAssessment,
    field_envelope: FieldEnvelopeSupervisor,
    last_field_envelope: FieldEnvelopeAssessment,
    capability_profile: CapabilityProfile,
    partition_recovery: PartitionRecoverySupervisor,
    last_partition_recovery: PartitionRecoveryAssessment,
    cognitive_interval: usize,
    fep_tau_factor: f32,
    last_free_energy: f64,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    last_command: SubterraneanCommand,
    hazard_supervisor: HazardSupervisor,
    last_raw_hazard: HazardAssessment,
    last_hazard: HazardAssessment,
    fallback_stage: SubterraneanFallbackStage,
    fallback_cycles_in_stage: u32,
    evidence: SafetyEvidenceLedger,
    team_coordinator: TeamCoordinator,
    last_team: TeamOperationalAssessment,
    mission_executive: MissionExecutive,
    last_executive: ExecutiveAssessment,
    last_effective_mission: SubterraneanMissionIntent,
    operator_authority: OperatorAuthority,
    degraded_supervisor: DegradedOperationsSupervisor,
    last_degraded_transition: DegradedTransition,
    operator_link_fresh: bool,
    control_loop_healthy: bool,
    checkpoint_valid: bool,
    reboot_count_in_window: u32,
    update_manager: Option<UpdateManager>,
}

impl SubterraneanEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self::with_config(genesis, SubterraneanConfig::default())
    }

    pub fn try_with_config(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self::with_config(genesis, config))
    }

    pub fn try_with_config_and_geology(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        geology: GeotechnicalProfile,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self::with_config_and_geology(genesis, config, geology))
    }

    pub fn with_config(genesis: &GenesisSeed, config: SubterraneanConfig) -> Self {
        Self::with_config_and_geology(genesis, config, GeotechnicalProfile::default())
    }

    pub fn with_config_and_geology(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        geology: GeotechnicalProfile,
    ) -> Self {
        Self::with_config_geology_and_agent(genesis, config, geology, AgentId::new(1))
    }

    pub fn with_config_geology_and_agent(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        geology: GeotechnicalProfile,
        local_agent: AgentId,
    ) -> Self {
        Self {
            controller: SubterraneanController::new(genesis, &config),
            simulator: SimpleSubterraneanSimulator::with_geology(geology),
            context_encoder: SubterraneanControlContextEncoder::new(genesis, 32),
            mission_manager: MissionManager::default(),
            fep: ActiveInferenceSubterraneanAgent::new(),
            observation_monitor: ChannelReliabilityMonitor::default(),
            observation_quality: ObservationQualityReport::nominal(),
            sensor_fusion: SensorFusionSupervisor::default(),
            pending_sensor_frame: None,
            last_sensor_fusion: SensorFusionReport::nominal(),
            actuator_isolation: ActuatorIsolationSupervisor::default(),
            last_actuator_isolation: ActuatorIsolationReport::nominal(),
            invariant_monitor: RuntimeInvariantMonitor::default(),
            last_invariant_assessment: InvariantAssessment::nominal(0),
            field_envelope: FieldEnvelopeSupervisor::default(),
            last_field_envelope: FieldEnvelopeAssessment::nominal(),
            capability_profile: CapabilityProfile::nominal(),
            partition_recovery: PartitionRecoverySupervisor::default(),
            last_partition_recovery: PartitionRecoveryAssessment::connected(),
            cognitive_interval: config.cognitive_interval.max(1),
            fep_tau_factor: 1.0,
            last_free_energy: 0.0,
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            last_command: SubterraneanCommand::zero(),
            hazard_supervisor: HazardSupervisor::new(),
            last_raw_hazard: HazardAssessment::clear(),
            last_hazard: HazardAssessment::clear(),
            fallback_stage: SubterraneanFallbackStage::Nominal,
            fallback_cycles_in_stage: 0,
            evidence: SafetyEvidenceLedger::new(config.evidence_capacity),
            team_coordinator: TeamCoordinator::new(local_agent),
            last_team: TeamOperationalAssessment::solo(),
            mission_executive: MissionExecutive::default(),
            last_executive: ExecutiveAssessment::idle(TunnelNodeId(0), TunnelNodeId(0)),
            last_effective_mission: SubterraneanMissionIntent::Explore,
            operator_authority: OperatorAuthority::default(),
            degraded_supervisor: DegradedOperationsSupervisor::default(),
            last_degraded_transition: DegradedTransition {
                previous: DegradedMode::Normal,
                current: DegradedMode::Normal,
                changed: false,
            },
            operator_link_fresh: true,
            control_loop_healthy: true,
            checkpoint_valid: true,
            reboot_count_in_window: 0,
            update_manager: None,
        }
    }

    pub fn with_checkpoint(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        checkpoint: &ControllerCheckpoint,
    ) -> Result<Self, CheckpointError> {
        let mut embodiment = Self::with_config(genesis, config);
        embodiment.controller.load_checkpoint(checkpoint)?;
        Ok(embodiment)
    }

    pub fn with_checkpoint_and_geology(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        geology: GeotechnicalProfile,
        checkpoint: &ControllerCheckpoint,
    ) -> Result<Self, CheckpointError> {
        let mut embodiment = Self::with_config_and_geology(genesis, config, geology);
        embodiment.controller.load_checkpoint(checkpoint)?;
        Ok(embodiment)
    }

    pub fn try_with_checkpoint(
        genesis: &GenesisSeed,
        config: SubterraneanConfig,
        checkpoint: &ControllerCheckpoint,
    ) -> Result<Self, EmbodimentBuildError> {
        config.validate()?;
        let mut embodiment = Self::with_config(genesis, config);
        embodiment.controller.load_checkpoint(checkpoint)?;
        Ok(embodiment)
    }

    pub fn controller_checkpoint(&self) -> ControllerCheckpoint {
        self.controller.checkpoint()
    }

    pub fn operational_checkpoint(&self) -> SubterraneanOperationalCheckpoint {
        SubterraneanOperationalCheckpoint {
            schema_version: OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
            controller: self.controller.checkpoint(),
            mission: self.mission_executive.checkpoint(),
            operator_authority: self.operator_authority.clone(),
            degraded_supervisor: self.degraded_supervisor.clone(),
            update_manager: self.update_manager.clone(),
            sensor_fusion: self.sensor_fusion.clone(),
            actuator_isolation: self.actuator_isolation.clone(),
            field_envelope: self.field_envelope.clone(),
            partition_recovery: self.partition_recovery.clone(),
        }
    }

    pub fn load_operational_checkpoint(
        &mut self,
        checkpoint: &SubterraneanOperationalCheckpoint,
    ) -> Result<(), OperationalCheckpointError> {
        if checkpoint.schema_version
            < crate::operational_checkpoint::MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
            || checkpoint.schema_version > OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
        {
            return Err(OperationalCheckpointError::UnsupportedSchema {
                found: checkpoint.schema_version,
                expected: OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
            });
        }
        if !checkpoint.operator_authority.validate() {
            return Err(OperationalCheckpointError::InvalidOperatorState);
        }
        if !checkpoint.degraded_supervisor.validate() {
            return Err(OperationalCheckpointError::InvalidDegradedState);
        }
        if checkpoint
            .update_manager
            .as_ref()
            .is_some_and(|manager| !manager.validate())
        {
            return Err(OperationalCheckpointError::InvalidUpdateState);
        }
        if !checkpoint.sensor_fusion.validate() {
            return Err(OperationalCheckpointError::InvalidSensorFusionState);
        }
        if !checkpoint.actuator_isolation.validate() {
            return Err(OperationalCheckpointError::InvalidActuatorIsolationState);
        }
        if !checkpoint.field_envelope.validate() {
            return Err(OperationalCheckpointError::InvalidFieldEnvelopeState);
        }
        if !checkpoint.partition_recovery.validate() {
            return Err(OperationalCheckpointError::InvalidPartitionRecoveryState);
        }
        // Validate every checkpoint domain before mutating any live state.
        let mut mission_probe = self.mission_executive.clone();
        mission_probe.load_checkpoint(&checkpoint.mission)?;
        self.controller.load_checkpoint(&checkpoint.controller)?;
        self.mission_executive = mission_probe;
        self.operator_authority = checkpoint.operator_authority.clone();
        self.degraded_supervisor = checkpoint.degraded_supervisor.clone();
        self.update_manager = checkpoint.update_manager.clone();
        self.sensor_fusion = checkpoint.sensor_fusion.clone();
        self.last_sensor_fusion = self.sensor_fusion.report();
        self.actuator_isolation = checkpoint.actuator_isolation.clone();
        self.last_actuator_isolation = self.actuator_isolation.report();
        self.field_envelope = checkpoint.field_envelope.clone();
        self.last_field_envelope = self.field_envelope.last_assessment();
        self.partition_recovery = checkpoint.partition_recovery.clone();
        self.last_partition_recovery = self.partition_recovery.assessment();
        self.capability_profile = CapabilityProfile::assess(
            self.last_sensor_fusion,
            self.last_actuator_isolation,
            self.last_field_envelope,
            self.mission_executive.maintenance().assessment(),
        );
        self.checkpoint_valid = true;
        self.last_executive = self
            .mission_executive
            .last_assessment()
            .cloned()
            .unwrap_or_else(|| ExecutiveAssessment::idle(TunnelNodeId(0), TunnelNodeId(0)));
        Ok(())
    }

    /// Apply moral gate from the ethics engine. Ahimsa forces Red
    /// (VentAndRetreat), a consent violation forces Orange, caution forces
    /// a Yellow cap. Previously this crate never overrode the trait's no-op
    /// default.
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral_safety =
            if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
                Some(MotorSafetyLevel::Red)
            } else if gate.consent_violation {
                Some(MotorSafetyLevel::Orange)
            } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
                Some(MotorSafetyLevel::Yellow)
            } else {
                None
            };
    }

    fn fallback_stage_for_action(action: RecoveryAction) -> SubterraneanFallbackStage {
        match action {
            RecoveryAction::Nominal | RecoveryAction::LimitedAutonomy => {
                SubterraneanFallbackStage::Nominal
            }
            RecoveryAction::GeologicalProbe => SubterraneanFallbackStage::GeologicalProbe,
            RecoveryAction::ThermalArrest => SubterraneanFallbackStage::ThermalArrest,
            RecoveryAction::FloodIsolation => SubterraneanFallbackStage::FloodIsolation,
            RecoveryAction::GasWithdrawal => SubterraneanFallbackStage::GasWithdrawal,
            RecoveryAction::RoofStabilization => SubterraneanFallbackStage::RoofStabilization,
            RecoveryAction::SpoilClearing => SubterraneanFallbackStage::SpoilClearing,
            RecoveryAction::ControlledWithdrawal => SubterraneanFallbackStage::ControlledWithdrawal,
            RecoveryAction::NavigationRecovery => SubterraneanFallbackStage::NavigationRecovery,
            RecoveryAction::SensorIsolation => SubterraneanFallbackStage::SensorIsolation,
            RecoveryAction::EnergyConservation => SubterraneanFallbackStage::EnergyConservation,
            RecoveryAction::ReserveProtectedReturn => {
                SubterraneanFallbackStage::ReserveProtectedReturn
            }
            RecoveryAction::TunnelYield => SubterraneanFallbackStage::TunnelYield,
            RecoveryAction::PolicyStop => SubterraneanFallbackStage::PolicyStop,
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        // Fuse declared redundant sensor paths before any safety assessment.
        // Multi-source frames require critical-channel quorum; the fallback
        // state is retained only so fail-closed planning has bounded inputs.
        let fallback_state = self.simulator.state().clone();
        let (fused_state, fusion_report) = match self.pending_sensor_frame.take() {
            Some(frame) => self.sensor_fusion.fuse(&frame, &fallback_state),
            None => self.sensor_fusion.fuse_local_state(&fallback_state),
        };
        *self.simulator.state_mut() = fused_state;
        self.last_sensor_fusion = fusion_report;

        // Physical hazards are assessed from the pre-actuation state. They are
        // an independent safety authority alongside phi, manual overrides, and
        // the moral gate.
        let integrity_before = self.simulator.state().integrity_report();
        if !integrity_before.is_valid() {
            self.observation_quality = self
                .observation_monitor
                .penalize_integrity_fault(integrity_before);
        }
        let local_return_path_before = self.simulator.return_path_assessment();
        let geology_before = self.simulator.geology_lookahead(6.0);
        self.last_degraded_transition = self.degraded_supervisor.update(DegradedObservation {
            operator_link_fresh: self.operator_link_fresh,
            control_loop_healthy: self.control_loop_healthy,
            checkpoint_valid: self.checkpoint_valid,
            reboot_count_in_window: self.reboot_count_in_window,
            battery_ratio: self.simulator.state().battery_ratio(),
            return_feasible: local_return_path_before.feasible,
            at_surface_or_service_bay: self.simulator.state().depth_m() <= 0.1,
        });
        let maintenance_before = self.mission_executive.maintenance().assessment();
        self.last_field_envelope = self.field_envelope.assess(
            self.simulator.state(),
            self.mission_executive.logistics().coolant_health,
            maintenance_before,
        );
        self.capability_profile = CapabilityProfile::assess(
            self.last_sensor_fusion,
            self.last_actuator_isolation,
            self.last_field_envelope,
            maintenance_before,
        );
        let operator_constraint = self.operator_authority.constraint();
        let degraded_constraint = match self.degraded_supervisor.mode() {
            DegradedMode::Normal | DegradedMode::OperatorLinkLost => OperatorConstraint::None,
            DegradedMode::AutonomousReturn => OperatorConstraint::ReturnHome,
            DegradedMode::SafeHold => OperatorConstraint::HoldPosition,
            DegradedMode::RecoveryRequired => OperatorConstraint::MaintenanceLock,
        };
        let runtime_constraint = operator_constraint.more_restrictive(degraded_constraint);
        let requested_mission = self
            .degraded_supervisor
            .mode()
            .mission_override()
            .or_else(|| operator_constraint.mission_override())
            .or_else(|| self.last_partition_recovery.mode.mission_override())
            .or_else(|| self.capability_profile.disposition.mission_override())
            .unwrap_or_else(|| self.mission_manager.requested());
        self.last_team = self.team_coordinator.assess(
            self.total_steps as u64,
            self.simulator.state().depth_m(),
            requested_mission.tunnel_direction(),
            requested_mission.reservation_priority(),
            5.0,
            1.0,
        );
        self.last_partition_recovery = self.partition_recovery.update(PartitionObservation {
            surface_reachable: self.last_team.status.known_peers == 0
                || self.last_team.surface_mesh.reachable,
            fresh_peers: self.last_team.status.fresh_peers,
            battery_ratio: self.simulator.state().battery_ratio(),
            return_feasible: local_return_path_before.feasible,
            local_map_revision: self.team_coordinator.local_map_revision(),
            highest_peer_map_revision: self.team_coordinator.highest_peer_revision(),
        });
        if !self.last_partition_recovery.team_state_authoritative {
            self.last_team.directive = crate::team_operations::TeamDirective::None;
        }
        let return_path_before = self
            .last_team
            .shared_route
            .conservative_return_fusion(local_return_path_before);
        let mut hazard_portfolio = assess_hazard_portfolio_with_operational_context(
            self.simulator.state(),
            return_path_before,
            geology_before,
            self.observation_quality,
        );
        if self.last_sensor_fusion.requires_fail_closed() {
            hazard_portfolio.set_max(SubterraneanHazard::SensorFault, 1.0);
        }
        if self.last_team.occupancy.conflict() {
            hazard_portfolio.set_max(
                SubterraneanHazard::TunnelConflict,
                self.last_team.occupancy.conflict_severity,
            );
        }
        self.last_hazard = self.hazard_supervisor.update_portfolio(hazard_portfolio);
        self.last_raw_hazard = self.hazard_supervisor.raw();
        if self.last_raw_hazard.primary == SubterraneanHazard::SensorFault {
            self.simulator.state_mut().sanitize_fail_closed();
        }

        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        if let Some(moral_level) = self.moral_safety {
            self.current_safety = self.current_safety.max(moral_level);
        }
        if let Some(operator_floor) = runtime_constraint.safety_floor() {
            self.current_safety = self.current_safety.max(operator_floor);
        }
        if let Some(degraded_floor) = self.degraded_supervisor.mode().safety_floor() {
            self.current_safety = self.current_safety.max(degraded_floor);
        }
        self.current_safety = self.current_safety.max(self.last_hazard.safety_level);

        if self.total_steps % self.cognitive_interval == 0 {
            let fep = self.fep.tick_with_precision(
                self.simulator.state(),
                self.observation_quality.aggregate_precision,
            );
            self.fep_tau_factor = fep.tau_factor;
            self.last_free_energy = fep.free_energy;
        }

        self.last_executive = self.mission_executive.assess(
            self.total_steps as u64,
            self.simulator.state(),
            self.last_hazard,
            self.last_team.directive,
            self.simulator.recovery_resources(),
        );
        let base_effective_mission = self.mission_manager.update_with_team(
            self.simulator.state(),
            self.last_hazard,
            self.last_team.directive,
        );
        let effective_mission = if self.last_hazard.primary == SubterraneanHazard::None
            && self.last_team.directive == crate::team_operations::TeamDirective::None
        {
            self.degraded_supervisor
                .mode()
                .mission_override()
                .or_else(|| operator_constraint.mission_override())
                .or_else(|| self.last_partition_recovery.mode.mission_override())
                .or_else(|| self.capability_profile.disposition.mission_override())
                .or_else(|| self.last_executive.directive.mission_override())
                .or(self.last_executive.work_mission)
                .unwrap_or(base_effective_mission)
        } else {
            base_effective_mission
        };
        self.last_effective_mission = effective_mission;
        let gain = self.current_safety.motor_gain();
        let control_context = self.context_encoder.encode(
            self.simulator.state(),
            Some(thought_hv),
            effective_mission,
        );
        let controller_dt = dt * self.fep_tau_factor;
        let mut cmd = self.controller.forward(&control_context, controller_dt);
        if gain < 1.0 && !matches!(self.current_safety, MotorSafetyLevel::Red) {
            for value in &mut cmd.torques {
                *value *= gain;
            }
        }
        cmd = runtime_constraint.constrain_nominal(cmd, self.simulator.state());
        if self.last_hazard.primary == SubterraneanHazard::None {
            cmd = self.last_partition_recovery.constrain_nominal(cmd);
        }

        let recovery_plan = plan_command_with_portfolio_resources(
            cmd,
            self.simulator.state(),
            self.last_hazard,
            self.hazard_supervisor.raw_portfolio(),
            self.current_safety,
            self.simulator.recovery_resources(),
        );
        cmd = recovery_plan.command;
        cmd = self.last_field_envelope.constrain(cmd);
        // Mechanical truth is applied after recovery planning: a failed pump or
        // track cannot receive authority merely because the desired fallback
        // would have used it. The resulting loss remains visible to the mission
        // executive on the next cycle.
        cmd = self.mission_executive.maintenance().derate_command(cmd);
        cmd = self.actuator_isolation.constrain(cmd);
        let (invariant_command, invariant_assessment) = self.invariant_monitor.enforce(
            cmd,
            InvariantContext {
                state: self.simulator.state(),
                safety_level: self.current_safety,
                primary_hazard: self.last_hazard.primary,
                tunnel_conflict: self.last_team.occupancy.conflict(),
                return_feasible: return_path_before.feasible,
                capability_disposition: self.capability_profile.disposition,
                actuator_isolation: self.last_actuator_isolation,
            },
        );
        cmd = invariant_command;
        self.last_invariant_assessment = invariant_assessment;
        if !self.last_invariant_assessment.passed() {
            self.current_safety = MotorSafetyLevel::Red;
        }
        self.fallback_stage = Self::fallback_stage_for_action(recovery_plan.action);
        if self.last_hazard.primary == SubterraneanHazard::None {
            self.fallback_stage = match self.degraded_supervisor.mode() {
                DegradedMode::AutonomousReturn => SubterraneanFallbackStage::DegradedReturn,
                DegradedMode::SafeHold => SubterraneanFallbackStage::DegradedHold,
                DegradedMode::RecoveryRequired => SubterraneanFallbackStage::RecoveryLock,
                DegradedMode::Normal | DegradedMode::OperatorLinkLost => {
                    match operator_constraint {
                        OperatorConstraint::EmergencyStop | OperatorConstraint::HoldPosition => {
                            SubterraneanFallbackStage::OperatorHold
                        }
                        OperatorConstraint::ReturnHome => SubterraneanFallbackStage::OperatorReturn,
                        OperatorConstraint::MaintenanceLock => {
                            SubterraneanFallbackStage::MaintenanceLock
                        }
                        OperatorConstraint::None | OperatorConstraint::Mission(_) => {
                            self.fallback_stage
                        }
                    }
                }
            };
        }

        if self.last_hazard.primary == SubterraneanHazard::None
            && self.fallback_stage == SubterraneanFallbackStage::Nominal
        {
            self.fallback_stage = match self.last_partition_recovery.mode {
                PartitionRecoveryMode::ReturnToMesh => SubterraneanFallbackStage::PartitionReturn,
                PartitionRecoveryMode::HoldAndBeacon => SubterraneanFallbackStage::PartitionHold,
                PartitionRecoveryMode::Reconciling => SubterraneanFallbackStage::PartitionReconcile,
                PartitionRecoveryMode::Connected
                | PartitionRecoveryMode::Grace
                | PartitionRecoveryMode::LocalAutonomy => {
                    match self.capability_profile.disposition {
                        CapabilityDisposition::ReturnOnly => {
                            SubterraneanFallbackStage::CapabilityReturn
                        }
                        CapabilityDisposition::HoldForRecovery => {
                            SubterraneanFallbackStage::CapabilityHold
                        }
                        CapabilityDisposition::ReducedWork
                            if self.last_field_envelope.mode != FieldEnvelopeMode::Nominal =>
                        {
                            SubterraneanFallbackStage::FieldDerating
                        }
                        CapabilityDisposition::FullMission | CapabilityDisposition::ReducedWork => {
                            SubterraneanFallbackStage::Nominal
                        }
                    }
                }
            };
        }

        if !self.last_invariant_assessment.passed() {
            self.fallback_stage = SubterraneanFallbackStage::InvariantStop;
        }

        if !matches!(self.fallback_stage, SubterraneanFallbackStage::Nominal) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
        } else {
            self.fallback_cycles_in_stage = 0;
        }

        self.last_command = cmd;
        self.last_control_effort = cmd.control_effort();

        // Produce a command-conditioned one-step prediction before applying
        // the command to the embodied plant. This is a genuine model residual,
        // not temporal novelty between two observations. In the deterministic
        // reference simulator it should be near zero; real sensor/plant
        // divergence will make it rise.
        let plant_before = self.simulator.state().clone();
        let mut prediction_model = self.simulator.clone();
        prediction_model.step(&cmd, dt as f64);
        let predicted_perception = self
            .context_encoder
            .encode_perception(prediction_model.state());

        self.simulator.step(&cmd, dt as f64);
        self.last_actuator_isolation =
            self.actuator_isolation
                .observe(&cmd, &plant_before, self.simulator.state());
        self.capability_profile = CapabilityProfile::assess(
            self.last_sensor_fusion,
            self.last_actuator_isolation,
            self.last_field_envelope,
            self.mission_executive.maintenance().assessment(),
        );
        let safe_work_progress = self.last_hazard.primary == SubterraneanHazard::None
            && self.current_safety == MotorSafetyLevel::Green
            && self.fallback_stage == SubterraneanFallbackStage::Nominal
            && self.capability_profile.mission_work_allowed
            && self.last_partition_recovery.motion_permitted
            && self.last_partition_recovery.team_state_authoritative;
        self.mission_executive.observe_post_step(
            &cmd,
            self.simulator.state(),
            dt as f64,
            safe_work_progress,
        );
        self.observation_quality = self
            .observation_monitor
            .update(prediction_model.state(), self.simulator.state());
        let perception = self
            .context_encoder
            .encode_perception(self.simulator.state());
        let pe = (1.0 - perception.similarity(&predicted_perception)).clamp(0.0, 1.0);
        self.last_prediction_error = pe;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        let observation_confidence = (grounding_from_prediction_error(pe)
            * self.observation_quality.aggregate_precision as f32)
            .clamp(0.0, 1.0);
        let resources = self.simulator.recovery_resources();
        let return_path = self
            .last_team
            .shared_route
            .conservative_return_fusion(self.simulator.return_path_assessment());
        let geology_sample = self.simulator.geology_sample();
        let geology_lookahead = self.simulator.geology_lookahead(6.0);
        self.evidence.push(SafetyEvidenceRecord {
            step: self.total_steps as u64,
            state_channels: self.simulator.state().channels,
            command: self.last_command,
            raw_hazard: self.last_raw_hazard.primary.label().to_string(),
            latched_hazard: self.last_hazard.primary.label().to_string(),
            raw_hazard_severity: self.last_raw_hazard.severity,
            latched_hazard_severity: self.last_hazard.severity,
            safety_level: safety_level_label(self.current_safety).to_string(),
            requested_mission: self.mission_manager.requested().label().to_string(),
            effective_mission: self.last_effective_mission.label().to_string(),
            fallback_stage: self.fallback_stage.label().to_string(),
            control_effort: self.last_control_effort,
            free_energy: self.last_free_energy,
            prediction_error: pe,
            observation_confidence,
            recovery_resource_limited: recovery_plan.resource_limited,
            addressed_hazards: recovery_plan.addressed_hazards.labels(),
            return_path: ReturnPathEvidenceSnapshot {
                distance_home_m: return_path.distance_home_m,
                path_confidence: return_path.path_confidence,
                obstruction_risk: return_path.obstruction_risk,
                estimated_battery_required: return_path.estimated_battery_required,
                battery_margin: return_path.battery_margin,
                feasible: return_path.feasible,
            },
            geology: GeologyEvidenceSnapshot {
                material: geology_sample.material.label().to_string(),
                lookahead_risk: geology_lookahead.risk_score,
                minimum_survey_confidence: geology_lookahead.minimum_survey_confidence,
                transition_count: geology_lookahead.transition_count,
                probe_required: geology_lookahead.probe_required,
            },
            sensor_quality: SensorQualityEvidenceSnapshot {
                aggregate_precision: self.observation_quality.aggregate_precision,
                minimum_reliability: self.observation_quality.minimum_reliability,
                maximum_residual: self.observation_quality.maximum_residual,
                degraded_channels: self.observation_quality.degraded_channels,
                critical_degraded_channels: self.observation_quality.critical_degraded_channels,
            },
            team: TeamEvidenceSnapshot {
                known_peers: self.last_team.status.known_peers,
                fresh_peers: self.last_team.status.fresh_peers,
                stale_peers: self.last_team.status.stale_peers,
                distressed_peers: self.last_team.status.distressed_peers,
                directive: self.last_team.directive.label().to_string(),
                conflicting_agent: self
                    .last_team
                    .occupancy
                    .conflicting_agent
                    .map(|agent| agent.0),
                conflict_severity: self.last_team.occupancy.conflict_severity,
                must_yield: self.last_team.occupancy.must_yield,
                surface_reachable: self.last_team.surface_mesh.reachable,
                mesh_bottleneck_quality: self.last_team.surface_mesh.bottleneck_quality,
                mesh_hops: self.last_team.surface_mesh.hops,
                shared_known_bins: self.last_team.shared_route.known_bins,
                shared_contributing_peers: self.last_team.shared_route.contributing_peers,
                shared_obstruction_risk: self.last_team.shared_route.maximum_obstruction_risk,
                rescue_state: self.last_team.rescue_state.label().to_string(),
            },
            executive: {
                let logistics = self.mission_executive.logistics();
                let admission = self.last_executive.admission;
                let outbound = self.last_executive.outbound_route.as_ref();
                let return_route = self.last_executive.return_route.as_ref();
                ExecutiveEvidenceSnapshot {
                    directive: executive_directive_label(self.last_executive.directive).to_string(),
                    active_work_order: self.last_executive.scheduler.active.map(|id| id.0),
                    queued_work_orders: self.last_executive.scheduler.queued,
                    completed_work_orders: self.last_executive.scheduler.completed,
                    failed_work_orders: self.last_executive.scheduler.failed,
                    work_admitted: admission.is_some_and(|value| value.admitted),
                    admission_refusal: admission
                        .and_then(|value| value.refusal)
                        .map(admission_refusal_label)
                        .map(str::to_string),
                    outbound_distance_m: outbound.map_or(0.0, |route| route.distance_m),
                    return_distance_m: return_route.map_or(0.0, |route| route.distance_m),
                    route_maximum_risk: outbound
                        .map_or(0.0, |route| route.maximum_risk)
                        .max(return_route.map_or(0.0, |route| route.maximum_risk)),
                    route_minimum_confidence: outbound
                        .map_or(1.0, |route| route.minimum_confidence)
                        .min(return_route.map_or(1.0, |route| route.minimum_confidence)),
                    battery_required: admission
                        .map_or(0.0, |value| value.envelope.battery_required),
                    battery_after_return: admission
                        .map_or(self.simulator.state().battery_ratio(), |value| {
                            value.envelope.battery_after_return
                        }),
                    minimum_component_health: self.last_executive.maintenance.minimum_health,
                    critical_component: self
                        .last_executive
                        .maintenance
                        .critical_component
                        .map(|component| component.label().to_string()),
                    maintenance_due: self.last_executive.maintenance.maintenance_due,
                    mission_abort_required: self.last_executive.maintenance.mission_abort_required,
                    sample_fill: logistics.sample_fill,
                    spoil_fill: logistics.spoil_fill,
                    coolant_health: logistics.coolant_health,
                }
            },
            survivability: SurvivabilityEvidenceSnapshot {
                declared_sensor_sources: self.last_sensor_fusion.declared_sources,
                accepted_sensor_sources: self.last_sensor_fusion.accepted_sources,
                critical_channels_without_quorum: self
                    .last_sensor_fusion
                    .critical_channels_without_quorum,
                maximum_sensor_disagreement: self
                    .last_sensor_fusion
                    .maximum_normalized_disagreement,
                minimum_source_reliability: self.last_sensor_fusion.minimum_source_reliability,
                isolated_actuators: self.last_actuator_isolation.isolated_count,
                total_actuator_isolations: self.actuator_isolation.total_isolations(),
                mobility_degraded: self.last_actuator_isolation.mobility_degraded,
                cooling_degraded: self.last_actuator_isolation.cooling_degraded,
                recovery_degraded: self.last_actuator_isolation.recovery_degraded,
                envelope_mode: self.last_field_envelope.mode.label().to_string(),
                power_margin: self.last_field_envelope.power_margin,
                thermal_margin: self.last_field_envelope.thermal_margin,
                capability_disposition: self.capability_profile.disposition.label().to_string(),
                mission_work_allowed: self.capability_profile.mission_work_allowed,
                partition_mode: self.last_partition_recovery.mode.label().to_string(),
                partition_steps: self.last_partition_recovery.partition_steps,
                reconciliation_steps: self.last_partition_recovery.reconciliation_steps,
                map_revision_gap: self.last_partition_recovery.map_revision_gap,
                team_state_authoritative: self.last_partition_recovery.team_state_authoritative,
            },
            authority: AuthorityEvidenceSnapshot {
                operator_constraint: self.operator_authority.constraint().label().to_string(),
                operator_accepted_commands: self.operator_authority.accepted_commands(),
                operator_rejected_commands: self.operator_authority.rejected_commands(),
                operator_last_proposal: self.operator_authority.last_applied_proposal(),
                degraded_mode: self.degraded_supervisor.mode().label().to_string(),
                degraded_transitions: self.degraded_supervisor.transitions(),
                operator_link_loss_steps: self.degraded_supervisor.operator_link_loss_steps(),
                update_state: self
                    .update_manager
                    .as_ref()
                    .map(|manager| manager.state().label().to_string()),
                successful_update_activations: self
                    .update_manager
                    .as_ref()
                    .map_or(0, UpdateManager::successful_activations),
                update_rollbacks: self
                    .update_manager
                    .as_ref()
                    .map_or(0, UpdateManager::rollbacks),
            },
            certification: CertificationEvidenceSnapshot {
                invariant_violations: self
                    .last_invariant_assessment
                    .violations
                    .iter()
                    .map(|violation| violation.code().to_string())
                    .collect(),
                invariant_command_modified: self.last_invariant_assessment.command_modified,
                total_invariant_breaches: self.last_invariant_assessment.total_breaches,
                consecutive_invariant_breach_frames: self
                    .last_invariant_assessment
                    .consecutive_breach_frames,
            },
            recovery_resources: RecoveryResourceSnapshot {
                sealant_ratio: resources.sealant_ratio,
                relay_units: resources.relay_units,
                roof_support_units: resources.roof_support_units,
                dewatering_health: resources.dewatering_health,
            },
        });
        EmbodimentResult {
            num_actuators: NUM_PHYSICAL_ACTUATORS,
            control_effort: self.last_control_effort,
            success: self.simulator.state().is_finite(),
            prediction_error: pe,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence,
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self
            .context_encoder
            .encode_perception(self.simulator.state());
        self.last_perception = Some(p.clone());
        p
    }
    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.context_encoder.reset();
        self.mission_manager.reset();
        self.fep.reset();
        self.observation_monitor.reset();
        self.observation_quality = ObservationQualityReport::nominal();
        self.sensor_fusion.reset_runtime();
        self.pending_sensor_frame = None;
        self.last_sensor_fusion = SensorFusionReport::nominal();
        self.actuator_isolation = ActuatorIsolationSupervisor::default();
        self.last_actuator_isolation = ActuatorIsolationReport::nominal();
        self.invariant_monitor.reset_runtime();
        self.last_invariant_assessment =
            InvariantAssessment::nominal(self.invariant_monitor.total_breaches());
        self.field_envelope.reset_runtime();
        self.last_field_envelope = FieldEnvelopeAssessment::nominal();
        self.capability_profile = CapabilityProfile::nominal();
        self.partition_recovery.reset_runtime();
        self.last_partition_recovery = PartitionRecoveryAssessment::connected();
        self.fep_tau_factor = 1.0;
        self.last_free_energy = 0.0;
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.last_command = SubterraneanCommand::zero();
        self.hazard_supervisor.reset();
        self.last_raw_hazard = HazardAssessment::clear();
        self.last_hazard = HazardAssessment::clear();
        self.fallback_stage = SubterraneanFallbackStage::Nominal;
        self.fallback_cycles_in_stage = 0;
        self.evidence.clear();
        self.team_coordinator.reset();
        self.last_team = TeamOperationalAssessment::solo();
        self.mission_executive.reset_runtime();
        self.last_executive = ExecutiveAssessment::idle(TunnelNodeId(0), TunnelNodeId(0));
        self.last_effective_mission = self.mission_manager.requested();
        self.operator_authority.reset_runtime();
        self.degraded_supervisor.reset_runtime();
        self.last_degraded_transition = DegradedTransition {
            previous: DegradedMode::Normal,
            current: DegradedMode::Normal,
            changed: false,
        };
        self.operator_link_fresh = true;
        self.control_loop_healthy = true;
        self.checkpoint_valid = true;
        self.reboot_count_in_window = 0;
    }
    pub fn fallback_stage(&self) -> SubterraneanFallbackStage {
        self.fallback_stage
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    pub fn last_command(&self) -> SubterraneanCommand {
        self.last_command
    }
    pub fn last_hazard(&self) -> HazardAssessment {
        self.last_hazard
    }
    pub fn last_raw_hazard(&self) -> HazardAssessment {
        self.last_raw_hazard
    }
    pub fn last_free_energy(&self) -> f64 {
        self.last_free_energy
    }
    pub fn fep_tau_factor(&self) -> f32 {
        self.fep_tau_factor
    }
    pub fn observation_quality(&self) -> ObservationQualityReport {
        self.observation_quality
    }
    pub fn ingest_redundant_sensor_frame(&mut self, frame: RedundantSensorFrame) {
        self.pending_sensor_frame = Some(frame);
    }
    pub fn sensor_fusion_report(&self) -> SensorFusionReport {
        self.last_sensor_fusion
    }
    pub fn actuator_isolation_report(&self) -> ActuatorIsolationReport {
        self.last_actuator_isolation
    }
    pub fn invariant_assessment(&self) -> &InvariantAssessment {
        &self.last_invariant_assessment
    }
    pub fn field_envelope_assessment(&self) -> FieldEnvelopeAssessment {
        self.last_field_envelope
    }
    pub fn capability_profile(&self) -> CapabilityProfile {
        self.capability_profile
    }
    pub fn service_isolated_actuator(&mut self, actuator: PhysicalActuator) {
        self.actuator_isolation.service(actuator);
        self.last_actuator_isolation = self.actuator_isolation.report();
    }
    pub fn partition_recovery_assessment(&self) -> PartitionRecoveryAssessment {
        self.last_partition_recovery
    }
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }
    pub fn ingest_operator_command(
        &mut self,
        envelope: OperatorCommandEnvelope,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        let physical_hazard_clear = self.last_hazard.primary == SubterraneanHazard::None
            && self.simulator.state().integrity_report().is_valid();
        let decision = self.operator_authority.ingest(
            envelope,
            self.total_steps as u64,
            physical_hazard_clear,
        );
        if decision.is_ok() {
            self.operator_link_fresh = true;
        }
        decision
    }
    pub fn ingest_operator_command_with_audit(
        &mut self,
        provider: &impl AuditDigestProvider,
        ledger: &mut AuditLedger,
        envelope: OperatorCommandEnvelope,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        let result = self.ingest_operator_command(envelope);
        ledger.append(
            provider,
            AuditEvent::OperatorCommand {
                operator_id: envelope.operator.0,
                proposal_id: envelope.proposal_id,
                command_code: envelope.command.code(),
                accepted: result.is_ok(),
            },
        );
        if result.is_ok() {
            ledger.append(
                provider,
                AuditEvent::OperatorConstraint {
                    constraint_code: self.operator_authority.constraint().code(),
                },
            );
        }
        result
    }
    pub fn operator_constraint(&self) -> OperatorConstraint {
        self.operator_authority.constraint()
    }
    pub fn operator_authority(&self) -> &OperatorAuthority {
        &self.operator_authority
    }
    pub fn degraded_mode(&self) -> DegradedMode {
        self.degraded_supervisor.mode()
    }
    pub fn last_degraded_transition(&self) -> DegradedTransition {
        self.last_degraded_transition
    }
    pub fn set_runtime_health(
        &mut self,
        operator_link_fresh: bool,
        control_loop_healthy: bool,
        checkpoint_valid: bool,
        reboot_count_in_window: u32,
    ) {
        self.operator_link_fresh = operator_link_fresh;
        self.control_loop_healthy = control_loop_healthy;
        self.checkpoint_valid = checkpoint_valid;
        self.reboot_count_in_window = reboot_count_in_window;
    }
    pub fn initialize_update_control(
        &mut self,
        current_digest: ArtifactDigest,
        current_epoch: u64,
    ) -> Result<(), UpdateRejection> {
        self.update_manager = Some(UpdateManager::new(current_digest, current_epoch)?);
        Ok(())
    }
    pub fn update_state(&self) -> Option<UpdateState> {
        self.update_manager.as_ref().map(UpdateManager::state)
    }
    pub fn update_preconditions(&self) -> UpdatePreconditions {
        let at_service_bay = self
            .mission_executive
            .graph()
            .node(self.last_executive.current_node)
            .is_some_and(|node| node.kind == TunnelNodeKind::ServiceBay);
        UpdatePreconditions {
            at_surface_or_service_bay: self.simulator.state().depth_m() <= 0.1 || at_service_bay,
            active_work: self.active_work_order().is_some(),
            physical_hazard_clear: self.last_hazard.primary == SubterraneanHazard::None
                && self.simulator.state().integrity_report().is_valid(),
            battery_ratio: self.simulator.state().battery_ratio(),
            operator_constraint: self.operator_authority.constraint(),
        }
    }
    pub fn stage_update(&mut self, manifest: UpdateManifest) -> Result<(), UpdateRejection> {
        let preconditions = self.update_preconditions();
        self.update_manager
            .as_mut()
            .ok_or(UpdateRejection::InvalidTransition)?
            .stage(
                manifest,
                self.total_steps as u64,
                OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
                preconditions,
            )
    }
    pub fn activate_staged_update(
        &mut self,
        health_window_steps: u64,
    ) -> Result<ArtifactDigest, UpdateRejection> {
        let preconditions = self.update_preconditions();
        self.update_manager
            .as_mut()
            .ok_or(UpdateRejection::InvalidTransition)?
            .activate(self.total_steps as u64, health_window_steps, preconditions)
    }
    pub fn observe_update_health(&mut self, healthy: bool) -> Result<UpdateState, UpdateRejection> {
        self.update_manager
            .as_mut()
            .ok_or(UpdateRejection::InvalidTransition)?
            .observe_health(healthy, self.total_steps as u64)
    }
    pub fn rollback_update(&mut self) -> Result<ArtifactDigest, UpdateRejection> {
        self.update_manager
            .as_mut()
            .ok_or(UpdateRejection::InvalidTransition)?
            .rollback()
    }
    pub fn authorize_degraded_recovery_clear(&mut self, externally_authorized: bool) -> bool {
        self.degraded_supervisor.authorize_recovery_clear(
            DegradedObservation {
                operator_link_fresh: self.operator_link_fresh,
                control_loop_healthy: self.control_loop_healthy,
                checkpoint_valid: self.checkpoint_valid,
                reboot_count_in_window: self.reboot_count_in_window,
                battery_ratio: self.simulator.state().battery_ratio(),
                return_feasible: self.simulator.return_path_assessment().feasible,
                at_surface_or_service_bay: self.simulator.state().depth_m() <= 0.1,
            },
            externally_authorized,
        )
    }
    pub fn set_mission_intent(&mut self, intent: SubterraneanMissionIntent) {
        self.mission_manager.set_requested(intent);
    }
    pub fn requested_mission_intent(&self) -> SubterraneanMissionIntent {
        self.mission_manager.requested()
    }
    pub fn effective_mission_intent(&self) -> SubterraneanMissionIntent {
        self.last_effective_mission
    }
    pub fn mission_executive_assessment(&self) -> &ExecutiveAssessment {
        &self.last_executive
    }
    pub fn add_tunnel_node(&mut self, node: TunnelNode) -> Result<(), MissionExecutiveError> {
        self.mission_executive.add_tunnel_node(node)
    }
    pub fn upsert_tunnel_edge(&mut self, edge: TunnelEdge) -> Result<(), MissionExecutiveError> {
        self.mission_executive.upsert_tunnel_edge(edge)
    }
    pub fn submit_work_order(&mut self, order: WorkOrder) -> Result<(), MissionExecutiveError> {
        self.mission_executive.submit_work(order)
    }
    pub fn active_work_order(&self) -> Option<WorkOrderId> {
        self.mission_executive
            .scheduler()
            .active_order()
            .map(|order| order.id)
    }
    pub fn geology_sample(&self) -> GeologySample {
        self.simulator.geology_sample()
    }
    pub fn geology_lookahead(&self, horizon_m: f64) -> GeologicalLookahead {
        self.simulator.geology_lookahead(horizon_m)
    }
    pub fn return_path_assessment(&self) -> ReturnPathAssessment {
        self.simulator.return_path_assessment()
    }
    pub fn team_return_path_assessment(&self) -> ReturnPathAssessment {
        self.last_team
            .shared_route
            .conservative_return_fusion(self.simulator.return_path_assessment())
    }
    pub fn local_agent_id(&self) -> AgentId {
        self.team_coordinator.local_agent()
    }
    pub fn team_assessment(&self) -> TeamOperationalAssessment {
        self.last_team
    }
    pub fn ingest_team_heartbeat(
        &mut self,
        heartbeat: TeamHeartbeat,
    ) -> Result<(), HeartbeatRejection> {
        self.team_coordinator
            .ingest_heartbeat(heartbeat, self.total_steps as u64)
    }
    pub fn merge_shared_tunnel_observation(
        &mut self,
        observation: SharedTunnelObservation,
    ) -> Result<(), SharedMapRejection> {
        self.team_coordinator.merge_tunnel_observation(observation)
    }
    pub fn ingest_tunnel_reservation(
        &mut self,
        reservation: TunnelReservation,
    ) -> Result<(), ReservationRejection> {
        self.team_coordinator.ingest_reservation(reservation)
    }
    pub fn merge_mesh_link(&mut self, link: MeshLink) -> Result<(), MeshLinkRejection> {
        self.team_coordinator.merge_mesh_link(link)
    }
    pub fn receive_rescue_request(
        &mut self,
        request: RescueRequest,
    ) -> Result<(), RescueTransitionError> {
        self.team_coordinator.receive_rescue_request(request)
    }
    pub fn evaluate_pending_rescue(&self) -> Option<RescueFeasibility> {
        self.team_coordinator.evaluate_pending_rescue(
            self.total_steps as u64,
            self.simulator.state(),
            self.simulator.return_path_assessment(),
        )
    }
    pub fn offer_rescue(
        &mut self,
        sequence: u64,
        feasibility: RescueFeasibility,
    ) -> Result<RescueOffer, RescueTransitionError> {
        self.team_coordinator
            .offer_rescue(sequence, self.total_steps as u64, feasibility)
    }
    pub fn accept_rescue(
        &mut self,
        requester: AgentId,
        case_id: crate::rescue::RescueCaseId,
        sequence: u64,
    ) -> Result<(), RescueTransitionError> {
        self.team_coordinator
            .accept_rescue(requester, case_id, sequence)
    }
    pub fn begin_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.team_coordinator.begin_rescue()
    }
    pub fn complete_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.team_coordinator.complete_rescue()
    }
    pub fn abort_rescue(&mut self) -> Result<(), RescueTransitionError> {
        self.team_coordinator.abort_rescue()
    }
    pub fn evidence_records(&self) -> Vec<SafetyEvidenceRecord> {
        self.evidence.records()
    }
    pub fn evidence_summary(&self) -> SafetyEvidenceSummary {
        self.evidence.summary()
    }
    pub fn evidence_json(&self) -> Result<String, serde_json::Error> {
        self.evidence.to_pretty_json()
    }
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: self.current_safety,
            platform: "subterranean".to_string(),
            num_actuators: NUM_PHYSICAL_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for SubterraneanEmbodiment {
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.step(thought_hv, dt, phi)
    }

    fn encode_perception(&mut self) -> ContinuousHV {
        self.encode_perception()
    }

    fn reset(&mut self) {
        self.reset()
    }

    fn safety_level(&self) -> MotorSafetyLevel {
        self.safety_level()
    }

    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.set_safety_override(level)
    }

    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }

    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }

    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Subterranean
    }

    fn num_actuators(&self) -> usize {
        NUM_PHYSICAL_ACTUATORS
    }

    fn total_steps(&self) -> usize {
        self.total_steps()
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}

impl SafeFallback for SubterraneanEmbodiment {
    fn platform_name(&self) -> &'static str {
        "subterranean"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: enclosed/underground, thermal risk to the vehicle itself
    }
    fn safe_fallback_description(&self) -> &'static str {
        "hazard-specific arrest, dewatering, sealing, relay deployment, roof support, withdrawal, or energy conservation"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn operator_envelope(
        operator: u64,
        sequence: u64,
        command: crate::operator_protocol::OperatorCommand,
    ) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: crate::operator_protocol::OperatorId(operator),
            role: crate::operator_protocol::OperatorRole::SafetyOfficer,
            authentication: crate::operator_protocol::AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id: 1,
            issued_step: 0,
            expires_step: 100,
            command,
        }
    }

    #[test]
    fn operator_hold_stops_nominal_motion_before_plant_step() {
        let mut embodiment =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("operator hold"));
        embodiment
            .ingest_operator_command(operator_envelope(
                1,
                1,
                crate::operator_protocol::OperatorCommand::HoldPosition,
            ))
            .expect("hold command should be accepted");
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 4242);
        embodiment.step(&thought, 0.005, 0.95);
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
        assert_eq!(embodiment.last_command().left_track(), 0.0);
        assert_eq!(embodiment.last_command().right_track(), 0.0);
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::OperatorHold
        );
    }

    #[test]
    fn physical_hazard_remains_authoritative_over_operator_mission() {
        let mut embodiment =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("operator hazard"));
        embodiment
            .ingest_operator_command(operator_envelope(
                1,
                1,
                crate::operator_protocol::OperatorCommand::SetMission(
                    SubterraneanMissionIntent::Explore,
                ),
            ))
            .expect("mission command should be accepted");
        embodiment.simulator.state_mut().channels[crate::types::GAS_RISK] = 1.0;
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 4243);
        embodiment.step(&thought, 0.005, 0.95);
        assert_ne!(
            embodiment.effective_mission_intent(),
            SubterraneanMissionIntent::Explore
        );
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
    }

    #[test]
    fn repeated_watchdog_failure_removes_nominal_motion_authority() {
        let mut embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("watchdog"));
        embodiment.set_runtime_health(true, false, true, 0);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 5001);
        for _ in 0..3 {
            embodiment.step(&thought, 0.005, 0.95);
        }
        assert_eq!(embodiment.degraded_mode(), DegradedMode::RecoveryRequired);
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
        assert_eq!(embodiment.last_command().left_track(), 0.0);
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::RecoveryLock
        );
    }

    #[test]
    fn operational_checkpoint_preserves_operator_and_degraded_authority() {
        let genesis = GenesisSeed::from_phrase("authority checkpoint");
        let mut source = SubterraneanEmbodiment::new(&genesis);
        source
            .ingest_operator_command(operator_envelope(
                1,
                1,
                crate::operator_protocol::OperatorCommand::HoldPosition,
            ))
            .expect("hold should be accepted");
        source.set_runtime_health(true, false, true, 0);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 7100);
        for _ in 0..3 {
            source.step(&thought, 0.005, 0.95);
        }
        let checkpoint = source.operational_checkpoint();
        let mut restored = SubterraneanEmbodiment::new(&genesis);
        restored
            .load_operational_checkpoint(&checkpoint)
            .expect("checkpoint should restore");
        assert_eq!(
            restored.operator_constraint(),
            OperatorConstraint::HoldPosition
        );
        assert_eq!(restored.degraded_mode(), DegradedMode::RecoveryRequired);
    }

    #[test]
    fn test_step() {
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_policy_red_does_not_zero_thermal_pump() {
        // Regression: asserts the resulting command, not just the safety enum.
        let mut embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("policy-red"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        embodiment.step(&thought, 0.005, 0.05);
        assert_eq!(embodiment.last_command().thermal_pump(), 1.0);
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::PolicyStop
        );
    }

    #[test]
    fn test_red_arrests_cutter_overheat() {
        // End-to-end: starting from a hot cutter, sustained Red-tier
        // stepping must cool it down (thermal_pump active), not let it
        // continue climbing toward the 180 C clamp the way a
        // zero-everything fallback would (cooling=0, and if a stale
        // controller command still requests boring, heat only grows).
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[crate::types::CUTTER_TEMP_C] = 150.0;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..50 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let temp = e.simulator.state().channels[crate::types::CUTTER_TEMP_C];
        assert!(
            temp < 150.0,
            "cutter must cool under sustained VentAndRetreat, started at 150.0, got {temp}"
        );
    }
    #[test]
    fn test_physical_hazard_escalates_safety_and_overrides_command() {
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.state_mut().channels[crate::types::GAS_RISK] = 0.95;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let result = e.step(&hv, 0.005, 0.9);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(
            e.last_hazard().primary,
            crate::safety::SubterraneanHazard::Gas
        );
        assert_eq!(e.last_command().cutter_head(), 0.0);
        assert!(e.last_command().left_track() < 0.0);
        assert_eq!(e.fallback_stage(), SubterraneanFallbackStage::GasWithdrawal);
    }

    #[test]
    fn runtime_hazard_latch_prevents_single_frame_clear() {
        let mut embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("hazard-latch"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 99);
        embodiment.simulator.state_mut().channels[crate::types::GAS_RISK] = 0.95;
        embodiment.step(&thought, 0.005, 0.9);
        assert_eq!(embodiment.safety_level(), MotorSafetyLevel::Red);

        embodiment.simulator.state_mut().channels[crate::types::GAS_RISK] = 0.0;
        embodiment.step(&thought, 0.005, 0.9);
        assert_eq!(
            embodiment.last_raw_hazard().safety_level,
            MotorSafetyLevel::Green
        );
        assert_eq!(embodiment.last_hazard().safety_level, MotorSafetyLevel::Red);
        assert_eq!(embodiment.safety_level(), MotorSafetyLevel::Red);
    }

    #[test]
    fn physical_hazard_overrides_requested_mission_symbol() {
        let mut embodiment =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("mission-override"));
        embodiment.set_mission_intent(SubterraneanMissionIntent::FollowVein);
        embodiment.simulator.state_mut().channels[crate::types::WATER_INGRESS_RATIO] = 0.95;
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 88);
        embodiment.step(&thought, 0.005, 0.9);
        assert_eq!(
            embodiment.requested_mission_intent(),
            SubterraneanMissionIntent::FollowVein
        );
        assert_eq!(
            embodiment.effective_mission_intent(),
            SubterraneanMissionIntent::EmergencySurface
        );
    }

    #[test]
    fn malformed_runtime_observation_fails_closed_without_poisoning_control() {
        let mut embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("sensor-fault"));
        embodiment.simulator.state_mut().channels[crate::types::GAS_RISK] = f64::NAN;
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 101);
        let result = embodiment.step(&thought, 0.005, 0.9);
        assert!(result.success);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(
            embodiment.last_raw_hazard().primary,
            SubterraneanHazard::SensorFault
        );
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::SensorIsolation
        );
        assert!(embodiment.simulator.state().is_finite());
        assert_eq!(embodiment.last_command().left_track(), 0.0);
    }

    #[test]
    fn bounded_runtime_evidence_records_actual_commands() {
        let genesis = GenesisSeed::from_phrase("evidence-ledger");
        let mut config = SubterraneanConfig::default();
        config.evidence_capacity = 2;
        let mut embodiment = SubterraneanEmbodiment::with_config(&genesis, config);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 55);
        for _ in 0..3 {
            embodiment.step(&thought, 0.005, 0.9);
        }
        let summary = embodiment.evidence_summary();
        assert_eq!(summary.total_records, 3);
        assert_eq!(summary.retained_records, 2);
        assert_eq!(summary.dropped_records, 1);
        assert_eq!(embodiment.evidence_records().len(), 2);
        assert!(embodiment.evidence_json().is_ok());
    }

    #[test]
    fn checked_constructor_rejects_invalid_configuration() {
        let genesis = GenesisSeed::from_phrase("invalid-config");
        let mut config = SubterraneanConfig::default();
        config.physics_hz = 0.0;
        assert!(matches!(
            SubterraneanEmbodiment::try_with_config(&genesis, config),
            Err(ConfigError::InvalidPhysicsRate)
        ));
    }

    #[test]
    fn deployed_fep_path_is_live() {
        let mut embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("runtime-fep"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 19);
        embodiment.step(&thought, 0.005, 0.9);
        assert!(embodiment.last_free_energy().is_finite());
        assert!(embodiment.fep_tau_factor().is_finite());
        assert!(embodiment.fep_tau_factor() > 0.0);
    }

    #[test]
    fn deterministic_reference_model_has_small_prediction_residual() {
        let mut embodiment =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("prediction-residual"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 20);
        let result = embodiment.step(&thought, 0.005, 0.9);
        assert!(
            result.prediction_error < 1e-4,
            "reference model and deterministic plant should agree, got {}",
            result.prediction_error
        );
    }

    #[test]
    fn custom_site_geology_is_available_through_deployment_api() {
        use crate::geology::{GeotechnicalProfile, MaterialClass};

        let embodiment = SubterraneanEmbodiment::with_config_and_geology(
            &GenesisSeed::from_phrase("custom-site-geology"),
            SubterraneanConfig::default(),
            GeotechnicalProfile::homogeneous(MaterialClass::Granite),
        );
        assert_eq!(embodiment.geology_sample().material, MaterialClass::Granite);
        assert!(embodiment.geology_lookahead(5.0).max_hardness > 0.9);
        assert!(embodiment.return_path_assessment().feasible);
    }
    #[test]
    fn peer_tunnel_conflict_becomes_command_level_safety_authority() {
        use crate::occupancy::{ReservationPriority, TunnelDirection, TunnelReservation};

        let genesis = GenesisSeed::from_phrase("team-collision");
        let mut embodiment = SubterraneanEmbodiment::with_config_geology_and_agent(
            &genesis,
            SubterraneanConfig::default(),
            GeotechnicalProfile::default(),
            AgentId::new(3),
        );
        assert_eq!(
            embodiment.ingest_tunnel_reservation(TunnelReservation {
                agent_id: AgentId::new(2),
                epoch: 1,
                sequence: 1,
                issued_step: 0,
                valid_from_step: 0,
                valid_until_step: 100,
                minimum_depth_m: 0.0,
                maximum_depth_m: 4.0,
                direction: TunnelDirection::Inbound,
                priority: ReservationPriority::Emergency,
            }),
            Ok(())
        );
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 404);
        let result = embodiment.step(&thought, 0.005, 0.9);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert_eq!(
            embodiment.last_hazard().primary,
            SubterraneanHazard::TunnelConflict
        );
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::TunnelYield
        );
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
        assert_eq!(embodiment.last_command().left_track(), 0.0);
        assert_eq!(
            embodiment.effective_mission_intent(),
            SubterraneanMissionIntent::YieldTunnel
        );
    }

    #[test]
    fn stale_temporal_frame_removes_productive_authority_same_cycle() {
        let genesis = GenesisSeed::from_phrase("temporal-stale-runtime");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        let revisions = temporal_runtime_revisions(
            0,
            HazardAssessment::clear(),
            &embodiment.mission_executive,
            embodiment.mission_manager.requested(),
        );
        let mut frame = TemporalRuntimeInputs::derive(0, 0, 0.005, revisions);
        frame.observations[0].observed_time_ns = 0;
        frame.observations[0].freshness_limit_ns = 1_000_000;
        embodiment.ingest_temporal_frame(frame);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 5501);
        let result = embodiment.step(&thought, 0.005, 0.9);
        assert_eq!(result.safety_level, MotorSafetyLevel::Orange);
        assert_eq!(embodiment.last_command().cutter_head(), 0.0);
        assert_eq!(
            embodiment.effective_mission_intent(),
            SubterraneanMissionIntent::ReturnHome
        );
        assert_eq!(
            embodiment.fallback_stage(),
            SubterraneanFallbackStage::TemporalReturn
        );
        assert_eq!(
            embodiment.temporal_assessment().authority,
            TemporalAuthority::ReturnOnly
        );
    }

    #[test]
    fn temporal_hold_survives_operational_checkpoint_restore() {
        let genesis = GenesisSeed::from_phrase("temporal-checkpoint-runtime");
        let mut original = SubterraneanEmbodiment::new(&genesis);
        let revisions = temporal_runtime_revisions(
            0,
            HazardAssessment::clear(),
            &original.mission_executive,
            original.mission_manager.requested(),
        );
        let mut frame = TemporalRuntimeInputs::derive(0, 0, 0.005, revisions);
        frame.clock_samples[0].event_time_ns = 100_000_000;
        original.ingest_temporal_frame(frame);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 5502);
        original.step(&thought, 0.005, 0.9);
        assert!(original.temporal_assessment().hold_latched);

        let checkpoint = original.operational_checkpoint();
        let mut restored = SubterraneanEmbodiment::new(&genesis);
        restored.load_operational_checkpoint(&checkpoint).unwrap();
        restored.step(&thought, 0.005, 0.9);
        assert!(restored.temporal_assessment().hold_latched);
        assert_eq!(restored.last_command().cutter_head(), 0.0);
        assert_eq!(
            restored.fallback_stage(),
            SubterraneanFallbackStage::TemporalHold
        );
    }
}
