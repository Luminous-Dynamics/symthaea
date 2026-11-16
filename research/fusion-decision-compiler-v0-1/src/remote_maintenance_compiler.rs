//! Design-for-remote-maintenance procedure compiler, v0.1.
//!
//! Validates the completeness of a declared replaceable assembly, interface
//! inventory, maintenance DAG, resource/tool allocation and scheduled outage.
//! It does not establish physical reachability, collision freedom, activated
//! handling, dose/contamination performance or mockup repair time.

use std::collections::{BTreeMap, BTreeSet};

pub const DFRM_COMPILER_VERSION: &str = "fusion-dfrm-compiler/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum InterfaceFamily {
    Coolant,
    Electrical,
    Instrumentation,
    Vacuum,
    Tritium,
    Structural,
}

impl InterfaceFamily {
    pub const fn requires_boundary_preparation(self) -> bool {
        matches!(self, Self::Coolant | Self::Vacuum | Self::Tritium)
    }

    pub const fn expected_integrity_test(self) -> IntegrityTestKind {
        match self {
            Self::Coolant => IntegrityTestKind::PressureLeak,
            Self::Electrical => IntegrityTestKind::ElectricalContinuity,
            Self::Instrumentation => IntegrityTestKind::SignalLoop,
            Self::Vacuum => IntegrityTestKind::VacuumLeak,
            Self::Tritium => IntegrityTestKind::TritiumBoundary,
            Self::Structural => IntegrityTestKind::AlignmentMetrology,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum IntegrityTestKind {
    PressureLeak,
    ElectricalContinuity,
    SignalLoop,
    VacuumLeak,
    TritiumBoundary,
    AlignmentMetrology,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MaintenanceEvidenceClass {
    Synthetic,
    Assumed,
    Modelled,
    KinematicSimulation,
    MockupValidated,
    Measured,
    IndependentlyVerified,
    PhysicallyQualified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MaintenanceAction {
    IsolateSystem,
    ProveSafeState,
    PrepareServiceBoundary,
    DisconnectInterface,
    ReleaseRestraints,
    ExtractModule,
    TransferToShieldedCask,
    HotCellInspectRepair,
    InstallModule,
    AlignMetrology,
    SecureRestraints,
    ReconnectInterface,
    IntegrityTest(IntegrityTestKind),
    SystemRequalification,
    RestartSystem,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplaceableAssembly {
    pub assembly_id: String,
    pub revision: String,
    pub mass_kg: f64,
    pub centre_of_gravity_x_m: f64,
    pub centre_of_gravity_y_m: f64,
    pub centre_of_gravity_z_m: f64,
    pub removal_envelope_x_m: f64,
    pub removal_envelope_y_m: f64,
    pub removal_envelope_z_m: f64,
    pub handling_datum_ref: String,
    pub removal_envelope_ref: String,
    pub shielded_cask_envelope_ref: String,
    pub hot_cell_interface_ref: String,
    pub hazard_profile_ref: String,
    pub isolation_plan_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceInterface {
    pub interface_id: String,
    pub mate_interface_id: String,
    /// True only for the connector physically mounted on the replaceable assembly.
    pub is_assembly_side: bool,
    pub family: InterfaceFamily,
    pub compatibility_key: String,
    pub access_side_ref: String,
    pub remote_actuation_capability: String,
    pub remote_actuation_proof_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterfaceProcedureBinding {
    pub interface_id: String,
    pub isolation_operation_id: String,
    pub prove_safe_operation_id: String,
    pub preparation_operation_id: Option<String>,
    pub disconnect_operation_id: String,
    pub reconnect_operation_id: String,
    pub integrity_test_operation_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceResource {
    pub resource_id: String,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaintenanceTool {
    pub tool_id: String,
    pub capabilities: Vec<String>,
    pub maximum_payload_kg: f64,
    pub maximum_centre_of_gravity_offset_m: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceOperation {
    pub operation_id: String,
    pub action: MaintenanceAction,
    pub interface_id: Option<String>,
    pub predecessor_ids: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub resource_id: String,
    pub tool_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub minimum_duration_seconds: u64,
    pub nominal_duration_seconds: u64,
    pub maximum_duration_seconds: u64,
    pub procedure_step_ref: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub evidence_class: MaintenanceEvidenceClass,
    pub clearance_reachability_ref: String,
    pub clearance_reachability_sha256: String,
    pub kinematic_model_ref: String,
    pub validity_domain_ref: String,
    pub failure_recovery_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemoteMaintenancePlan {
    pub campaign_id: String,
    pub procedure_revision: String,
    pub service_boundary_ref: String,
    pub outage_start_second: u64,
    pub outage_end_second: u64,
    pub assembly: ReplaceableAssembly,
    pub interfaces: Vec<ServiceInterface>,
    pub interface_bindings: Vec<InterfaceProcedureBinding>,
    pub resources: Vec<MaintenanceResource>,
    pub tools: Vec<MaintenanceTool>,
    pub operations: Vec<MaintenanceOperation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DfrmDisposition {
    NotEvaluable,
    ValidDeclaredProcedure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DfrmDiagnostic {
    EmptyCampaignIdentity,
    EmptyProcedureRevision,
    EmptyServiceBoundary,
    InvalidOutageHorizon,
    EmptyAssemblyIdentity,
    InvalidAssemblyMass,
    InvalidAssemblyCentreOfGravity,
    InvalidRemovalEnvelope,
    MissingHandlingDatum,
    MissingRemovalEnvelopeReference,
    MissingShieldedCaskEnvelope,
    MissingHotCellInterface,
    MissingHazardProfile,
    MissingIsolationPlan,
    DuplicateInterfaceIdentity,
    MissingInterfaceMate,
    IncompatibleInterfaceMate,
    MissingInterfaceAccess,
    MissingRemoteActuationProof,
    DuplicateInterfaceBinding,
    MissingInterfaceBinding,
    UnknownBindingInterface,
    UnknownIsolationOperation,
    UnknownProveSafeOperation,
    UnknownPreparationOperation,
    UnknownDisconnectOperation,
    UnknownReconnectOperation,
    UnknownIntegrityTestOperation,
    WrongIsolationAction,
    WrongProveSafeAction,
    WrongPreparationAction,
    WrongDisconnectAction,
    WrongReconnectAction,
    WrongIntegrityTestAction,
    MissingRequiredPreparation,
    UnexpectedPreparation,
    MissingResource,
    DuplicateResourceIdentity,
    MissingTool,
    DuplicateToolIdentity,
    EmptyOperationIdentity,
    DuplicateOperationIdentity,
    EmptyProcedureStepReference,
    EmptyEvidenceReference,
    InvalidEvidenceDigest,
    MissingClearanceReachabilityProof,
    InvalidClearanceReachabilityDigest,
    MissingKinematicModelReference,
    MissingKinematicValidityDomain,
    MissingFailureRecoveryReference,
    UnknownPredecessor,
    DuplicatePredecessor,
    DependencyCycle,
    MissingAction(MaintenanceAction),
    InvalidOperationDuration,
    OperationOutsideOutage,
    ScheduleDurationOutsideDeclaredRange,
    ResourceCapabilityMissing,
    ToolCapabilityMissing,
    ResourceScheduleCollision,
    ToolScheduleCollision,
    PayloadLimitExceeded,
    CentreOfGravityLimitExceeded,
    DependencyScheduleViolation,
    MissingRequiredPredecessor,
    RestraintReleaseBeforeDisconnect,
    ExtractionBeforeRestraintRelease,
    TransferBeforeExtraction,
    HotCellBeforeTransfer,
    InstallBeforeInspection,
    AlignmentBeforeInstall,
    RestraintSecureBeforeAlignment,
    ReconnectBeforeRestraintSecure,
    IntegrityTestBeforeReconnect,
    RequalificationBeforeIntegrityTest,
    RestartBeforeRequalification,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceUtilization {
    pub resource_id: String,
    pub busy_seconds: u64,
    pub idle_seconds_within_campaign: u64,
    pub operation_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceClassCount {
    pub evidence_class: MaintenanceEvidenceClass,
    pub operation_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationEvidenceReceipt {
    pub operation_id: String,
    pub action: MaintenanceAction,
    pub resource_id: String,
    pub tool_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub duration_low_seconds: u64,
    pub duration_nominal_seconds: u64,
    pub duration_high_seconds: u64,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub evidence_class: MaintenanceEvidenceClass,
    pub clearance_reachability_ref: String,
    pub clearance_reachability_sha256: String,
    pub kinematic_model_ref: String,
    pub validity_domain_ref: String,
    pub failure_recovery_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfrmReport {
    pub contract_version: &'static str,
    pub campaign_id: String,
    pub assembly_id: String,
    pub disposition: DfrmDisposition,
    pub diagnostics: Vec<DfrmDiagnostic>,
    pub operation_count: usize,
    pub interface_count: usize,
    pub resource_count: usize,
    pub tool_count: usize,
    pub scheduled_outage_seconds: Option<u64>,
    /// Critical path computed from the maximum declared operation durations.
    /// It is a model bound, not a measured repair time.
    pub critical_path_high_seconds: Option<u64>,
    pub scheduled_resource_utilization: Vec<ResourceUtilization>,
    pub evidence_class_counts: Vec<EvidenceClassCount>,
    pub operation_receipts: Vec<OperationEvidenceReceipt>,
    pub claim_ceiling: &'static str,
}

pub fn compile_remote_maintenance_plan(plan: &RemoteMaintenancePlan) -> DfrmReport {
    let mut diagnostics = validate_plan(plan);
    diagnostics.sort_by_key(|diagnostic| format!("{diagnostic:?}"));
    diagnostics.dedup();

    let operation_receipts = canonical_operation_receipts(plan);
    if !diagnostics.is_empty() {
        return DfrmReport {
            contract_version: DFRM_COMPILER_VERSION,
            campaign_id: plan.campaign_id.clone(),
            assembly_id: plan.assembly.assembly_id.clone(),
            disposition: DfrmDisposition::NotEvaluable,
            diagnostics,
            operation_count: plan.operations.len(),
            interface_count: plan.interfaces.len(),
            resource_count: plan.resources.len(),
            tool_count: plan.tools.len(),
            scheduled_outage_seconds: None,
            critical_path_high_seconds: None,
            scheduled_resource_utilization: Vec::new(),
            evidence_class_counts: Vec::new(),
            operation_receipts,
            claim_ceiling: "declared maintenance-procedure completeness only",
        };
    }

    let topological_order = topological_order(&plan.operations).unwrap_or_default();
    let operation_map: BTreeMap<&str, &MaintenanceOperation> = plan
        .operations
        .iter()
        .map(|operation| (operation.operation_id.as_str(), operation))
        .collect();
    let mut critical_path_ends: BTreeMap<String, u64> = BTreeMap::new();

    for operation_id in topological_order {
        let operation = operation_map[operation_id.as_str()];
        let predecessor_end = operation
            .predecessor_ids
            .iter()
            .filter_map(|predecessor| critical_path_ends.get(predecessor))
            .copied()
            .max()
            .unwrap_or(0);
        critical_path_ends.insert(
            operation_id,
            predecessor_end.saturating_add(operation.maximum_duration_seconds),
        );
    }
    let critical_path_high_seconds = critical_path_ends.values().copied().max().unwrap_or(0);

    let mut utilization: BTreeMap<String, ResourceUtilization> = BTreeMap::new();
    let campaign_duration = plan.outage_end_second - plan.outage_start_second;
    for resource in &plan.resources {
        utilization.insert(
            resource.resource_id.clone(),
            ResourceUtilization {
                resource_id: resource.resource_id.clone(),
                busy_seconds: 0,
                idle_seconds_within_campaign: campaign_duration,
                operation_count: 0,
            },
        );
    }
    for operation in &plan.operations {
        let item = utilization
            .get_mut(&operation.resource_id)
            .expect("validated resource");
        let duration = operation.end_second - operation.start_second;
        item.busy_seconds = item.busy_seconds.saturating_add(duration);
        item.idle_seconds_within_campaign =
            item.idle_seconds_within_campaign.saturating_sub(duration);
        item.operation_count += 1;
    }

    let mut class_counts: BTreeMap<MaintenanceEvidenceClass, usize> = BTreeMap::new();
    for operation in &plan.operations {
        *class_counts.entry(operation.evidence_class).or_default() += 1;
    }
    let evidence_class_counts = class_counts
        .into_iter()
        .map(|(evidence_class, operation_count)| EvidenceClassCount {
            evidence_class,
            operation_count,
        })
        .collect();

    DfrmReport {
        contract_version: DFRM_COMPILER_VERSION,
        campaign_id: plan.campaign_id.clone(),
        assembly_id: plan.assembly.assembly_id.clone(),
        disposition: DfrmDisposition::ValidDeclaredProcedure,
        diagnostics,
        operation_count: plan.operations.len(),
        interface_count: plan.interfaces.len(),
        resource_count: plan.resources.len(),
        tool_count: plan.tools.len(),
        scheduled_outage_seconds: Some(campaign_duration),
        critical_path_high_seconds: Some(critical_path_high_seconds),
        scheduled_resource_utilization: utilization.into_values().collect(),
        evidence_class_counts,
        operation_receipts,
        claim_ceiling: "declared maintenance-procedure completeness only",
    }
}

fn validate_plan(plan: &RemoteMaintenancePlan) -> Vec<DfrmDiagnostic> {
    let mut diagnostics = Vec::new();
    if plan.campaign_id.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::EmptyCampaignIdentity);
    }
    if plan.procedure_revision.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::EmptyProcedureRevision);
    }
    if plan.service_boundary_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::EmptyServiceBoundary);
    }
    if plan.outage_end_second <= plan.outage_start_second {
        diagnostics.push(DfrmDiagnostic::InvalidOutageHorizon);
    }

    let assembly = &plan.assembly;
    if assembly.assembly_id.trim().is_empty() || assembly.revision.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::EmptyAssemblyIdentity);
    }
    if !assembly.mass_kg.is_finite() || assembly.mass_kg <= 0.0 {
        diagnostics.push(DfrmDiagnostic::InvalidAssemblyMass);
    }
    let cg_values = [
        assembly.centre_of_gravity_x_m,
        assembly.centre_of_gravity_y_m,
        assembly.centre_of_gravity_z_m,
    ];
    if cg_values.iter().any(|value| !value.is_finite()) {
        diagnostics.push(DfrmDiagnostic::InvalidAssemblyCentreOfGravity);
    }
    let envelope = [
        assembly.removal_envelope_x_m,
        assembly.removal_envelope_y_m,
        assembly.removal_envelope_z_m,
    ];
    if envelope
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        diagnostics.push(DfrmDiagnostic::InvalidRemovalEnvelope);
    }
    if assembly.handling_datum_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingHandlingDatum);
    }
    if assembly.removal_envelope_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingRemovalEnvelopeReference);
    }
    if assembly.shielded_cask_envelope_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingShieldedCaskEnvelope);
    }
    if assembly.hot_cell_interface_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingHotCellInterface);
    }
    if assembly.hazard_profile_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingHazardProfile);
    }
    if assembly.isolation_plan_ref.trim().is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingIsolationPlan);
    }

    let mut interface_map: BTreeMap<&str, &ServiceInterface> = BTreeMap::new();
    for interface in &plan.interfaces {
        if interface.interface_id.trim().is_empty() || interface.compatibility_key.trim().is_empty()
        {
            diagnostics.push(DfrmDiagnostic::DuplicateInterfaceIdentity);
        }
        if interface.access_side_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::MissingInterfaceAccess);
        }
        if interface.remote_actuation_capability.trim().is_empty()
            || interface.remote_actuation_proof_ref.trim().is_empty()
        {
            diagnostics.push(DfrmDiagnostic::MissingRemoteActuationProof);
        }
        if interface_map
            .insert(interface.interface_id.as_str(), interface)
            .is_some()
        {
            diagnostics.push(DfrmDiagnostic::DuplicateInterfaceIdentity);
        }
    }
    for interface in &plan.interfaces {
        match interface_map.get(interface.mate_interface_id.as_str()) {
            None => diagnostics.push(DfrmDiagnostic::MissingInterfaceMate),
            Some(mate)
                if mate.mate_interface_id != interface.interface_id
                    || mate.family != interface.family
                    || mate.compatibility_key != interface.compatibility_key
                    || mate.is_assembly_side == interface.is_assembly_side =>
            {
                diagnostics.push(DfrmDiagnostic::IncompatibleInterfaceMate);
            }
            Some(_) => {}
        }
    }

    let mut resources = BTreeMap::new();
    for resource in &plan.resources {
        if resource.resource_id.trim().is_empty()
            || resources
                .insert(resource.resource_id.as_str(), resource)
                .is_some()
        {
            diagnostics.push(DfrmDiagnostic::DuplicateResourceIdentity);
        }
    }
    let mut tools = BTreeMap::new();
    for tool in &plan.tools {
        if tool.tool_id.trim().is_empty() || tools.insert(tool.tool_id.as_str(), tool).is_some() {
            diagnostics.push(DfrmDiagnostic::DuplicateToolIdentity);
        }
        if !tool.maximum_payload_kg.is_finite()
            || tool.maximum_payload_kg <= 0.0
            || !tool.maximum_centre_of_gravity_offset_m.is_finite()
            || tool.maximum_centre_of_gravity_offset_m < 0.0
        {
            diagnostics.push(DfrmDiagnostic::PayloadLimitExceeded);
        }
    }

    let mut operations: BTreeMap<&str, &MaintenanceOperation> = BTreeMap::new();
    for operation in &plan.operations {
        if operation.operation_id.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::EmptyOperationIdentity);
        }
        if operations
            .insert(operation.operation_id.as_str(), operation)
            .is_some()
        {
            diagnostics.push(DfrmDiagnostic::DuplicateOperationIdentity);
        }
        if operation.procedure_step_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::EmptyProcedureStepReference);
        }
        if operation.evidence_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::EmptyEvidenceReference);
        }
        if !is_sha256(&operation.evidence_sha256) {
            diagnostics.push(DfrmDiagnostic::InvalidEvidenceDigest);
        }
        if operation.clearance_reachability_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::MissingClearanceReachabilityProof);
        }
        if !is_sha256(&operation.clearance_reachability_sha256) {
            diagnostics.push(DfrmDiagnostic::InvalidClearanceReachabilityDigest);
        }
        if operation.kinematic_model_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::MissingKinematicModelReference);
        }
        if operation.validity_domain_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::MissingKinematicValidityDomain);
        }
        if operation.failure_recovery_ref.trim().is_empty() {
            diagnostics.push(DfrmDiagnostic::MissingFailureRecoveryReference);
        }
        if operation.maximum_duration_seconds == 0
            || operation.minimum_duration_seconds == 0
            || operation.minimum_duration_seconds > operation.nominal_duration_seconds
            || operation.nominal_duration_seconds > operation.maximum_duration_seconds
        {
            diagnostics.push(DfrmDiagnostic::InvalidOperationDuration);
        }
        if operation.start_second < plan.outage_start_second
            || operation.end_second > plan.outage_end_second
            || operation.end_second <= operation.start_second
        {
            diagnostics.push(DfrmDiagnostic::OperationOutsideOutage);
        } else {
            let scheduled = operation.end_second - operation.start_second;
            if scheduled < operation.minimum_duration_seconds
                || scheduled > operation.maximum_duration_seconds
            {
                diagnostics.push(DfrmDiagnostic::ScheduleDurationOutsideDeclaredRange);
            }
        }

        let resource = resources.get(operation.resource_id.as_str());
        if resource.is_none() {
            diagnostics.push(DfrmDiagnostic::MissingResource);
        }
        let tool = tools.get(operation.tool_id.as_str());
        if tool.is_none() {
            diagnostics.push(DfrmDiagnostic::MissingTool);
        }
        for capability in &operation.required_capabilities {
            if resource.is_some_and(|item| !item.capabilities.contains(capability)) {
                diagnostics.push(DfrmDiagnostic::ResourceCapabilityMissing);
            }
            if tool.is_some_and(|item| !item.capabilities.contains(capability)) {
                diagnostics.push(DfrmDiagnostic::ToolCapabilityMissing);
            }
        }
        if matches!(
            operation.action,
            MaintenanceAction::ExtractModule
                | MaintenanceAction::TransferToShieldedCask
                | MaintenanceAction::InstallModule
        ) {
            if let Some(tool) = tool {
                if assembly.mass_kg > tool.maximum_payload_kg {
                    diagnostics.push(DfrmDiagnostic::PayloadLimitExceeded);
                }
                let cg_offset = (assembly.centre_of_gravity_x_m.powi(2)
                    + assembly.centre_of_gravity_y_m.powi(2)
                    + assembly.centre_of_gravity_z_m.powi(2))
                .sqrt();
                if cg_offset > tool.maximum_centre_of_gravity_offset_m {
                    diagnostics.push(DfrmDiagnostic::CentreOfGravityLimitExceeded);
                }
            }
        }
    }

    let operation_ids: BTreeSet<&str> = operations.keys().copied().collect();
    for operation in &plan.operations {
        let mut seen_predecessors = BTreeSet::new();
        for predecessor in &operation.predecessor_ids {
            if !operation_ids.contains(predecessor.as_str()) {
                diagnostics.push(DfrmDiagnostic::UnknownPredecessor);
            }
            if !seen_predecessors.insert(predecessor.as_str()) {
                diagnostics.push(DfrmDiagnostic::DuplicatePredecessor);
            }
        }
    }

    let Some(topological) = topological_order(&plan.operations) else {
        diagnostics.push(DfrmDiagnostic::DependencyCycle);
        return diagnostics;
    };
    let ancestors = ancestor_map(&plan.operations, &topological);

    validate_interface_bindings(
        plan,
        &interface_map,
        &operations,
        &ancestors,
        &mut diagnostics,
    );
    validate_action_ordering(plan, &ancestors, &mut diagnostics);
    validate_schedules(plan, &mut diagnostics);

    diagnostics
}

fn validate_interface_bindings(
    plan: &RemoteMaintenancePlan,
    interfaces: &BTreeMap<&str, &ServiceInterface>,
    operations: &BTreeMap<&str, &MaintenanceOperation>,
    ancestors: &BTreeMap<String, BTreeSet<String>>,
    diagnostics: &mut Vec<DfrmDiagnostic>,
) {
    let mut binding_map = BTreeMap::new();
    for binding in &plan.interface_bindings {
        if binding_map
            .insert(binding.interface_id.as_str(), binding)
            .is_some()
        {
            diagnostics.push(DfrmDiagnostic::DuplicateInterfaceBinding);
        }
    }
    for interface in plan.interfaces.iter().filter(|item| item.is_assembly_side) {
        let Some(binding) = binding_map.get(interface.interface_id.as_str()) else {
            diagnostics.push(DfrmDiagnostic::MissingInterfaceBinding);
            continue;
        };
        let isolation = operations.get(binding.isolation_operation_id.as_str());
        if isolation.is_none() {
            diagnostics.push(DfrmDiagnostic::UnknownIsolationOperation);
        } else if isolation
            .is_some_and(|operation| operation.action != MaintenanceAction::IsolateSystem)
        {
            diagnostics.push(DfrmDiagnostic::WrongIsolationAction);
        }
        let safe = operations.get(binding.prove_safe_operation_id.as_str());
        if safe.is_none() {
            diagnostics.push(DfrmDiagnostic::UnknownProveSafeOperation);
        } else if safe
            .is_some_and(|operation| operation.action != MaintenanceAction::ProveSafeState)
        {
            diagnostics.push(DfrmDiagnostic::WrongProveSafeAction);
        }
        if let (Some(isolation), Some(safe)) = (isolation, safe) {
            require_ancestor(
                safe.operation_id.as_str(),
                isolation.operation_id.as_str(),
                ancestors,
                diagnostics,
                DfrmDiagnostic::WrongProveSafeAction,
            );
        }

        match (
            interface.family.requires_boundary_preparation(),
            binding.preparation_operation_id.as_deref(),
        ) {
            (true, None) => diagnostics.push(DfrmDiagnostic::MissingRequiredPreparation),
            (false, Some(_)) => diagnostics.push(DfrmDiagnostic::UnexpectedPreparation),
            (true, Some(preparation_id)) => match operations.get(preparation_id) {
                None => diagnostics.push(DfrmDiagnostic::UnknownPreparationOperation),
                Some(operation) => {
                    if operation.action != MaintenanceAction::PrepareServiceBoundary {
                        diagnostics.push(DfrmDiagnostic::WrongPreparationAction);
                    }
                    require_ancestor(
                        operation.operation_id.as_str(),
                        binding.prove_safe_operation_id.as_str(),
                        ancestors,
                        diagnostics,
                        DfrmDiagnostic::WrongPreparationAction,
                    );
                }
            },
            (false, None) => {}
        }

        let disconnect = operations.get(binding.disconnect_operation_id.as_str());
        if disconnect.is_none() {
            diagnostics.push(DfrmDiagnostic::UnknownDisconnectOperation);
        } else if disconnect.is_some_and(|operation| {
            operation.action != MaintenanceAction::DisconnectInterface
                || operation.interface_id.as_deref() != Some(interface.interface_id.as_str())
        }) {
            diagnostics.push(DfrmDiagnostic::WrongDisconnectAction);
        }
        if let Some(disconnect) = disconnect {
            require_ancestor(
                disconnect.operation_id.as_str(),
                binding.prove_safe_operation_id.as_str(),
                ancestors,
                diagnostics,
                DfrmDiagnostic::WrongDisconnectAction,
            );
            if let Some(preparation_id) = binding.preparation_operation_id.as_deref() {
                require_ancestor(
                    disconnect.operation_id.as_str(),
                    preparation_id,
                    ancestors,
                    diagnostics,
                    DfrmDiagnostic::WrongDisconnectAction,
                );
            }
        }

        let reconnect = operations.get(binding.reconnect_operation_id.as_str());
        if reconnect.is_none() {
            diagnostics.push(DfrmDiagnostic::UnknownReconnectOperation);
        } else if reconnect.is_some_and(|operation| {
            operation.action != MaintenanceAction::ReconnectInterface
                || operation.interface_id.as_deref() != Some(interface.interface_id.as_str())
        }) {
            diagnostics.push(DfrmDiagnostic::WrongReconnectAction);
        }
        for operation in disconnect.into_iter().chain(reconnect) {
            if !operation
                .required_capabilities
                .contains(&interface.remote_actuation_capability)
            {
                diagnostics.push(DfrmDiagnostic::ToolCapabilityMissing);
            }
        }

        let integrity = operations.get(binding.integrity_test_operation_id.as_str());
        if integrity.is_none() {
            diagnostics.push(DfrmDiagnostic::UnknownIntegrityTestOperation);
        } else if integrity.is_some_and(|operation| {
            operation.action
                != MaintenanceAction::IntegrityTest(interface.family.expected_integrity_test())
                || operation.interface_id.as_deref() != Some(interface.interface_id.as_str())
        }) {
            diagnostics.push(DfrmDiagnostic::WrongIntegrityTestAction);
        }
        if let (Some(reconnect), Some(integrity)) = (reconnect, integrity) {
            require_ancestor(
                integrity.operation_id.as_str(),
                reconnect.operation_id.as_str(),
                ancestors,
                diagnostics,
                DfrmDiagnostic::WrongIntegrityTestAction,
            );
        }
        if !interfaces.contains_key(interface.interface_id.as_str()) {
            diagnostics.push(DfrmDiagnostic::UnknownBindingInterface);
        }
    }
    for binding in &plan.interface_bindings {
        if !interfaces
            .get(binding.interface_id.as_str())
            .is_some_and(|interface| interface.is_assembly_side)
        {
            diagnostics.push(DfrmDiagnostic::UnknownBindingInterface);
        }
    }
    for operation in &plan.operations {
        if let Some(interface_id) = operation.interface_id.as_deref()
            && !interfaces
                .get(interface_id)
                .is_some_and(|interface| interface.is_assembly_side)
        {
            diagnostics.push(DfrmDiagnostic::UnknownBindingInterface);
        }
    }
}

fn validate_action_ordering(
    plan: &RemoteMaintenancePlan,
    ancestors: &BTreeMap<String, BTreeSet<String>>,
    diagnostics: &mut Vec<DfrmDiagnostic>,
) {
    let find_actions = |action: MaintenanceAction| {
        plan.operations
            .iter()
            .filter(|operation| operation.action == action)
            .collect::<Vec<_>>()
    };
    let require_all = |successor: &MaintenanceOperation,
                       predecessors: &[&MaintenanceOperation],
                       diagnostic: DfrmDiagnostic,
                       diagnostics: &mut Vec<DfrmDiagnostic>| {
        for predecessor in predecessors {
            require_ancestor(
                successor.operation_id.as_str(),
                predecessor.operation_id.as_str(),
                ancestors,
                diagnostics,
                diagnostic,
            );
        }
    };

    let isolates = find_actions(MaintenanceAction::IsolateSystem);
    let safe_states = find_actions(MaintenanceAction::ProveSafeState);
    let disconnects = find_actions(MaintenanceAction::DisconnectInterface);
    let releases = find_actions(MaintenanceAction::ReleaseRestraints);
    let extracts = find_actions(MaintenanceAction::ExtractModule);
    let transfers = find_actions(MaintenanceAction::TransferToShieldedCask);
    let hot_cell = find_actions(MaintenanceAction::HotCellInspectRepair);
    let installs = find_actions(MaintenanceAction::InstallModule);
    let aligns = find_actions(MaintenanceAction::AlignMetrology);
    let secures = find_actions(MaintenanceAction::SecureRestraints);
    let reconnects = find_actions(MaintenanceAction::ReconnectInterface);
    let integrity_tests: Vec<_> = plan
        .operations
        .iter()
        .filter(|operation| matches!(operation.action, MaintenanceAction::IntegrityTest(_)))
        .collect();
    let requalifications = find_actions(MaintenanceAction::SystemRequalification);
    let restarts = find_actions(MaintenanceAction::RestartSystem);

    for safe in &safe_states {
        require_all(
            safe,
            &isolates,
            DfrmDiagnostic::WrongProveSafeAction,
            diagnostics,
        );
    }
    for disconnect in &disconnects {
        require_all(
            disconnect,
            &safe_states,
            DfrmDiagnostic::WrongDisconnectAction,
            diagnostics,
        );
    }
    for release in &releases {
        require_all(
            release,
            &disconnects,
            DfrmDiagnostic::RestraintReleaseBeforeDisconnect,
            diagnostics,
        );
    }
    for extract in &extracts {
        require_all(
            extract,
            &releases,
            DfrmDiagnostic::ExtractionBeforeRestraintRelease,
            diagnostics,
        );
    }
    for transfer in &transfers {
        require_all(
            transfer,
            &extracts,
            DfrmDiagnostic::TransferBeforeExtraction,
            diagnostics,
        );
    }
    for inspect in &hot_cell {
        require_all(
            inspect,
            &transfers,
            DfrmDiagnostic::HotCellBeforeTransfer,
            diagnostics,
        );
    }
    for install in &installs {
        require_all(
            install,
            &hot_cell,
            DfrmDiagnostic::InstallBeforeInspection,
            diagnostics,
        );
    }
    for align in &aligns {
        require_all(
            align,
            &installs,
            DfrmDiagnostic::AlignmentBeforeInstall,
            diagnostics,
        );
    }
    for secure in &secures {
        require_all(
            secure,
            &aligns,
            DfrmDiagnostic::RestraintSecureBeforeAlignment,
            diagnostics,
        );
    }
    for reconnect in &reconnects {
        require_all(
            reconnect,
            &secures,
            DfrmDiagnostic::ReconnectBeforeRestraintSecure,
            diagnostics,
        );
    }
    for test in &integrity_tests {
        require_all(
            test,
            &reconnects,
            DfrmDiagnostic::IntegrityTestBeforeReconnect,
            diagnostics,
        );
    }
    for requalify in &requalifications {
        require_all(
            requalify,
            &integrity_tests,
            DfrmDiagnostic::RequalificationBeforeIntegrityTest,
            diagnostics,
        );
    }
    for restart in &restarts {
        require_all(
            restart,
            &requalifications,
            DfrmDiagnostic::RestartBeforeRequalification,
            diagnostics,
        );
    }

    if isolates.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::IsolateSystem,
        ));
    }
    if safe_states.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::ProveSafeState,
        ));
    }
    if disconnects.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::DisconnectInterface,
        ));
    }
    if releases.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::ReleaseRestraints,
        ));
    }
    if extracts.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::ExtractModule,
        ));
    }
    if transfers.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::TransferToShieldedCask,
        ));
    }
    if hot_cell.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::HotCellInspectRepair,
        ));
    }
    if installs.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::InstallModule,
        ));
    }
    if aligns.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::AlignMetrology,
        ));
    }
    if secures.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::SecureRestraints,
        ));
    }
    if reconnects.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::ReconnectInterface,
        ));
    }
    if integrity_tests.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::IntegrityTest(IntegrityTestKind::PressureLeak),
        ));
    }
    if requalifications.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::SystemRequalification,
        ));
    }
    if restarts.is_empty() {
        diagnostics.push(DfrmDiagnostic::MissingAction(
            MaintenanceAction::RestartSystem,
        ));
    }

    // Restart is terminal: every other declared operation must be its ancestor.
    for restart in &restarts {
        for operation in plan.operations.iter().filter(|operation| {
            operation.operation_id != restart.operation_id
                && operation.action != MaintenanceAction::Other
        }) {
            require_all(
                restart,
                &[operation],
                DfrmDiagnostic::RestartBeforeRequalification,
                diagnostics,
            );
        }
    }
}

fn validate_schedules(plan: &RemoteMaintenancePlan, diagnostics: &mut Vec<DfrmDiagnostic>) {
    for (index, first) in plan.operations.iter().enumerate() {
        for second in plan.operations.iter().skip(index + 1) {
            let overlaps =
                first.start_second < second.end_second && second.start_second < first.end_second;
            if !overlaps {
                continue;
            }
            if first.resource_id == second.resource_id {
                diagnostics.push(DfrmDiagnostic::ResourceScheduleCollision);
            }
            if first.tool_id == second.tool_id {
                diagnostics.push(DfrmDiagnostic::ToolScheduleCollision);
            }
        }
    }
    let operation_map: BTreeMap<&str, &MaintenanceOperation> = plan
        .operations
        .iter()
        .map(|operation| (operation.operation_id.as_str(), operation))
        .collect();
    for operation in &plan.operations {
        for predecessor_id in &operation.predecessor_ids {
            if let Some(predecessor) = operation_map.get(predecessor_id.as_str())
                && operation.start_second < predecessor.end_second
            {
                diagnostics.push(DfrmDiagnostic::DependencyScheduleViolation);
            }
        }
    }
}

fn require_ancestor(
    successor_id: &str,
    predecessor_id: &str,
    ancestors: &BTreeMap<String, BTreeSet<String>>,
    diagnostics: &mut Vec<DfrmDiagnostic>,
    diagnostic: DfrmDiagnostic,
) {
    if !ancestors
        .get(successor_id)
        .is_some_and(|set| set.contains(predecessor_id))
    {
        diagnostics.push(diagnostic);
    }
}

fn topological_order(operations: &[MaintenanceOperation]) -> Option<Vec<String>> {
    let mut in_degree: BTreeMap<String, usize> = operations
        .iter()
        .map(|operation| {
            (
                operation.operation_id.clone(),
                operation.predecessor_ids.len(),
            )
        })
        .collect();
    let mut downstream: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for operation in operations {
        for predecessor in &operation.predecessor_ids {
            downstream
                .entry(predecessor.clone())
                .or_default()
                .push(operation.operation_id.clone());
        }
    }
    let mut ready: BTreeSet<String> = in_degree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut result = Vec::with_capacity(operations.len());
    while let Some(id) = ready.pop_first() {
        result.push(id.clone());
        for successor in downstream.get(&id).into_iter().flatten() {
            if let Some(degree) = in_degree.get_mut(successor) {
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    ready.insert(successor.clone());
                }
            }
        }
    }
    (result.len() == operations.len()).then_some(result)
}

fn ancestor_map(
    operations: &[MaintenanceOperation],
    topological: &[String],
) -> BTreeMap<String, BTreeSet<String>> {
    let by_id: BTreeMap<&str, &MaintenanceOperation> = operations
        .iter()
        .map(|operation| (operation.operation_id.as_str(), operation))
        .collect();
    let mut ancestors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for operation_id in topological {
        let mut set = BTreeSet::new();
        if let Some(operation) = by_id.get(operation_id.as_str()) {
            for predecessor_id in &operation.predecessor_ids {
                set.insert(predecessor_id.clone());
                if let Some(parent_ancestors) = ancestors.get(predecessor_id) {
                    set.extend(parent_ancestors.iter().cloned());
                }
            }
        }
        ancestors.insert(operation_id.clone(), set);
    }
    ancestors
}

fn canonical_operation_receipts(plan: &RemoteMaintenancePlan) -> Vec<OperationEvidenceReceipt> {
    let mut operations = plan.operations.clone();
    operations.sort_by(|left, right| {
        (
            left.start_second,
            left.end_second,
            left.operation_id.as_str(),
        )
            .cmp(&(
                right.start_second,
                right.end_second,
                right.operation_id.as_str(),
            ))
    });
    operations
        .into_iter()
        .map(|operation| OperationEvidenceReceipt {
            operation_id: operation.operation_id,
            action: operation.action,
            resource_id: operation.resource_id,
            tool_id: operation.tool_id,
            start_second: operation.start_second,
            end_second: operation.end_second,
            duration_low_seconds: operation.minimum_duration_seconds,
            duration_nominal_seconds: operation.nominal_duration_seconds,
            duration_high_seconds: operation.maximum_duration_seconds,
            evidence_ref: operation.evidence_ref,
            evidence_sha256: operation.evidence_sha256,
            evidence_class: operation.evidence_class,
            clearance_reachability_ref: operation.clearance_reachability_ref,
            clearance_reachability_sha256: operation.clearance_reachability_sha256,
            kinematic_model_ref: operation.kinematic_model_ref,
            validity_domain_ref: operation.validity_domain_ref,
            failure_recovery_ref: operation.failure_recovery_ref,
        })
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn operation(
        id: &str,
        action: MaintenanceAction,
        interface_id: Option<&str>,
        predecessors: &[&str],
        start: u64,
        end: u64,
    ) -> MaintenanceOperation {
        MaintenanceOperation {
            operation_id: id.to_string(),
            action,
            interface_id: interface_id.map(str::to_string),
            predecessor_ids: predecessors
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            required_capabilities: vec!["remote-maintenance".to_string()],
            resource_id: "robot-a".to_string(),
            tool_id: "tool-a".to_string(),
            start_second: start,
            end_second: end,
            minimum_duration_seconds: end - start,
            nominal_duration_seconds: end - start,
            maximum_duration_seconds: end - start,
            procedure_step_ref: format!("step-{id}"),
            evidence_ref: format!("evidence-{id}"),
            evidence_sha256: sha('a'),
            evidence_class: MaintenanceEvidenceClass::Modelled,
            clearance_reachability_ref: format!("reachability-{id}"),
            clearance_reachability_sha256: sha('b'),
            kinematic_model_ref: "kinematic-model-v1".to_string(),
            validity_domain_ref: "mockup-envelope-v1".to_string(),
            failure_recovery_ref: format!("recovery-{id}"),
        }
    }

    fn interface(
        id: &str,
        mate: &str,
        is_assembly_side: bool,
        family: InterfaceFamily,
        compatibility_key: &str,
    ) -> ServiceInterface {
        ServiceInterface {
            interface_id: id.to_string(),
            mate_interface_id: mate.to_string(),
            is_assembly_side,
            family,
            compatibility_key: compatibility_key.to_string(),
            access_side_ref: format!("access-{id}"),
            remote_actuation_capability: "remote-maintenance".to_string(),
            remote_actuation_proof_ref: format!("actuation-proof-{id}"),
        }
    }

    fn base_plan() -> RemoteMaintenancePlan {
        let mut operations = vec![
            operation(
                "isolate",
                MaintenanceAction::IsolateSystem,
                None,
                &[],
                0,
                10,
            ),
            operation(
                "safe",
                MaintenanceAction::ProveSafeState,
                None,
                &["isolate"],
                10,
                20,
            ),
            operation(
                "prep-coolant",
                MaintenanceAction::PrepareServiceBoundary,
                None,
                &["safe"],
                20,
                30,
            ),
            operation(
                "disconnect-coolant",
                MaintenanceAction::DisconnectInterface,
                Some("coolant-a"),
                &["safe", "prep-coolant"],
                30,
                40,
            ),
            operation(
                "disconnect-power",
                MaintenanceAction::DisconnectInterface,
                Some("power-a"),
                &["safe"],
                40,
                50,
            ),
            operation(
                "release",
                MaintenanceAction::ReleaseRestraints,
                None,
                &["disconnect-coolant", "disconnect-power"],
                50,
                60,
            ),
            operation(
                "extract",
                MaintenanceAction::ExtractModule,
                None,
                &["release"],
                60,
                80,
            ),
            operation(
                "transfer",
                MaintenanceAction::TransferToShieldedCask,
                None,
                &["extract"],
                80,
                100,
            ),
            operation(
                "inspect",
                MaintenanceAction::HotCellInspectRepair,
                None,
                &["transfer"],
                100,
                150,
            ),
            operation(
                "install",
                MaintenanceAction::InstallModule,
                None,
                &["inspect"],
                150,
                170,
            ),
            operation(
                "align",
                MaintenanceAction::AlignMetrology,
                None,
                &["install"],
                170,
                180,
            ),
            operation(
                "secure",
                MaintenanceAction::SecureRestraints,
                None,
                &["align"],
                180,
                190,
            ),
            operation(
                "reconnect-coolant",
                MaintenanceAction::ReconnectInterface,
                Some("coolant-a"),
                &["secure"],
                190,
                200,
            ),
            operation(
                "reconnect-power",
                MaintenanceAction::ReconnectInterface,
                Some("power-a"),
                &["secure"],
                200,
                210,
            ),
            operation(
                "test-coolant",
                MaintenanceAction::IntegrityTest(IntegrityTestKind::PressureLeak),
                Some("coolant-a"),
                &["reconnect-coolant"],
                210,
                220,
            ),
            operation(
                "test-power",
                MaintenanceAction::IntegrityTest(IntegrityTestKind::ElectricalContinuity),
                Some("power-a"),
                &["reconnect-power"],
                220,
                230,
            ),
            operation(
                "requalify",
                MaintenanceAction::SystemRequalification,
                None,
                &["test-coolant", "test-power"],
                230,
                240,
            ),
            operation(
                "restart",
                MaintenanceAction::RestartSystem,
                None,
                &["requalify"],
                240,
                250,
            ),
        ];
        operations.reverse();

        RemoteMaintenancePlan {
            campaign_id: "synthetic-blanket-replacement-v1".to_string(),
            procedure_revision: "procedure-r1".to_string(),
            service_boundary_ref: "plant-net-electric-v1".to_string(),
            outage_start_second: 0,
            outage_end_second: 300,
            assembly: ReplaceableAssembly {
                assembly_id: "blanket-module-7".to_string(),
                revision: "module-r4".to_string(),
                mass_kg: 1200.0,
                centre_of_gravity_x_m: 0.1,
                centre_of_gravity_y_m: 0.1,
                centre_of_gravity_z_m: 0.1,
                removal_envelope_x_m: 2.0,
                removal_envelope_y_m: 1.5,
                removal_envelope_z_m: 1.2,
                handling_datum_ref: "handling-datum-v1".to_string(),
                removal_envelope_ref: "removal-envelope-v1".to_string(),
                shielded_cask_envelope_ref: "shielded-cask-route-v1".to_string(),
                hot_cell_interface_ref: "hot-cell-interface-v1".to_string(),
                hazard_profile_ref: "activation-tritium-profile-v1".to_string(),
                isolation_plan_ref: "coolant-tritium-isolation-plan-v1".to_string(),
            },
            interfaces: vec![
                interface(
                    "coolant-a",
                    "coolant-b",
                    true,
                    InterfaceFamily::Coolant,
                    "DN200-PN40",
                ),
                interface(
                    "coolant-b",
                    "coolant-a",
                    false,
                    InterfaceFamily::Coolant,
                    "DN200-PN40",
                ),
                interface(
                    "power-a",
                    "power-b",
                    true,
                    InterfaceFamily::Electrical,
                    "HV-remote-plug",
                ),
                interface(
                    "power-b",
                    "power-a",
                    false,
                    InterfaceFamily::Electrical,
                    "HV-remote-plug",
                ),
            ],
            interface_bindings: vec![
                InterfaceProcedureBinding {
                    interface_id: "coolant-a".to_string(),
                    isolation_operation_id: "isolate".to_string(),
                    prove_safe_operation_id: "safe".to_string(),
                    preparation_operation_id: Some("prep-coolant".to_string()),
                    disconnect_operation_id: "disconnect-coolant".to_string(),
                    reconnect_operation_id: "reconnect-coolant".to_string(),
                    integrity_test_operation_id: "test-coolant".to_string(),
                },
                InterfaceProcedureBinding {
                    interface_id: "power-a".to_string(),
                    isolation_operation_id: "isolate".to_string(),
                    prove_safe_operation_id: "safe".to_string(),
                    preparation_operation_id: None,
                    disconnect_operation_id: "disconnect-power".to_string(),
                    reconnect_operation_id: "reconnect-power".to_string(),
                    integrity_test_operation_id: "test-power".to_string(),
                },
            ],
            resources: vec![MaintenanceResource {
                resource_id: "robot-a".to_string(),
                capabilities: vec!["remote-maintenance".to_string()],
            }],
            tools: vec![MaintenanceTool {
                tool_id: "tool-a".to_string(),
                capabilities: vec!["remote-maintenance".to_string()],
                maximum_payload_kg: 2000.0,
                maximum_centre_of_gravity_offset_m: 0.5,
            }],
            operations,
        }
    }

    #[test]
    fn complete_declared_procedure_is_accepted_with_bounded_claim() {
        let report = compile_remote_maintenance_plan(&base_plan());
        assert_eq!(
            report.disposition,
            DfrmDisposition::ValidDeclaredProcedure,
            "unexpected DFRM diagnostics: {:?}",
            report.diagnostics
        );
        assert_eq!(report.operation_count, 18);
        assert_eq!(report.interface_count, 4);
        assert_eq!(report.scheduled_outage_seconds, Some(300));
        assert_eq!(report.critical_path_high_seconds, Some(250));
        assert_eq!(
            report.claim_ceiling,
            "declared maintenance-procedure completeness only"
        );
    }

    #[test]
    fn plant_side_mating_connector_does_not_require_duplicate_procedure_binding() {
        let mut plan = base_plan();
        plan.interface_bindings.push(InterfaceProcedureBinding {
            interface_id: "coolant-b".to_string(),
            isolation_operation_id: "isolate".to_string(),
            prove_safe_operation_id: "safe".to_string(),
            preparation_operation_id: Some("prep-coolant".to_string()),
            disconnect_operation_id: "disconnect-coolant".to_string(),
            reconnect_operation_id: "reconnect-coolant".to_string(),
            integrity_test_operation_id: "test-coolant".to_string(),
        });
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::UnknownBindingInterface)
        );
    }

    #[test]
    fn coolant_disconnect_requires_prepare_after_prove_safe() {
        let mut plan = base_plan();
        plan.interface_bindings[0].preparation_operation_id = None;
        let report = compile_remote_maintenance_plan(&plan);
        assert_eq!(report.disposition, DfrmDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::MissingRequiredPreparation)
        );
    }

    #[test]
    fn restraint_release_before_every_disconnect_is_refused() {
        let mut plan = base_plan();
        let release = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "release")
            .unwrap();
        release.predecessor_ids = vec!["disconnect-coolant".to_string()];
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::RestraintReleaseBeforeDisconnect)
        );
    }

    #[test]
    fn extraction_before_restraint_release_is_refused() {
        let mut plan = base_plan();
        let extract = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "extract")
            .unwrap();
        extract.predecessor_ids.clear();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::ExtractionBeforeRestraintRelease)
        );
    }

    #[test]
    fn wrong_mate_compatibility_is_refused() {
        let mut plan = base_plan();
        plan.interfaces[1].compatibility_key = "wrong-size".to_string();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::IncompatibleInterfaceMate)
        );
    }

    #[test]
    fn resource_overlap_is_refused() {
        let mut plan = base_plan();
        let operation = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "safe")
            .unwrap();
        operation.start_second = 5;
        operation.end_second = 15;
        operation.minimum_duration_seconds = 10;
        operation.nominal_duration_seconds = 10;
        operation.maximum_duration_seconds = 10;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::ResourceScheduleCollision)
        );
    }

    #[test]
    fn tool_overlap_is_refused_even_when_resources_differ() {
        let mut plan = base_plan();
        plan.resources.push(MaintenanceResource {
            resource_id: "robot-b".to_string(),
            capabilities: vec!["remote-maintenance".to_string()],
        });
        let operation = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "safe")
            .unwrap();
        operation.resource_id = "robot-b".to_string();
        operation.start_second = 5;
        operation.end_second = 15;
        operation.minimum_duration_seconds = 10;
        operation.nominal_duration_seconds = 10;
        operation.maximum_duration_seconds = 10;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::ToolScheduleCollision)
        );
    }

    #[test]
    fn incompatible_tool_capability_is_refused() {
        let mut plan = base_plan();
        plan.tools[0].capabilities.clear();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::ToolCapabilityMissing)
        );
    }

    #[test]
    fn insufficient_tool_payload_or_centre_of_gravity_limit_is_refused() {
        let mut plan = base_plan();
        plan.tools[0].maximum_payload_kg = 1000.0;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::PayloadLimitExceeded)
        );

        let mut plan = base_plan();
        plan.tools[0].maximum_centre_of_gravity_offset_m = 0.1;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::CentreOfGravityLimitExceeded)
        );
    }

    #[test]
    fn missing_reachability_proof_or_stale_digest_is_refused() {
        let mut plan = base_plan();
        plan.operations[0].clearance_reachability_ref.clear();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::MissingClearanceReachabilityProof)
        );

        let mut plan = base_plan();
        plan.operations[0].clearance_reachability_sha256 = "bad".to_string();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::InvalidClearanceReachabilityDigest)
        );
    }

    #[test]
    fn restart_must_follow_requalification_and_integrity_tests() {
        let mut plan = base_plan();
        let restart = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "restart")
            .unwrap();
        restart.predecessor_ids.clear();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::RestartBeforeRequalification)
        );
    }

    #[test]
    fn cycle_is_refused() {
        let mut plan = base_plan();
        let isolate = plan
            .operations
            .iter_mut()
            .find(|item| item.operation_id == "isolate")
            .unwrap();
        isolate.predecessor_ids = vec!["restart".to_string()];
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::DependencyCycle)
        );
    }

    #[test]
    fn durations_must_be_ordered_and_schedule_within_declared_range() {
        let mut plan = base_plan();
        plan.operations[0].minimum_duration_seconds = 100;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::InvalidOperationDuration)
        );

        let mut plan = base_plan();
        plan.operations[0].maximum_duration_seconds = 1;
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::ScheduleDurationOutsideDeclaredRange)
        );
    }

    #[test]
    fn duplicate_predecessor_or_missing_failure_recovery_is_refused() {
        let mut plan = base_plan();
        plan.operations[0].predecessor_ids = vec!["safe".to_string(), "safe".to_string()];
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::DuplicatePredecessor)
        );

        let mut plan = base_plan();
        plan.operations[0].failure_recovery_ref.clear();
        let report = compile_remote_maintenance_plan(&plan);
        assert!(
            report
                .diagnostics
                .contains(&DfrmDiagnostic::MissingFailureRecoveryReference)
        );
    }

    #[test]
    fn event_input_order_does_not_change_canonical_operation_receipts() {
        let plan = base_plan();
        let mut reordered = base_plan();
        reordered.operations.reverse();
        let first = compile_remote_maintenance_plan(&plan);
        let second = compile_remote_maintenance_plan(&reordered);
        assert_eq!(first.operation_receipts, second.operation_receipts);
        assert_eq!(first.disposition, second.disposition);
    }

    #[test]
    fn modelled_evidence_is_not_promoted_to_mockup_validated() {
        let report = compile_remote_maintenance_plan(&base_plan());
        assert_eq!(report.evidence_class_counts.len(), 1);
        assert_eq!(
            report.evidence_class_counts[0].evidence_class,
            MaintenanceEvidenceClass::Modelled
        );
        assert_eq!(report.evidence_class_counts[0].operation_count, 18);
    }
}
