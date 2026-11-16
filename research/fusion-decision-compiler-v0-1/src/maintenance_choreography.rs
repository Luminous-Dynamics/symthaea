//! Resource-constrained remote maintenance choreography.
//!
//! This module schedules a frozen precedence DAG against renewable resources
//! such as remote-handling robots, access ports, transfer casks, hot-cell bays,
//! crews and metrology systems. It is deterministic v0.1: duration uncertainty,
//! failure probability and adaptive rescheduling are deliberately out of scope.

use std::collections::{BTreeMap, BTreeSet};

use super::lifecycle_service_ledger::{
    AvailabilityEvent, LifecycleEvidenceClass, OutageCategory, ServiceImpact,
};

pub const MAINTENANCE_SCHEDULER_VERSION: &str = "fusion-maintenance-choreography/0.1";
pub const EXACT_ORACLE_MAX_TASKS: usize = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MaintenancePhase {
    IsolateAndVerify,
    DisconnectInterfaces,
    RemoveActivatedComponent,
    TransferToHotCell,
    InspectOrRepair,
    PrepareReplacement,
    TransferReplacement,
    InsertAndAlign,
    ReconnectAndTest,
    RequalifyAndRestart,
    Other,
}

impl MaintenancePhase {
    pub const fn id(self) -> &'static str {
        match self {
            Self::IsolateAndVerify => "isolate-and-verify",
            Self::DisconnectInterfaces => "disconnect-interfaces",
            Self::RemoveActivatedComponent => "remove-activated-component",
            Self::TransferToHotCell => "transfer-to-hot-cell",
            Self::InspectOrRepair => "inspect-or-repair",
            Self::PrepareReplacement => "prepare-replacement",
            Self::TransferReplacement => "transfer-replacement",
            Self::InsertAndAlign => "insert-and-align",
            Self::ReconnectAndTest => "reconnect-and-test",
            Self::RequalifyAndRestart => "requalify-and-restart",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenewableResource {
    pub resource_id: String,
    pub capacity: u32,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceDemand {
    pub resource_id: String,
    pub units: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessAssessmentDisposition {
    AnalyticFixturePass,
    SimulationFeasibleForMockup,
    MockupDemonstrated,
    IndependentlyVerified,
    PhysicalQualification,
    Blocked,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessCertificateBinding {
    pub certificate_ref: String,
    pub certificate_sha256: String,
    pub bound_candidate_revision: String,
    pub bound_geometry_sha256: String,
    pub bound_task_definition_ref: String,
    pub bound_task_definition_sha256: String,
    pub robot_model_sha256: String,
    pub tool_model_sha256: String,
    pub clearance_profile_ref: String,
    pub disposition: AccessAssessmentDisposition,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceTask {
    pub task_id: String,
    pub phase: MaintenancePhase,
    pub subsystem_id: String,
    pub failure_mode_id: String,
    pub duration_seconds: u64,
    pub predecessor_ids: Vec<String>,
    pub resource_demands: Vec<ResourceDemand>,
    pub outage_category: OutageCategory,
    pub service_impact: ServiceImpact,
    pub task_definition_ref: String,
    pub task_definition_sha256: String,
    pub access_certificate: AccessCertificateBinding,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceSchedulePlan {
    pub campaign_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub campaign_start_second: u64,
    pub priority_order: Vec<String>,
    pub resources: Vec<RenewableResource>,
    pub tasks: Vec<MaintenanceTask>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledMaintenanceTask {
    pub task_id: String,
    pub phase: MaintenancePhase,
    pub subsystem_id: String,
    pub failure_mode_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub resource_demands: Vec<ResourceDemand>,
    pub outage_category: OutageCategory,
    pub service_impact: ServiceImpact,
    pub task_definition_ref: String,
    pub task_definition_sha256: String,
    pub access_certificate: AccessCertificateBinding,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduleDiagnostic {
    EmptyCampaignId,
    DuplicateTaskId,
    EmptyTaskIdentity,
    EmptySubsystemIdentity,
    EmptyFailureModeIdentity,
    EmptyTaskEvidence,
    InvalidTaskDefinitionDigest,
    EmptyCandidateRevision,
    InvalidGeometryDigest,
    MissingAccessCertificate,
    InvalidAccessCertificateDigest,
    AccessCandidateRevisionMismatch,
    AccessGeometryDigestMismatch,
    AccessTaskDefinitionMismatch,
    InvalidRobotModelDigest,
    InvalidToolModelDigest,
    MissingClearanceProfile,
    AccessAssessmentBlocked,
    AccessAssessmentUnknown,
    InsufficientAccessEvidenceClass,
    ZeroTaskDuration,
    DuplicatePredecessor,
    MissingPredecessor,
    SelfPredecessor,
    PrecedenceCycle,
    DuplicateResourceId,
    EmptyResourceIdentity,
    ZeroResourceCapacity,
    EmptyResourceEvidence,
    UnknownResource,
    ZeroResourceDemand,
    DemandExceedsCapacity,
    DuplicateResourceDemand,
    InvalidPriorityOrder,
    ScheduleTaskCountMismatch,
    ScheduledTaskIdentityMismatch,
    PrecedenceViolation,
    ResourceCapacityViolation,
    ScheduleDurationMismatch,
    ScheduleStartsBeforeCampaign,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleError {
    pub diagnostic: ScheduleDiagnostic,
    pub subject_id: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledTaskReceipt {
    pub task_id: String,
    pub phase: MaintenancePhase,
    pub start_second: u64,
    pub end_second: u64,
    pub resource_demands: Vec<ResourceDemand>,
    pub task_definition_ref: String,
    pub task_definition_sha256: String,
    pub access_certificate: AccessCertificateBinding,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceScheduleReport {
    pub contract_version: &'static str,
    pub campaign_id: String,
    pub scheduled_tasks: Vec<ScheduledMaintenanceTask>,
    pub task_receipts: Vec<ScheduledTaskReceipt>,
    pub makespan_seconds: u64,
    pub serial_baseline_makespan_seconds: u64,
    pub critical_path_lower_bound_seconds: u64,
    pub parallel_speedup_over_serial: (u64, u64),
    /// Plant-outage window from the first to the last task requiring full
    /// outage. Actual outage intervals are passed to the lifecycle ledger.
    pub full_outage_window: Option<(u64, u64)>,
    pub full_outage_window_seconds: u64,
    pub lifecycle_events: Vec<AvailabilityEvent>,
    pub claim_ceiling: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactOracleError {
    TooManyTasks { maximum: usize, actual: usize },
    InvalidPlan(Vec<ScheduleError>),
}

#[derive(Clone, Debug)]
struct PlannedTask {
    task: MaintenanceTask,
    start_second: u64,
    end_second: u64,
}

fn error(
    diagnostic: ScheduleDiagnostic,
    subject_id: impl Into<String>,
    detail: impl Into<String>,
) -> ScheduleError {
    ScheduleError {
        diagnostic,
        subject_id: subject_id.into(),
        detail: detail.into(),
    }
}

pub fn generate_maintenance_schedule(
    plan: &MaintenanceSchedulePlan,
) -> Result<MaintenanceScheduleReport, Vec<ScheduleError>> {
    let validation = validate_plan(plan);
    if !validation.is_empty() {
        return Err(validation);
    }
    let order = topological_order(plan).map_err(|diagnostic| {
        vec![error(
            diagnostic,
            &plan.campaign_id,
            "task graph is not schedulable",
        )]
    })?;
    let mut scheduled: Vec<PlannedTask> = Vec::with_capacity(plan.tasks.len());
    let resources: BTreeMap<&str, &RenewableResource> = plan
        .resources
        .iter()
        .map(|resource| (resource.resource_id.as_str(), resource))
        .collect();

    for task_id in order {
        let task = task_by_id(plan, &task_id).expect("validated task ID must exist");
        let predecessor_finish = task
            .predecessor_ids
            .iter()
            .filter_map(|predecessor| {
                scheduled
                    .iter()
                    .find(|entry| entry.task.task_id == *predecessor)
                    .map(|entry| entry.end_second)
            })
            .max()
            .unwrap_or(plan.campaign_start_second);
        let start = earliest_resource_start(task, predecessor_finish, &scheduled, &resources)
            .ok_or_else(|| {
                vec![error(
                    ScheduleDiagnostic::ResourceCapacityViolation,
                    &task.task_id,
                    "no feasible resource allocation found",
                )]
            })?;
        let end = start.checked_add(task.duration_seconds).ok_or_else(|| {
            vec![error(
                ScheduleDiagnostic::ScheduleDurationMismatch,
                &task.task_id,
                "task end time overflows u64",
            )]
        })?;
        scheduled.push(PlannedTask {
            task: task.clone(),
            start_second: start,
            end_second: end,
        });
    }

    let baseline = serial_baseline_makespan(plan).map_err(|diagnostic| {
        vec![error(
            diagnostic,
            &plan.campaign_id,
            "unable to construct serial baseline",
        )]
    })?;
    let critical_path = critical_path_lower_bound(plan).map_err(|diagnostic| {
        vec![error(
            diagnostic,
            &plan.campaign_id,
            "unable to compute critical path",
        )]
    })?;
    let makespan = scheduled
        .iter()
        .map(|entry| entry.end_second)
        .max()
        .unwrap_or(plan.campaign_start_second)
        .saturating_sub(plan.campaign_start_second);

    scheduled.sort_by(|left, right| {
        (
            left.start_second,
            left.end_second,
            left.task.task_id.as_str(),
        )
            .cmp(&(
                right.start_second,
                right.end_second,
                right.task.task_id.as_str(),
            ))
    });

    let scheduled_tasks: Vec<ScheduledMaintenanceTask> = scheduled
        .iter()
        .map(|entry| ScheduledMaintenanceTask {
            task_id: entry.task.task_id.clone(),
            phase: entry.task.phase,
            subsystem_id: entry.task.subsystem_id.clone(),
            failure_mode_id: entry.task.failure_mode_id.clone(),
            start_second: entry.start_second,
            end_second: entry.end_second,
            resource_demands: canonical_demands(&entry.task.resource_demands),
            outage_category: entry.task.outage_category,
            service_impact: entry.task.service_impact,
            task_definition_ref: entry.task.task_definition_ref.clone(),
            task_definition_sha256: entry.task.task_definition_sha256.clone(),
            access_certificate: entry.task.access_certificate.clone(),
            evidence_ref: entry.task.evidence_ref.clone(),
            evidence_class: entry.task.evidence_class,
        })
        .collect();

    let full_outage_window = full_outage_window(&scheduled_tasks);
    let full_outage_window_seconds = full_outage_window
        .map(|(start, end)| end.saturating_sub(start))
        .unwrap_or(0);
    let lifecycle_events = scheduled_tasks.iter().map(to_lifecycle_event).collect();
    let task_receipts = scheduled_tasks
        .iter()
        .map(|task| ScheduledTaskReceipt {
            task_id: task.task_id.clone(),
            phase: task.phase,
            start_second: task.start_second,
            end_second: task.end_second,
            resource_demands: task.resource_demands.clone(),
            task_definition_ref: task.task_definition_ref.clone(),
            task_definition_sha256: task.task_definition_sha256.clone(),
            access_certificate: task.access_certificate.clone(),
            evidence_ref: task.evidence_ref.clone(),
            evidence_class: task.evidence_class,
        })
        .collect();

    Ok(MaintenanceScheduleReport {
        contract_version: MAINTENANCE_SCHEDULER_VERSION,
        campaign_id: plan.campaign_id.clone(),
        scheduled_tasks,
        task_receipts,
        makespan_seconds: makespan,
        serial_baseline_makespan_seconds: baseline,
        critical_path_lower_bound_seconds: critical_path,
        parallel_speedup_over_serial: reduce_ratio(baseline, makespan),
        full_outage_window,
        full_outage_window_seconds,
        lifecycle_events,
        claim_ceiling: "deterministic resource/precedence schedule only",
    })
}

/// Independently validate a schedule against the frozen task graph and
/// resource capacities. This is intentionally separate from generation.
pub fn validate_maintenance_schedule(
    plan: &MaintenanceSchedulePlan,
    report: &MaintenanceScheduleReport,
) -> Vec<ScheduleError> {
    let mut errors = validate_plan(plan);
    if report.campaign_id != plan.campaign_id {
        errors.push(error(
            ScheduleDiagnostic::EmptyCampaignId,
            &report.campaign_id,
            "schedule campaign identity does not match the plan",
        ));
    }
    if report.scheduled_tasks.len() != plan.tasks.len() {
        errors.push(error(
            ScheduleDiagnostic::ScheduleTaskCountMismatch,
            &report.campaign_id,
            "schedule does not contain exactly one entry per task",
        ));
    }

    let plan_tasks: BTreeMap<&str, &MaintenanceTask> = plan
        .tasks
        .iter()
        .map(|task| (task.task_id.as_str(), task))
        .collect();
    let mut schedule_by_id = BTreeMap::new();
    for entry in &report.scheduled_tasks {
        if schedule_by_id
            .insert(entry.task_id.as_str(), entry)
            .is_some()
        {
            errors.push(error(
                ScheduleDiagnostic::ScheduledTaskIdentityMismatch,
                &entry.task_id,
                "task appears more than once in schedule",
            ));
        }
        let Some(task) = plan_tasks.get(entry.task_id.as_str()) else {
            errors.push(error(
                ScheduleDiagnostic::ScheduledTaskIdentityMismatch,
                &entry.task_id,
                "scheduled task is not in the frozen plan",
            ));
            continue;
        };
        if entry.start_second < plan.campaign_start_second {
            errors.push(error(
                ScheduleDiagnostic::ScheduleStartsBeforeCampaign,
                &entry.task_id,
                "scheduled interval begins before campaign start",
            ));
        }
        if entry.end_second.saturating_sub(entry.start_second) != task.duration_seconds {
            errors.push(error(
                ScheduleDiagnostic::ScheduleDurationMismatch,
                &entry.task_id,
                "scheduled duration differs from frozen task duration",
            ));
        }
        if entry.phase != task.phase
            || entry.subsystem_id != task.subsystem_id
            || entry.failure_mode_id != task.failure_mode_id
            || entry.outage_category != task.outage_category
            || entry.service_impact != task.service_impact
            || entry.task_definition_ref != task.task_definition_ref
            || entry.task_definition_sha256 != task.task_definition_sha256
            || entry.access_certificate != task.access_certificate
            || entry.evidence_ref != task.evidence_ref
            || entry.evidence_class != task.evidence_class
            || entry.resource_demands != canonical_demands(&task.resource_demands)
        {
            errors.push(error(
                ScheduleDiagnostic::ScheduledTaskIdentityMismatch,
                &entry.task_id,
                "scheduled task metadata differs from the frozen task definition",
            ));
        }
    }

    let resources: BTreeMap<&str, &RenewableResource> = plan
        .resources
        .iter()
        .map(|resource| (resource.resource_id.as_str(), resource))
        .collect();
    for entry in &report.scheduled_tasks {
        let Some(task) = plan_tasks.get(entry.task_id.as_str()) else {
            continue;
        };
        for predecessor_id in &task.predecessor_ids {
            match schedule_by_id.get(predecessor_id.as_str()) {
                Some(predecessor) if predecessor.end_second <= entry.start_second => {}
                _ => errors.push(error(
                    ScheduleDiagnostic::PrecedenceViolation,
                    &entry.task_id,
                    format!("predecessor {predecessor_id} does not finish before task starts"),
                )),
            }
        }
        if resource_capacity_exceeded(entry, &report.scheduled_tasks, &resources) {
            errors.push(error(
                ScheduleDiagnostic::ResourceCapacityViolation,
                &entry.task_id,
                "resource pool capacity is exceeded during this interval",
            ));
        }
    }

    let expected_makespan = report
        .scheduled_tasks
        .iter()
        .map(|task| task.end_second)
        .max()
        .unwrap_or(plan.campaign_start_second)
        .saturating_sub(plan.campaign_start_second);
    if report.makespan_seconds != expected_makespan {
        errors.push(error(
            ScheduleDiagnostic::ScheduleDurationMismatch,
            &report.campaign_id,
            "reported makespan does not match scheduled task intervals",
        ));
    }
    errors
}

/// Return the exact minimum makespan over all active serial schedule
/// generation orders for small frozen task graphs. This is a benchmark oracle,
/// not the production scheduler; factorial growth is capped at nine tasks.
pub fn exact_small_instance_makespan(
    plan: &MaintenanceSchedulePlan,
) -> Result<u64, ExactOracleError> {
    if plan.tasks.len() > EXACT_ORACLE_MAX_TASKS {
        return Err(ExactOracleError::TooManyTasks {
            maximum: EXACT_ORACLE_MAX_TASKS,
            actual: plan.tasks.len(),
        });
    }
    let validation = validate_plan(plan);
    if !validation.is_empty() {
        return Err(ExactOracleError::InvalidPlan(validation));
    }

    let resources: BTreeMap<&str, &RenewableResource> = plan
        .resources
        .iter()
        .map(|resource| (resource.resource_id.as_str(), resource))
        .collect();
    let mut best = u64::MAX;
    let mut scheduled = Vec::with_capacity(plan.tasks.len());
    let mut used = BTreeSet::new();
    enumerate_active_orders(plan, &resources, &mut scheduled, &mut used, &mut best);
    Ok(best.saturating_sub(plan.campaign_start_second))
}

fn enumerate_active_orders(
    plan: &MaintenanceSchedulePlan,
    resources: &BTreeMap<&str, &RenewableResource>,
    scheduled: &mut Vec<PlannedTask>,
    used: &mut BTreeSet<String>,
    best_makespan: &mut u64,
) {
    if scheduled.len() == plan.tasks.len() {
        let makespan = scheduled
            .iter()
            .map(|entry| entry.end_second)
            .max()
            .unwrap_or(plan.campaign_start_second);
        *best_makespan = (*best_makespan).min(makespan);
        return;
    }

    let mut ready: Vec<&MaintenanceTask> = plan
        .tasks
        .iter()
        .filter(|task| {
            !used.contains(&task.task_id)
                && task
                    .predecessor_ids
                    .iter()
                    .all(|predecessor| used.contains(predecessor))
        })
        .collect();
    ready.sort_by(|left, right| left.task_id.cmp(&right.task_id));

    for task in ready {
        let predecessor_finish = task
            .predecessor_ids
            .iter()
            .filter_map(|predecessor| {
                scheduled
                    .iter()
                    .find(|entry| entry.task.task_id == *predecessor)
                    .map(|entry| entry.end_second)
            })
            .max()
            .unwrap_or(plan.campaign_start_second);
        let Some(start) = earliest_resource_start(task, predecessor_finish, scheduled, resources)
        else {
            continue;
        };
        let Some(end) = start.checked_add(task.duration_seconds) else {
            continue;
        };
        used.insert(task.task_id.clone());
        scheduled.push(PlannedTask {
            task: task.clone(),
            start_second: start,
            end_second: end,
        });
        enumerate_active_orders(plan, resources, scheduled, used, best_makespan);
        scheduled.pop();
        used.remove(&task.task_id);
    }
}

fn validate_plan(plan: &MaintenanceSchedulePlan) -> Vec<ScheduleError> {
    let mut errors = Vec::new();
    if plan.campaign_id.trim().is_empty() {
        errors.push(error(
            ScheduleDiagnostic::EmptyCampaignId,
            "",
            "campaign identity is empty",
        ));
    }
    if plan.candidate_revision.trim().is_empty() {
        errors.push(error(
            ScheduleDiagnostic::EmptyCandidateRevision,
            &plan.campaign_id,
            "candidate revision is required for physical-access binding",
        ));
    }
    if !is_sha256(&plan.geometry_sha256) {
        errors.push(error(
            ScheduleDiagnostic::InvalidGeometryDigest,
            &plan.campaign_id,
            "campaign geometry SHA-256 is malformed",
        ));
    }

    let mut resources = BTreeMap::new();
    for resource in &plan.resources {
        if resource.resource_id.trim().is_empty() {
            errors.push(error(
                ScheduleDiagnostic::EmptyResourceIdentity,
                "",
                "resource identity is empty",
            ));
        }
        if resource.capacity == 0 {
            errors.push(error(
                ScheduleDiagnostic::ZeroResourceCapacity,
                &resource.resource_id,
                "resource capacity must be positive",
            ));
        }
        if resource.evidence_ref.trim().is_empty() {
            errors.push(error(
                ScheduleDiagnostic::EmptyResourceEvidence,
                &resource.resource_id,
                "resource pool needs provenance",
            ));
        }
        if resources
            .insert(resource.resource_id.as_str(), resource)
            .is_some()
        {
            errors.push(error(
                ScheduleDiagnostic::DuplicateResourceId,
                &resource.resource_id,
                "resource ID is duplicated",
            ));
        }
    }

    let mut tasks = BTreeMap::new();
    for task in &plan.tasks {
        if task.task_id.trim().is_empty()
            || task.subsystem_id.trim().is_empty()
            || task.failure_mode_id.trim().is_empty()
        {
            errors.push(error(
                ScheduleDiagnostic::EmptyTaskIdentity,
                &task.task_id,
                "task, subsystem and failure-mode identities are required",
            ));
        }
        if task.duration_seconds == 0 {
            errors.push(error(
                ScheduleDiagnostic::ZeroTaskDuration,
                &task.task_id,
                "task duration must be positive",
            ));
        }
        if task.task_definition_ref.trim().is_empty() || task.evidence_ref.trim().is_empty() {
            errors.push(error(
                ScheduleDiagnostic::EmptyTaskEvidence,
                &task.task_id,
                "task definition and evidence references are required",
            ));
        }
        if !is_sha256(&task.task_definition_sha256) {
            errors.push(error(
                ScheduleDiagnostic::InvalidTaskDefinitionDigest,
                &task.task_id,
                "task-definition SHA-256 is malformed",
            ));
        }
        let access = &task.access_certificate;
        if access.certificate_ref.trim().is_empty() {
            errors.push(error(
                ScheduleDiagnostic::MissingAccessCertificate,
                &task.task_id,
                "source-owned access certificate reference is required",
            ));
        }
        if !is_sha256(&access.certificate_sha256) {
            errors.push(error(
                ScheduleDiagnostic::InvalidAccessCertificateDigest,
                &task.task_id,
                "access certificate SHA-256 is malformed",
            ));
        }
        if access.bound_candidate_revision != plan.candidate_revision {
            errors.push(error(
                ScheduleDiagnostic::AccessCandidateRevisionMismatch,
                &task.task_id,
                "access certificate is bound to a different candidate revision",
            ));
        }
        if access.bound_geometry_sha256 != plan.geometry_sha256 {
            errors.push(error(
                ScheduleDiagnostic::AccessGeometryDigestMismatch,
                &task.task_id,
                "access certificate is bound to different geometry",
            ));
        }
        if access.bound_task_definition_ref != task.task_definition_ref
            || access.bound_task_definition_sha256 != task.task_definition_sha256
        {
            errors.push(error(
                ScheduleDiagnostic::AccessTaskDefinitionMismatch,
                &task.task_id,
                "access certificate is bound to a different task definition",
            ));
        }
        if !is_sha256(&access.robot_model_sha256) {
            errors.push(error(
                ScheduleDiagnostic::InvalidRobotModelDigest,
                &task.task_id,
                "robot-model SHA-256 is malformed",
            ));
        }
        if !is_sha256(&access.tool_model_sha256) {
            errors.push(error(
                ScheduleDiagnostic::InvalidToolModelDigest,
                &task.task_id,
                "tool-model SHA-256 is malformed",
            ));
        }
        if access.clearance_profile_ref.trim().is_empty() {
            errors.push(error(
                ScheduleDiagnostic::MissingClearanceProfile,
                &task.task_id,
                "declared clearance/tolerance profile is required",
            ));
        }
        match access.disposition {
            AccessAssessmentDisposition::Blocked => errors.push(error(
                ScheduleDiagnostic::AccessAssessmentBlocked,
                &task.task_id,
                "physical access certificate reports a blocked task",
            )),
            AccessAssessmentDisposition::Unknown => errors.push(error(
                ScheduleDiagnostic::AccessAssessmentUnknown,
                &task.task_id,
                "physical access is unknown or out of domain",
            )),
            _ => {}
        }
        if !access_evidence_supports(access.disposition, access.evidence_class) {
            errors.push(error(
                ScheduleDiagnostic::InsufficientAccessEvidenceClass,
                &task.task_id,
                "access evidence class does not support its declared assessment disposition",
            ));
        }
        if tasks.insert(task.task_id.as_str(), task).is_some() {
            errors.push(error(
                ScheduleDiagnostic::DuplicateTaskId,
                &task.task_id,
                "task ID is duplicated",
            ));
        }

        let mut predecessors = BTreeSet::new();
        for predecessor in &task.predecessor_ids {
            if !predecessors.insert(predecessor.as_str()) {
                errors.push(error(
                    ScheduleDiagnostic::DuplicatePredecessor,
                    &task.task_id,
                    format!("predecessor {predecessor} is duplicated"),
                ));
            }
            if predecessor == &task.task_id {
                errors.push(error(
                    ScheduleDiagnostic::SelfPredecessor,
                    &task.task_id,
                    "task cannot depend on itself",
                ));
            }
        }

        let mut demands = BTreeSet::new();
        for demand in &task.resource_demands {
            if demand.units == 0 {
                errors.push(error(
                    ScheduleDiagnostic::ZeroResourceDemand,
                    &task.task_id,
                    format!("resource {} demand is zero", demand.resource_id),
                ));
            }
            if !demands.insert(demand.resource_id.as_str()) {
                errors.push(error(
                    ScheduleDiagnostic::DuplicateResourceDemand,
                    &task.task_id,
                    format!("resource {} is demanded more than once", demand.resource_id),
                ));
            }
            match resources.get(demand.resource_id.as_str()) {
                None => errors.push(error(
                    ScheduleDiagnostic::UnknownResource,
                    &task.task_id,
                    format!("resource {} is not declared", demand.resource_id),
                )),
                Some(resource) if demand.units > resource.capacity => errors.push(error(
                    ScheduleDiagnostic::DemandExceedsCapacity,
                    &task.task_id,
                    format!(
                        "demand {} exceeds {} capacity {}",
                        demand.units, demand.resource_id, resource.capacity
                    ),
                )),
                Some(_) => {}
            }
        }
    }

    for task in &plan.tasks {
        for predecessor in &task.predecessor_ids {
            if !tasks.contains_key(predecessor.as_str()) {
                errors.push(error(
                    ScheduleDiagnostic::MissingPredecessor,
                    &task.task_id,
                    format!("predecessor {predecessor} does not exist"),
                ));
            }
        }
    }

    if plan.priority_order.len() != plan.tasks.len() {
        errors.push(error(
            ScheduleDiagnostic::InvalidPriorityOrder,
            &plan.campaign_id,
            "priority order must contain every task exactly once",
        ));
    }
    let mut priority = BTreeSet::new();
    for task_id in &plan.priority_order {
        if !priority.insert(task_id.as_str()) || !tasks.contains_key(task_id.as_str()) {
            errors.push(error(
                ScheduleDiagnostic::InvalidPriorityOrder,
                task_id,
                "priority order has a duplicate or unknown task",
            ));
        }
    }
    if errors.is_empty() && topological_order(plan).is_err() {
        errors.push(error(
            ScheduleDiagnostic::PrecedenceCycle,
            &plan.campaign_id,
            "task precedence graph contains a cycle",
        ));
    }
    errors
}

fn topological_order(plan: &MaintenanceSchedulePlan) -> Result<Vec<String>, ScheduleDiagnostic> {
    let priority: BTreeMap<&str, usize> = plan
        .priority_order
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    let mut indegree: BTreeMap<&str, usize> = BTreeMap::new();
    let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for task in &plan.tasks {
        indegree.insert(task.task_id.as_str(), task.predecessor_ids.len());
        for predecessor in &task.predecessor_ids {
            children
                .entry(predecessor.as_str())
                .or_default()
                .push(task.task_id.as_str());
        }
    }

    let mut order = Vec::with_capacity(plan.tasks.len());
    let mut ready: BTreeSet<(usize, &str)> = indegree
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(task_id, _)| (*priority.get(task_id).unwrap_or(&usize::MAX), *task_id))
        .collect();
    while let Some((_, task_id)) = ready.pop_first() {
        order.push(task_id.to_string());
        if let Some(descendants) = children.get(task_id) {
            for child in descendants {
                let count = indegree.get_mut(child).expect("child task must exist");
                *count -= 1;
                if *count == 0 {
                    ready.insert((*priority.get(child).unwrap_or(&usize::MAX), child));
                }
            }
        }
    }
    if order.len() != plan.tasks.len() {
        return Err(ScheduleDiagnostic::PrecedenceCycle);
    }
    Ok(order)
}

fn task_by_id<'a>(plan: &'a MaintenanceSchedulePlan, task_id: &str) -> Option<&'a MaintenanceTask> {
    plan.tasks.iter().find(|task| task.task_id == task_id)
}

fn earliest_resource_start(
    task: &MaintenanceTask,
    earliest_start: u64,
    scheduled: &[PlannedTask],
    resources: &BTreeMap<&str, &RenewableResource>,
) -> Option<u64> {
    let mut candidate = earliest_start;
    for _ in 0..=scheduled.len() {
        let end = candidate.checked_add(task.duration_seconds)?;
        let mut bump_to: Option<u64> = None;

        for demand in &task.resource_demands {
            let capacity = resources.get(demand.resource_id.as_str())?.capacity;
            let mut boundaries = BTreeSet::new();
            boundaries.insert(candidate);
            for existing in scheduled {
                if existing.start_second < end && candidate < existing.end_second {
                    boundaries.insert(existing.start_second.max(candidate));
                    boundaries.insert(existing.end_second.min(end));
                }
            }

            for instant in boundaries {
                if instant >= end {
                    continue;
                }
                let used: u64 = scheduled
                    .iter()
                    .filter(|existing| {
                        existing.start_second <= instant && instant < existing.end_second
                    })
                    .flat_map(|existing| {
                        existing
                            .task
                            .resource_demands
                            .iter()
                            .filter(|item| item.resource_id == demand.resource_id)
                    })
                    .map(|item| u64::from(item.units))
                    .sum();
                if used + u64::from(demand.units) > u64::from(capacity) {
                    let next_release = scheduled
                        .iter()
                        .filter(|existing| {
                            existing.start_second <= instant && instant < existing.end_second
                        })
                        .filter(|existing| {
                            existing
                                .task
                                .resource_demands
                                .iter()
                                .any(|item| item.resource_id == demand.resource_id)
                        })
                        .map(|existing| existing.end_second)
                        .min()?;
                    bump_to =
                        Some(bump_to.map_or(next_release, |current| current.min(next_release)));
                    break;
                }
            }
        }

        match bump_to {
            Some(next) if next > candidate => candidate = next,
            Some(_) => return None,
            None => return Some(candidate),
        }
    }
    None
}

fn resource_capacity_exceeded(
    subject: &ScheduledMaintenanceTask,
    all: &[ScheduledMaintenanceTask],
    resources: &BTreeMap<&str, &RenewableResource>,
) -> bool {
    for demand in &subject.resource_demands {
        let Some(resource) = resources.get(demand.resource_id.as_str()) else {
            return true;
        };
        let mut points = BTreeSet::new();
        points.insert(subject.start_second);
        for other in all {
            if other.task_id == subject.task_id
                || other.start_second >= subject.end_second
                || subject.start_second >= other.end_second
            {
                continue;
            }
            points.insert(other.start_second.max(subject.start_second));
            points.insert(other.end_second.min(subject.end_second));
        }
        for instant in points {
            if instant >= subject.end_second {
                continue;
            }
            let used: u64 = all
                .iter()
                .filter(|other| other.start_second <= instant && instant < other.end_second)
                .flat_map(|other| {
                    other
                        .resource_demands
                        .iter()
                        .filter(|item| item.resource_id == demand.resource_id)
                })
                .map(|item| u64::from(item.units))
                .sum();
            if used > u64::from(resource.capacity) {
                return true;
            }
        }
    }
    false
}

fn serial_baseline_makespan(plan: &MaintenanceSchedulePlan) -> Result<u64, ScheduleDiagnostic> {
    let order = topological_order(plan)?;
    let mut cursor = plan.campaign_start_second;
    for task_id in order {
        let task = task_by_id(plan, &task_id).ok_or(ScheduleDiagnostic::InvalidPriorityOrder)?;
        cursor = cursor
            .checked_add(task.duration_seconds)
            .ok_or(ScheduleDiagnostic::ScheduleDurationMismatch)?;
    }
    Ok(cursor.saturating_sub(plan.campaign_start_second))
}

fn critical_path_lower_bound(plan: &MaintenanceSchedulePlan) -> Result<u64, ScheduleDiagnostic> {
    let order = topological_order(plan)?;
    let mut finish: BTreeMap<String, u64> = BTreeMap::new();
    for task_id in order {
        let task = task_by_id(plan, &task_id).ok_or(ScheduleDiagnostic::InvalidPriorityOrder)?;
        let earliest = task
            .predecessor_ids
            .iter()
            .filter_map(|predecessor| finish.get(predecessor).copied())
            .max()
            .unwrap_or(0);
        finish.insert(
            task_id,
            earliest
                .checked_add(task.duration_seconds)
                .ok_or(ScheduleDiagnostic::ScheduleDurationMismatch)?,
        );
    }
    Ok(finish.values().copied().max().unwrap_or(0))
}

fn full_outage_window(tasks: &[ScheduledMaintenanceTask]) -> Option<(u64, u64)> {
    let starts: Vec<u64> = tasks
        .iter()
        .filter(|task| task.service_impact == ServiceImpact::FullOutage)
        .map(|task| task.start_second)
        .collect();
    let ends: Vec<u64> = tasks
        .iter()
        .filter(|task| task.service_impact == ServiceImpact::FullOutage)
        .map(|task| task.end_second)
        .collect();
    Some((*starts.iter().min()?, *ends.iter().max()?))
}

fn to_lifecycle_event(task: &ScheduledMaintenanceTask) -> AvailabilityEvent {
    AvailabilityEvent {
        event_id: task.task_id.clone(),
        category: task.outage_category,
        impact: task.service_impact,
        subsystem_id: task.subsystem_id.clone(),
        failure_mode_id: task.failure_mode_id.clone(),
        start_second: task.start_second,
        end_second: task.end_second,
        evidence_ref: task.evidence_ref.clone(),
        evidence_class: task.evidence_class,
    }
}

fn canonical_demands(demands: &[ResourceDemand]) -> Vec<ResourceDemand> {
    let mut result = demands.to_vec();
    result.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
    result
}

fn reduce_ratio(numerator: u64, denominator: u64) -> (u64, u64) {
    if denominator == 0 {
        return (0, 1);
    }
    let divisor = gcd(numerator, denominator);
    (numerator / divisor, denominator / divisor)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

fn access_evidence_supports(
    disposition: AccessAssessmentDisposition,
    evidence: LifecycleEvidenceClass,
) -> bool {
    match disposition {
        AccessAssessmentDisposition::AnalyticFixturePass => {
            evidence == LifecycleEvidenceClass::Synthetic
        }
        AccessAssessmentDisposition::SimulationFeasibleForMockup => matches!(
            evidence,
            LifecycleEvidenceClass::Modelled
                | LifecycleEvidenceClass::MockupValidated
                | LifecycleEvidenceClass::Measured
                | LifecycleEvidenceClass::IndependentlyVerified
                | LifecycleEvidenceClass::PhysicallyQualified
        ),
        AccessAssessmentDisposition::MockupDemonstrated => matches!(
            evidence,
            LifecycleEvidenceClass::MockupValidated
                | LifecycleEvidenceClass::Measured
                | LifecycleEvidenceClass::IndependentlyVerified
                | LifecycleEvidenceClass::PhysicallyQualified
        ),
        AccessAssessmentDisposition::IndependentlyVerified => matches!(
            evidence,
            LifecycleEvidenceClass::IndependentlyVerified
                | LifecycleEvidenceClass::PhysicallyQualified
        ),
        AccessAssessmentDisposition::PhysicalQualification => {
            evidence == LifecycleEvidenceClass::PhysicallyQualified
        }
        AccessAssessmentDisposition::Blocked | AccessAssessmentDisposition::Unknown => false,
    }
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

    fn resource(id: &str, capacity: u32) -> RenewableResource {
        RenewableResource {
            resource_id: id.to_string(),
            capacity,
            evidence_ref: format!("resource-evidence-{id}"),
            evidence_class: LifecycleEvidenceClass::Synthetic,
        }
    }

    fn access_binding(id: &str, task_sha: String) -> AccessCertificateBinding {
        AccessCertificateBinding {
            certificate_ref: format!("analytic-access-certificate-{id}"),
            certificate_sha256: sha('c'),
            bound_candidate_revision: "candidate-r1".to_string(),
            bound_geometry_sha256: sha('d'),
            bound_task_definition_ref: format!("task-definition-{id}"),
            bound_task_definition_sha256: task_sha,
            robot_model_sha256: sha('e'),
            tool_model_sha256: sha('f'),
            clearance_profile_ref: "synthetic-clearance-profile-v1".to_string(),
            disposition: AccessAssessmentDisposition::AnalyticFixturePass,
            evidence_class: LifecycleEvidenceClass::Synthetic,
        }
    }

    fn task(
        id: &str,
        duration: u64,
        predecessors: &[&str],
        resource_id: &str,
        impact: ServiceImpact,
    ) -> MaintenanceTask {
        MaintenanceTask {
            task_id: id.to_string(),
            phase: MaintenancePhase::Other,
            subsystem_id: format!("subsystem-{id}"),
            failure_mode_id: format!("failure-{id}"),
            duration_seconds: duration,
            predecessor_ids: predecessors
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            resource_demands: if resource_id.is_empty() {
                Vec::new()
            } else {
                vec![ResourceDemand {
                    resource_id: resource_id.to_string(),
                    units: 1,
                }]
            },
            outage_category: OutageCategory::Planned,
            service_impact: impact,
            task_definition_ref: format!("task-definition-{id}"),
            task_definition_sha256: sha('a'),
            access_certificate: access_binding(id, sha('a')),
            evidence_ref: format!("task-evidence-{id}"),
            evidence_class: LifecycleEvidenceClass::Synthetic,
        }
    }

    fn plan(
        tasks: Vec<MaintenanceTask>,
        resources: Vec<RenewableResource>,
    ) -> MaintenanceSchedulePlan {
        let mut priority_order: Vec<String> =
            tasks.iter().map(|task| task.task_id.clone()).collect();
        priority_order.sort();
        MaintenanceSchedulePlan {
            campaign_id: "synthetic-maintenance-campaign-v1".to_string(),
            candidate_revision: "candidate-r1".to_string(),
            geometry_sha256: sha('d'),
            campaign_start_second: 1_000,
            priority_order,
            resources,
            tasks,
        }
    }

    #[test]
    fn independent_tasks_parallelize_when_resources_allow() {
        let plan = plan(
            vec![
                task("a", 100, &[], "robot-a", ServiceImpact::FullOutage),
                task("b", 100, &[], "robot-b", ServiceImpact::FullOutage),
                task("c", 50, &["a", "b"], "metrology", ServiceImpact::Derated),
            ],
            vec![
                resource("robot-a", 1),
                resource("robot-b", 1),
                resource("metrology", 1),
            ],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.makespan_seconds, 150);
        assert_eq!(schedule.serial_baseline_makespan_seconds, 250);
        assert_eq!(schedule.critical_path_lower_bound_seconds, 150);
        assert_eq!(schedule.parallel_speedup_over_serial, (5, 3));
        assert!(validate_maintenance_schedule(&plan, &schedule).is_empty());
    }

    #[test]
    fn shared_capacity_one_robot_serializes_tasks() {
        let plan = plan(
            vec![
                task("a", 100, &[], "robot", ServiceImpact::FullOutage),
                task("b", 100, &[], "robot", ServiceImpact::FullOutage),
            ],
            vec![resource("robot", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.makespan_seconds, 200);
    }

    #[test]
    fn capacity_two_robot_pool_allows_two_tasks_to_run_in_parallel() {
        let plan = plan(
            vec![
                task("a", 100, &[], "robot-fleet", ServiceImpact::FullOutage),
                task("b", 100, &[], "robot-fleet", ServiceImpact::FullOutage),
            ],
            vec![resource("robot-fleet", 2)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.makespan_seconds, 100);
    }

    #[test]
    fn access_port_is_a_capacity_one_spatial_exclusion_resource() {
        let mut first = task("a", 100, &[], "robot-a", ServiceImpact::FullOutage);
        let mut second = task("b", 100, &[], "robot-b", ServiceImpact::FullOutage);
        first.resource_demands.push(ResourceDemand {
            resource_id: "port-1".to_string(),
            units: 1,
        });
        second.resource_demands.push(ResourceDemand {
            resource_id: "port-1".to_string(),
            units: 1,
        });
        let plan = plan(
            vec![first, second],
            vec![
                resource("robot-a", 1),
                resource("robot-b", 1),
                resource("port-1", 1),
            ],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.makespan_seconds, 200);
    }

    #[test]
    fn predecessor_task_must_finish_before_dependent_starts() {
        let plan = plan(
            vec![
                task("a", 120, &[], "robot-a", ServiceImpact::FullOutage),
                task("b", 50, &["a"], "robot-b", ServiceImpact::FullOutage),
            ],
            vec![resource("robot-a", 1), resource("robot-b", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        let a = schedule
            .scheduled_tasks
            .iter()
            .find(|task| task.task_id == "a")
            .unwrap();
        let b = schedule
            .scheduled_tasks
            .iter()
            .find(|task| task.task_id == "b")
            .unwrap();
        assert_eq!(b.start_second, a.end_second);
    }

    #[test]
    fn overlapping_full_outage_tasks_emit_one_shared_outage_window() {
        let plan = plan(
            vec![
                task("a", 100, &[], "robot-a", ServiceImpact::FullOutage),
                task("b", 100, &[], "robot-b", ServiceImpact::FullOutage),
            ],
            vec![resource("robot-a", 1), resource("robot-b", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.full_outage_window, Some((1_000, 1_100)));
        assert_eq!(schedule.full_outage_window_seconds, 100);
        assert_eq!(schedule.lifecycle_events.len(), 2);
    }

    #[test]
    fn derated_tasks_remain_visible_without_expanding_full_outage_window() {
        let plan = plan(
            vec![
                task(
                    "full-outage",
                    100,
                    &[],
                    "robot-a",
                    ServiceImpact::FullOutage,
                ),
                task(
                    "derated-inspection",
                    200,
                    &[],
                    "robot-b",
                    ServiceImpact::Derated,
                ),
            ],
            vec![resource("robot-a", 1), resource("robot-b", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.full_outage_window_seconds, 100);
        assert_eq!(schedule.lifecycle_events.len(), 2);
    }

    #[test]
    fn task_input_order_does_not_change_frozen_priority_schedule() {
        let tasks = vec![
            task("a", 100, &[], "robot-a", ServiceImpact::FullOutage),
            task("b", 100, &[], "robot-b", ServiceImpact::FullOutage),
            task("c", 50, &["a", "b"], "metrology", ServiceImpact::Derated),
        ];
        let first = plan(
            tasks.clone(),
            vec![
                resource("robot-a", 1),
                resource("robot-b", 1),
                resource("metrology", 1),
            ],
        );
        let mut reversed = tasks;
        reversed.reverse();
        let second = plan(
            reversed,
            vec![
                resource("metrology", 1),
                resource("robot-b", 1),
                resource("robot-a", 1),
            ],
        );
        assert_eq!(
            generate_maintenance_schedule(&first).unwrap(),
            generate_maintenance_schedule(&second).unwrap()
        );
    }

    #[test]
    fn missing_predecessor_is_rejected() {
        let plan = plan(
            vec![task(
                "a",
                10,
                &["missing"],
                "robot",
                ServiceImpact::FullOutage,
            )],
            vec![resource("robot", 1)],
        );
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::MissingPredecessor)
        );
    }

    #[test]
    fn precedence_cycle_is_rejected() {
        let plan = plan(
            vec![
                task("a", 10, &["b"], "robot-a", ServiceImpact::FullOutage),
                task("b", 10, &["a"], "robot-b", ServiceImpact::FullOutage),
            ],
            vec![resource("robot-a", 1), resource("robot-b", 1)],
        );
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::PrecedenceCycle)
        );
    }

    #[test]
    fn demand_above_resource_capacity_is_rejected() {
        let mut first = task("a", 10, &[], "robot", ServiceImpact::FullOutage);
        first.resource_demands[0].units = 2;
        let plan = plan(vec![first], vec![resource("robot", 1)]);
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::DemandExceedsCapacity)
        );
    }

    #[test]
    fn missing_resource_pool_is_not_unlimited_capacity() {
        let plan = plan(
            vec![task(
                "a",
                10,
                &[],
                "missing-robot",
                ServiceImpact::FullOutage,
            )],
            vec![],
        );
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::UnknownResource)
        );
    }

    #[test]
    fn wrong_priority_order_is_rejected() {
        let mut plan = plan(
            vec![
                task("a", 10, &[], "robot-a", ServiceImpact::FullOutage),
                task("b", 10, &[], "robot-b", ServiceImpact::FullOutage),
            ],
            vec![resource("robot-a", 1), resource("robot-b", 1)],
        );
        plan.priority_order.pop();
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::InvalidPriorityOrder)
        );
    }

    #[test]
    fn exact_small_instance_oracle_is_no_worse_than_list_schedule() {
        let plan = plan(
            vec![
                task("a", 80, &[], "robot", ServiceImpact::FullOutage),
                task("b", 40, &[], "robot", ServiceImpact::FullOutage),
                task("c", 60, &["a"], "port", ServiceImpact::FullOutage),
                task("d", 30, &["b"], "port", ServiceImpact::Derated),
                task("e", 20, &["c", "d"], "metrology", ServiceImpact::Derated),
            ],
            vec![
                resource("robot", 1),
                resource("port", 1),
                resource("metrology", 1),
            ],
        );
        let heuristic = generate_maintenance_schedule(&plan).unwrap();
        let oracle = exact_small_instance_makespan(&plan).unwrap();
        assert!(oracle <= heuristic.makespan_seconds);
        assert!(oracle >= heuristic.critical_path_lower_bound_seconds);
    }

    #[test]
    fn exact_oracle_refuses_large_instances() {
        let mut plan = plan(
            vec![task("a", 1, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        for index in 0..EXACT_ORACLE_MAX_TASKS {
            plan.tasks.push(task(
                &format!("b{index}"),
                1,
                &[],
                "robot",
                ServiceImpact::FullOutage,
            ));
        }
        plan.priority_order = plan.tasks.iter().map(|task| task.task_id.clone()).collect();
        assert_eq!(
            exact_small_instance_makespan(&plan),
            Err(ExactOracleError::TooManyTasks {
                maximum: EXACT_ORACLE_MAX_TASKS,
                actual: EXACT_ORACLE_MAX_TASKS + 1
            })
        );
    }

    #[test]
    fn tampered_resource_interval_is_rejected_by_independent_validator() {
        let plan = plan(
            vec![
                task("a", 100, &[], "robot", ServiceImpact::FullOutage),
                task("b", 100, &[], "robot", ServiceImpact::FullOutage),
            ],
            vec![resource("robot", 1)],
        );
        let mut schedule = generate_maintenance_schedule(&plan).unwrap();
        schedule.scheduled_tasks[1].start_second = schedule.scheduled_tasks[0].start_second;
        let errors = validate_maintenance_schedule(&plan, &schedule);
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::ResourceCapacityViolation)
        );
    }

    #[test]
    fn scheduled_task_events_preserve_evidence_and_service_impact() {
        let plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::Derated)],
            vec![resource("robot", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        let event = &schedule.lifecycle_events[0];
        assert_eq!(event.event_id, "a");
        assert_eq!(event.impact, ServiceImpact::Derated);
        assert_eq!(event.evidence_class, LifecycleEvidenceClass::Synthetic);
    }

    #[test]
    fn missing_or_stale_access_certificate_refuses_task_schedule() {
        let mut first_plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        first_plan.tasks[0]
            .access_certificate
            .certificate_ref
            .clear();
        let errors = generate_maintenance_schedule(&first_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::MissingAccessCertificate)
        );

        let mut first_plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        first_plan.tasks[0].access_certificate.bound_geometry_sha256 = sha('b');
        let errors = generate_maintenance_schedule(&first_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::AccessGeometryDigestMismatch)
        );
    }

    #[test]
    fn blocked_or_unknown_access_cannot_be_scheduled_as_feasible() {
        let mut first_plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        first_plan.tasks[0].access_certificate.disposition = AccessAssessmentDisposition::Blocked;
        let errors = generate_maintenance_schedule(&first_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::AccessAssessmentBlocked)
        );

        let mut first_plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        first_plan.tasks[0].access_certificate.disposition = AccessAssessmentDisposition::Unknown;
        let errors = generate_maintenance_schedule(&first_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic == ScheduleDiagnostic::AccessAssessmentUnknown)
        );
    }

    #[test]
    fn simulated_access_requires_modelled_or_stronger_evidence() {
        let mut plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        plan.tasks[0].access_certificate.disposition =
            AccessAssessmentDisposition::SimulationFeasibleForMockup;
        plan.tasks[0].access_certificate.evidence_class = LifecycleEvidenceClass::Synthetic;
        let errors = generate_maintenance_schedule(&plan).unwrap_err();
        assert!(
            errors.iter().any(
                |error| error.diagnostic == ScheduleDiagnostic::InsufficientAccessEvidenceClass
            )
        );

        plan.tasks[0].access_certificate.evidence_class = LifecycleEvidenceClass::Modelled;
        assert!(generate_maintenance_schedule(&plan).is_ok());
    }

    #[test]
    fn mockup_and_physical_access_dispositions_require_matching_evidence_class() {
        let mut plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        plan.tasks[0].access_certificate.disposition =
            AccessAssessmentDisposition::MockupDemonstrated;
        plan.tasks[0].access_certificate.evidence_class = LifecycleEvidenceClass::Modelled;
        assert!(generate_maintenance_schedule(&plan).is_err());
        plan.tasks[0].access_certificate.evidence_class = LifecycleEvidenceClass::MockupValidated;
        assert!(generate_maintenance_schedule(&plan).is_ok());

        plan.tasks[0].access_certificate.disposition =
            AccessAssessmentDisposition::PhysicalQualification;
        plan.tasks[0].access_certificate.evidence_class = LifecycleEvidenceClass::MockupValidated;
        assert!(generate_maintenance_schedule(&plan).is_err());
        plan.tasks[0].access_certificate.evidence_class =
            LifecycleEvidenceClass::PhysicallyQualified;
        assert!(generate_maintenance_schedule(&plan).is_ok());
    }

    #[test]
    fn campaign_start_is_preserved_in_absolute_schedule_times() {
        let plan = plan(
            vec![task("a", 10, &[], "robot", ServiceImpact::FullOutage)],
            vec![resource("robot", 1)],
        );
        let schedule = generate_maintenance_schedule(&plan).unwrap();
        assert_eq!(schedule.scheduled_tasks[0].start_second, 1_000);
        assert_eq!(schedule.scheduled_tasks[0].end_second, 1_010);
    }
}
