// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-carrying mission task verification (SX-032).
//!
//! Tasks declare the capabilities, support components, compatible connections,
//! and operating environments they require. The evaluator composes SX-031's
//! mission graph with those explicit task declarations. It does not infer
//! requirements, authorize EVA, or qualify hardware.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::suit_interop::{SuitInteropReportStatus, SuitOperatingEnvironment};
use crate::suit_mission_configuration::{
    digest_mission_configuration_report, evaluate_mission_configuration,
    MissionConfiguration, MissionConfigurationError, MissionConfigurationReport,
    MissionConfigurationStatus,
};
use crate::suit_profile::{
    SuitCapability, SuitCapabilityFindingStatus, SuitEvidenceLevel,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTaskRequirement {
    pub task_id: String,
    pub required: bool,
    pub required_capabilities: BTreeSet<SuitCapability>,
    pub required_component_ids: BTreeSet<String>,
    pub required_connection_ids: BTreeSet<String>,
    /// Every required dependency must be admissible in all these environments.
    pub required_environments: BTreeSet<SuitOperatingEnvironment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionTaskFindingCode {
    MissingCapabilityEvidence,
    CapabilityEvidenceBelowPolicy,
    ConflictingCapabilityClaims,
    MissingComponent,
    ComponentNotReachable,
    MissingConnection,
    ConnectionBlocked,
    ConnectionEnvironmentMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionTaskFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTaskFinding {
    pub task_id: String,
    pub subject_id: String,
    pub code: MissionTaskFindingCode,
    pub severity: MissionTaskFindingSeverity,
    pub evidence_level: Option<SuitEvidenceLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionTaskStatus {
    SatisfiedUnderDeclaredPolicy,
    Blocked,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTaskEvaluation {
    pub task_id: String,
    pub required: bool,
    pub required_capabilities: BTreeSet<SuitCapability>,
    pub required_component_ids: BTreeSet<String>,
    pub required_connection_ids: BTreeSet<String>,
    pub required_environments: BTreeSet<SuitOperatingEnvironment>,
    pub findings: Vec<MissionTaskFinding>,
    pub status: MissionTaskStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionTaskPlanStatus {
    Blocked,
    Degraded,
    MeetsDeclaredTaskPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTaskPlanReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub mission_configuration_report: MissionConfigurationReport,
    pub task_evaluations: Vec<MissionTaskEvaluation>,
    pub status: MissionTaskPlanStatus,
    pub required_task_count: usize,
    pub satisfied_required_task_count: usize,
    pub blocked_required_task_count: usize,
    pub optional_task_count: usize,
    pub degraded_optional_task_count: usize,
    /// Always false: task traceability is not physical qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTaskReceipt {
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub mission_configuration_report_digest_sha256: String,
    pub task_report_digest_sha256: String,
    /// Deterministic change-evidence, not a signature or authentication claim.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionTaskError {
    InvalidMissionConfiguration(MissionConfigurationError),
    EmptyTaskSet,
    InvalidTaskIdentity,
    DuplicateTaskIdentity,
    InvalidTaskEnvironment,
    TaskHasNoDeclaredDependencies,
}

/// Evaluate a non-empty, explicit task plan against the exact mission
/// configuration and its policy-admissible interface graph.
pub fn evaluate_mission_tasks(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
) -> Result<(MissionTaskPlanReport, MissionTaskReceipt), MissionTaskError> {
    validate_tasks(tasks)?;
    let (mission_report, _) = evaluate_mission_configuration(configuration)
        .map_err(MissionTaskError::InvalidMissionConfiguration)?;

    let configuration_digest_sha256 = mission_report.configuration_digest_sha256.clone();
    let task_plan_digest_sha256 = digest_task_plan(tasks);
    let mut ordered_tasks: Vec<&MissionTaskRequirement> = tasks.iter().collect();
    ordered_tasks.sort_by(|left, right| left.task_id.cmp(&right.task_id));

    let component_ids: BTreeSet<String> = configuration
        .components
        .iter()
        .map(|component| component.component_id.clone())
        .collect();
    let connections: BTreeMap<&str, _> = mission_report
        .connection_evaluations
        .iter()
        .map(|connection| (connection.connection_id.as_str(), connection))
        .collect();
    // Task plans often reuse the same environment profile. Calculate graph
    // reachability once per distinct environment set, not once per task.
    let mut reachability_cache: BTreeMap<
        BTreeSet<SuitOperatingEnvironment>,
        BTreeSet<String>,
    > = BTreeMap::new();
    let mut task_evaluations = Vec::with_capacity(ordered_tasks.len());
    for task in ordered_tasks {
        let environments = task.required_environments.clone();
        if !reachability_cache.contains_key(&environments) {
            let reachable = reachable_for_task(
                configuration,
                &mission_report,
                &environments,
            );
            reachability_cache.insert(environments.clone(), reachable);
        }
        let reachable = reachability_cache
            .get(&environments)
            .expect("reachability cache entry was just inserted");
        task_evaluations.push(evaluate_task(
            &mission_report,
            &component_ids,
            &connections,
            reachable,
            task,
        ));
    }

    let required_task_count = task_evaluations.iter().filter(|task| task.required).count();
    let satisfied_required_task_count = task_evaluations
        .iter()
        .filter(|task| {
            task.required && task.status == MissionTaskStatus::SatisfiedUnderDeclaredPolicy
        })
        .count();
    let blocked_required_task_count = task_evaluations
        .iter()
        .filter(|task| task.required && task.status == MissionTaskStatus::Blocked)
        .count();
    let optional_task_count = task_evaluations.iter().filter(|task| !task.required).count();
    let degraded_optional_task_count = task_evaluations
        .iter()
        .filter(|task| {
            !task.required && task.status != MissionTaskStatus::SatisfiedUnderDeclaredPolicy
        })
        .count();

    let task_blocker_count = task_evaluations.iter().flat_map(|task| &task.findings)
        .filter(|finding| finding.severity == MissionTaskFindingSeverity::Blocker)
        .count();
    let task_degradation_count = task_evaluations
        .iter()
        .flat_map(|task| &task.findings)
        .filter(|finding| finding.severity == MissionTaskFindingSeverity::Degradation)
        .count();
    let status = if mission_report.status == MissionConfigurationStatus::Blocked
        || task_blocker_count > 0
    {
        MissionTaskPlanStatus::Blocked
    } else if mission_report.status == MissionConfigurationStatus::Degraded
        || task_degradation_count > 0
    {
        MissionTaskPlanStatus::Degraded
    } else {
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy
    };

    let mut report = MissionTaskPlanReport {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        configuration_digest_sha256: configuration_digest_sha256.clone(),
        task_plan_digest_sha256: task_plan_digest_sha256.clone(),
        mission_configuration_report: mission_report,
        task_evaluations,
        status,
        required_task_count,
        satisfied_required_task_count,
        blocked_required_task_count,
        optional_task_count,
        degraded_optional_task_count,
        represents_physical_qualification: false,
    };

    // Canonical task order is established above; findings within each task are
    // independently sorted. Digest binds the complete embedded graph report.
    let task_report_digest_sha256 = digest_task_report(&report);
    let mut receipt = MissionTaskReceipt {
        configuration_digest_sha256,
        task_plan_digest_sha256,
        mission_configuration_report_digest_sha256:
            digest_mission_configuration_report(&report.mission_configuration_report),
        task_report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_task_receipt(&receipt);

    // Prevent accidental future changes to status/count assembly from creating
    // an inconsistent report. These counters are checked by the digest verifier.
    report.task_evaluations.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    Ok((report, receipt))
}

/// Re-run the evaluator; altered task requirements, graph evidence, or report
/// fields fail verification. This does not authenticate evidence sources.
pub fn verify_mission_task_receipt(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    expected_report: &MissionTaskPlanReport,
    expected_receipt: &MissionTaskReceipt,
) -> bool {
    match evaluate_mission_tasks(configuration, tasks) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn validate_tasks(tasks: &[MissionTaskRequirement]) -> Result<(), MissionTaskError> {
    if tasks.is_empty() {
        return Err(MissionTaskError::EmptyTaskSet);
    }
    let mut ids = BTreeSet::new();
    for task in tasks {
        if !clean_identifier(&task.task_id) {
            return Err(MissionTaskError::InvalidTaskIdentity);
        }
        if !ids.insert(task.task_id.as_str()) {
            return Err(MissionTaskError::DuplicateTaskIdentity);
        }
        if task.required_environments.is_empty() {
            return Err(MissionTaskError::InvalidTaskEnvironment);
        }
        if task.required_capabilities.is_empty()
            && task.required_component_ids.is_empty()
            && task.required_connection_ids.is_empty()
        {
            return Err(MissionTaskError::TaskHasNoDeclaredDependencies);
        }
        if task.required_component_ids.iter().any(|id| !clean_identifier(id))
            || task.required_connection_ids.iter().any(|id| !clean_identifier(id))
        {
            return Err(MissionTaskError::InvalidTaskIdentity);
        }
    }
    Ok(())
}

fn evaluate_task(
    mission_report: &MissionConfigurationReport,
    component_ids: &BTreeSet<String>,
    connections: &BTreeMap<&str, &crate::suit_mission_configuration::MissionConnectionEvaluation>,
    reachable: &BTreeSet<String>,
    task: &MissionTaskRequirement,
) -> MissionTaskEvaluation {
    let severity = if task.required {
        MissionTaskFindingSeverity::Blocker
    } else {
        MissionTaskFindingSeverity::Degradation
    };
    let mut findings = Vec::new();

    for capability in &task.required_capabilities {
        let matching = mission_report
            .suit_profile_evaluation
            .findings
            .iter()
            .find(|finding| finding.capability == *capability);
        match matching {
            None => findings.push(MissionTaskFinding {
                task_id: task.task_id.clone(),
                subject_id: capability_tag(*capability).into(),
                code: MissionTaskFindingCode::MissingCapabilityEvidence,
                severity,
                evidence_level: None,
            }),
            Some(finding) => match finding.status {
                SuitCapabilityFindingStatus::MeetsDeclaredPolicy => {}
                SuitCapabilityFindingStatus::MissingRequiredCapability => {
                    findings.push(MissionTaskFinding {
                        task_id: task.task_id.clone(),
                        subject_id: capability_tag(*capability).into(),
                        code: MissionTaskFindingCode::MissingCapabilityEvidence,
                        severity,
                        evidence_level: None,
                    });
                }
                SuitCapabilityFindingStatus::EvidenceBelowPolicy => {
                    findings.push(MissionTaskFinding {
                        task_id: task.task_id.clone(),
                        subject_id: capability_tag(*capability).into(),
                        code: MissionTaskFindingCode::CapabilityEvidenceBelowPolicy,
                        severity,
                        evidence_level: finding.evidence_level,
                    });
                }
                SuitCapabilityFindingStatus::ConflictingClaims => {
                    findings.push(MissionTaskFinding {
                        task_id: task.task_id.clone(),
                        subject_id: capability_tag(*capability).into(),
                        code: MissionTaskFindingCode::ConflictingCapabilityClaims,
                        severity,
                        evidence_level: None,
                    });
                }
            },
        }
    }

    for component_id in &task.required_component_ids {
        if !component_ids.contains(component_id) {
            findings.push(MissionTaskFinding {
                task_id: task.task_id.clone(),
                subject_id: component_id.clone(),
                code: MissionTaskFindingCode::MissingComponent,
                severity,
                evidence_level: None,
            });
        } else if !reachable.contains(component_id) {
            findings.push(MissionTaskFinding {
                task_id: task.task_id.clone(),
                subject_id: component_id.clone(),
                code: MissionTaskFindingCode::ComponentNotReachable,
                severity,
                evidence_level: None,
            });
        }
    }

    for connection_id in &task.required_connection_ids {
        match connections.get(connection_id.as_str()) {
            None => findings.push(MissionTaskFinding {
                task_id: task.task_id.clone(),
                subject_id: connection_id.clone(),
                code: MissionTaskFindingCode::MissingConnection,
                severity,
                evidence_level: None,
            }),
            Some(connection) => {
                let in_scope = task.required_environments.is_subset(
                    &connection.interoperability.policy.required_environments,
                );
                if !in_scope {
                    findings.push(MissionTaskFinding {
                        task_id: task.task_id.clone(),
                        subject_id: connection_id.clone(),
                        code: MissionTaskFindingCode::ConnectionEnvironmentMismatch,
                        severity,
                        evidence_level: None,
                    });
                }
                if connection.interoperability.status
                    != SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
                {
                    findings.push(MissionTaskFinding {
                        task_id: task.task_id.clone(),
                        subject_id: connection_id.clone(),
                        code: MissionTaskFindingCode::ConnectionBlocked,
                        severity,
                        evidence_level: None,
                    });
                }
            }
        }
    }

    findings.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then(left.subject_id.cmp(&right.subject_id))
            .then(left.severity.cmp(&right.severity))
            .then(left.evidence_level.cmp(&right.evidence_level))
    });
    let status = if findings.is_empty() {
        MissionTaskStatus::SatisfiedUnderDeclaredPolicy
    } else if task.required {
        MissionTaskStatus::Blocked
    } else {
        MissionTaskStatus::Degraded
    };
    MissionTaskEvaluation {
        task_id: task.task_id.clone(),
        required: task.required,
        required_capabilities: task.required_capabilities.clone(),
        required_component_ids: task.required_component_ids.clone(),
        required_connection_ids: task.required_connection_ids.clone(),
        required_environments: task.required_environments.clone(),
        findings,
        status,
    }
}

fn reachable_for_task(
    configuration: &MissionConfiguration,
    mission_report: &MissionConfigurationReport,
    environments: &BTreeSet<SuitOperatingEnvironment>,
) -> BTreeSet<String> {
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = configuration
        .components
        .iter()
        .map(|component| (component.component_id.clone(), BTreeSet::new()))
        .collect();
    for connection in &mission_report.connection_evaluations {
        if connection.interoperability.status
            != SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
            || !environments.is_subset(
                &connection.interoperability.policy.required_environments,
            )
        {
            continue;
        }
        adjacency
            .entry(connection.left_component_id.clone())
            .or_default()
            .insert(connection.right_component_id.clone());
        adjacency
            .entry(connection.right_component_id.clone())
            .or_default()
            .insert(connection.left_component_id.clone());
    }

    let mut reached = BTreeSet::new();
    let mut pending = VecDeque::from([configuration.suit_component_id.clone()]);
    while let Some(component_id) = pending.pop_front() {
        if !reached.insert(component_id.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&component_id) {
            pending.extend(neighbors.iter().cloned());
        }
    }
    reached
}

fn digest_task_plan(tasks: &[MissionTaskRequirement]) -> String {
    let mut writer = HashWriter::new("symthaea.sx032.task-plan.v1");
    writer.u64(1);
    let mut ordered: Vec<&MissionTaskRequirement> = tasks.iter().collect();
    ordered.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    writer.usize(ordered.len());
    for task in ordered {
        writer.string(&task.task_id);
        writer.boolean(task.required);
        writer.usize(task.required_capabilities.len());
        for capability in &task.required_capabilities {
            writer.string(capability_tag(*capability));
        }
        writer.usize(task.required_component_ids.len());
        for component_id in &task.required_component_ids {
            writer.string(component_id);
        }
        writer.usize(task.required_connection_ids.len());
        for connection_id in &task.required_connection_ids {
            writer.string(connection_id);
        }
        writer.usize(task.required_environments.len());
        for environment in &task.required_environments {
            writer.string(environment_tag(*environment));
        }
    }
    writer.finish()
}

fn digest_task_report(report: &MissionTaskPlanReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx032.task-report.v1");
    writer.u64(report.schema_version as u64);
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.configuration_digest_sha256);
    writer.string(&report.task_plan_digest_sha256);
    writer.string(&digest_mission_configuration_report(&report.mission_configuration_report));
    writer.string(task_plan_status_tag(report.status));
    writer.usize(report.required_task_count);
    writer.usize(report.satisfied_required_task_count);
    writer.usize(report.blocked_required_task_count);
    writer.usize(report.optional_task_count);
    writer.usize(report.degraded_optional_task_count);
    writer.boolean(report.represents_physical_qualification);
    writer.usize(report.task_evaluations.len());
    for task in &report.task_evaluations {
        writer.string(&task.task_id);
        writer.boolean(task.required);
        writer.string(task_status_tag(task.status));
        writer.usize(task.required_capabilities.len());
        for capability in &task.required_capabilities {
            writer.string(capability_tag(*capability));
        }
        writer.usize(task.required_component_ids.len());
        for id in &task.required_component_ids {
            writer.string(id);
        }
        writer.usize(task.required_connection_ids.len());
        for id in &task.required_connection_ids {
            writer.string(id);
        }
        writer.usize(task.required_environments.len());
        for environment in &task.required_environments {
            writer.string(environment_tag(*environment));
        }
        writer.usize(task.findings.len());
        for finding in &task.findings {
            writer.string(&finding.task_id);
            writer.string(&finding.subject_id);
            writer.string(task_finding_code_tag(finding.code));
            writer.string(task_finding_severity_tag(finding.severity));
            match finding.evidence_level {
                Some(level) => {
                    writer.boolean(true);
                    writer.string(evidence_tag(level));
                }
                None => writer.boolean(false),
            }
        }
    }
    writer.finish()
}

fn digest_task_receipt(receipt: &MissionTaskReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx032.task-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.mission_configuration_report_digest_sha256);
    writer.string(&receipt.task_report_digest_sha256);
    writer.finish()
}

fn clean_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

struct HashWriter(Sha256);

impl HashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn boolean(&mut self, value: bool) {
        self.raw(&[u8::from(value)]);
    }

    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        format!(
            "sha256:{}",
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        )
    }
}

fn capability_tag(value: SuitCapability) -> &'static str {
    match value {
        SuitCapability::PressureBoundaryIntegrity => "pressure_boundary_integrity",
        SuitCapability::PrimaryLifeSupportInterface => "primary_life_support_interface",
        SuitCapability::ThermalManagement => "thermal_management",
        SuitCapability::IndependentEmergencyReserve => "independent_emergency_reserve",
        SuitCapability::ManualFallback => "manual_fallback",
        SuitCapability::CommunicationsAndTelemetry => "communications_and_telemetry",
        SuitCapability::FaultContainmentAndSafeDegradation => {
            "fault_containment_and_safe_degradation"
        },
        SuitCapability::MicrogravityTranslationAndRestraint => {
            "microgravity_translation_and_restraint"
        },
        SuitCapability::TetherAndAnchorInterfaces => "tether_and_anchor_interfaces",
        SuitCapability::OrbitalWorksiteInterfaces => "orbital_worksite_interfaces",
        SuitCapability::PlanetaryLocomotion => "planetary_locomotion",
        SuitCapability::DustMitigation => "dust_mitigation",
        SuitCapability::ElectrostaticChargeManagement => "electrostatic_charge_management",
        SuitCapability::AbrasionAndPunctureResistance => "abrasion_and_puncture_resistance",
        SuitCapability::HighDexterityGloveInterfaces => "high_dexterity_glove_interfaces",
        SuitCapability::SurfaceNavigationInterface => "surface_navigation_interface",
        SuitCapability::HeavyToolAndLoadBearingInterfaces => {
            "heavy_tool_and_load_bearing_interfaces"
        },
        SuitCapability::ContaminationContainment => "contamination_containment",
        SuitCapability::SuitportMechanicalAndPressureCompatibility => {
            "suitport_mechanical_and_pressure_compatibility"
        },
        SuitCapability::ResourceServiceInterfaces => "resource_service_interfaces",
        SuitCapability::CrewRescueHandlingInterfaces => "crew_rescue_handling_interfaces",
        SuitCapability::FieldRepairableSubsystems => "field_repairable_subsystems",
        SuitCapability::ModularLifeSupportReplacementInterfaces => {
            "modular_life_support_replacement_interfaces"
        },
        SuitCapability::RadiationMonitoringInterface => "radiation_monitoring_interface",
    }
}

fn environment_tag(value: SuitOperatingEnvironment) -> &'static str {
    match value {
        SuitOperatingEnvironment::MicrogravityVacuum => "microgravity_vacuum",
        SuitOperatingEnvironment::LunarSunlitSurface => "lunar_sunlit_surface",
        SuitOperatingEnvironment::LunarShadowOrPolarRegion => "lunar_shadow_or_polar_region",
        SuitOperatingEnvironment::MarsSurface => "mars_surface",
        SuitOperatingEnvironment::PressurizedCabin => "pressurized_cabin",
        SuitOperatingEnvironment::ThermalVacuumGroundTest => "thermal_vacuum_ground_test",
        SuitOperatingEnvironment::GroundLaboratory => "ground_laboratory",
    }
}

fn evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn task_status_tag(value: MissionTaskStatus) -> &'static str {
    match value {
        MissionTaskStatus::SatisfiedUnderDeclaredPolicy => "satisfied_under_declared_policy",
        MissionTaskStatus::Blocked => "blocked",
        MissionTaskStatus::Degraded => "degraded",
    }
}

fn task_plan_status_tag(value: MissionTaskPlanStatus) -> &'static str {
    match value {
        MissionTaskPlanStatus::Blocked => "blocked",
        MissionTaskPlanStatus::Degraded => "degraded",
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy => "meets_declared_task_policy",
    }
}

fn task_finding_code_tag(value: MissionTaskFindingCode) -> &'static str {
    match value {
        MissionTaskFindingCode::MissingCapabilityEvidence => "missing_capability_evidence",
        MissionTaskFindingCode::CapabilityEvidenceBelowPolicy => "capability_evidence_below_policy",
        MissionTaskFindingCode::ConflictingCapabilityClaims => "conflicting_capability_claims",
        MissionTaskFindingCode::MissingComponent => "missing_component",
        MissionTaskFindingCode::ComponentNotReachable => "component_not_reachable",
        MissionTaskFindingCode::MissingConnection => "missing_connection",
        MissionTaskFindingCode::ConnectionBlocked => "connection_blocked",
        MissionTaskFindingCode::ConnectionEnvironmentMismatch => "connection_environment_mismatch",
    }
}

fn task_finding_severity_tag(value: MissionTaskFindingSeverity) -> &'static str {
    match value {
        MissionTaskFindingSeverity::Degradation => "degradation",
        MissionTaskFindingSeverity::Blocker => "blocker",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suit_interop::{
        SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
    };
    use crate::suit_mission_configuration::{
        MissionComponent, MissionComponentKind, MissionConnection,
    };
    use crate::suit_profile::{SuitCapabilityClaim, SuitMissionProfile, SuitVariantDeclaration};

    fn fixture() -> MissionConfiguration {
        let suit_revision = "suit-r3";
        let lander_revision = "lander-r2";
        let variant_id = "lunar-suit-v1";
        let environments = BTreeSet::from([
            SuitOperatingEnvironment::LunarShadowOrPolarRegion,
        ]);
        let endpoints = |component_id: &str, role: SuitInterfaceRole| {
            vec![SuitInterfaceEndpoint {
                interface_id: "grounding.esd.v1".into(),
                endpoint_id: format!("{component_id}.grounding"),
                kind: SuitInterfaceKind::GroundingAndEsd,
                role,
                contract_revision: "1.0.0".into(),
                schema_digest_sha256: format!("sha256:{}", "a".repeat(64)),
                connector_profile_id: "synthetic-grounding-profile-v1".into(),
                applicable_environments: environments.clone(),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-evidence:{component_id}:grounding"),
                required: true,
            }]
        };
        let components = vec![
            MissionComponent {
                component_id: "suit".into(),
                kind: MissionComponentKind::Suit,
                hardware_revision: suit_revision.into(),
                interfaces: SuitInterfaceManifest {
                    component_id: "suit".into(),
                    hardware_revision: suit_revision.into(),
                    endpoints: endpoints("suit", SuitInterfaceRole::Consumer),
                },
            },
            MissionComponent {
                component_id: "lander".into(),
                kind: MissionComponentKind::LanderOrVehicle,
                hardware_revision: lander_revision.into(),
                interfaces: SuitInterfaceManifest {
                    component_id: "lander".into(),
                    hardware_revision: lander_revision.into(),
                    endpoints: endpoints("lander", SuitInterfaceRole::Provider),
                },
            },
        ];
        let capability_claims = SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .into_iter()
            .map(|capability| SuitCapabilityClaim {
                capability,
                applies_to_variant_id: variant_id.into(),
                applies_to_hardware_revision: suit_revision.into(),
                implementation_id: format!("synthetic-{}", capability_tag(capability)),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-evidence-{}", capability_tag(capability)),
            })
            .collect();

        MissionConfiguration {
            schema_version: 1,
            mission_id: "synthetic-lunar-task-test".into(),
            configuration_revision: "cfg-r1".into(),
            suit_component_id: "suit".into(),
            suit_profile: SuitMissionProfile::LunarSurfaceEva,
            suit_variant: SuitVariantDeclaration {
                variant_id: variant_id.into(),
                hardware_revision: suit_revision.into(),
                capability_claims,
            },
            minimum_capability_evidence: SuitEvidenceLevel::BenchTested,
            required_component_kinds: BTreeSet::from([
                MissionComponentKind::Suit,
                MissionComponentKind::LanderOrVehicle,
            ]),
            required_connection_ids: BTreeSet::from(["suit-lander-grounding".into()]),
            components,
            connections: vec![MissionConnection {
                connection_id: "suit-lander-grounding".into(),
                left_component_id: "suit".into(),
                right_component_id: "lander".into(),
                policy: crate::suit_interop::SuitInteropPolicy {
                    minimum_evidence_level: SuitEvidenceLevel::Simulated,
                    required_environments: environments,
                },
            }],
        }
    }

    fn task(task_id: &str) -> MissionTaskRequirement {
        MissionTaskRequirement {
            task_id: task_id.into(),
            required: true,
            required_capabilities: BTreeSet::from([
                SuitCapability::ElectrostaticChargeManagement,
            ]),
            required_component_ids: BTreeSet::from(["lander".into()]),
            required_connection_ids: BTreeSet::from(["suit-lander-grounding".into()]),
            required_environments: BTreeSet::from([
                SuitOperatingEnvironment::LunarShadowOrPolarRegion,
            ]),
        }
    }

    #[test]
    fn satisfied_task_requires_capability_component_connection_and_environment() {
        let config = fixture();
        let tasks = vec![task("inspect-lunar-grounding")];
        let (report, receipt) = evaluate_mission_tasks(&config, &tasks).unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::MeetsDeclaredTaskPolicy);
        assert_eq!(report.required_task_count, 1);
        assert_eq!(report.satisfied_required_task_count, 1);
        assert!(!report.represents_physical_qualification);
        assert!(verify_mission_task_receipt(&config, &tasks, &report, &receipt));
    }

    #[test]
    fn missing_capability_evidence_blocks_required_task() {
        let config = fixture();
        let tasks = vec![MissionTaskRequirement {
            required_capabilities: BTreeSet::from([SuitCapability::RadiationMonitoringInterface]),
            ..task("radiation-monitoring")
        }];
        let (report, _) = evaluate_mission_tasks(&config, &tasks).unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::Blocked);
        assert!(report.task_evaluations[0].findings.iter().any(|finding| {
            finding.code == MissionTaskFindingCode::MissingCapabilityEvidence
                && finding.subject_id == "radiation_monitoring_interface"
        }));
    }

    #[test]
    fn below_floor_capability_evidence_blocks_task() {
        let mut config = fixture();
        for claim in &mut config.suit_variant.capability_claims {
            if claim.capability == SuitCapability::ElectrostaticChargeManagement {
                claim.evidence_level = SuitEvidenceLevel::Declared;
            }
        }
        let (report, _) = evaluate_mission_tasks(&config, &[task("grounding")]).unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::Blocked);
        assert!(report.task_evaluations[0].findings.iter().any(|finding| {
            finding.code == MissionTaskFindingCode::CapabilityEvidenceBelowPolicy
        }));
    }

    #[test]
    fn missing_component_and_connection_are_not_inferred() {
        let config = fixture();
        let tasks = vec![MissionTaskRequirement {
            required_component_ids: BTreeSet::from(["rescue-unit".into()]),
            required_connection_ids: BTreeSet::from(["suit-rescue".into()]),
            ..task("rescue-contingency")
        }];
        let (report, _) = evaluate_mission_tasks(&config, &tasks).unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::Blocked);
        assert!(report.task_evaluations[0].findings.iter().any(|f| {
            f.code == MissionTaskFindingCode::MissingComponent && f.subject_id == "rescue-unit"
        }));
        assert!(report.task_evaluations[0].findings.iter().any(|f| {
            f.code == MissionTaskFindingCode::MissingConnection && f.subject_id == "suit-rescue"
        }));
    }

    #[test]
    fn environment_scope_must_cover_the_task_environment() {
        let mut config = fixture();
        config.connections[0].policy.required_environments =
            BTreeSet::from([SuitOperatingEnvironment::LunarSunlitSurface]);
        let (report, _) = evaluate_mission_tasks(&config, &[task("shadowed-work")]).unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::Blocked);
        assert!(report.task_evaluations[0].findings.iter().any(|f| {
            f.code == MissionTaskFindingCode::ConnectionEnvironmentMismatch
                && f.subject_id == "suit-lander-grounding"
        }));
    }

    #[test]
    fn optional_task_failure_degrades_when_required_task_passes() {
        let config = fixture();
        let optional = MissionTaskRequirement {
            task_id: "optional-radiation-survey".into(),
            required: false,
            required_capabilities: BTreeSet::from([
                SuitCapability::RadiationMonitoringInterface,
            ]),
            required_component_ids: BTreeSet::new(),
            required_connection_ids: BTreeSet::new(),
            required_environments: BTreeSet::from([
                SuitOperatingEnvironment::LunarShadowOrPolarRegion,
            ]),
        };
        let (report, _) = evaluate_mission_tasks(
            &config,
            &[task("core-grounding"), optional],
        )
        .unwrap();
        assert_eq!(report.status, MissionTaskPlanStatus::Degraded);
        assert_eq!(report.satisfied_required_task_count, 1);
        assert_eq!(report.degraded_optional_task_count, 1);
    }

    #[test]
    fn task_order_does_not_change_report_or_receipt() {
        let config = fixture();
        let tasks = vec![task("inspect-grounding"), task("inspect-charging")];
        let (report_a, receipt_a) = evaluate_mission_tasks(&config, &tasks).unwrap();
        let (report_b, receipt_b) =
            evaluate_mission_tasks(&config, &[tasks[1].clone(), tasks[0].clone()]).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);
    }

    #[test]
    fn changed_task_contract_invalidates_receipt() {
        let config = fixture();
        let tasks = vec![task("inspect-grounding")];
        let (report, receipt) = evaluate_mission_tasks(&config, &tasks).unwrap();
        let mut changed_tasks = tasks.clone();
        changed_tasks[0].required = false;
        assert!(!verify_mission_task_receipt(
            &config,
            &changed_tasks,
            &report,
            &receipt
        ));
    }

    #[test]
    fn tampered_embedded_graph_report_invalidates_receipt() {
        let config = fixture();
        let tasks = vec![task("inspect-grounding")];
        let (mut report, receipt) = evaluate_mission_tasks(&config, &tasks).unwrap();
        report.mission_configuration_report.mission_id.push_str("-tampered");
        assert!(!verify_mission_task_receipt(&config, &tasks, &report, &receipt));
    }

    #[test]
    fn empty_or_ambiguous_task_declarations_fail_closed() {
        let config = fixture();
        assert_eq!(
            evaluate_mission_tasks(&config, &[]).unwrap_err(),
            MissionTaskError::EmptyTaskSet
        );
        let mut no_dependencies = task("empty-task");
        no_dependencies.required_capabilities.clear();
        no_dependencies.required_component_ids.clear();
        no_dependencies.required_connection_ids.clear();
        assert_eq!(
            evaluate_mission_tasks(&config, &[no_dependencies]).unwrap_err(),
            MissionTaskError::TaskHasNoDeclaredDependencies
        );
        assert_eq!(
            evaluate_mission_tasks(&config, &[task("same"), task("same")]).unwrap_err(),
            MissionTaskError::DuplicateTaskIdentity
        );
    }

    #[test]
    fn receipt_is_independent_of_component_vector_permutation() {
        let mut config = fixture();
        let tasks = vec![task("inspect-grounding")];
        let (report_a, receipt_a) = evaluate_mission_tasks(&config, &tasks).unwrap();
        config.components.reverse();
        config.connections.reverse();
        config.suit_variant.capability_claims.reverse();
        let (report_b, receipt_b) = evaluate_mission_tasks(&config, &tasks).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);
    }
}
