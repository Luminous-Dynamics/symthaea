// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read-only evidence join for aircraft electrical fault-containment analysis.
//! This crate does not simulate faults, calculate fault current, evaluate trip
//! curves, infer isolation, or establish safety.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MODEL_VERSION: &str = "airroad-fault-evidence-join-v0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PowerElementKind {
    Source,
    BusSegment,
    BusTie,
    Branch,
    Feeder,
    Cable,
    Connector,
    Contactor,
    ProtectionDevice,
    Propulsor,
    Sensor,
    CoolingSystem,
}

/// Exact reference to the upstream canonical AsBuiltPowerTree, not a local clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AircraftPowerTreeRef {
    pub tree_id: String,
    pub tree_digest: String,
    pub tree_model_version: String,
    pub configuration_digest: String,
    pub as_built_manifest_digest: String,
}

/// Crosswalk from an aircraft-domain ID to one canonical power-tree element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AircraftElementBinding {
    pub aircraft_element_id: String,
    pub power_tree_element_id: String,
    pub kind: PowerElementKind,
    pub configuration_digest: String,
    pub element_evidence_manifest_digest: String,
    pub physical_zone_id: String,
    pub common_cause_group_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropulsionPathBinding {
    pub drive_path_id: String,
    pub propulsor_element_id: String,
    pub bus_element_id: String,
    pub branch_element_id: String,
    pub feeder_element_id: String,
    pub protection_device_element_ids: Vec<String>,
    pub configuration_digest: String,
    pub mission_drive_profile_digest: String,
}

/// Device-specific declared metadata. No generic trip curve or rating is inferred.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtectionDeviceBinding {
    pub protection_device_id: String,
    pub aircraft_element_id: String,
    pub model_or_part_id: String,
    pub serial_or_instance_id: String,
    pub configuration_digest: String,
    pub rated_voltage_v: f64,
    pub interrupting_capacity_a: Option<f64>,
    pub trip_curve_digest: Option<String>,
    pub interrupting_capacity_evidence_digest: Option<String>,
    pub evidence_manifest_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FaultKind {
    BranchOpenCircuit,
    BranchShortCircuit,
    FeederToGroundFault,
    BusSegmentShort,
    ContactorStuckOpen,
    ContactorWeldedClosed,
    ProtectionFailedToTrip,
    ProtectionOutsideDeclaredCurve,
    BusTieStuckClosed,
    SourceUnavailable,
    CommonCoolingFailure,
    SharedSensorOrCommunicationLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsequenceDomain {
    Electrical,
    MissionCapability,
    Thermal,
    EmcOrPowerIntegrity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AircraftFaultScenario {
    pub scenario_id: String,
    pub fault_kind: FaultKind,
    pub target_aircraft_element_ids: Vec<String>,
    pub required_protection_device_ids: Vec<String>,
    pub mission_segment_id: String,
    pub operating_context_digest: String,
    pub affected_consequence_domains: Vec<ConsequenceDomain>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FaultEvidenceKind {
    AsBuiltTopology,
    FaultWaveform,
    ProtectionTimeCurrentCurve,
    InterruptingCapacity,
    ClearingTime,
    IsolationState,
    PostFaultElectricalAvailability,
    PostFaultAircraftCapability,
    ThermalConsequence,
    EmcOrPowerIntegrityConsequence,
    AuthorityDisposition,
    CertificationApproval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceClass {
    EngineeringAnalysis,
    SimulatedResult,
    MeasuredTest,
    AuthorityDecision,
    CertificationApproval,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProvenance {
    pub class: EvidenceClass,
    pub source_id: String,
    pub source_revision: String,
    pub input_manifest_digest: Option<String>,
    pub output_manifest_digest: Option<String>,
    pub validation_manifest_digest: Option<String>,
    pub procedure_digest: Option<String>,
    pub calibration_manifest_digest: Option<String>,
    pub raw_data_digest: Option<String>,
    pub decision_or_certificate_id: Option<String>,
    pub approval_scope_digest: Option<String>,
}

/// Opaque evidence reference. Digests are checked for identity equality but not
/// recomputed from bytes or cryptographically authenticated by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultEvidenceArtifact {
    pub artifact_id: String,
    pub artifact_digest: String,
    pub kind: FaultEvidenceKind,
    pub subject_id: String,
    pub scenario_id: String,
    pub configuration_digest: String,
    pub topology_digest: String,
    pub operating_context_digest: String,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaultEvidenceJoinInput {
    pub configuration_digest: String,
    pub topology: AircraftPowerTreeRef,
    pub element_bindings: Vec<AircraftElementBinding>,
    pub propulsion_paths: Vec<PropulsionPathBinding>,
    pub protection_devices: Vec<ProtectionDeviceBinding>,
    pub scenarios: Vec<AircraftFaultScenario>,
    pub artifacts: Vec<FaultEvidenceArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceRequirement {
    pub kind: FaultEvidenceKind,
    pub subject_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceJoinDisposition {
    ReadyForIndependentReview,
    UnknownMissingEvidence,
    NonPassIdentityOrProvenanceMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultScenarioEvidenceReport {
    pub scenario_id: String,
    pub disposition: EvidenceJoinDisposition,
    pub required_evidence: Vec<EvidenceRequirement>,
    pub missing_evidence: Vec<EvidenceRequirement>,
    pub mismatches: Vec<String>,
    pub artifact_ids: Vec<String>,
    pub claim_classes_present: Vec<EvidenceClass>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultEvidenceJoinReport {
    pub model_version: &'static str,
    pub configuration_digest: String,
    pub topology_id: String,
    pub topology_digest: String,
    pub scenario_reports: Vec<FaultScenarioEvidenceReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultEvidenceJoinError {
    InvalidInput(&'static str),
    DuplicateIdentity(&'static str),
    MissingElement(String),
    WrongElementKind {
        element_id: String,
        expected: PowerElementKind,
    },
    MissingProtectionDevice(String),
}

impl fmt::Display for FaultEvidenceJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FaultEvidenceJoinError {}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn optional_non_empty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|value| non_empty(value))
}

fn required_evidence_for(
    scenario: &AircraftFaultScenario,
    tree_id: &str,
) -> Vec<EvidenceRequirement> {
    let mut required = BTreeSet::new();
    let mut add = |kind, subject: &str| {
        required.insert(EvidenceRequirement {
            kind,
            subject_id: subject.to_owned(),
        });
    };

    add(FaultEvidenceKind::AsBuiltTopology, tree_id);
    add(FaultEvidenceKind::IsolationState, &scenario.scenario_id);
    add(
        FaultEvidenceKind::PostFaultElectricalAvailability,
        &scenario.scenario_id,
    );
    add(
        FaultEvidenceKind::PostFaultAircraftCapability,
        &scenario.scenario_id,
    );

    match scenario.fault_kind {
        FaultKind::BranchShortCircuit
        | FaultKind::FeederToGroundFault
        | FaultKind::BusSegmentShort
        | FaultKind::ProtectionFailedToTrip
        | FaultKind::ProtectionOutsideDeclaredCurve
        | FaultKind::BusTieStuckClosed
        | FaultKind::SourceUnavailable => {
            add(FaultEvidenceKind::FaultWaveform, &scenario.scenario_id);
            add(FaultEvidenceKind::ClearingTime, &scenario.scenario_id);
        }
        FaultKind::BranchOpenCircuit
        | FaultKind::ContactorStuckOpen
        | FaultKind::ContactorWeldedClosed
        | FaultKind::CommonCoolingFailure
        | FaultKind::SharedSensorOrCommunicationLoss => {}
    }

    for device_id in &scenario.required_protection_device_ids {
        add(FaultEvidenceKind::ProtectionTimeCurrentCurve, device_id);
        add(FaultEvidenceKind::InterruptingCapacity, device_id);
    }

    if scenario
        .affected_consequence_domains
        .contains(&ConsequenceDomain::Thermal)
    {
        add(FaultEvidenceKind::ThermalConsequence, &scenario.scenario_id);
    }
    if scenario
        .affected_consequence_domains
        .contains(&ConsequenceDomain::EmcOrPowerIntegrity)
    {
        add(
            FaultEvidenceKind::EmcOrPowerIntegrityConsequence,
            &scenario.scenario_id,
        );
    }

    required.into_iter().collect()
}

fn validate_provenance(provenance: &EvidenceProvenance) -> Result<(), &'static str> {
    if !non_empty(&provenance.source_id) || !non_empty(&provenance.source_revision) {
        return Err("source identity and revision are required");
    }

    match provenance.class {
        EvidenceClass::EngineeringAnalysis => Ok(()),
        EvidenceClass::SimulatedResult => {
            if optional_non_empty(&provenance.input_manifest_digest)
                && optional_non_empty(&provenance.output_manifest_digest)
                && optional_non_empty(&provenance.validation_manifest_digest)
            {
                Ok(())
            } else {
                Err("simulation requires input, output and model-validation manifest digests")
            }
        }
        EvidenceClass::MeasuredTest => {
            if optional_non_empty(&provenance.procedure_digest)
                && optional_non_empty(&provenance.calibration_manifest_digest)
                && optional_non_empty(&provenance.raw_data_digest)
            {
                Ok(())
            } else {
                Err("measured test requires procedure, calibration and raw-data digests")
            }
        }
        EvidenceClass::AuthorityDecision | EvidenceClass::CertificationApproval => {
            if optional_non_empty(&provenance.decision_or_certificate_id)
                && optional_non_empty(&provenance.approval_scope_digest)
            {
                Ok(())
            } else {
                Err("authority/certification evidence requires decision ID and exact scope digest")
            }
        }
    }
}

fn validate_topology(
    input: &FaultEvidenceJoinInput,
) -> Result<BTreeMap<String, AircraftElementBinding>, FaultEvidenceJoinError> {
    if !non_empty(&input.configuration_digest)
        || !non_empty(&input.topology.tree_id)
        || !non_empty(&input.topology.tree_digest)
        || !non_empty(&input.topology.tree_model_version)
        || !non_empty(&input.topology.configuration_digest)
        || !non_empty(&input.topology.as_built_manifest_digest)
        || input.topology.configuration_digest != input.configuration_digest
        || input.element_bindings.is_empty()
        || input.propulsion_paths.is_empty()
        || input.scenarios.is_empty()
    {
        return Err(FaultEvidenceJoinError::InvalidInput(
            "configuration, exact power tree, crosswalk, propulsion paths and scenarios are required",
        ));
    }

    let mut by_aircraft_id = BTreeMap::new();
    let mut power_tree_ids = BTreeSet::new();
    for binding in &input.element_bindings {
        if !non_empty(&binding.aircraft_element_id)
            || !non_empty(&binding.power_tree_element_id)
            || !non_empty(&binding.configuration_digest)
            || !non_empty(&binding.element_evidence_manifest_digest)
            || !non_empty(&binding.physical_zone_id)
            || !non_empty(&binding.common_cause_group_id)
            || binding.configuration_digest != input.configuration_digest
        {
            return Err(FaultEvidenceJoinError::InvalidInput(
                "element binding identity, zone, common-cause and configuration evidence are required",
            ));
        }
        if by_aircraft_id
            .insert(binding.aircraft_element_id.clone(), binding.clone())
            .is_some()
        {
            return Err(FaultEvidenceJoinError::DuplicateIdentity(
                "aircraft element ID",
            ));
        }
        if !power_tree_ids.insert(binding.power_tree_element_id.clone()) {
            return Err(FaultEvidenceJoinError::DuplicateIdentity(
                "canonical power-tree element ID",
            ));
        }
    }

    let mut path_ids = BTreeSet::new();
    for path in &input.propulsion_paths {
        if !non_empty(&path.drive_path_id)
            || !non_empty(&path.configuration_digest)
            || !non_empty(&path.mission_drive_profile_digest)
            || path.configuration_digest != input.configuration_digest
        {
            return Err(FaultEvidenceJoinError::InvalidInput(
                "propulsion path and mission profile evidence identities are required",
            ));
        }
        if !path_ids.insert(path.drive_path_id.clone()) {
            return Err(FaultEvidenceJoinError::DuplicateIdentity("drive path ID"));
        }
        for (element_id, expected) in [
            (&path.propulsor_element_id, PowerElementKind::Propulsor),
            (&path.bus_element_id, PowerElementKind::BusSegment),
            (&path.branch_element_id, PowerElementKind::Branch),
            (&path.feeder_element_id, PowerElementKind::Feeder),
        ] {
            let binding = by_aircraft_id
                .get(element_id)
                .ok_or_else(|| FaultEvidenceJoinError::MissingElement(element_id.clone()))?;
            if binding.kind != expected {
                return Err(FaultEvidenceJoinError::WrongElementKind {
                    element_id: element_id.clone(),
                    expected,
                });
            }
        }
        for device_element_id in &path.protection_device_element_ids {
            let binding = by_aircraft_id
                .get(device_element_id)
                .ok_or_else(|| FaultEvidenceJoinError::MissingElement(device_element_id.clone()))?;
            if binding.kind != PowerElementKind::ProtectionDevice {
                return Err(FaultEvidenceJoinError::WrongElementKind {
                    element_id: device_element_id.clone(),
                    expected: PowerElementKind::ProtectionDevice,
                });
            }
        }
    }

    let mut device_ids = BTreeSet::new();
    for device in &input.protection_devices {
        if !non_empty(&device.protection_device_id)
            || !non_empty(&device.aircraft_element_id)
            || !non_empty(&device.model_or_part_id)
            || !non_empty(&device.serial_or_instance_id)
            || !non_empty(&device.configuration_digest)
            || !non_empty(&device.evidence_manifest_digest)
            || device.configuration_digest != input.configuration_digest
            || !device.rated_voltage_v.is_finite()
            || device.rated_voltage_v <= 0.0
        {
            return Err(FaultEvidenceJoinError::InvalidInput(
                "device identity, exact instance, rating, evidence and configuration are required",
            ));
        }
        if let Some(value) = device.interrupting_capacity_a {
            if !value.is_finite() || value <= 0.0 {
                return Err(FaultEvidenceJoinError::InvalidInput(
                    "interrupting capacity, when supplied, must be finite and positive",
                ));
            }
        }
        if !device_ids.insert(device.protection_device_id.clone()) {
            return Err(FaultEvidenceJoinError::DuplicateIdentity(
                "protection device ID",
            ));
        }
        let binding = by_aircraft_id
            .get(&device.aircraft_element_id)
            .ok_or_else(|| FaultEvidenceJoinError::MissingElement(device.aircraft_element_id.clone()))?;
        if binding.kind != PowerElementKind::ProtectionDevice {
            return Err(FaultEvidenceJoinError::WrongElementKind {
                element_id: device.aircraft_element_id.clone(),
                expected: PowerElementKind::ProtectionDevice,
            });
        }
    }

    for scenario in &input.scenarios {
        if !non_empty(&scenario.scenario_id)
            || !non_empty(&scenario.mission_segment_id)
            || !non_empty(&scenario.operating_context_digest)
            || scenario.target_aircraft_element_ids.is_empty()
        {
            return Err(FaultEvidenceJoinError::InvalidInput(
                "scenario ID, target, mission segment and context are required",
            ));
        }
        for target in &scenario.target_aircraft_element_ids {
            if !by_aircraft_id.contains_key(target) {
                return Err(FaultEvidenceJoinError::MissingElement(target.clone()));
            }
        }
        for device_id in &scenario.required_protection_device_ids {
            if !device_ids.contains(device_id) {
                return Err(FaultEvidenceJoinError::MissingProtectionDevice(
                    device_id.clone(),
                ));
            }
        }
    }

    Ok(by_aircraft_id)
}

/// Join scenario evidence against the exact aircraft crosswalk.
///
/// ReadyForIndependentReview means required references and scope identities are
/// present. It is not a safety PASS, airworthiness result or certification.
pub fn join_fault_evidence(
    input: &FaultEvidenceJoinInput,
) -> Result<FaultEvidenceJoinReport, FaultEvidenceJoinError> {
    let _crosswalk = validate_topology(input)?;

    let mut scenario_ids = BTreeSet::new();
    for scenario in &input.scenarios {
        if !scenario_ids.insert(scenario.scenario_id.clone()) {
            return Err(FaultEvidenceJoinError::DuplicateIdentity("scenario ID"));
        }
    }

    let mut artifact_ids = BTreeSet::new();
    for artifact in &input.artifacts {
        if !non_empty(&artifact.artifact_id) {
            return Err(FaultEvidenceJoinError::InvalidInput("artifact ID must not be empty"));
        }
        if !artifact_ids.insert(artifact.artifact_id.clone()) {
            return Err(FaultEvidenceJoinError::DuplicateIdentity("artifact ID"));
        }
    }

    let mut scenarios = input.scenarios.clone();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut reports = Vec::with_capacity(scenarios.len());

    for scenario in &scenarios {
        let required = required_evidence_for(scenario, &input.topology.tree_id);
        let mut missing = Vec::new();
        let mut mismatches = Vec::new();
        let mut used_ids = BTreeSet::new();
        let mut claim_classes = BTreeSet::new();
        let artifacts = input
            .artifacts
            .iter()
            .filter(|artifact| artifact.scenario_id == scenario.scenario_id)
            .collect::<Vec<_>>();

        for artifact in &artifacts {
            if !non_empty(&artifact.artifact_digest)
                || !non_empty(&artifact.subject_id)
                || !non_empty(&artifact.configuration_digest)
                || !non_empty(&artifact.topology_digest)
                || !non_empty(&artifact.operating_context_digest)
            {
                mismatches.push(format!("artifact {} has incomplete identity or scope", artifact.artifact_id));
                continue;
            }
            if artifact.configuration_digest != input.configuration_digest
                || artifact.topology_digest != input.topology.tree_digest
                || artifact.operating_context_digest != scenario.operating_context_digest
            {
                mismatches.push(format!(
                    "artifact {} belongs to another configuration, topology or operating context",
                    artifact.artifact_id
                ));
                continue;
            }
            if let Err(reason) = validate_provenance(&artifact.provenance) {
                mismatches.push(format!(
                    "artifact {} has invalid {:?} provenance: {reason}",
                    artifact.artifact_id, artifact.provenance.class
                ));
                continue;
            }
            claim_classes.insert(artifact.provenance.class);
            used_ids.insert(artifact.artifact_id.clone());
        }

        for requirement in &required {
            let matches = artifacts
                .iter()
                .filter(|artifact| {
                    artifact.kind == requirement.kind
                        && artifact.subject_id == requirement.subject_id
                        && artifact.configuration_digest == input.configuration_digest
                        && artifact.topology_digest == input.topology.tree_digest
                        && artifact.operating_context_digest == scenario.operating_context_digest
                        && validate_provenance(&artifact.provenance).is_ok()
                })
                .collect::<Vec<_>>();

            if matches.is_empty() {
                missing.push(requirement.clone());
                continue;
            }

            if requirement.kind == FaultEvidenceKind::AsBuiltTopology
                && matches.iter().all(|artifact| {
                    artifact.artifact_digest != input.topology.as_built_manifest_digest
                })
            {
                mismatches.push(format!(
                    "as-built topology for scenario {} does not match the pinned manifest",
                    scenario.scenario_id
                ));
            }

            if requirement.kind == FaultEvidenceKind::ProtectionTimeCurrentCurve {
                if let Some(device) = input.protection_devices.iter().find(|device| {
                    device.protection_device_id == requirement.subject_id
                }) {
                    if device.trip_curve_digest.as_deref().is_none()
                        || matches.iter().all(|artifact| {
                            Some(artifact.artifact_digest.as_str())
                                != device.trip_curve_digest.as_deref()
                        })
                    {
                        mismatches.push(format!(
                            "protection curve for device {} does not match its pinned digest",
                            requirement.subject_id
                        ));
                    }
                }
            }

            if requirement.kind == FaultEvidenceKind::InterruptingCapacity {
                if let Some(device) = input.protection_devices.iter().find(|device| {
                    device.protection_device_id == requirement.subject_id
                }) {
                    if device.interrupting_capacity_a.is_none()
                        || device.interrupting_capacity_evidence_digest.as_deref().is_none()
                        || matches.iter().all(|artifact| {
                            Some(artifact.artifact_digest.as_str())
                                != device.interrupting_capacity_evidence_digest.as_deref()
                        })
                    {
                        mismatches.push(format!(
                            "interrupting capacity for device {} is absent or has a different digest",
                            requirement.subject_id
                        ));
                    }
                }
            }
        }

        let disposition = if !mismatches.is_empty() {
            EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch
        } else if !missing.is_empty() {
            EvidenceJoinDisposition::UnknownMissingEvidence
        } else {
            EvidenceJoinDisposition::ReadyForIndependentReview
        };

        reports.push(FaultScenarioEvidenceReport {
            scenario_id: scenario.scenario_id.clone(),
            disposition,
            required_evidence: required,
            missing_evidence: missing,
            mismatches,
            artifact_ids: used_ids.into_iter().collect(),
            claim_classes_present: claim_classes.into_iter().collect(),
        });
    }

    Ok(FaultEvidenceJoinReport {
        model_version: MODEL_VERSION,
        configuration_digest: input.configuration_digest.clone(),
        topology_id: input.topology.tree_id.clone(),
        topology_digest: input.topology.tree_digest.clone(),
        scenario_reports: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(class: EvidenceClass) -> EvidenceProvenance {
        let mut value = EvidenceProvenance {
            class,
            source_id: "source-01".to_owned(),
            source_revision: "rev-1".to_owned(),
            input_manifest_digest: None,
            output_manifest_digest: None,
            validation_manifest_digest: None,
            procedure_digest: None,
            calibration_manifest_digest: None,
            raw_data_digest: None,
            decision_or_certificate_id: None,
            approval_scope_digest: None,
        };
        match class {
            EvidenceClass::EngineeringAnalysis => {}
            EvidenceClass::SimulatedResult => {
                value.input_manifest_digest = Some("input".to_owned());
                value.output_manifest_digest = Some("output".to_owned());
                value.validation_manifest_digest = Some("solver-validation".to_owned());
            }
            EvidenceClass::MeasuredTest => {
                value.procedure_digest = Some("procedure".to_owned());
                value.calibration_manifest_digest = Some("calibration".to_owned());
                value.raw_data_digest = Some("raw-data".to_owned());
            }
            EvidenceClass::AuthorityDecision | EvidenceClass::CertificationApproval => {
                value.decision_or_certificate_id = Some("decision-or-certificate".to_owned());
                value.approval_scope_digest = Some("approval-scope".to_owned());
            }
        }
        value
    }

    fn artifact(
        id: &str,
        kind: FaultEvidenceKind,
        subject: &str,
        scenario: &str,
    ) -> FaultEvidenceArtifact {
        FaultEvidenceArtifact {
            artifact_id: id.to_owned(),
            artifact_digest: format!("digest-{id}"),
            kind,
            subject_id: subject.to_owned(),
            scenario_id: scenario.to_owned(),
            configuration_digest: "aircraft-config".to_owned(),
            topology_digest: "topology-digest".to_owned(),
            operating_context_digest: "hover-hot-context".to_owned(),
            provenance: provenance(EvidenceClass::MeasuredTest),
        }
    }

    fn base_input() -> FaultEvidenceJoinInput {
        let topology = AircraftPowerTreeRef {
            tree_id: "as-built-power-tree".to_owned(),
            tree_digest: "topology-digest".to_owned(),
            tree_model_version: "power-tree-v1".to_owned(),
            configuration_digest: "aircraft-config".to_owned(),
            as_built_manifest_digest: "as-built-manifest".to_owned(),
        };
        let element = |aircraft_id: &str, tree_id: &str, kind| AircraftElementBinding {
            aircraft_element_id: aircraft_id.to_owned(),
            power_tree_element_id: tree_id.to_owned(),
            kind,
            configuration_digest: "aircraft-config".to_owned(),
            element_evidence_manifest_digest: format!("evidence-{aircraft_id}"),
            physical_zone_id: "zone-a".to_owned(),
            common_cause_group_id: "common-cause-a".to_owned(),
        };
        let elements = vec![
            element("source-a", "tree-source-a", PowerElementKind::Source),
            element("bus-a", "tree-bus-a", PowerElementKind::BusSegment),
            element("branch-a", "tree-branch-a", PowerElementKind::Branch),
            element("feeder-a", "tree-feeder-a", PowerElementKind::Feeder),
            element("propulsor-a", "tree-propulsor-a", PowerElementKind::Propulsor),
            element("protection-a", "tree-protection-a", PowerElementKind::ProtectionDevice),
        ];
        let path = PropulsionPathBinding {
            drive_path_id: "drive-left".to_owned(),
            propulsor_element_id: "propulsor-a".to_owned(),
            bus_element_id: "bus-a".to_owned(),
            branch_element_id: "branch-a".to_owned(),
            feeder_element_id: "feeder-a".to_owned(),
            protection_device_element_ids: vec!["protection-a".to_owned()],
            configuration_digest: "aircraft-config".to_owned(),
            mission_drive_profile_digest: "mission-drive-profile".to_owned(),
        };
        let protection = ProtectionDeviceBinding {
            protection_device_id: "breaker-left".to_owned(),
            aircraft_element_id: "protection-a".to_owned(),
            model_or_part_id: "breaker-part-123".to_owned(),
            serial_or_instance_id: "breaker-serial-456".to_owned(),
            configuration_digest: "aircraft-config".to_owned(),
            rated_voltage_v: 800.0,
            interrupting_capacity_a: Some(5_000.0),
            trip_curve_digest: Some("curve-digest".to_owned()),
            interrupting_capacity_evidence_digest: Some("interrupt-digest".to_owned()),
            evidence_manifest_digest: "breaker-evidence".to_owned(),
        };
        let scenario = AircraftFaultScenario {
            scenario_id: "fault-left-short".to_owned(),
            fault_kind: FaultKind::BranchShortCircuit,
            target_aircraft_element_ids: vec!["branch-a".to_owned()],
            required_protection_device_ids: vec!["breaker-left".to_owned()],
            mission_segment_id: "hover-segment".to_owned(),
            operating_context_digest: "hover-hot-context".to_owned(),
            affected_consequence_domains: vec![
                ConsequenceDomain::Electrical,
                ConsequenceDomain::MissionCapability,
                ConsequenceDomain::Thermal,
            ],
        };
        let mut artifacts = vec![
            artifact("topology", FaultEvidenceKind::AsBuiltTopology, "as-built-power-tree", "fault-left-short"),
            artifact("waveform", FaultEvidenceKind::FaultWaveform, "fault-left-short", "fault-left-short"),
            artifact("clearing", FaultEvidenceKind::ClearingTime, "fault-left-short", "fault-left-short"),
            artifact("isolation", FaultEvidenceKind::IsolationState, "fault-left-short", "fault-left-short"),
            artifact("availability", FaultEvidenceKind::PostFaultElectricalAvailability, "fault-left-short", "fault-left-short"),
            artifact("capability", FaultEvidenceKind::PostFaultAircraftCapability, "fault-left-short", "fault-left-short"),
            artifact("thermal", FaultEvidenceKind::ThermalConsequence, "fault-left-short", "fault-left-short"),
            artifact("curve", FaultEvidenceKind::ProtectionTimeCurrentCurve, "breaker-left", "fault-left-short"),
            artifact("interrupt", FaultEvidenceKind::InterruptingCapacity, "breaker-left", "fault-left-short"),
        ];
        artifacts[0].artifact_digest = "as-built-manifest".to_owned();
        artifacts[7].artifact_digest = "curve-digest".to_owned();
        artifacts[8].artifact_digest = "interrupt-digest".to_owned();
        FaultEvidenceJoinInput {
            configuration_digest: "aircraft-config".to_owned(),
            topology,
            element_bindings: elements,
            propulsion_paths: vec![path],
            protection_devices: vec![protection],
            scenarios: vec![scenario],
            artifacts,
        }
    }

    #[test]
    fn complete_join_is_ready_only_for_independent_review() {
        let report = join_fault_evidence(&base_input()).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::ReadyForIndependentReview);
        assert!(report.scenario_reports[0].missing_evidence.is_empty());
        assert_eq!(report.scenario_reports[0].artifact_ids.len(), 9);
    }

    #[test]
    fn missing_post_fault_capability_is_unknown_not_pass() {
        let mut input = base_input();
        input.artifacts.retain(|item| item.kind != FaultEvidenceKind::PostFaultAircraftCapability);
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::UnknownMissingEvidence);
    }

    #[test]
    fn stale_configuration_or_topology_evidence_is_non_pass() {
        let mut input = base_input();
        input.artifacts[1].topology_digest = "stale-topology".to_owned();
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn protection_curve_must_match_the_exact_device_digest() {
        let mut input = base_input();
        input.artifacts.iter_mut().find(|item| item.kind == FaultEvidenceKind::ProtectionTimeCurrentCurve).unwrap().artifact_digest = "other-curve".to_owned();
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn missing_protection_device_is_not_silently_ignored() {
        let mut input = base_input();
        input.protection_devices.clear();
        assert!(matches!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::MissingProtectionDevice(_))));
    }

    #[test]
    fn missing_curve_or_interrupt_rating_is_not_review_ready() {
        let mut input = base_input();
        input.protection_devices[0].trip_curve_digest = None;
        input.protection_devices[0].interrupting_capacity_a = None;
        input.protection_devices[0].interrupting_capacity_evidence_digest = None;
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn simulated_result_requires_model_validation_manifest() {
        let mut input = base_input();
        let item = input.artifacts.iter_mut().find(|item| item.kind == FaultEvidenceKind::FaultWaveform).unwrap();
        item.provenance.class = EvidenceClass::SimulatedResult;
        item.provenance.input_manifest_digest = Some("input".to_owned());
        item.provenance.output_manifest_digest = Some("output".to_owned());
        item.provenance.validation_manifest_digest = None;
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn measured_test_requires_procedure_calibration_and_raw_data() {
        let mut input = base_input();
        input.artifacts.iter_mut().find(|item| item.kind == FaultEvidenceKind::FaultWaveform).unwrap().provenance.calibration_manifest_digest = None;
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn authority_and_certification_claims_are_preserved_without_promotion() {
        let mut input = base_input();
        let mut authority = artifact("authority", FaultEvidenceKind::AuthorityDisposition, "fault-left-short", "fault-left-short");
        authority.provenance = provenance(EvidenceClass::AuthorityDecision);
        let mut certificate = artifact("certificate", FaultEvidenceKind::CertificationApproval, "fault-left-short", "fault-left-short");
        certificate.provenance = provenance(EvidenceClass::CertificationApproval);
        input.artifacts.extend([authority, certificate]);
        let report = join_fault_evidence(&input).unwrap();
        assert!(report.scenario_reports[0].claim_classes_present.contains(&EvidenceClass::AuthorityDecision));
        assert!(report.scenario_reports[0].claim_classes_present.contains(&EvidenceClass::CertificationApproval));
    }

    #[test]
    fn thermal_and_emc_consequence_domains_are_separately_required() {
        let mut input = base_input();
        input.scenarios[0].affected_consequence_domains.push(ConsequenceDomain::EmcOrPowerIntegrity);
        input.artifacts.retain(|item| item.kind != FaultEvidenceKind::EmcOrPowerIntegrityConsequence);
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::UnknownMissingEvidence);
        assert!(report.scenario_reports[0].missing_evidence.iter().any(|item| item.kind == FaultEvidenceKind::EmcOrPowerIntegrityConsequence));
    }

    #[test]
    fn duplicate_element_and_artifact_identities_fail_closed() {
        let mut input = base_input();
        input.element_bindings.push(input.element_bindings[0].clone());
        assert_eq!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::DuplicateIdentity("aircraft element ID")));
        let mut input = base_input();
        input.artifacts.push(input.artifacts[0].clone());
        assert_eq!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::DuplicateIdentity("artifact ID")));
    }

    #[test]
    fn dangling_propulsion_path_element_fails_closed() {
        let mut input = base_input();
        input.propulsion_paths[0].feeder_element_id = "missing-feeder".to_owned();
        assert!(matches!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::MissingElement(id)) if id == "missing-feeder"));
    }

    #[test]
    fn cross_configuration_element_binding_fails_closed() {
        let mut input = base_input();
        input.element_bindings[0].configuration_digest = "other-config".to_owned();
        assert!(matches!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::InvalidInput(_))));
    }

    #[test]
    fn scenario_targets_must_exist_in_the_exact_crosswalk() {
        let mut input = base_input();
        input.scenarios[0].target_aircraft_element_ids = vec!["unknown-target".to_owned()];
        assert!(matches!(join_fault_evidence(&input), Err(FaultEvidenceJoinError::MissingElement(id)) if id == "unknown-target"));
    }

    #[test]
    fn evidence_from_a_different_operating_context_is_non_pass() {
        let mut input = base_input();
        input.artifacts[0].operating_context_digest = "cold-context".to_owned();
        let report = join_fault_evidence(&input).unwrap();
        assert_eq!(report.scenario_reports[0].disposition, EvidenceJoinDisposition::NonPassIdentityOrProvenanceMismatch);
    }

    #[test]
    fn scenario_order_is_canonical_and_deterministic() {
        let mut input = base_input();
        let mut second = input.scenarios[0].clone();
        second.scenario_id = "fault-left-open".to_owned();
        second.fault_kind = FaultKind::BranchOpenCircuit;
        second.required_protection_device_ids.clear();
        second.affected_consequence_domains = vec![ConsequenceDomain::Electrical, ConsequenceDomain::MissionCapability];
        let mut topology = artifact("topology-open", FaultEvidenceKind::AsBuiltTopology, "as-built-power-tree", "fault-left-open");
        topology.artifact_digest = "as-built-manifest".to_owned();
        let other = vec![
            topology,
            artifact("isolation-open", FaultEvidenceKind::IsolationState, "fault-left-open", "fault-left-open"),
            artifact("availability-open", FaultEvidenceKind::PostFaultElectricalAvailability, "fault-left-open", "fault-left-open"),
            artifact("capability-open", FaultEvidenceKind::PostFaultAircraftCapability, "fault-left-open", "fault-left-open"),
        ];
        input.scenarios.push(second);
        input.artifacts.extend(other);
        let first = join_fault_evidence(&input).unwrap();
        input.scenarios.reverse();
        input.artifacts.reverse();
        let second = join_fault_evidence(&input).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.scenario_reports[0].scenario_id, "fault-left-open");
    }
}
