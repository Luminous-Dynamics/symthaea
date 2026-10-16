// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-level suit configuration verification (SX-031).
//!
//! This module composes declared suit-profile evidence and pairwise interface
//! contracts into a deterministic mission graph result. It is a software
//! contract checker, not a physical model, hardware qualification, flight
//! readiness determination, or safety authority. It has no actuator surface.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::suit_interop::{
    evaluate_suit_interoperability, SuitInterfaceManifest, SuitInteropError,
    SuitInteropPolicy, SuitInteropReceipt, SuitInteropReport, SuitInteropReportStatus,
};
use crate::suit_profile::{
    evaluate_suit_profile, SuitCapability, SuitEvidenceLevel, SuitMissionProfile,
    SuitProfileError, SuitProfileEvaluation, SuitProfileEvaluationStatus,
    SuitProfileEvidencePolicy, SuitVariantDeclaration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MissionComponentKind {
    Suit,
    LanderOrVehicle,
    Habitat,
    Rover,
    ServiceStation,
    RescueCapability,
    CommunicationsRelay,
    Worksite,
    GroundSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionComponent {
    pub component_id: String,
    pub kind: MissionComponentKind,
    pub hardware_revision: String,
    pub interfaces: SuitInterfaceManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionConnection {
    pub connection_id: String,
    pub left_component_id: String,
    pub right_component_id: String,
    pub policy: SuitInteropPolicy,
}

/// The required sets are mission-author choices, not inferred from the profile.
/// Required connection IDs are intentionally separate from the connection
/// declarations so a missing required link can be reported rather than vanish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionConfiguration {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub suit_component_id: String,
    pub suit_profile: SuitMissionProfile,
    pub suit_variant: SuitVariantDeclaration,
    pub minimum_capability_evidence: SuitEvidenceLevel,
    pub required_component_kinds: BTreeSet<MissionComponentKind>,
    pub required_connection_ids: BTreeSet<String>,
    pub components: Vec<MissionComponent>,
    pub connections: Vec<MissionConnection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionFindingCode {
    MissingRequiredComponent,
    MissingRequiredConnection,
    RequiredComponentDisconnected,
    SuitProfileBlocked,
    RequiredConnectionHasNoRequiredInterfaces,
    RequiredConnectionBlocked,
    OptionalConnectionDegraded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionFinding {
    pub code: MissionFindingCode,
    pub subject_id: String,
    pub severity: MissionFindingSeverity,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionConnectionEvaluation {
    pub connection_id: String,
    pub left_component_id: String,
    pub right_component_id: String,
    pub required: bool,
    pub interoperability: SuitInteropReport,
    pub interoperability_receipt: SuitInteropReceipt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionConfigurationStatus {
    Blocked,
    Degraded,
    MeetsDeclaredMissionConfigurationPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionConfigurationReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub configuration_digest_sha256: String,
    pub suit_profile_evaluation: SuitProfileEvaluation,
    pub connection_evaluations: Vec<MissionConnectionEvaluation>,
    pub findings: Vec<MissionFinding>,
    pub status: MissionConfigurationStatus,
    pub component_count: usize,
    pub connection_count: usize,
    pub required_component_kind_count: usize,
    pub present_required_component_kind_count: usize,
    pub required_connection_count: usize,
    pub missing_required_connection_count: usize,
    pub compatible_required_connection_count: usize,
    pub blocked_required_connection_count: usize,
    pub missing_required_component_kind_count: usize,
    pub disconnected_required_component_count: usize,
    /// Always false: graph consistency is not physical qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionConfigurationReceipt {
    pub configuration_digest_sha256: String,
    pub report_digest_sha256: String,
    /// Deterministic change-evidence; not a signature or authentication claim.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionConfigurationError {
    UnsupportedSchemaVersion,
    InvalidMissionIdentity,
    InvalidMissionRequirements,
    InvalidSuitVariant,
    MissingSuitComponent,
    DuplicateComponentId,
    InvalidComponentManifest,
    ComponentManifestMismatch,
    DuplicateConnectionId,
    InvalidConnection,
    DuplicateComponentPair,
}

impl MissionConfiguration {
    /// Validate only structural identity and ambiguity. Missing declared
    /// requirements are evaluated into blocking findings instead of being
    /// rejected as malformed input.
    pub fn is_structurally_valid(&self) -> Result<(), MissionConfigurationError> {
        if self.schema_version != 1 {
            return Err(MissionConfigurationError::UnsupportedSchemaVersion);
        }
        if !clean_identifier(&self.mission_id)
            || !clean_identifier(&self.configuration_revision)
            || !clean_identifier(&self.suit_component_id)
        {
            return Err(MissionConfigurationError::InvalidMissionIdentity);
        }
        if !self.required_component_kinds.contains(&MissionComponentKind::Suit) {
            return Err(MissionConfigurationError::InvalidMissionRequirements);
        }
        if self.required_component_kinds.is_empty()
            || self.required_connection_ids.iter().any(|id| !clean_identifier(id))
        {
            return Err(MissionConfigurationError::InvalidMissionRequirements);
        }
        if !self.suit_variant.is_valid() {
            return Err(MissionConfigurationError::InvalidSuitVariant);
        }

        let mut component_ids = BTreeSet::new();
        for component in &self.components {
            if !clean_identifier(&component.component_id)
                || !clean_identifier(&component.hardware_revision)
            {
                return Err(MissionConfigurationError::InvalidComponentManifest);
            }
            if !component_ids.insert(component.component_id.as_str()) {
                return Err(MissionConfigurationError::DuplicateComponentId);
            }
            if !component.interfaces.is_valid() {
                return Err(MissionConfigurationError::InvalidComponentManifest);
            }
            if component.interfaces.component_id != component.component_id
                || component.interfaces.hardware_revision != component.hardware_revision
            {
                return Err(MissionConfigurationError::ComponentManifestMismatch);
            }
        }

        let suit = self
            .components
            .iter()
            .find(|component| component.component_id == self.suit_component_id)
            .ok_or(MissionConfigurationError::MissingSuitComponent)?;
        if suit.kind != MissionComponentKind::Suit
            || suit.hardware_revision != self.suit_variant.hardware_revision
        {
            return Err(MissionConfigurationError::ComponentManifestMismatch);
        }

        let mut connection_ids = BTreeSet::new();
        let mut component_pairs = BTreeSet::new();
        for connection in &self.connections {
            if !clean_identifier(&connection.connection_id)
                || !clean_identifier(&connection.left_component_id)
                || !clean_identifier(&connection.right_component_id)
                || connection.left_component_id == connection.right_component_id
                || !connection.policy.is_valid()
            {
                return Err(MissionConfigurationError::InvalidConnection);
            }
            if !connection_ids.insert(connection.connection_id.as_str()) {
                return Err(MissionConfigurationError::DuplicateConnectionId);
            }
            if !component_ids.contains(connection.left_component_id.as_str())
                || !component_ids.contains(connection.right_component_id.as_str())
            {
                return Err(MissionConfigurationError::InvalidConnection);
            }
            let (first, second) = if connection.left_component_id < connection.right_component_id {
                (&connection.left_component_id, &connection.right_component_id)
            } else {
                (&connection.right_component_id, &connection.left_component_id)
            };
            if !component_pairs.insert((first.as_str(), second.as_str())) {
                return Err(MissionConfigurationError::DuplicateComponentPair);
            }
        }
        Ok(())
    }
}

/// Evaluate a mission graph using exact component revisions and caller-selected
/// evidence floors. Incompatible required links and absent required support
/// block the software policy result; incompatible optional links degrade it.
pub fn evaluate_mission_configuration(
    configuration: &MissionConfiguration,
) -> Result<(MissionConfigurationReport, MissionConfigurationReceipt), MissionConfigurationError> {
    configuration.is_structurally_valid()?;

    let profile_policy = SuitProfileEvidencePolicy {
        minimum_required_evidence: configuration.minimum_capability_evidence,
    };
    let profile = evaluate_suit_profile(
        configuration.suit_profile,
        &configuration.suit_variant,
        profile_policy,
    )
    .map_err(map_profile_error)?;

    let configuration_digest_sha256 = digest_configuration(configuration);
    let components: BTreeMap<&str, &MissionComponent> = configuration
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component))
        .collect();

    let mut findings = Vec::new();
    if profile.status != SuitProfileEvaluationStatus::MeetsDeclaredEvidencePolicy {
        findings.push(MissionFinding {
            code: MissionFindingCode::SuitProfileBlocked,
            subject_id: configuration.suit_variant.variant_id.clone(),
            severity: MissionFindingSeverity::Blocker,
            detail: format!(
                "Suit profile does not meet caller evidence policy: {:?}",
                profile.status
            ),
        });
    }

    let mut present_required_kinds = BTreeSet::new();
    for component in &configuration.components {
        if configuration.required_component_kinds.contains(&component.kind) {
            present_required_kinds.insert(component.kind);
        }
    }
    let missing_required_component_kind_count = configuration
        .required_component_kinds
        .iter()
        .filter(|kind| !present_required_kinds.contains(kind))
        .count();
    for kind in &configuration.required_component_kinds {
        if !present_required_kinds.contains(kind) {
            findings.push(MissionFinding {
                code: MissionFindingCode::MissingRequiredComponent,
                subject_id: component_kind_tag(*kind).to_string(),
                severity: MissionFindingSeverity::Blocker,
                detail: format!("Required component kind is absent: {}", component_kind_tag(*kind)),
            });
        }
    }

    let actual_connection_ids: BTreeSet<&str> = configuration
        .connections
        .iter()
        .map(|connection| connection.connection_id.as_str())
        .collect();
    let missing_required_connection_count = configuration
        .required_connection_ids
        .iter()
        .filter(|id| !actual_connection_ids.contains(id.as_str()))
        .count();
    for connection_id in &configuration.required_connection_ids {
        if !actual_connection_ids.contains(connection_id.as_str()) {
            findings.push(MissionFinding {
                code: MissionFindingCode::MissingRequiredConnection,
                subject_id: connection_id.clone(),
                severity: MissionFindingSeverity::Blocker,
                detail: "Required connection declaration is absent".to_string(),
            });
        }
    }

    let mut sorted_connections: Vec<&MissionConnection> =
        configuration.connections.iter().collect();
    sorted_connections.sort_by(|left, right| left.connection_id.cmp(&right.connection_id));

    let mut adjacency: BTreeMap<String, BTreeSet<String>> = configuration
        .components
        .iter()
        .map(|component| (component.component_id.clone(), BTreeSet::new()))
        .collect();
    let mut connection_evaluations = Vec::new();
    let mut compatible_required_connection_count = 0;
    let mut blocked_required_connection_count = 0;

    for connection in sorted_connections {
        let left = components[connection.left_component_id.as_str()];
        let right = components[connection.right_component_id.as_str()];
        let required = configuration
            .required_connection_ids
            .contains(&connection.connection_id);
        let (interop_report, interoperability_receipt) =
            evaluate_suit_interoperability(&left.interfaces, &right.interfaces, &connection.policy)
                .map_err(map_interop_error)?;

        // Reachability represents policy-admissible paths, not merely declared
        // topology. A failed optional edge must not connect a required resource
        // to the suit in the mission-level graph.
        if interop_report.status == SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy {
            adjacency
                .get_mut(&connection.left_component_id)
                .expect("validated component has adjacency entry")
                .insert(connection.right_component_id.clone());
            adjacency
                .get_mut(&connection.right_component_id)
                .expect("validated component has adjacency entry")
                .insert(connection.left_component_id.clone());
        }

        match interop_report.status {
            SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy if required => {
                compatible_required_connection_count += 1;
            }
            _ if required => {
                blocked_required_connection_count += 1;
                let (code, detail) = if interop_report.status
                    == SuitInteropReportStatus::NoRequiredInterfacesToVerify
                {
                    (
                        MissionFindingCode::RequiredConnectionHasNoRequiredInterfaces,
                        "Required connection has no required interfaces; not an implicit pass",
                    )
                } else {
                    (
                        MissionFindingCode::RequiredConnectionBlocked,
                        "Required connection fails its declared interoperability policy",
                    )
                };
                findings.push(MissionFinding {
                    code,
                    subject_id: connection.connection_id.clone(),
                    severity: MissionFindingSeverity::Blocker,
                    detail: detail.to_string(),
                });
            }
            _ if interop_report.status
                != SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy => {
                findings.push(MissionFinding {
                    code: MissionFindingCode::OptionalConnectionDegraded,
                    subject_id: connection.connection_id.clone(),
                    severity: MissionFindingSeverity::Degradation,
                    detail: format!(
                        "Optional connection does not meet declared policy: {:?}",
                        interop_report.status
                    ),
                });
            }
            _ => {}
        }

        connection_evaluations.push(MissionConnectionEvaluation {
            connection_id: connection.connection_id.clone(),
            left_component_id: connection.left_component_id.clone(),
            right_component_id: connection.right_component_id.clone(),
            required,
            interoperability: interop_report,
            interoperability_receipt,
        });
    }

    let reachable = reachable_from(&configuration.suit_component_id, &adjacency);
    let mut disconnected_required_component_count = 0;
    for component in &configuration.components {
        if configuration.required_component_kinds.contains(&component.kind)
            && !reachable.contains(&component.component_id)
        {
            disconnected_required_component_count += 1;
            findings.push(MissionFinding {
                code: MissionFindingCode::RequiredComponentDisconnected,
                subject_id: component.component_id.clone(),
                severity: MissionFindingSeverity::Blocker,
                detail: "Required component has no declared path to the suit node".to_string(),
            });
        }
    }

    findings.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then(left.subject_id.cmp(&right.subject_id))
            .then(left.severity.cmp(&right.severity))
            .then(left.detail.cmp(&right.detail))
    });

    let blocker_count = findings
        .iter()
        .filter(|finding| finding.severity == MissionFindingSeverity::Blocker)
        .count();
    let degradation_count = findings
        .iter()
        .filter(|finding| finding.severity == MissionFindingSeverity::Degradation)
        .count();
    let status = if blocker_count > 0 {
        MissionConfigurationStatus::Blocked
    } else if degradation_count > 0 {
        MissionConfigurationStatus::Degraded
    } else {
        MissionConfigurationStatus::MeetsDeclaredMissionConfigurationPolicy
    };
    let present_required_component_count = configuration
        .components
        .iter()
        .filter(|component| configuration.required_component_kinds.contains(&component.kind))
        .map(|component| component.kind)
        .collect::<BTreeSet<_>>()
        .len();
    let report = MissionConfigurationReport {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        configuration_digest_sha256: configuration_digest_sha256.clone(),
        suit_profile_evaluation: profile,
        connection_evaluations,
        findings,
        status,
        component_count: configuration.components.len(),
        connection_count: configuration.connections.len(),
        required_component_kind_count: configuration.required_component_kinds.len(),
        present_required_component_kind_count: present_required_component_count,
        required_connection_count: configuration.required_connection_ids.len(),
        missing_required_connection_count,
        compatible_required_connection_count,
        blocked_required_connection_count,
        missing_required_component_kind_count,
        disconnected_required_component_count,
        represents_physical_qualification: false,
    };

    let report_digest_sha256 = digest_report(&report);
    let mut receipt = MissionConfigurationReceipt {
        configuration_digest_sha256,
        report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_mission_receipt(&receipt);
    Ok((report, receipt))
}

/// Re-evaluate the entire configuration and require byte-for-byte equivalent
/// typed report and receipt values. This detects changed manifests, policy,
/// profile evidence, graph topology, and report content; it is not a signature.
pub fn verify_mission_configuration_receipt(
    configuration: &MissionConfiguration,
    expected_report: &MissionConfigurationReport,
    expected_receipt: &MissionConfigurationReceipt,
) -> bool {
    match evaluate_mission_configuration(configuration) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn reachable_from(
    start: &str,
    adjacency: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    let mut reached = BTreeSet::new();
    let mut pending = VecDeque::from([start.to_string()]);
    while let Some(node) = pending.pop_front() {
        if !reached.insert(node.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            pending.extend(neighbors.iter().cloned());
        }
    }
    reached
}

fn map_profile_error(error: SuitProfileError) -> MissionConfigurationError {
    match error {
        SuitProfileError::InvalidVariantId
        | SuitProfileError::InvalidHardwareRevision
        | SuitProfileError::InvalidCapabilityClaim
        | SuitProfileError::EvidenceScopeMismatch => MissionConfigurationError::InvalidSuitVariant,
    }
}

fn map_interop_error(_: SuitInteropError) -> MissionConfigurationError {
    MissionConfigurationError::InvalidComponentManifest
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

    fn option_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.boolean(true);
                self.string(value);
            }
            None => self.boolean(false),
        }
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        format!(
            "sha256:{}",
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        )
    }
}

fn digest_configuration(configuration: &MissionConfiguration) -> String {
    let mut writer = HashWriter::new("symthaea.sx031.mission-configuration.v1");
    writer.u64(configuration.schema_version as u64);
    writer.string(&configuration.mission_id);
    writer.string(&configuration.configuration_revision);
    writer.string(&configuration.suit_component_id);
    writer.string(profile_tag(configuration.suit_profile));
    writer.string(evidence_tag(configuration.minimum_capability_evidence));

    writer.usize(configuration.required_component_kinds.len());
    for kind in &configuration.required_component_kinds {
        writer.string(component_kind_tag(*kind));
    }
    writer.usize(configuration.required_connection_ids.len());
    for id in &configuration.required_connection_ids {
        writer.string(id);
    }

    let mut claims: Vec<_> = configuration.suit_variant.capability_claims.iter().collect();
    claims.sort_by(|left, right| {
        left.capability
            .cmp(&right.capability)
            .then(left.applies_to_variant_id.cmp(&right.applies_to_variant_id))
            .then(left.applies_to_hardware_revision.cmp(&right.applies_to_hardware_revision))
            .then(left.implementation_id.cmp(&right.implementation_id))
            .then(left.evidence_level.cmp(&right.evidence_level))
            .then(left.evidence_reference.cmp(&right.evidence_reference))
    });
    writer.string(&configuration.suit_variant.variant_id);
    writer.string(&configuration.suit_variant.hardware_revision);
    writer.usize(claims.len());
    for claim in claims {
        writer.string(capability_tag(claim.capability));
        writer.string(&claim.applies_to_variant_id);
        writer.string(&claim.applies_to_hardware_revision);
        writer.string(&claim.implementation_id);
        writer.string(evidence_tag(claim.evidence_level));
        writer.string(&claim.evidence_reference);
    }

    let mut components: Vec<_> = configuration.components.iter().collect();
    components.sort_by(|left, right| left.component_id.cmp(&right.component_id));
    writer.usize(components.len());
    for component in components {
        writer.string(&component.component_id);
        writer.string(component_kind_tag(component.kind));
        writer.string(&component.hardware_revision);
        write_manifest(&mut writer, &component.interfaces);
    }

    let mut connections: Vec<_> = configuration.connections.iter().collect();
    connections.sort_by(|left, right| left.connection_id.cmp(&right.connection_id));
    writer.usize(connections.len());
    for connection in connections {
        writer.string(&connection.connection_id);
        writer.string(&connection.left_component_id);
        writer.string(&connection.right_component_id);
        write_policy(&mut writer, &connection.policy);
    }
    writer.finish()
}

fn write_manifest(writer: &mut HashWriter, manifest: &SuitInterfaceManifest) {
    writer.string(&manifest.component_id);
    writer.string(&manifest.hardware_revision);
    let mut endpoints: Vec<_> = manifest.endpoints.iter().collect();
    endpoints.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));
    writer.usize(endpoints.len());
    for endpoint in endpoints {
        writer.string(&endpoint.interface_id);
        writer.string(&endpoint.endpoint_id);
        writer.string(interface_kind_tag(endpoint.kind));
        writer.string(interface_role_tag(endpoint.role));
        writer.string(&endpoint.contract_revision);
        writer.string(&endpoint.schema_digest_sha256);
        writer.string(&endpoint.connector_profile_id);
        writer.usize(endpoint.applicable_environments.len());
        for environment in &endpoint.applicable_environments {
            writer.string(environment_tag(*environment));
        }
        writer.string(evidence_tag(endpoint.evidence_level));
        writer.string(&endpoint.evidence_reference);
        writer.boolean(endpoint.required);
    }
}

fn write_policy(writer: &mut HashWriter, policy: &SuitInteropPolicy) {
    writer.string(evidence_tag(policy.minimum_evidence_level));
    writer.usize(policy.required_environments.len());
    for environment in &policy.required_environments {
        writer.string(environment_tag(*environment));
    }
}

fn digest_report(report: &MissionConfigurationReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx031.mission-report.v1");
    writer.string(&report.configuration_digest_sha256);
    writer.string(mission_status_tag(report.status));
    writer.string(profile_status_tag(report.suit_profile_evaluation.status));
    writer.string(profile_tag(report.suit_profile_evaluation.profile));
    writer.string(&report.suit_profile_evaluation.variant_id);
    writer.string(&report.suit_profile_evaluation.hardware_revision);
    writer.usize(report.suit_profile_evaluation.missing_required_count);
    writer.usize(report.suit_profile_evaluation.below_policy_required_count);
    writer.usize(report.suit_profile_evaluation.conflicting_required_count);
    writer.usize(report.suit_profile_evaluation.required_capabilities.len());
    for capability in &report.suit_profile_evaluation.required_capabilities {
        writer.string(capability_tag(*capability));
    }
    writer.usize(report.suit_profile_evaluation.findings.len());
    for finding in &report.suit_profile_evaluation.findings {
        writer.string(capability_tag(finding.capability));
        writer.boolean(finding.required_for_profile);
        writer.string(capability_finding_status_tag(finding.status));
        writer.usize(finding.claim_count);
        writer.option_string(finding.implementation_id.as_deref());
        match finding.evidence_level {
            Some(level) => {
                writer.boolean(true);
                writer.string(evidence_tag(level));
            }
            None => writer.boolean(false),
        }
        writer.option_string(finding.evidence_reference.as_deref());
    }
    writer.boolean(report.represents_physical_qualification);

    writer.usize(report.findings.len());
    for finding in &report.findings {
        writer.string(finding_code_tag(finding.code));
        writer.string(&finding.subject_id);
        writer.string(finding_severity_tag(finding.severity));
        writer.string(&finding.detail);
    }
    writer.usize(report.connection_evaluations.len());
    for connection in &report.connection_evaluations {
        writer.string(&connection.connection_id);
        writer.string(&connection.left_component_id);
        writer.string(&connection.right_component_id);
        writer.boolean(connection.required);
        let interop = &connection.interoperability;
        writer.string(match interop.evidence_class {
            crate::suit_interop::SuitInteropEvidenceClass::DeclaredSoftwareContract => {
                "declared_software_contract"
            }
        });
        writer.string(&interop.left_component_id);
        writer.string(&interop.left_hardware_revision);
        writer.string(&interop.right_component_id);
        writer.string(&interop.right_hardware_revision);
        write_policy(&mut writer, &interop.policy);
        writer.usize(interop.findings.len());
        for finding in &interop.findings {
            writer.string(&finding.component_id);
            writer.string(&finding.endpoint_id);
            writer.string(&finding.interface_id);
            writer.string(interface_kind_tag(finding.interface_kind));
            writer.string(interface_role_tag(finding.role));
            writer.boolean(finding.required);
            writer.string(&finding.peer_component_id);
            writer.option_string(finding.peer_endpoint_id.as_deref());
            writer.string(interop_finding_status_tag(finding.status));
        }
        writer.string(interop_status_tag(interop.status));
        writer.usize(interop.required_interface_count);
        writer.usize(interop.required_compatible_count);
        writer.usize(interop.blocked_required_count);
        writer.boolean(interop.represents_physical_qualification);

        let pair_receipt = &connection.interoperability_receipt;
        writer.string(&pair_receipt.left_manifest_digest_sha256);
        writer.string(&pair_receipt.right_manifest_digest_sha256);
        writer.string(&pair_receipt.policy_digest_sha256);
        writer.string(&pair_receipt.result_digest_sha256);
        writer.string(&pair_receipt.receipt_digest_sha256);
    }

    writer.usize(report.component_count);
    writer.usize(report.connection_count);
    writer.usize(report.required_component_kind_count);
    writer.usize(report.present_required_component_kind_count);
    writer.usize(report.required_connection_count);
    writer.usize(report.missing_required_connection_count);
    writer.usize(report.compatible_required_connection_count);
    writer.usize(report.blocked_required_connection_count);
    writer.usize(report.missing_required_component_kind_count);
    writer.usize(report.disconnected_required_component_count);
    writer.finish()
}

fn digest_mission_receipt(receipt: &MissionConfigurationReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx031.mission-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn component_kind_tag(value: MissionComponentKind) -> &'static str {
    match value {
        MissionComponentKind::Suit => "suit",
        MissionComponentKind::LanderOrVehicle => "lander_or_vehicle",
        MissionComponentKind::Habitat => "habitat",
        MissionComponentKind::Rover => "rover",
        MissionComponentKind::ServiceStation => "service_station",
        MissionComponentKind::RescueCapability => "rescue_capability",
        MissionComponentKind::CommunicationsRelay => "communications_relay",
        MissionComponentKind::Worksite => "worksite",
        MissionComponentKind::GroundSupport => "ground_support",
    }
}

fn profile_tag(value: SuitMissionProfile) -> &'static str {
    match value {
        SuitMissionProfile::OrbitalEva => "orbital_eva",
        SuitMissionProfile::LunarSurfaceEva => "lunar_surface_eva",
        SuitMissionProfile::MarsSurfaceEva => "mars_surface_eva",
        SuitMissionProfile::ConstructionAndHeavyTooling => "construction_and_heavy_tooling",
        SuitMissionProfile::DustIntensiveSurfaceWork => "dust_intensive_surface_work",
        SuitMissionProfile::SuitportAndHabitatServicing => "suitport_and_habitat_servicing",
        SuitMissionProfile::CrewRescueAndRecovery => "crew_rescue_and_recovery",
        SuitMissionProfile::LongDurationOrbitalMaintenance => "long_duration_orbital_maintenance",
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

fn evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn environment_tag(value: crate::suit_interop::SuitOperatingEnvironment) -> &'static str {
    use crate::suit_interop::SuitOperatingEnvironment::*;
    match value {
        MicrogravityVacuum => "microgravity_vacuum",
        LunarSunlitSurface => "lunar_sunlit_surface",
        LunarShadowOrPolarRegion => "lunar_shadow_or_polar_region",
        MarsSurface => "mars_surface",
        PressurizedCabin => "pressurized_cabin",
        ThermalVacuumGroundTest => "thermal_vacuum_ground_test",
        GroundLaboratory => "ground_laboratory",
    }
}

fn interface_kind_tag(value: crate::suit_interop::SuitInterfaceKind) -> &'static str {
    use crate::suit_interop::SuitInterfaceKind::*;
    match value {
        PressureBoundary => "pressure_boundary",
        BreathableGas => "breathable_gas",
        Coolant => "coolant",
        ThermalExchange => "thermal_exchange",
        ElectricalPower => "electrical_power",
        GroundingAndEsd => "grounding_and_esd",
        Communications => "communications",
        TelemetryAndData => "telemetry_and_data",
        MechanicalCapture => "mechanical_capture",
        RestraintAndAnchor => "restraint_and_anchor",
        ConsumableTransfer => "consumable_transfer",
        DustMitigationService => "dust_mitigation_service",
        RescueHandling => "rescue_handling",
    }
}

fn interface_role_tag(value: crate::suit_interop::SuitInterfaceRole) -> &'static str {
    use crate::suit_interop::SuitInterfaceRole::*;
    match value {
        Provider => "provider",
        Consumer => "consumer",
        Bidirectional => "bidirectional",
    }
}

fn capability_finding_status_tag(
    value: crate::suit_profile::SuitCapabilityFindingStatus,
) -> &'static str {
    use crate::suit_profile::SuitCapabilityFindingStatus::*;
    match value {
        MeetsDeclaredPolicy => "meets_declared_policy",
        MissingRequiredCapability => "missing_required_capability",
        EvidenceBelowPolicy => "evidence_below_policy",
        ConflictingClaims => "conflicting_claims",
    }
}

fn profile_status_tag(value: SuitProfileEvaluationStatus) -> &'static str {
    match value {
        SuitProfileEvaluationStatus::RequirementsMissing => "requirements_missing",
        SuitProfileEvaluationStatus::RequiredEvidenceBelowPolicy => {
            "required_evidence_below_policy"
        },
        SuitProfileEvaluationStatus::ConflictingRequiredClaims => "conflicting_required_claims",
        SuitProfileEvaluationStatus::MeetsDeclaredEvidencePolicy => {
            "meets_declared_evidence_policy"
        },
    }
}

fn mission_status_tag(value: MissionConfigurationStatus) -> &'static str {
    match value {
        MissionConfigurationStatus::Blocked => "blocked",
        MissionConfigurationStatus::Degraded => "degraded",
        MissionConfigurationStatus::MeetsDeclaredMissionConfigurationPolicy => {
            "meets_declared_mission_configuration_policy"
        }
    }
}

fn interop_finding_status_tag(
    value: crate::suit_interop::SuitInteropFindingStatus,
) -> &'static str {
    use crate::suit_interop::SuitInteropFindingStatus::*;
    match value {
        CompatibleUnderPolicy => "compatible_under_policy",
        MissingPeerInterface => "missing_peer_interface",
        AmbiguousPeerInterfaces => "ambiguous_peer_interfaces",
        InterfaceKindMismatch => "interface_kind_mismatch",
        InterfaceRoleMismatch => "interface_role_mismatch",
        ContractRevisionMismatch => "contract_revision_mismatch",
        SchemaDigestMismatch => "schema_digest_mismatch",
        ConnectorProfileMismatch => "connector_profile_mismatch",
        OperatingEnvironmentMismatch => "operating_environment_mismatch",
        InsufficientEvidence => "insufficient_evidence",
    }
}

fn interop_status_tag(value: SuitInteropReportStatus) -> &'static str {
    match value {
        SuitInteropReportStatus::NoRequiredInterfacesToVerify => "no_required_interfaces_to_verify",
        SuitInteropReportStatus::RequiredInterfacesBlocked => "required_interfaces_blocked",
        SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy => {
            "meets_declared_compatibility_policy"
        }
    }
}

fn finding_code_tag(value: MissionFindingCode) -> &'static str {
    match value {
        MissionFindingCode::MissingRequiredComponent => "missing_required_component",
        MissionFindingCode::MissingRequiredConnection => "missing_required_connection",
        MissionFindingCode::RequiredComponentDisconnected => "required_component_disconnected",
        MissionFindingCode::SuitProfileBlocked => "suit_profile_blocked",
        MissionFindingCode::RequiredConnectionHasNoRequiredInterfaces => {
            "required_connection_has_no_required_interfaces"
        }
        MissionFindingCode::RequiredConnectionBlocked => "required_connection_blocked",
        MissionFindingCode::OptionalConnectionDegraded => "optional_connection_degraded",
    }
}

fn finding_severity_tag(value: MissionFindingSeverity) -> &'static str {
    match value {
        MissionFindingSeverity::Degradation => "degradation",
        MissionFindingSeverity::Blocker => "blocker",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suit_interop::{
        SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceRole, SuitOperatingEnvironment,
    };
    use crate::suit_profile::SuitCapabilityClaim;

    fn manifest(
        component_id: &str,
        hardware_revision: &str,
        role: SuitInterfaceRole,
    ) -> SuitInterfaceManifest {
        let endpoint = SuitInterfaceEndpoint {
            interface_id: "grounding.esd.v1".into(),
            endpoint_id: format!("{component_id}.grounding"),
            kind: SuitInterfaceKind::GroundingAndEsd,
            role,
            contract_revision: "1.0.0".into(),
            schema_digest_sha256: format!("sha256:{}", "a".repeat(64)),
            connector_profile_id: "synthetic-grounding-profile-v1".into(),
            applicable_environments: BTreeSet::from([
                SuitOperatingEnvironment::LunarShadowOrPolarRegion,
            ]),
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: format!("synthetic-test:{component_id}:grounding-v1"),
            required: true,
        };
        SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: hardware_revision.into(),
            endpoints: vec![endpoint],
        }
    }

    fn claim(
        capability: SuitCapability,
        variant_id: &str,
        hardware_revision: &str,
    ) -> SuitCapabilityClaim {
        SuitCapabilityClaim {
            capability,
            applies_to_variant_id: variant_id.into(),
            applies_to_hardware_revision: hardware_revision.into(),
            implementation_id: format!("impl-{}", capability_tag(capability)),
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: format!("synthetic-capability:{}", capability_tag(capability)),
        }
    }

    fn fixture() -> MissionConfiguration {
        let variant_id = "lunar-demo-suit";
        let suit_revision = "suit-r3";
        let vehicle_revision = "lander-r2";
        let capability_claims = SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .into_iter()
            .map(|capability| claim(capability, variant_id, suit_revision))
            .collect();
        let suit = MissionComponent {
            component_id: "suit".into(),
            kind: MissionComponentKind::Suit,
            hardware_revision: suit_revision.into(),
            interfaces: manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
        };
        let lander = MissionComponent {
            component_id: "lander".into(),
            kind: MissionComponentKind::LanderOrVehicle,
            hardware_revision: vehicle_revision.into(),
            interfaces: manifest("lander", vehicle_revision, SuitInterfaceRole::Provider),
        };
        MissionConfiguration {
            schema_version: 1,
            mission_id: "synthetic-lunar-eva-demo".into(),
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
            components: vec![suit, lander],
            connections: vec![MissionConnection {
                connection_id: "suit-lander-grounding".into(),
                left_component_id: "suit".into(),
                right_component_id: "lander".into(),
                policy: SuitInteropPolicy {
                    minimum_evidence_level: SuitEvidenceLevel::Simulated,
                    required_environments: BTreeSet::from([
                        SuitOperatingEnvironment::LunarShadowOrPolarRegion,
                    ]),
                },
            }],
        }
    }

    #[test]
    fn valid_synthetic_mission_meets_declared_policy_but_not_qualification() {
        let configuration = fixture();
        let (report, receipt) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(
            report.status,
            MissionConfigurationStatus::MeetsDeclaredMissionConfigurationPolicy
        );
        assert!(report.findings.is_empty());
        assert_eq!(report.compatible_required_connection_count, 1);
        assert_eq!(report.blocked_required_connection_count, 0);
        assert!(!report.represents_physical_qualification);
        assert!(verify_mission_configuration_receipt(
            &configuration,
            &report,
            &receipt
        ));
    }

    #[test]
    fn missing_required_component_is_a_blocker_not_an_input_parse_error() {
        let mut configuration = fixture();
        configuration.required_component_kinds.insert(MissionComponentKind::RescueCapability);
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert_eq!(report.missing_required_component_kind_count, 1);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::MissingRequiredComponent
                && finding.subject_id == "rescue_capability"
        }));
    }

    #[test]
    fn missing_required_connection_is_not_silently_ignored() {
        let mut configuration = fixture();
        configuration.connections.clear();
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::MissingRequiredConnection
                && finding.subject_id == "suit-lander-grounding"
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::RequiredComponentDisconnected
                && finding.subject_id == "lander"
        }));
    }

    #[test]
    fn incompatible_required_connection_blocks_configuration() {
        let mut configuration = fixture();
        configuration.connections[0].policy.minimum_evidence_level = SuitEvidenceLevel::Qualified;
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert_eq!(report.blocked_required_connection_count, 1);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::RequiredConnectionBlocked
        }));
    }

    #[test]
    fn optional_connection_failure_degrades_instead_of_blocking() {
        let mut configuration = fixture();
        configuration.required_connection_ids.clear();
        configuration.connections[0].policy.minimum_evidence_level = SuitEvidenceLevel::Qualified;
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Degraded);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::OptionalConnectionDegraded
        }));
        assert_eq!(report.blocked_required_connection_count, 0);
    }

    #[test]
    fn incompatible_optional_edge_does_not_connect_required_worksite() {
        let mut configuration = fixture();
        configuration.required_component_kinds.insert(MissionComponentKind::Worksite);
        configuration.components.push(MissionComponent {
            component_id: "worksite".into(),
            kind: MissionComponentKind::Worksite,
            hardware_revision: "worksite-r1".into(),
            interfaces: manifest("worksite", "worksite-r1", SuitInterfaceRole::Provider),
        });
        configuration.connections.push(MissionConnection {
            connection_id: "suit-worksite-optional".into(),
            left_component_id: "suit".into(),
            right_component_id: "worksite".into(),
            policy: SuitInteropPolicy {
                minimum_evidence_level: SuitEvidenceLevel::Qualified,
                required_environments: BTreeSet::from([
                    SuitOperatingEnvironment::LunarShadowOrPolarRegion,
                ]),
            },
        });
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert_eq!(report.compatible_required_connection_count, 1);
        assert_eq!(report.blocked_required_connection_count, 0);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::RequiredComponentDisconnected
                && finding.subject_id == "worksite"
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::OptionalConnectionDegraded
                && finding.subject_id == "suit-worksite-optional"
        }));
    }

    #[test]
    fn empty_required_interface_set_is_not_an_implicit_pass() {
        let mut configuration = fixture();
        configuration.components[0].interfaces.endpoints[0].required = false;
        configuration.components[1].interfaces.endpoints[0].required = false;
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::RequiredConnectionHasNoRequiredInterfaces
        }));
    }

    #[test]
    fn evidence_does_not_transfer_across_hardware_revisions() {
        let mut configuration = fixture();
        configuration.suit_variant.hardware_revision = "suit-r4".into();
        let error = evaluate_mission_configuration(&configuration).unwrap_err();
        assert_eq!(error, MissionConfigurationError::InvalidSuitVariant);
    }

    #[test]
    fn receipt_detects_changed_required_policy() {
        let configuration = fixture();
        let (report, receipt) = evaluate_mission_configuration(&configuration).unwrap();
        let mut changed = configuration.clone();
        changed.required_component_kinds.insert(MissionComponentKind::RescueCapability);
        assert!(!verify_mission_configuration_receipt(&changed, &report, &receipt));
    }

    #[test]
    fn component_and_connection_permutation_is_deterministic() {
        let configuration = fixture();
        let (report_a, receipt_a) = evaluate_mission_configuration(&configuration).unwrap();
        let mut permuted = configuration.clone();
        permuted.components.reverse();
        permuted.connections.reverse();
        permuted.suit_variant.capability_claims.reverse();
        let (report_b, receipt_b) = evaluate_mission_configuration(&permuted).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);
    }

    #[test]
    fn report_digest_binds_embedded_pairwise_findings() {
        let configuration = fixture();
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        let original_digest = digest_report(&report);
        let mut tampered = report.clone();
        tampered.connection_evaluations[0].interoperability.findings[0]
            .component_id
            .push_str("-tampered");
        assert_ne!(digest_report(&tampered), original_digest);
    }

    #[test]
    fn duplicate_component_pairs_are_rejected_as_ambiguous() {
        let mut configuration = fixture();
        let mut duplicate = configuration.connections[0].clone();
        duplicate.connection_id = "duplicate-link-id".into();
        configuration.connections.push(duplicate);
        assert_eq!(
            evaluate_mission_configuration(&configuration).unwrap_err(),
            MissionConfigurationError::DuplicateComponentPair
        );
    }

    #[test]
    fn duplicate_connection_ids_are_rejected() {
        let mut configuration = fixture();
        configuration.connections.push(configuration.connections[0].clone());
        assert_eq!(
            evaluate_mission_configuration(&configuration).unwrap_err(),
            MissionConfigurationError::DuplicateConnectionId
        );
    }

    #[test]
    fn invalid_component_manifest_binding_is_rejected() {
        let mut configuration = fixture();
        configuration.components[1].interfaces.hardware_revision = "different-revision".into();
        assert_eq!(
            evaluate_mission_configuration(&configuration).unwrap_err(),
            MissionConfigurationError::ComponentManifestMismatch
        );
    }

    #[test]
    fn profile_evidence_below_policy_blocks_mission_graph() {
        let mut configuration = fixture();
        for claim in &mut configuration.suit_variant.capability_claims {
            claim.evidence_level = SuitEvidenceLevel::Declared;
        }
        let (report, _) = evaluate_mission_configuration(&configuration).unwrap();
        assert_eq!(report.status, MissionConfigurationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionFindingCode::SuitProfileBlocked
        }));
    }
}
