// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned suit/vehicle/habitat interface compatibility contracts (SX-030).
//!
//! Compatibility means only that declared software contracts match under the
//! supplied policy. It is not evidence of physical, electrical, thermal,
//! pressure-boundary, breathing-gas, grounding, or human-rating qualification.
//! This module has no actuator or life-support command surface.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use crate::suit_profile::SuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SuitInterfaceKind {
    PressureBoundary,
    BreathableGas,
    Coolant,
    ThermalExchange,
    ElectricalPower,
    GroundingAndEsd,
    Communications,
    TelemetryAndData,
    MechanicalCapture,
    RestraintAndAnchor,
    ConsumableTransfer,
    DustMitigationService,
    RescueHandling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SuitInterfaceRole {
    Provider,
    Consumer,
    Bidirectional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SuitOperatingEnvironment {
    MicrogravityVacuum,
    LunarSunlitSurface,
    LunarShadowOrPolarRegion,
    MarsSurface,
    PressurizedCabin,
    ThermalVacuumGroundTest,
    GroundLaboratory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitInteropEvidenceClass {
    DeclaredSoftwareContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInterfaceEndpoint {
    /// Stable logical interface name inside the declared ecosystem, not a
    /// connector model or proof of physical interoperability.
    pub interface_id: String,
    pub endpoint_id: String,
    pub kind: SuitInterfaceKind,
    pub role: SuitInterfaceRole,
    /// Exact contract revision. No implicit semver/backward compatibility.
    pub contract_revision: String,
    /// SHA-256 of the externally defined interface schema/ICD artifact.
    /// A hash identifies declared bytes; it does not validate the design.
    pub schema_digest_sha256: String,
    /// Explicit connector/geometry/termination profile identifier.
    pub connector_profile_id: String,
    /// Environments in which this endpoint's compatibility claim applies.
    pub applicable_environments: BTreeSet<SuitOperatingEnvironment>,
    pub evidence_level: SuitEvidenceLevel,
    /// URI, requirements ID, report ID, or artifact digest; not authenticated here.
    pub evidence_reference: String,
    /// If true, incompatibility blocks the declared software-level policy result.
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInterfaceManifest {
    pub component_id: String,
    pub hardware_revision: String,
    pub endpoints: Vec<SuitInterfaceEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInteropPolicy {
    pub minimum_evidence_level: SuitEvidenceLevel,
    /// Non-empty environment context is mandatory to avoid claiming
    /// environment-free compatibility from an unspecified operating condition.
    pub required_environments: BTreeSet<SuitOperatingEnvironment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuitInteropFindingStatus {
    CompatibleUnderPolicy,
    MissingPeerInterface,
    AmbiguousPeerInterfaces,
    InterfaceKindMismatch,
    InterfaceRoleMismatch,
    ContractRevisionMismatch,
    SchemaDigestMismatch,
    ConnectorProfileMismatch,
    OperatingEnvironmentMismatch,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitInteropReportStatus {
    NoRequiredInterfacesToVerify,
    RequiredInterfacesBlocked,
    MeetsDeclaredCompatibilityPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInteropFinding {
    pub component_id: String,
    pub endpoint_id: String,
    pub interface_id: String,
    pub interface_kind: SuitInterfaceKind,
    pub role: SuitInterfaceRole,
    pub required: bool,
    pub peer_component_id: String,
    pub peer_endpoint_id: Option<String>,
    pub status: SuitInteropFindingStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInteropReport {
    pub evidence_class: SuitInteropEvidenceClass,
    pub left_component_id: String,
    pub left_hardware_revision: String,
    pub right_component_id: String,
    pub right_hardware_revision: String,
    pub policy: SuitInteropPolicy,
    pub findings: Vec<SuitInteropFinding>,
    pub status: SuitInteropReportStatus,
    pub required_interface_count: usize,
    pub required_compatible_count: usize,
    pub blocked_required_count: usize,
    /// Always false: software contract matching is not hardware qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuitInteropReceipt {
    pub left_manifest_digest_sha256: String,
    pub right_manifest_digest_sha256: String,
    pub policy_digest_sha256: String,
    pub result_digest_sha256: String,
    /// Deterministic change-evidence, not a signature or authentication claim.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitInteropError {
    InvalidLeftManifest,
    InvalidRightManifest,
    InvalidPolicy,
}

impl SuitInterfaceManifest {
    pub fn is_valid(&self) -> bool {
        if !clean_identifier(&self.component_id)
            || !clean_identifier(&self.hardware_revision)
        {
            return false;
        }
        let mut endpoint_ids = BTreeSet::new();
        self.endpoints.iter().all(|endpoint| {
            endpoint_ids.insert(endpoint.endpoint_id.as_str())
                && clean_identifier(&endpoint.endpoint_id)
                && clean_identifier(&endpoint.interface_id)
                && clean_identifier(&endpoint.contract_revision)
                && valid_sha256(&endpoint.schema_digest_sha256)
                && clean_identifier(&endpoint.connector_profile_id)
                && !endpoint.applicable_environments.is_empty()
                && clean_identifier(&endpoint.evidence_reference)
        })
    }
}

impl SuitInteropPolicy {
    pub fn is_valid(&self) -> bool {
        !self.required_environments.is_empty()
    }
}

/// Compare declared interface manifests in a concrete operating-environment
/// context. Every Consumer/Bidirectional endpoint is checked against the peer.
/// No revision coercion, digest relaxation, inferred adapter, or default
/// interface is applied. All findings are returned in canonical order.
pub fn evaluate_suit_interoperability(
    left: &SuitInterfaceManifest,
    right: &SuitInterfaceManifest,
    policy: &SuitInteropPolicy,
) -> Result<(SuitInteropReport, SuitInteropReceipt), SuitInteropError> {
    if !left.is_valid() {
        return Err(SuitInteropError::InvalidLeftManifest);
    }
    if !right.is_valid() {
        return Err(SuitInteropError::InvalidRightManifest);
    }
    if !policy.is_valid() {
        return Err(SuitInteropError::InvalidPolicy);
    }

    let mut findings = Vec::new();
    findings.extend(evaluate_manifest_side(left, right, policy));
    findings.extend(evaluate_manifest_side(right, left, policy));
    findings.sort_by(compare_findings);

    let required_interface_count = findings.iter().filter(|finding| finding.required).count();
    let required_compatible_count = findings
        .iter()
        .filter(|finding| {
            finding.required
                && finding.status == SuitInteropFindingStatus::CompatibleUnderPolicy
        })
        .count();
    let blocked_required_count = required_interface_count - required_compatible_count;
    let status = if required_interface_count == 0 {
        SuitInteropReportStatus::NoRequiredInterfacesToVerify
    } else if blocked_required_count > 0 {
        SuitInteropReportStatus::RequiredInterfacesBlocked
    } else {
        SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
    };

    let report = SuitInteropReport {
        evidence_class: SuitInteropEvidenceClass::DeclaredSoftwareContract,
        left_component_id: left.component_id.clone(),
        left_hardware_revision: left.hardware_revision.clone(),
        right_component_id: right.component_id.clone(),
        right_hardware_revision: right.hardware_revision.clone(),
        policy: policy.clone(),
        findings,
        status,
        required_interface_count,
        required_compatible_count,
        blocked_required_count,
        represents_physical_qualification: false,
    };

    let mut receipt = SuitInteropReceipt {
        left_manifest_digest_sha256: digest_manifest(left),
        right_manifest_digest_sha256: digest_manifest(right),
        policy_digest_sha256: digest_policy(policy),
        result_digest_sha256: digest_report(&report),
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_receipt(&receipt);
    Ok((report, receipt))
}

/// Verify that the exact supplied manifests/policy and report match the receipt.
/// This does not authenticate the source of the manifests or qualify hardware.
pub fn verify_suit_interop_receipt(
    left: &SuitInterfaceManifest,
    right: &SuitInterfaceManifest,
    policy: &SuitInteropPolicy,
    report: &SuitInteropReport,
    receipt: &SuitInteropReceipt,
) -> bool {
    valid_sha256(&receipt.left_manifest_digest_sha256)
        && valid_sha256(&receipt.right_manifest_digest_sha256)
        && valid_sha256(&receipt.policy_digest_sha256)
        && valid_sha256(&receipt.result_digest_sha256)
        && valid_sha256(&receipt.receipt_digest_sha256)
        && receipt.left_manifest_digest_sha256 == digest_manifest(left)
        && receipt.right_manifest_digest_sha256 == digest_manifest(right)
        && receipt.policy_digest_sha256 == digest_policy(policy)
        && receipt.result_digest_sha256 == digest_report(report)
        && receipt.receipt_digest_sha256 == digest_receipt(receipt)
        && report.left_component_id == left.component_id
        && report.left_hardware_revision == left.hardware_revision
        && report.right_component_id == right.component_id
        && report.right_hardware_revision == right.hardware_revision
        && report.policy == *policy
        && !report.represents_physical_qualification
}

fn evaluate_manifest_side(
    subject: &SuitInterfaceManifest,
    peer: &SuitInterfaceManifest,
    policy: &SuitInteropPolicy,
) -> Vec<SuitInteropFinding> {
    let mut endpoints = subject.endpoints.clone();
    endpoints.sort_by(compare_endpoints);
    endpoints
        .iter()
        .filter(|endpoint| {
            matches!(
                endpoint.role,
                SuitInterfaceRole::Consumer | SuitInterfaceRole::Bidirectional
            )
        })
        .map(|endpoint| {
            let matching: Vec<_> = peer
                .endpoints
                .iter()
                .filter(|candidate| candidate.interface_id == endpoint.interface_id)
                .collect();
            let (status, peer_endpoint_id) = compare_endpoint(endpoint, &matching, policy);
            SuitInteropFinding {
                component_id: subject.component_id.clone(),
                endpoint_id: endpoint.endpoint_id.clone(),
                interface_id: endpoint.interface_id.clone(),
                interface_kind: endpoint.kind,
                role: endpoint.role,
                required: endpoint.required,
                peer_component_id: peer.component_id.clone(),
                peer_endpoint_id,
                status,
            }
        })
        .collect()
}

fn compare_endpoint(
    endpoint: &SuitInterfaceEndpoint,
    matching: &[&SuitInterfaceEndpoint],
    policy: &SuitInteropPolicy,
) -> (SuitInteropFindingStatus, Option<String>) {
    if matching.is_empty() {
        return (SuitInteropFindingStatus::MissingPeerInterface, None);
    }
    if matching.len() != 1 {
        return (SuitInteropFindingStatus::AmbiguousPeerInterfaces, None);
    }
    let peer = matching[0];
    let peer_endpoint = Some(peer.endpoint_id.clone());
    let fail = |status| (status, peer_endpoint.clone());

    if endpoint.kind != peer.kind {
        return fail(SuitInteropFindingStatus::InterfaceKindMismatch);
    }
    let compatible_roles = match endpoint.role {
        SuitInterfaceRole::Consumer => {
            matches!(peer.role, SuitInterfaceRole::Provider | SuitInterfaceRole::Bidirectional)
        }
        SuitInterfaceRole::Bidirectional => peer.role == SuitInterfaceRole::Bidirectional,
        SuitInterfaceRole::Provider => true,
    };
    if !compatible_roles {
        return fail(SuitInteropFindingStatus::InterfaceRoleMismatch);
    }
    if endpoint.contract_revision != peer.contract_revision {
        return fail(SuitInteropFindingStatus::ContractRevisionMismatch);
    }
    if endpoint.schema_digest_sha256 != peer.schema_digest_sha256 {
        return fail(SuitInteropFindingStatus::SchemaDigestMismatch);
    }
    if endpoint.connector_profile_id != peer.connector_profile_id {
        return fail(SuitInteropFindingStatus::ConnectorProfileMismatch);
    }
    if !policy
        .required_environments
        .is_subset(&endpoint.applicable_environments)
        || !policy
            .required_environments
            .is_subset(&peer.applicable_environments)
    {
        return fail(SuitInteropFindingStatus::OperatingEnvironmentMismatch);
    }
    if endpoint.evidence_level < policy.minimum_evidence_level
        || peer.evidence_level < policy.minimum_evidence_level
    {
        return fail(SuitInteropFindingStatus::InsufficientEvidence);
    }
    (SuitInteropFindingStatus::CompatibleUnderPolicy, peer_endpoint)
}

fn compare_endpoints(
    left: &SuitInterfaceEndpoint,
    right: &SuitInterfaceEndpoint,
) -> std::cmp::Ordering {
    left.interface_id
        .cmp(&right.interface_id)
        .then(left.endpoint_id.cmp(&right.endpoint_id))
        .then(left.kind.cmp(&right.kind))
        .then(left.role.cmp(&right.role))
        .then(left.contract_revision.cmp(&right.contract_revision))
        .then(left.connector_profile_id.cmp(&right.connector_profile_id))
}

fn compare_findings(left: &SuitInteropFinding, right: &SuitInteropFinding) -> std::cmp::Ordering {
    left.component_id
        .cmp(&right.component_id)
        .then(left.endpoint_id.cmp(&right.endpoint_id))
        .then(left.interface_id.cmp(&right.interface_id))
        .then(left.interface_kind.cmp(&right.interface_kind))
        .then(left.role.cmp(&right.role))
        .then(left.required.cmp(&right.required))
        .then(left.peer_component_id.cmp(&right.peer_component_id))
        .then(left.peer_endpoint_id.cmp(&right.peer_endpoint_id))
        .then(left.status.cmp(&right.status))
}

fn clean_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
        && value[7..] == value[7..].to_ascii_lowercase()
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
                self.raw(&[1]);
                self.string(value);
            }
            None => self.raw(&[0]),
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

fn evidence_tag(level: SuitEvidenceLevel) -> &'static str {
    match level {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn kind_tag(kind: SuitInterfaceKind) -> &'static str {
    match kind {
        SuitInterfaceKind::PressureBoundary => "pressure_boundary",
        SuitInterfaceKind::BreathableGas => "breathable_gas",
        SuitInterfaceKind::Coolant => "coolant",
        SuitInterfaceKind::ThermalExchange => "thermal_exchange",
        SuitInterfaceKind::ElectricalPower => "electrical_power",
        SuitInterfaceKind::GroundingAndEsd => "grounding_and_esd",
        SuitInterfaceKind::Communications => "communications",
        SuitInterfaceKind::TelemetryAndData => "telemetry_and_data",
        SuitInterfaceKind::MechanicalCapture => "mechanical_capture",
        SuitInterfaceKind::RestraintAndAnchor => "restraint_and_anchor",
        SuitInterfaceKind::ConsumableTransfer => "consumable_transfer",
        SuitInterfaceKind::DustMitigationService => "dust_mitigation_service",
        SuitInterfaceKind::RescueHandling => "rescue_handling",
    }
}

fn role_tag(role: SuitInterfaceRole) -> &'static str {
    match role {
        SuitInterfaceRole::Provider => "provider",
        SuitInterfaceRole::Consumer => "consumer",
        SuitInterfaceRole::Bidirectional => "bidirectional",
    }
}

fn environment_tag(environment: SuitOperatingEnvironment) -> &'static str {
    match environment {
        SuitOperatingEnvironment::MicrogravityVacuum => "microgravity_vacuum",
        SuitOperatingEnvironment::LunarSunlitSurface => "lunar_sunlit_surface",
        SuitOperatingEnvironment::LunarShadowOrPolarRegion => "lunar_shadow_or_polar_region",
        SuitOperatingEnvironment::MarsSurface => "mars_surface",
        SuitOperatingEnvironment::PressurizedCabin => "pressurized_cabin",
        SuitOperatingEnvironment::ThermalVacuumGroundTest => "thermal_vacuum_ground_test",
        SuitOperatingEnvironment::GroundLaboratory => "ground_laboratory",
    }
}

fn finding_status_tag(status: SuitInteropFindingStatus) -> &'static str {
    match status {
        SuitInteropFindingStatus::CompatibleUnderPolicy => "compatible_under_policy",
        SuitInteropFindingStatus::MissingPeerInterface => "missing_peer_interface",
        SuitInteropFindingStatus::AmbiguousPeerInterfaces => "ambiguous_peer_interfaces",
        SuitInteropFindingStatus::InterfaceKindMismatch => "interface_kind_mismatch",
        SuitInteropFindingStatus::InterfaceRoleMismatch => "interface_role_mismatch",
        SuitInteropFindingStatus::ContractRevisionMismatch => "contract_revision_mismatch",
        SuitInteropFindingStatus::SchemaDigestMismatch => "schema_digest_mismatch",
        SuitInteropFindingStatus::ConnectorProfileMismatch => "connector_profile_mismatch",
        SuitInteropFindingStatus::OperatingEnvironmentMismatch => "operating_environment_mismatch",
        SuitInteropFindingStatus::InsufficientEvidence => "insufficient_evidence",
    }
}

fn report_status_tag(status: SuitInteropReportStatus) -> &'static str {
    match status {
        SuitInteropReportStatus::NoRequiredInterfacesToVerify => "no_required_interfaces_to_verify",
        SuitInteropReportStatus::RequiredInterfacesBlocked => "required_interfaces_blocked",
        SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy => {
            "meets_declared_compatibility_policy"
        },
    }
}

fn evidence_class_tag(class: SuitInteropEvidenceClass) -> &'static str {
    match class {
        SuitInteropEvidenceClass::DeclaredSoftwareContract => "declared_software_contract",
    }
}

fn digest_manifest(manifest: &SuitInterfaceManifest) -> String {
    let mut endpoints = manifest.endpoints.clone();
    endpoints.sort_by(compare_endpoints);
    let mut h = HashWriter::new("symthaea.suit.interop.manifest.v1");
    h.string(&manifest.component_id);
    h.string(&manifest.hardware_revision);
    h.usize(endpoints.len());
    for endpoint in &endpoints {
        h.string(&endpoint.interface_id);
        h.string(&endpoint.endpoint_id);
        h.string(kind_tag(endpoint.kind));
        h.string(role_tag(endpoint.role));
        h.string(&endpoint.contract_revision);
        h.string(&endpoint.schema_digest_sha256);
        h.string(&endpoint.connector_profile_id);
        h.usize(endpoint.applicable_environments.len());
        for environment in &endpoint.applicable_environments {
            h.string(environment_tag(*environment));
        }
        h.string(evidence_tag(endpoint.evidence_level));
        h.string(&endpoint.evidence_reference);
        h.boolean(endpoint.required);
    }
    h.finish()
}

fn digest_policy(policy: &SuitInteropPolicy) -> String {
    let mut h = HashWriter::new("symthaea.suit.interop.policy.v1");
    h.string(evidence_tag(policy.minimum_evidence_level));
    h.usize(policy.required_environments.len());
    for environment in &policy.required_environments {
        h.string(environment_tag(*environment));
    }
    h.finish()
}

fn digest_report(report: &SuitInteropReport) -> String {
    let mut h = HashWriter::new("symthaea.suit.interop.report.v1");
    h.string(evidence_class_tag(report.evidence_class));
    h.string(&report.left_component_id);
    h.string(&report.left_hardware_revision);
    h.string(&report.right_component_id);
    h.string(&report.right_hardware_revision);
    h.string(evidence_tag(report.policy.minimum_evidence_level));
    h.usize(report.policy.required_environments.len());
    for environment in &report.policy.required_environments {
        h.string(environment_tag(*environment));
    }
    h.usize(report.findings.len());
    for finding in &report.findings {
        h.string(&finding.component_id);
        h.string(&finding.endpoint_id);
        h.string(&finding.interface_id);
        h.string(kind_tag(finding.interface_kind));
        h.string(role_tag(finding.role));
        h.boolean(finding.required);
        h.string(&finding.peer_component_id);
        h.option_string(finding.peer_endpoint_id.as_deref());
        h.string(finding_status_tag(finding.status));
    }
    h.string(report_status_tag(report.status));
    h.usize(report.required_interface_count);
    h.usize(report.required_compatible_count);
    h.usize(report.blocked_required_count);
    h.boolean(report.represents_physical_qualification);
    h.finish()
}

fn digest_receipt(receipt: &SuitInteropReceipt) -> String {
    let mut h = HashWriter::new("symthaea.suit.interop.receipt.v1");
    h.string(&receipt.left_manifest_digest_sha256);
    h.string(&receipt.right_manifest_digest_sha256);
    h.string(&receipt.policy_digest_sha256);
    h.string(&receipt.result_digest_sha256);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(
        id: &str,
        role: SuitInterfaceRole,
        kind: SuitInterfaceKind,
    ) -> SuitInterfaceEndpoint {
        SuitInterfaceEndpoint {
            interface_id: id.to_string(),
            endpoint_id: format!("endpoint-{id}"),
            kind,
            role,
            contract_revision: "v1".to_string(),
            schema_digest_sha256: format!("sha256:{}", "a".repeat(64)),
            connector_profile_id: "connector-demo-r1".to_string(),
            applicable_environments: [
                SuitOperatingEnvironment::LunarSunlitSurface,
                SuitOperatingEnvironment::LunarShadowOrPolarRegion,
            ]
            .into_iter()
            .collect(),
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "sim://fixture/interface-contract".to_string(),
            required: true,
        }
    }

    fn manifest(id: &str, endpoints: Vec<SuitInterfaceEndpoint>) -> SuitInterfaceManifest {
        SuitInterfaceManifest {
            component_id: id.to_string(),
            hardware_revision: "proto-r1".to_string(),
            endpoints,
        }
    }

    fn policy() -> SuitInteropPolicy {
        SuitInteropPolicy {
            minimum_evidence_level: SuitEvidenceLevel::Simulated,
            required_environments: [SuitOperatingEnvironment::LunarShadowOrPolarRegion]
                .into_iter()
                .collect(),
        }
    }

    fn pair() -> (SuitInterfaceManifest, SuitInterfaceManifest) {
        (
            manifest(
                "suit-lunar-r1",
                vec![endpoint(
                    "vehicle-grounding",
                    SuitInterfaceRole::Consumer,
                    SuitInterfaceKind::GroundingAndEsd,
                )],
            ),
            manifest(
                "lander-lunar-r1",
                vec![endpoint(
                    "vehicle-grounding",
                    SuitInterfaceRole::Provider,
                    SuitInterfaceKind::GroundingAndEsd,
                )],
            ),
        )
    }

    #[test]
    fn matching_grounding_interface_is_only_declared_compatible_under_context() {
        let (left, right) = pair();
        let (report, receipt) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert_eq!(
            report.status,
            SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
        );
        assert_eq!(report.required_interface_count, 1);
        assert_eq!(report.required_compatible_count, 1);
        assert!(!report.represents_physical_qualification);
        assert!(verify_suit_interop_receipt(&left, &right, &policy(), &report, &receipt));
    }

    #[test]
    fn a_missing_peer_interface_blocks_the_required_policy() {
        let (left, _) = pair();
        let right = manifest("lander-no-grounding", vec![]);
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert_eq!(report.status, SuitInteropReportStatus::RequiredInterfacesBlocked);
        assert_eq!(
            report.findings[0].status,
            SuitInteropFindingStatus::MissingPeerInterface
        );
    }

    #[test]
    fn equal_label_with_a_different_schema_digest_is_not_compatible() {
        let (left, mut right) = pair();
        right.endpoints[0].schema_digest_sha256 = format!("sha256:{}", "b".repeat(64));
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::SchemaDigestMismatch
        }));
        assert_eq!(report.status, SuitInteropReportStatus::RequiredInterfacesBlocked);
    }

    #[test]
    fn pressure_and_grounding_interfaces_are_not_treated_as_substitutable() {
        let (left, mut right) = pair();
        right.endpoints[0].kind = SuitInterfaceKind::PressureBoundary;
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::InterfaceKindMismatch
        }));
        assert_eq!(report.status, SuitInteropReportStatus::RequiredInterfacesBlocked);
    }

    #[test]
    fn revision_and_connector_mismatches_are_not_coerced() {
        let (left, mut right) = pair();
        right.endpoints[0].contract_revision = "v2".to_string();
        let (revision_report, _) =
            evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(revision_report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::ContractRevisionMismatch
        }));

        right.endpoints[0].contract_revision = "v1".to_string();
        right.endpoints[0].connector_profile_id = "connector-other".to_string();
        let (connector_report, _) =
            evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(connector_report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::ConnectorProfileMismatch
        }));
    }

    #[test]
    fn unsupported_operating_environment_blocks_the_claim() {
        let (left, mut right) = pair();
        right.endpoints[0].applicable_environments = [
            SuitOperatingEnvironment::LunarSunlitSurface,
        ]
        .into_iter()
        .collect();
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::OperatingEnvironmentMismatch
        }));
    }

    #[test]
    fn ambiguous_peer_interfaces_do_not_pick_a_convenient_provider() {
        let (left, right) = pair();
        let mut duplicated = right.clone();
        let mut second = duplicated.endpoints[0].clone();
        second.endpoint_id = "endpoint-second-grounding-provider".into();
        duplicated.endpoints.push(second);
        let (report, _) = evaluate_suit_interoperability(&left, &duplicated, &policy()).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::AmbiguousPeerInterfaces
        }));
    }

    #[test]
    fn insufficient_evidence_is_reported_without_promoting_a_claim() {
        let (left, mut right) = pair();
        right.endpoints[0].evidence_level = SuitEvidenceLevel::Declared;
        let strict = SuitInteropPolicy {
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            ..policy()
        };
        let (report, _) = evaluate_suit_interoperability(&left, &right, &strict).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::InsufficientEvidence
        }));
    }

    #[test]
    fn bidirectional_contract_does_not_match_a_one_way_provider() {
        let (mut left, right) = pair();
        left.endpoints[0].role = SuitInterfaceRole::Bidirectional;
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert!(report.findings.iter().any(|finding| {
            finding.status == SuitInteropFindingStatus::InterfaceRoleMismatch
        }));
    }

    #[test]
    fn manifests_are_order_invariant_and_changes_break_the_receipt() {
        let (left, right) = pair();
        let p = policy();
        let (report_a, receipt_a) = evaluate_suit_interoperability(&left, &right, &p).unwrap();

        let mut reversed_left = left.clone();
        reversed_left.endpoints.reverse();
        let mut reversed_right = right.clone();
        reversed_right.endpoints.reverse();
        let (report_b, receipt_b) =
            evaluate_suit_interoperability(&reversed_left, &reversed_right, &p).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);

        let mut changed = right.clone();
        changed.endpoints[0].contract_revision = "v2".to_string();
        assert!(!verify_suit_interop_receipt(&left, &changed, &p, &report_a, &receipt_a));
    }

    #[test]
    fn absence_of_required_consumer_interfaces_is_not_a_compatibility_pass() {
        let mut offered = endpoint(
            "optional-diagnostic-data",
            SuitInterfaceRole::Provider,
            SuitInterfaceKind::TelemetryAndData,
        );
        offered.required = false;
        let left = manifest("optional-provider", vec![offered]);
        let right = manifest("unconnected-component", vec![]);
        let (report, _) = evaluate_suit_interoperability(&left, &right, &policy()).unwrap();
        assert_eq!(
            report.status,
            SuitInteropReportStatus::NoRequiredInterfacesToVerify
        );
        assert_eq!(report.required_interface_count, 0);
        assert!(!report.represents_physical_qualification);
    }

    #[test]
    fn invalid_digest_duplicate_endpoint_and_empty_context_fail_closed() {
        let (left, right) = pair();
        let mut invalid_digest = right.clone();
        invalid_digest.endpoints[0].schema_digest_sha256 = "not-a-digest".into();
        assert_eq!(
            evaluate_suit_interoperability(&left, &invalid_digest, &policy()),
            Err(SuitInteropError::InvalidRightManifest)
        );

        let mut duplicate_endpoint = right.clone();
        duplicate_endpoint.endpoints.push(duplicate_endpoint.endpoints[0].clone());
        assert_eq!(
            evaluate_suit_interoperability(&left, &duplicate_endpoint, &policy()),
            Err(SuitInteropError::InvalidRightManifest)
        );

        let empty_policy = SuitInteropPolicy {
            minimum_evidence_level: SuitEvidenceLevel::Declared,
            required_environments: BTreeSet::new(),
        };
        assert_eq!(
            evaluate_suit_interoperability(&left, &right, &empty_policy),
            Err(SuitInteropError::InvalidPolicy)
        );
    }
}
