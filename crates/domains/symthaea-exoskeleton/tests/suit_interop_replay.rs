// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay for SX-030's versioned interface fixture.
//! Digest byte encoding is reimplemented here rather than calling production
//! digest helpers, so the test can cross-check its output receipt.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use symthaea_exoskeleton::suit_interop::{
    evaluate_suit_interoperability, SuitInterfaceEndpoint, SuitInterfaceKind,
    SuitInterfaceManifest, SuitInterfaceRole, SuitInteropEvidenceClass,
    SuitInteropFinding, SuitInteropFindingStatus, SuitInteropPolicy, SuitInteropReceipt,
    SuitInteropReport, SuitInteropReportStatus, SuitOperatingEnvironment,
    verify_suit_interop_receipt,
};
use symthaea_exoskeleton::suit_profile::SuitEvidenceLevel;

const FIXTURE: &str = include_str!("fixtures/suit_interop_lunar_grounding_v1.json");

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: u32,
    fixture_id: String,
    left: SuitInterfaceManifest,
    right: SuitInterfaceManifest,
    policy: SuitInteropPolicy,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    status: SuitInteropReportStatus,
    required_interface_count: usize,
    required_compatible_count: usize,
    blocked_required_count: usize,
    finding_count: usize,
    finding_status: SuitInteropFindingStatus,
    represents_physical_qualification: bool,
}

#[test]
fn replays_grounding_fixture_and_independently_verifies_receipt() {
    let corpus: Corpus = serde_json::from_str(FIXTURE)
        .expect("versioned interface fixture must parse");
    assert_eq!(corpus.schema_version, 1);
    assert_eq!(corpus.fixture_id, "synthetic-lunar-grounding-interface-pair-v1");
    assert_eq!(corpus.left.endpoints.len(), 1);
    assert_eq!(corpus.right.endpoints.len(), 1);

    let (report, receipt) = evaluate_suit_interoperability(
        &corpus.left,
        &corpus.right,
        &corpus.policy,
    )
    .expect("synthetic declared contracts should evaluate");

    assert_eq!(report.status, corpus.expected.status);
    assert_eq!(
        report.required_interface_count,
        corpus.expected.required_interface_count
    );
    assert_eq!(
        report.required_compatible_count,
        corpus.expected.required_compatible_count
    );
    assert_eq!(
        report.blocked_required_count,
        corpus.expected.blocked_required_count
    );
    assert_eq!(report.findings.len(), corpus.expected.finding_count);
    assert_eq!(report.findings[0].status, corpus.expected.finding_status);
    assert_eq!(
        report.represents_physical_qualification,
        corpus.expected.represents_physical_qualification
    );
    assert_eq!(
        report.evidence_class,
        SuitInteropEvidenceClass::DeclaredSoftwareContract
    );

    assert_eq!(
        receipt.left_manifest_digest_sha256,
        independent_manifest_digest(&corpus.left)
    );
    assert_eq!(
        receipt.right_manifest_digest_sha256,
        independent_manifest_digest(&corpus.right)
    );
    assert_eq!(
        receipt.policy_digest_sha256,
        independent_policy_digest(&corpus.policy)
    );
    assert_eq!(
        receipt.result_digest_sha256,
        independent_report_digest(&report)
    );
    assert_eq!(
        receipt.receipt_digest_sha256,
        independent_receipt_digest(&receipt)
    );
    assert!(verify_suit_interop_receipt(
        &corpus.left,
        &corpus.right,
        &corpus.policy,
        &report,
        &receipt,
    ));
}

#[test]
fn input_order_is_invariant_and_contract_changes_invalidate_the_receipt() {
    let corpus: Corpus = serde_json::from_str(FIXTURE).unwrap();
    let (report_a, receipt_a) = evaluate_suit_interoperability(
        &corpus.left,
        &corpus.right,
        &corpus.policy,
    )
    .unwrap();

    let mut reversed_left = corpus.left.clone();
    reversed_left.endpoints.reverse();
    let mut reversed_right = corpus.right.clone();
    reversed_right.endpoints.reverse();
    let (report_b, receipt_b) = evaluate_suit_interoperability(
        &reversed_left,
        &reversed_right,
        &corpus.policy,
    )
    .unwrap();
    assert_eq!(report_a, report_b);
    assert_eq!(receipt_a, receipt_b);

    let mut changed = corpus.right.clone();
    changed.endpoints[0].contract_revision = "2.0.0".into();
    assert!(!verify_suit_interop_receipt(
        &corpus.left,
        &changed,
        &corpus.policy,
        &report_a,
        &receipt_a,
    ));
}

struct IndependentHashWriter(Sha256);

impl IndependentHashWriter {
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

fn independent_manifest_digest(manifest: &SuitInterfaceManifest) -> String {
    let mut endpoints = manifest.endpoints.clone();
    endpoints.sort_by(compare_endpoints);
    let mut h = IndependentHashWriter::new("symthaea.suit.interop.manifest.v1");
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

fn independent_policy_digest(policy: &SuitInteropPolicy) -> String {
    let mut h = IndependentHashWriter::new("symthaea.suit.interop.policy.v1");
    h.string(evidence_tag(policy.minimum_evidence_level));
    h.usize(policy.required_environments.len());
    for environment in &policy.required_environments {
        h.string(environment_tag(*environment));
    }
    h.finish()
}

fn independent_report_digest(report: &SuitInteropReport) -> String {
    let mut h = IndependentHashWriter::new("symthaea.suit.interop.report.v1");
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

fn independent_receipt_digest(receipt: &SuitInteropReceipt) -> String {
    let mut h = IndependentHashWriter::new("symthaea.suit.interop.receipt.v1");
    h.string(&receipt.left_manifest_digest_sha256);
    h.string(&receipt.right_manifest_digest_sha256);
    h.string(&receipt.policy_digest_sha256);
    h.string(&receipt.result_digest_sha256);
    h.finish()
}

// Keep the types in this module's public contract visible during fixture review.
#[allow(dead_code)]
fn _finding_type_anchor(_: &SuitInteropFinding) {}
