// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay and deterministic size-corpus checks for SX-031.
//! The scenarios are synthetic software-contract fixtures, not hardware evidence.

use std::collections::BTreeSet;

use symthaea_exoskeleton::suit_interop::{
    evaluate_suit_interoperability, verify_suit_interop_receipt, SuitInterfaceEndpoint,
    SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole, SuitInteropPolicy,
    SuitOperatingEnvironment,
};
use symthaea_exoskeleton::suit_mission_configuration::{
    evaluate_mission_configuration, verify_mission_configuration_receipt,
    MissionComponent, MissionComponentKind, MissionConfiguration, MissionConfigurationStatus,
    MissionConnection, MissionFindingCode,
};
use symthaea_exoskeleton::suit_profile::{
    SuitCapabilityClaim, SuitEvidenceLevel, SuitMissionProfile,
    SuitVariantDeclaration,
};

fn endpoint(component_id: &str, role: SuitInterfaceRole) -> SuitInterfaceEndpoint {
    SuitInterfaceEndpoint {
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
    }
}

fn component(
    component_id: &str,
    kind: MissionComponentKind,
    revision: &str,
    role: SuitInterfaceRole,
) -> MissionComponent {
    MissionComponent {
        component_id: component_id.into(),
        kind,
        hardware_revision: revision.into(),
        interfaces: SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: revision.into(),
            endpoints: vec![endpoint(component_id, role)],
        },
    }
}

fn scenario(worksite_count: usize) -> MissionConfiguration {
    let variant_id = "synthetic-lunar-suit";
    let suit_revision = "suit-r3";
    let mut capability_claims: Vec<SuitCapabilityClaim> =
        SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .into_iter()
            .map(|capability| SuitCapabilityClaim {
                capability,
                applies_to_variant_id: variant_id.into(),
                applies_to_hardware_revision: suit_revision.into(),
                implementation_id: format!("synthetic-impl-{capability:?}"),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-evidence-{capability:?}"),
            })
            .collect();

    let suit = component(
        "suit",
        MissionComponentKind::Suit,
        suit_revision,
        SuitInterfaceRole::Consumer,
    );
    let lander = component(
        "lander",
        MissionComponentKind::LanderOrVehicle,
        "lander-r2",
        SuitInterfaceRole::Provider,
    );
    let mut components = vec![suit, lander];
    let mut connections = vec![MissionConnection {
        connection_id: "suit-lander-grounding".into(),
        left_component_id: "suit".into(),
        right_component_id: "lander".into(),
        policy: policy(),
    }];
    let mut required_component_kinds = BTreeSet::from([
        MissionComponentKind::Suit,
        MissionComponentKind::LanderOrVehicle,
    ]);

    for index in 0..worksite_count {
        let component_id = format!("worksite-{index:04}");
        components.push(component(
            &component_id,
            MissionComponentKind::Worksite,
            "worksite-r1",
            SuitInterfaceRole::Provider,
        ));
        connections.push(MissionConnection {
            connection_id: format!("suit-{component_id}-grounding"),
            left_component_id: "suit".into(),
            right_component_id: component_id,
            policy: policy(),
        });
    }
    if worksite_count > 0 {
        required_component_kinds.insert(MissionComponentKind::Worksite);
    }
    capability_claims.sort_by(|left, right| {
        left.capability
            .cmp(&right.capability)
            .then(left.implementation_id.cmp(&right.implementation_id))
    });

    MissionConfiguration {
        schema_version: 1,
        mission_id: "synthetic-lunar-eva".into(),
        configuration_revision: "fixture-r1".into(),
        suit_component_id: "suit".into(),
        suit_profile: SuitMissionProfile::LunarSurfaceEva,
        suit_variant: SuitVariantDeclaration {
            variant_id: variant_id.into(),
            hardware_revision: suit_revision.into(),
            capability_claims,
        },
        minimum_capability_evidence: SuitEvidenceLevel::BenchTested,
        required_component_kinds,
        required_connection_ids: BTreeSet::from(["suit-lander-grounding".into()]),
        components,
        connections,
    }
}

fn policy() -> SuitInteropPolicy {
    SuitInteropPolicy {
        minimum_evidence_level: SuitEvidenceLevel::Simulated,
        required_environments: BTreeSet::from([
            SuitOperatingEnvironment::LunarShadowOrPolarRegion,
        ]),
    }
}

#[test]
fn independent_replay_checks_pairwise_and_mission_receipts() {
    let configuration = scenario(0);
    let suit = configuration
        .components
        .iter()
        .find(|component| component.component_id == "suit")
        .unwrap();
    let lander = configuration
        .components
        .iter()
        .find(|component| component.component_id == "lander")
        .unwrap();
    let (pair_report, pair_receipt) = evaluate_suit_interoperability(
        &suit.interfaces,
        &lander.interfaces,
        &policy(),
    )
    .unwrap();
    assert!(verify_suit_interop_receipt(
        &suit.interfaces,
        &lander.interfaces,
        &policy(),
        &pair_report,
        &pair_receipt,
    ));

    let (report, receipt) = evaluate_mission_configuration(&configuration).unwrap();
    assert_eq!(
        report.status,
        MissionConfigurationStatus::MeetsDeclaredMissionConfigurationPolicy
    );
    assert!(!report.represents_physical_qualification);
    assert!(verify_mission_configuration_receipt(
        &configuration,
        &report,
        &receipt,
    ));

    let mut changed = configuration.clone();
    changed.configuration_revision = "fixture-r2".into();
    assert!(!verify_mission_configuration_receipt(
        &changed,
        &report,
        &receipt,
    ));
}

#[test]
fn deterministic_size_corpus_catches_missing_links_and_disconnected_nodes() {
    for count in [0_usize, 1, 8, 32, 128] {
        let configuration = scenario(count);
        let (report_a, receipt_a) = evaluate_mission_configuration(&configuration).unwrap();
        let mut permuted = configuration.clone();
        permuted.components.reverse();
        permuted.connections.reverse();
        permuted.suit_variant.capability_claims.reverse();
        let (report_b, receipt_b) = evaluate_mission_configuration(&permuted).unwrap();
        assert_eq!(report_a, report_b, "component count {}", count + 2);
        assert_eq!(receipt_a, receipt_b, "component count {}", count + 2);
        assert_eq!(report_a.component_count, count + 2);
        assert_eq!(report_a.connection_count, count + 1);
        assert_eq!(
            report_a.status,
            MissionConfigurationStatus::MeetsDeclaredMissionConfigurationPolicy
        );
    }

    let mut missing_link = scenario(0);
    missing_link.connections.clear();
    let (report, _) = evaluate_mission_configuration(&missing_link).unwrap();
    assert_eq!(report.status, MissionConfigurationStatus::Blocked);
    assert!(report.findings.iter().any(|finding| {
        finding.code == MissionFindingCode::MissingRequiredConnection
    }));
    assert!(report.findings.iter().any(|finding| {
        finding.code == MissionFindingCode::RequiredComponentDisconnected
            && finding.subject_id == "lander"
    }));
}
