// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end synthetic mission-task replay with explicit expected outcomes.

use std::collections::BTreeSet;

use symthaea_exoskeleton::suit_interop::{
    SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
    SuitInteropPolicy, SuitOperatingEnvironment,
};
use symthaea_exoskeleton::suit_mission_configuration::{
    MissionComponent, MissionComponentKind, MissionConfiguration, MissionConnection,
};
use symthaea_exoskeleton::suit_mission_tasks::{
    evaluate_mission_tasks, verify_mission_task_receipt, MissionTaskFindingCode,
    MissionTaskPlanStatus, MissionTaskRequirement,
};
use symthaea_exoskeleton::suit_profile::{
    SuitCapability, SuitCapabilityClaim, SuitEvidenceLevel, SuitMissionProfile,
    SuitVariantDeclaration,
};

fn fixture() -> MissionConfiguration {
    let variant_id = "synthetic-lunar-suit";
    let suit_revision = "suit-r3";
    let lander_revision = "lander-r2";
    let environments = BTreeSet::from([
        SuitOperatingEnvironment::LunarShadowOrPolarRegion,
    ]);
    let manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
        SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: revision.into(),
            endpoints: vec![SuitInterfaceEndpoint {
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
            }],
        }
    };
    let claims = SuitMissionProfile::LunarSurfaceEva
        .required_capabilities()
        .into_iter()
        .map(|capability| SuitCapabilityClaim {
            capability,
            applies_to_variant_id: variant_id.into(),
            applies_to_hardware_revision: suit_revision.into(),
            implementation_id: format!("synthetic-implementation:{capability:?}"),
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: format!("synthetic-evidence:{capability:?}"),
        })
        .collect();

    MissionConfiguration {
        schema_version: 1,
        mission_id: "synthetic-lunar-task-replay".into(),
        configuration_revision: "configuration-r1".into(),
        suit_component_id: "suit".into(),
        suit_profile: SuitMissionProfile::LunarSurfaceEva,
        suit_variant: SuitVariantDeclaration {
            variant_id: variant_id.into(),
            hardware_revision: suit_revision.into(),
            capability_claims: claims,
        },
        minimum_capability_evidence: SuitEvidenceLevel::BenchTested,
        required_component_kinds: BTreeSet::from([
            MissionComponentKind::Suit,
            MissionComponentKind::LanderOrVehicle,
        ]),
        required_connection_ids: BTreeSet::from(["suit-lander-grounding".into()]),
        components: vec![
            MissionComponent {
                component_id: "suit".into(),
                kind: MissionComponentKind::Suit,
                hardware_revision: suit_revision.into(),
                interfaces: manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
            },
            MissionComponent {
                component_id: "lander".into(),
                kind: MissionComponentKind::LanderOrVehicle,
                hardware_revision: lander_revision.into(),
                interfaces: manifest("lander", lander_revision, SuitInterfaceRole::Provider),
            },
        ],
        connections: vec![MissionConnection {
            connection_id: "suit-lander-grounding".into(),
            left_component_id: "suit".into(),
            right_component_id: "lander".into(),
            policy: SuitInteropPolicy {
                minimum_evidence_level: SuitEvidenceLevel::Simulated,
                required_environments: environments,
            },
        }],
    }
}

fn task() -> MissionTaskRequirement {
    MissionTaskRequirement {
        task_id: "prepare-for-lunar-shadow-work".into(),
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
fn successful_synthetic_task_requires_explicit_evidence_and_exact_graph() {
    let configuration = fixture();
    let tasks = vec![task()];
    let (report, receipt) = evaluate_mission_tasks(&configuration, &tasks).unwrap();

    assert_eq!(report.schema_version, 1);
    assert_eq!(report.mission_id, "synthetic-lunar-task-replay");
    assert_eq!(
        report.status,
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy
    );
    assert_eq!(report.required_task_count, 1);
    assert_eq!(report.satisfied_required_task_count, 1);
    assert_eq!(report.blocked_required_task_count, 0);
    assert_eq!(
        report.task_evaluations[0].task_id,
        "prepare-for-lunar-shadow-work"
    );
    assert!(report.task_evaluations[0].findings.is_empty());
    assert!(!report.represents_physical_qualification);
    assert!(verify_mission_task_receipt(
        &configuration,
        &tasks,
        &report,
        &receipt,
    ));

    let mut changed_configuration = configuration.clone();
    changed_configuration.configuration_revision = "configuration-r2".into();
    assert!(!verify_mission_task_receipt(
        &changed_configuration,
        &tasks,
        &report,
        &receipt,
    ));
}

#[test]
fn missing_capability_resource_connection_and_environment_are_negative_controls() {
    let configuration = fixture();
    let tasks = vec![MissionTaskRequirement {
        task_id: "rescue-in-shadow".into(),
        required: true,
        required_capabilities: BTreeSet::from([
            SuitCapability::RadiationMonitoringInterface,
        ]),
        required_component_ids: BTreeSet::from(["rescue-unit".into()]),
        required_connection_ids: BTreeSet::from(["suit-rescue-link".into()]),
        required_environments: BTreeSet::from([
            SuitOperatingEnvironment::LunarSunlitSurface,
        ]),
    }];
    let (report, _) = evaluate_mission_tasks(&configuration, &tasks).unwrap();

    assert_eq!(report.status, MissionTaskPlanStatus::Blocked);
    let findings = &report.task_evaluations[0].findings;
    assert!(findings.iter().any(|finding| {
        finding.code == MissionTaskFindingCode::MissingCapabilityEvidence
    }));
    assert!(findings.iter().any(|finding| {
        finding.code == MissionTaskFindingCode::MissingComponent
    }));
    assert!(findings.iter().any(|finding| {
        finding.code == MissionTaskFindingCode::MissingConnection
    }));
}
