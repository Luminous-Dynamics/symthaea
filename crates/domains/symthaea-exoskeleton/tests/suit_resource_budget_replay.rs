// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent scenario replay for SX-033 conservative resource budgeting.

use std::collections::BTreeSet;

use symthaea_exoskeleton::suit_interop::{
    SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
    SuitInteropPolicy, SuitOperatingEnvironment,
};
use symthaea_exoskeleton::suit_mission_configuration::{
    MissionComponent, MissionComponentKind, MissionConfiguration, MissionConnection,
};
use symthaea_exoskeleton::suit_mission_tasks::{
    evaluate_mission_tasks, MissionTaskPlanStatus, MissionTaskReceipt, MissionTaskRequirement,
};
use symthaea_exoskeleton::suit_profile::{
    SuitCapability, SuitCapabilityClaim, SuitEvidenceLevel, SuitMissionProfile,
    SuitVariantDeclaration,
};
use symthaea_exoskeleton::suit_resource_budget::{
    evaluate_mission_resource_budget, verify_mission_resource_budget_receipt,
    MissionResourceBudgetPlan, MissionResourceBudgetStatus, MissionResourceDemand,
    MissionResourceKind, MissionResourcePool, ResourceQuantityInterval,
};

fn fixture() -> (
    MissionConfiguration,
    Vec<MissionTaskRequirement>,
    MissionTaskReceipt,
    MissionResourceBudgetPlan,
) {
    let variant_id = "replay-lunar-suit";
    let suit_revision = "suit-replay-r1";
    let vehicle_revision = "vehicle-replay-r2";
    let environments = BTreeSet::from([
        SuitOperatingEnvironment::LunarShadowOrPolarRegion,
    ]);
    let make_manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
        SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: revision.into(),
            endpoints: vec![SuitInterfaceEndpoint {
                interface_id: "consumable-service.v1".into(),
                endpoint_id: format!("{component_id}.consumable-service"),
                kind: SuitInterfaceKind::ConsumableTransfer,
                role,
                contract_revision: "1.0.0".into(),
                schema_digest_sha256: format!("sha256:{}", "d".repeat(64)),
                connector_profile_id: "synthetic-consumable-service-v1".into(),
                applicable_environments: environments.clone(),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-interface:{component_id}"),
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
            evidence_reference: format!("synthetic-capability-evidence:{capability:?}"),
        })
        .collect();

    let configuration = MissionConfiguration {
        schema_version: 1,
        mission_id: "synthetic-resource-replay".into(),
        configuration_revision: "config-r1".into(),
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
        required_connection_ids: BTreeSet::from(["suit-vehicle-resource-link".into()]),
        components: vec![
            MissionComponent {
                component_id: "suit".into(),
                kind: MissionComponentKind::Suit,
                hardware_revision: suit_revision.into(),
                interfaces: make_manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
            },
            MissionComponent {
                component_id: "vehicle".into(),
                kind: MissionComponentKind::LanderOrVehicle,
                hardware_revision: vehicle_revision.into(),
                interfaces: make_manifest("vehicle", vehicle_revision, SuitInterfaceRole::Provider),
            },
        ],
        connections: vec![MissionConnection {
            connection_id: "suit-vehicle-resource-link".into(),
            left_component_id: "suit".into(),
            right_component_id: "vehicle".into(),
            policy: SuitInteropPolicy {
                minimum_evidence_level: SuitEvidenceLevel::Simulated,
                required_environments: environments.clone(),
            },
        }],
    };
    let tasks = vec![MissionTaskRequirement {
        task_id: "required-shadow-service".into(),
        required: true,
        required_capabilities: BTreeSet::from([
            SuitCapability::ElectrostaticChargeManagement,
        ]),
        required_component_ids: BTreeSet::from(["vehicle".into()]),
        required_connection_ids: BTreeSet::from(["suit-vehicle-resource-link".into()]),
        required_environments: environments.clone(),
    }];
    let (task_report, task_receipt) =
        evaluate_mission_tasks(&configuration, &tasks).unwrap();
    assert_eq!(task_report.status, MissionTaskPlanStatus::MeetsDeclaredTaskPolicy);

    let plan = MissionResourceBudgetPlan {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
        expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
        minimum_evidence_level: SuitEvidenceLevel::BenchTested,
        reserve_basis_points: 2_000,
        pools: vec![MissionResourcePool {
            pool_id: "vehicle-oxygen-pool".into(),
            resource_kind: MissionResourceKind::OxygenMassMicrograms,
            provider_component_id: "vehicle".into(),
            provider_hardware_revision: vehicle_revision.into(),
            available: ResourceQuantityInterval {
                lower: 1_000,
                nominal: 1_100,
                upper: 1_200,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "synthetic-supply-record:oxygen".into(),
            applicable_environments: environments,
        }],
        demands: vec![MissionResourceDemand {
            demand_id: "task-oxygen-demand".into(),
            task_id: "required-shadow-service".into(),
            pool_id: "vehicle-oxygen-pool".into(),
            resource_kind: MissionResourceKind::OxygenMassMicrograms,
            quantity: ResourceQuantityInterval {
                lower: 100,
                nominal: 200,
                upper: 400,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "synthetic-demand-model:oxygen".into(),
        }],
    };
    (configuration, tasks, task_receipt, plan)
}

#[test]
fn positive_fixture_replays_conservative_capacity_and_receipt() {
    let (configuration, tasks, task_receipt, plan) = fixture();
    let (report, receipt) =
        evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();

    assert_eq!(report.status, MissionResourceBudgetStatus::MeetsDeclaredResourcePolicy);
    assert_eq!(report.required_pool_count, 1);
    assert_eq!(report.required_pool_count_within_budget, 1);
    assert_eq!(report.required_pool_count_over_budget, 0);
    assert_eq!(report.pools.len(), 1);
    assert_eq!(report.pools[0].required_demand_upper, 400);
    assert_eq!(report.pools[0].reserve_quantity, 80);
    assert_eq!(report.pools[0].required_capacity, 480);
    assert_eq!(report.pools[0].required_margin, Some(520));
    assert!(report.findings.is_empty());
    assert!(!report.represents_physical_qualification);
    assert!(verify_mission_resource_budget_receipt(
        &configuration,
        &tasks,
        &task_receipt,
        &plan,
        &report,
        &receipt,
    ));
}

#[test]
fn replay_blocks_required_shortfall_and_detects_budget_mutation() {
    let (configuration, tasks, task_receipt, mut plan) = fixture();
    plan.pools[0].available.lower = 479;
    plan.pools[0].available.nominal = 500;
    plan.pools[0].available.upper = 600;
    let (report, receipt) =
        evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
    assert_eq!(report.status, MissionResourceBudgetStatus::Blocked);
    assert_eq!(report.pools[0].required_shortfall, 1);
    assert!(!report.findings.is_empty());

    let mut changed = plan.clone();
    changed.demands[0].quantity.upper += 1;
    assert!(!verify_mission_resource_budget_receipt(
        &configuration,
        &tasks,
        &task_receipt,
        &changed,
        &report,
        &receipt,
    ));
}
