// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent scenario replay for SX-033 conservative resource budgeting.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};
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
    assert_eq!(
        receipt.task_receipt_digest_sha256,
        independent_task_receipt_digest(&task_receipt)
    );
    assert_eq!(
        report.resource_plan_digest_sha256,
        independent_resource_plan_digest(&plan)
    );
    assert_eq!(
        receipt.resource_plan_digest_sha256,
        independent_resource_plan_digest(&plan)
    );
    assert_eq!(
        receipt.report_digest_sha256,
        independent_resource_report_digest(&report)
    );
    assert_eq!(
        receipt.receipt_digest_sha256,
        independent_resource_receipt_digest(&receipt)
    );
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
    assert_eq!(
        receipt.report_digest_sha256,
        independent_resource_report_digest(&report)
    );
    assert_eq!(
        receipt.receipt_digest_sha256,
        independent_resource_receipt_digest(&receipt)
    );

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
                self.boolean(true);
                self.string(value);
            }
            None => self.boolean(false),
        }
    }

    fn option_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.boolean(true);
                self.u64(value);
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

fn independent_resource_plan_digest(plan: &MissionResourceBudgetPlan) -> String {
    let mut writer = IndependentHashWriter::new("symthaea.sx033.resource-plan.v1");
    writer.u64(plan.schema_version as u64);
    writer.string(&plan.mission_id);
    writer.string(&plan.configuration_revision);
    writer.string(&plan.expected_configuration_digest_sha256);
    writer.string(&plan.expected_task_plan_digest_sha256);
    writer.string(independent_evidence_tag(plan.minimum_evidence_level));
    writer.u64(u64::from(plan.reserve_basis_points));

    let mut pools: Vec<_> = plan.pools.iter().collect();
    pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
    writer.usize(pools.len());
    for pool in pools {
        writer.string(&pool.pool_id);
        writer.string(independent_resource_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        independent_interval(&mut writer, pool.available);
        writer.string(independent_evidence_tag(pool.evidence_level));
        writer.string(&pool.evidence_reference);
        writer.usize(pool.applicable_environments.len());
        for environment in &pool.applicable_environments {
            writer.string(independent_environment_tag(*environment));
        }
    }

    let mut demands: Vec<_> = plan.demands.iter().collect();
    demands.sort_by(|left, right| left.demand_id.cmp(&right.demand_id));
    writer.usize(demands.len());
    for demand in demands {
        writer.string(&demand.demand_id);
        writer.string(&demand.task_id);
        writer.string(&demand.pool_id);
        writer.string(independent_resource_tag(demand.resource_kind));
        independent_interval(&mut writer, demand.quantity);
        writer.string(independent_evidence_tag(demand.evidence_level));
        writer.string(&demand.evidence_reference);
    }
    writer.finish()
}

fn independent_resource_report_digest(
    report: &symthaea_exoskeleton::suit_resource_budget::MissionResourceBudgetReport,
) -> String {
    use symthaea_exoskeleton::suit_resource_budget::MissionResourceFindingSeverity;

    let mut writer = IndependentHashWriter::new("symthaea.sx033.resource-report.v1");
    writer.u64(report.schema_version as u64);
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.configuration_digest_sha256);
    writer.string(&report.task_plan_digest_sha256);
    writer.string(&report.resource_plan_digest_sha256);
    writer.string(independent_task_status_tag(report.mission_task_plan_status));
    writer.string(independent_budget_status_tag(report.status));
    writer.boolean(report.represents_physical_qualification);

    writer.usize(report.pools.len());
    for pool in &report.pools {
        writer.string(&pool.pool_id);
        writer.string(independent_resource_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        independent_interval(&mut writer, pool.available);
        writer.u64(pool.required_demand_upper);
        writer.u64(pool.optional_demand_upper);
        writer.u64(pool.reserve_quantity);
        writer.u64(pool.required_capacity);
        writer.option_u64(pool.required_margin);
        writer.u64(pool.required_shortfall);
        writer.u64(pool.total_planned_capacity);
        writer.option_u64(pool.remaining_after_all_demand);
        writer.u64(pool.optional_shortfall);
        writer.string(independent_pool_status_tag(pool.status));
    }

    writer.usize(report.findings.len());
    for finding in &report.findings {
        writer.string(&finding.pool_id);
        writer.option_string(finding.task_id.as_deref());
        writer.option_string(finding.demand_id.as_deref());
        writer.string(independent_finding_code_tag(finding.code));
        writer.string(match finding.severity {
            MissionResourceFindingSeverity::Degradation => "degradation",
            MissionResourceFindingSeverity::Blocker => "blocker",
        });
        match finding.observed_evidence_level {
            Some(level) => {
                writer.boolean(true);
                writer.string(independent_evidence_tag(level));
            }
            None => writer.boolean(false),
        }
    }

    writer.usize(report.required_pool_count);
    writer.usize(report.required_pool_count_within_budget);
    writer.usize(report.required_pool_count_over_budget);
    writer.usize(report.optional_pool_count_over_budget);
    writer.usize(report.required_demand_count);
    writer.usize(report.optional_demand_count);
    writer.finish()
}

fn independent_task_receipt_digest(
    receipt: &symthaea_exoskeleton::suit_mission_tasks::MissionTaskReceipt,
) -> String {
    let mut writer = IndependentHashWriter::new("symthaea.sx033.bound-task-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.mission_configuration_report_digest_sha256);
    writer.string(&receipt.task_report_digest_sha256);
    writer.string(&receipt.receipt_digest_sha256);
    writer.finish()
}

fn independent_resource_receipt_digest(
    receipt: &symthaea_exoskeleton::suit_resource_budget::MissionResourceBudgetReceipt,
) -> String {
    let mut writer = IndependentHashWriter::new("symthaea.sx033.resource-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.task_receipt_digest_sha256);
    writer.string(&receipt.resource_plan_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn independent_interval(
    writer: &mut IndependentHashWriter,
    value: ResourceQuantityInterval,
) {
    writer.u64(value.lower);
    writer.u64(value.nominal);
    writer.u64(value.upper);
}

fn independent_resource_tag(value: MissionResourceKind) -> &'static str {
    match value {
        MissionResourceKind::OxygenMassMicrograms => "oxygen_mass_micrograms",
        MissionResourceKind::WaterVolumeMicroliters => "water_volume_microliters",
        MissionResourceKind::ConsumableMassMicrograms => "consumable_mass_micrograms",
        MissionResourceKind::ElectricalEnergyMicrowattHours => {
            "electrical_energy_microwatt_hours"
        }
        MissionResourceKind::DataVolumeBytes => "data_volume_bytes",
        MissionResourceKind::CrewTimeMilliseconds => "crew_time_milliseconds",
    }
}

fn independent_evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn independent_environment_tag(value: SuitOperatingEnvironment) -> &'static str {
    match value {
        SuitOperatingEnvironment::MicrogravityVacuum => "microgravity_vacuum",
        SuitOperatingEnvironment::LunarSunlitSurface => "lunar_sunlit_surface",
        SuitOperatingEnvironment::LunarShadowOrPolarRegion => {
            "lunar_shadow_or_polar_region"
        }
        SuitOperatingEnvironment::MarsSurface => "mars_surface",
        SuitOperatingEnvironment::PressurizedCabin => "pressurized_cabin",
        SuitOperatingEnvironment::ThermalVacuumGroundTest => "thermal_vacuum_ground_test",
        SuitOperatingEnvironment::GroundLaboratory => "ground_laboratory",
    }
}

fn independent_task_status_tag(value: MissionTaskPlanStatus) -> &'static str {
    match value {
        MissionTaskPlanStatus::Blocked => "blocked",
        MissionTaskPlanStatus::Degraded => "degraded",
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy => "meets_declared_task_policy",
    }
}

fn independent_budget_status_tag(value: MissionResourceBudgetStatus) -> &'static str {
    match value {
        MissionResourceBudgetStatus::Blocked => "blocked",
        MissionResourceBudgetStatus::Degraded => "degraded",
        MissionResourceBudgetStatus::MeetsDeclaredResourcePolicy => {
            "meets_declared_resource_policy"
        }
    }
}

fn independent_pool_status_tag(
    value: symthaea_exoskeleton::suit_resource_budget::MissionResourcePoolStatus,
) -> &'static str {
    use symthaea_exoskeleton::suit_resource_budget::MissionResourcePoolStatus::*;
    match value {
        MeetsDeclaredBudget => "meets_declared_budget",
        RequiredDemandExceedsSupply => "required_demand_exceeds_supply",
        OptionalDemandExceedsRemainingCapacity => "optional_demand_exceeds_remaining_capacity",
    }
}

fn independent_finding_code_tag(
    value: symthaea_exoskeleton::suit_resource_budget::MissionResourceFindingCode,
) -> &'static str {
    use symthaea_exoskeleton::suit_resource_budget::MissionResourceFindingCode::*;
    match value {
        MissingProviderComponent => "missing_provider_component",
        ProviderHardwareRevisionMismatch => "provider_hardware_revision_mismatch",
        ProviderNotReachable => "provider_not_reachable",
        EnvironmentScopeMismatch => "environment_scope_mismatch",
        SupplyEvidenceBelowPolicy => "supply_evidence_below_policy",
        DemandEvidenceBelowPolicy => "demand_evidence_below_policy",
        RequiredDemandExceedsSupply => "required_demand_exceeds_supply",
        OptionalDemandExceedsRemainingCapacity => "optional_demand_exceeds_remaining_capacity",
    }
}
