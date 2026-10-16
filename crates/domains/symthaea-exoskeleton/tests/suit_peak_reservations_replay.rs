// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay and digest oracle for SX-034 peak reservations.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};
use symthaea_exoskeleton::suit_interop::{
    SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
    SuitOperatingEnvironment,
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
use symthaea_exoskeleton::suit_peak_reservations::{
    evaluate_peak_resource_reservations, verify_peak_resource_reservation_receipt,
    MissionTimeWindow, PeakQuantityInterval, PeakResourceFindingCode, PeakResourceFindingSeverity,
    PeakResourceKind, PeakResourcePool, PeakResourceReservation, PeakResourceReservationPlan,
    PeakResourceReservationReceipt, PeakResourceReservationReport,
    PeakResourceReservationStatus, PeakWindowStatus,
};

fn fixture() -> (
    MissionConfiguration,
    Vec<MissionTaskRequirement>,
    MissionTaskReceipt,
    PeakResourceReservationPlan,
) {
    let variant_id = "replay-peak-suit";
    let suit_revision = "suit-r7";
    let vehicle_revision = "vehicle-r7";
    let environments =
        BTreeSet::from([SuitOperatingEnvironment::LunarShadowOrPolarRegion]);
    let manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
        SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: revision.into(),
            endpoints: vec![SuitInterfaceEndpoint {
                interface_id: "peak-power-distribution.v1".into(),
                endpoint_id: format!("{component_id}.power"),
                kind: SuitInterfaceKind::ElectricalPower,
                role,
                contract_revision: "1.0.0".into(),
                schema_digest_sha256: format!("sha256:{}", "f".repeat(64)),
                connector_profile_id: "synthetic-peak-power-bus-v1".into(),
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
        mission_id: "synthetic-peak-replay".into(),
        configuration_revision: "cfg-replay-r1".into(),
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
        required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
        components: vec![
            MissionComponent {
                component_id: "suit".into(),
                kind: MissionComponentKind::Suit,
                hardware_revision: suit_revision.into(),
                interfaces: manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
            },
            MissionComponent {
                component_id: "vehicle".into(),
                kind: MissionComponentKind::LanderOrVehicle,
                hardware_revision: vehicle_revision.into(),
                interfaces: manifest("vehicle", vehicle_revision, SuitInterfaceRole::Provider),
            },
        ],
        connections: vec![MissionConnection {
            connection_id: "suit-vehicle-power".into(),
            left_component_id: "suit".into(),
            right_component_id: "vehicle".into(),
            policy: symthaea_exoskeleton::suit_interop::SuitInteropPolicy {
                minimum_evidence_level: SuitEvidenceLevel::Simulated,
                required_environments: environments.clone(),
            },
        }],
    };
    let tasks = vec![
        MissionTaskRequirement {
            task_id: "required-life-support".into(),
            required: true,
            required_capabilities: BTreeSet::from([SuitCapability::ThermalManagement]),
            required_component_ids: BTreeSet::from(["vehicle".into()]),
            required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
            required_environments: environments.clone(),
        },
        MissionTaskRequirement {
            task_id: "optional-science".into(),
            required: false,
            required_capabilities: BTreeSet::from([
                SuitCapability::ElectrostaticChargeManagement,
            ]),
            required_component_ids: BTreeSet::from(["vehicle".into()]),
            required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
            required_environments: environments.clone(),
        },
    ];
    let (task_report, task_receipt) =
        evaluate_mission_tasks(&configuration, &tasks).unwrap();
    assert_eq!(task_report.status, MissionTaskPlanStatus::MeetsDeclaredTaskPolicy);

    let plan = PeakResourceReservationPlan {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
        expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
        minimum_evidence_level: SuitEvidenceLevel::BenchTested,
        reserve_basis_points: 1_000,
        pools: vec![PeakResourcePool {
            pool_id: "vehicle-power-pool".into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            provider_component_id: "vehicle".into(),
            provider_hardware_revision: vehicle_revision.into(),
            required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
            required_interface_kinds: BTreeSet::from([SuitInterfaceKind::ElectricalPower]),
            capacity: PeakQuantityInterval {
                lower: 1_000,
                nominal: 1_100,
                upper: 1_200,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "synthetic-capacity-record:power".into(),
            applicable_environments: environments,
        }],
        reservations: vec![
            reservation("required-load", "required-life-support", 0, 100, 500),
            reservation("optional-load", "optional-science", 50, 150, 600),
        ],
    };
    (configuration, tasks, task_receipt, plan)
}

fn reservation(
    id: &str,
    task_id: &str,
    start_ms: u64,
    end_ms: u64,
    upper: u64,
) -> PeakResourceReservation {
    PeakResourceReservation {
        reservation_id: id.into(),
        task_id: task_id.into(),
        pool_id: "vehicle-power-pool".into(),
        resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
        window: MissionTimeWindow { start_ms, end_ms },
        demand: PeakQuantityInterval {
            lower: upper / 2,
            nominal: (upper * 3) / 4,
            upper,
        },
        evidence_level: SuitEvidenceLevel::BenchTested,
        evidence_reference: format!("synthetic-reservation:{id}"),
    }
}

#[test]
fn independent_oracle_verifies_plan_report_and_receipt_digests() {
    let (configuration, tasks, task_receipt, mut plan) = fixture();
    plan.reservations.retain(|item| item.task_id == "required-life-support");
    let (report, receipt) =
        evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
            .unwrap();
    assert_eq!(
        report.status,
        PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy
    );
    assert_eq!(report.windows.len(), 1);
    assert_eq!(report.windows[0].required_demand_upper, 500);
    assert_eq!(report.windows[0].total_shortfall, 0);
    assert_eq!(
        receipt.reservation_plan_digest_sha256,
        independent_plan_digest(&plan)
    );
    assert_eq!(
        receipt.report_digest_sha256,
        independent_report_digest(&report)
    );
    assert_eq!(
        receipt.receipt_digest_sha256,
        independent_receipt_digest(&receipt)
    );
    assert!(verify_peak_resource_reservation_receipt(
        &configuration,
        &tasks,
        &task_receipt,
        &plan,
        &report,
        &receipt
    ));
}

#[test]
fn independent_replay_distinguishes_optional_and_required_overbooking() {
    let (configuration, tasks, task_receipt, plan) = fixture();
    let (report, receipt) =
        evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
            .unwrap();
    assert_eq!(report.status, PeakResourceReservationStatus::Degraded);
    assert_eq!(
        report.windows.iter().find(|window| {
            window.window.start_ms == 50 && window.window.end_ms == 100
        }).unwrap().status,
        PeakWindowStatus::OptionalOversubscribed
    );
    assert_eq!(
        receipt.report_digest_sha256,
        independent_report_digest(&report)
    );

    let mut required_overload = plan.clone();
    required_overload.reservations = vec![
        reservation("required-a", "required-life-support", 0, 100, 500),
        reservation("required-b", "required-life-support", 50, 80, 500),
    ];
    let (blocked, blocked_receipt) = evaluate_peak_resource_reservations(
        &configuration, &tasks, &task_receipt, &required_overload
    ).unwrap();
    assert_eq!(blocked.status, PeakResourceReservationStatus::Blocked);
    assert!(blocked.findings.iter().any(|finding| {
        finding.code == PeakResourceFindingCode::RequiredOversubscription
            && finding.severity == PeakResourceFindingSeverity::Blocker
    }));
    assert_eq!(
        blocked_receipt.report_digest_sha256,
        independent_report_digest(&blocked)
    );
    assert_eq!(
        blocked_receipt.receipt_digest_sha256,
        independent_receipt_digest(&blocked_receipt)
    );
}

struct IndependentHashWriter(Sha256);

impl IndependentHashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }
    fn raw(&mut self, bytes: &[u8]) { self.0.update(bytes); }
    fn u64(&mut self, value: u64) { self.raw(&value.to_le_bytes()); }
    fn usize(&mut self, value: usize) { self.u64(value as u64); }
    fn boolean(&mut self, value: bool) { self.raw(&[u8::from(value)]); }
    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }
    fn option_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => { self.boolean(true); self.string(value); }
            None => self.boolean(false),
        }
    }
    fn finish(self) -> String {
        let bytes = self.0.finalize();
        format!("sha256:{}", bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
    }
}

fn independent_plan_digest(plan: &PeakResourceReservationPlan) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx034.reservation-plan.v1");
    h.u64(plan.schema_version as u64);
    h.string(&plan.mission_id);
    h.string(&plan.configuration_revision);
    h.string(&plan.expected_configuration_digest_sha256);
    h.string(&plan.expected_task_plan_digest_sha256);
    h.string(evidence_tag(plan.minimum_evidence_level));
    h.u64(u64::from(plan.reserve_basis_points));

    let mut pools: Vec<_> = plan.pools.iter().collect();
    pools.sort_by(|a, b| a.pool_id.cmp(&b.pool_id));
    h.usize(pools.len());
    for pool in pools {
        h.string(&pool.pool_id);
        h.string(resource_tag(pool.resource_kind));
        h.string(&pool.provider_component_id);
        h.string(&pool.provider_hardware_revision);
        h.usize(pool.required_connection_ids.len());
        for id in &pool.required_connection_ids { h.string(id); }
        h.usize(pool.required_interface_kinds.len());
        for kind in &pool.required_interface_kinds {
            h.string(interface_kind_tag(*kind));
        }
        interval(&mut h, pool.capacity);
        h.string(evidence_tag(pool.evidence_level));
        h.string(&pool.evidence_reference);
        h.usize(pool.applicable_environments.len());
        for environment in &pool.applicable_environments {
            h.string(environment_tag(*environment));
        }
    }
    let mut reservations: Vec<_> = plan.reservations.iter().collect();
    reservations.sort_by(|a, b| a.reservation_id.cmp(&b.reservation_id));
    h.usize(reservations.len());
    for reservation in reservations {
        h.string(&reservation.reservation_id);
        h.string(&reservation.task_id);
        h.string(&reservation.pool_id);
        h.string(resource_tag(reservation.resource_kind));
        h.u64(reservation.window.start_ms);
        h.u64(reservation.window.end_ms);
        interval(&mut h, reservation.demand);
        h.string(evidence_tag(reservation.evidence_level));
        h.string(&reservation.evidence_reference);
    }
    h.finish()
}

fn independent_report_digest(report: &PeakResourceReservationReport) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx034.reservation-report.v1");
    h.u64(report.schema_version as u64);
    h.string(&report.mission_id);
    h.string(&report.configuration_revision);
    h.string(&report.configuration_digest_sha256);
    h.string(&report.task_plan_digest_sha256);
    h.string(&report.reservation_plan_digest_sha256);
    h.string(task_status_tag(report.task_plan_status));
    h.string(reservation_status_tag(report.status));
    h.boolean(report.represents_physical_qualification);
    h.usize(report.pool_count);
    h.usize(report.reservation_count);
    h.usize(report.evaluated_window_count);
    h.usize(report.required_oversubscribed_window_count);
    h.usize(report.optional_oversubscribed_window_count);

    h.usize(report.pools.len());
    for pool in &report.pools {
        h.string(&pool.pool_id);
        h.string(resource_tag(pool.resource_kind));
        h.string(&pool.provider_component_id);
        h.string(&pool.provider_hardware_revision);
        interval(&mut h, pool.capacity);
        h.u64(pool.reserve_quantity);
        h.u64(pool.usable_capacity);
        h.u64(pool.peak_required_demand_upper);
        h.u64(pool.peak_optional_demand_upper);
        h.u64(pool.peak_total_demand_upper);
        h.u64(pool.minimum_required_headroom);
        h.u64(pool.maximum_required_shortfall);
        h.u64(pool.maximum_total_shortfall);
        h.usize(pool.required_oversubscribed_window_count);
        h.usize(pool.optional_oversubscribed_window_count);
    }
    h.usize(report.windows.len());
    for window in &report.windows {
        h.string(&window.pool_id);
        h.string(resource_tag(window.resource_kind));
        h.u64(window.window.start_ms);
        h.u64(window.window.end_ms);
        h.usize(window.active_reservation_ids.len());
        for id in &window.active_reservation_ids { h.string(id); }
        h.u64(window.required_demand_upper);
        h.u64(window.optional_demand_upper);
        h.u64(window.reserve_quantity);
        h.u64(window.usable_capacity);
        h.u64(window.required_shortfall);
        h.u64(window.total_shortfall);
        h.string(window_status_tag(window.status));
    }
    h.usize(report.findings.len());
    for finding in &report.findings {
        h.string(&finding.pool_id);
        h.option_string(finding.task_id.as_deref());
        h.option_string(finding.reservation_id.as_deref());
        match finding.window {
            Some(window) => {
                h.boolean(true);
                h.u64(window.start_ms);
                h.u64(window.end_ms);
            }
            None => h.boolean(false),
        }
        h.string(finding_code_tag(finding.code));
        h.string(match finding.severity {
            PeakResourceFindingSeverity::Degradation => "degradation",
            PeakResourceFindingSeverity::Blocker => "blocker",
        });
        match finding.observed_evidence_level {
            Some(level) => {
                h.boolean(true);
                h.string(evidence_tag(level));
            }
            None => h.boolean(false),
        }
    }
    h.finish()
}

fn independent_receipt_digest(receipt: &PeakResourceReservationReceipt) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx034.reservation-receipt.v1");
    h.string(&receipt.configuration_digest_sha256);
    h.string(&receipt.task_plan_digest_sha256);
    h.string(&receipt.task_receipt_digest_sha256);
    h.string(&receipt.reservation_plan_digest_sha256);
    h.string(&receipt.report_digest_sha256);
    h.finish()
}

fn interval(h: &mut IndependentHashWriter, value: PeakQuantityInterval) {
    h.u64(value.lower);
    h.u64(value.nominal);
    h.u64(value.upper);
}

fn resource_tag(value: PeakResourceKind) -> &'static str {
    match value {
        PeakResourceKind::ElectricalPowerMilliwatts => "electrical_power_milliwatts",
        PeakResourceKind::CoolantFlowMicrolitersPerSecond => "coolant_flow_microliters_per_second",
        PeakResourceKind::DataRateBytesPerSecond => "data_rate_bytes_per_second",
        PeakResourceKind::ComputeBudgetUnitsPerSecond => "compute_budget_units_per_second",
        PeakResourceKind::SharedEquipmentSlots => "shared_equipment_slots",
    }
}
fn interface_kind_tag(value: SuitInterfaceKind) -> &'static str {
    match value {
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
fn evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
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
fn task_status_tag(value: MissionTaskPlanStatus) -> &'static str {
    match value {
        MissionTaskPlanStatus::Blocked => "blocked",
        MissionTaskPlanStatus::Degraded => "degraded",
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy => "meets_declared_task_policy",
    }
}
fn reservation_status_tag(value: PeakResourceReservationStatus) -> &'static str {
    match value {
        PeakResourceReservationStatus::Blocked => "blocked",
        PeakResourceReservationStatus::Degraded => "degraded",
        PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy => {
            "meets_declared_peak_resource_policy"
        }
    }
}
fn window_status_tag(value: PeakWindowStatus) -> &'static str {
    match value {
        PeakWindowStatus::WithinDeclaredCapacity => "within_declared_capacity",
        PeakWindowStatus::RequiredOversubscribed => "required_oversubscribed",
        PeakWindowStatus::OptionalOversubscribed => "optional_oversubscribed",
    }
}
fn finding_code_tag(value: PeakResourceFindingCode) -> &'static str {
    match value {
        PeakResourceFindingCode::MissingProviderComponent => "missing_provider_component",
        PeakResourceFindingCode::ProviderHardwareRevisionMismatch => {
            "provider_hardware_revision_mismatch"
        },
        PeakResourceFindingCode::ProviderNotReachable => "provider_not_reachable",
        PeakResourceFindingCode::MissingProviderConnection => "missing_provider_connection",
        PeakResourceFindingCode::ProviderConnectionDoesNotBindProvider => {
            "provider_connection_does_not_bind_provider"
        },
        PeakResourceFindingCode::ProviderConnectionResourceKindMismatch => {
            "provider_connection_resource_kind_mismatch"
        },
        PeakResourceFindingCode::ProviderConnectionBlocked => "provider_connection_blocked",
        PeakResourceFindingCode::ProviderConnectionEnvironmentMismatch => {
            "provider_connection_environment_mismatch"
        },
        PeakResourceFindingCode::EnvironmentScopeMismatch => "environment_scope_mismatch",
        PeakResourceFindingCode::SupplyEvidenceBelowPolicy => "supply_evidence_below_policy",
        PeakResourceFindingCode::ReservationEvidenceBelowPolicy => {
            "reservation_evidence_below_policy"
        },
        PeakResourceFindingCode::RequiredOversubscription => "required_oversubscription",
        PeakResourceFindingCode::OptionalOversubscription => "optional_oversubscription",
    }
}
