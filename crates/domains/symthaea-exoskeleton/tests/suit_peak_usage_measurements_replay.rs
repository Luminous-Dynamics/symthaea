// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay oracle for the SX-035 measurement ledger.

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
    evaluate_mission_tasks, MissionTaskReceipt, MissionTaskRequirement,
};
use symthaea_exoskeleton::suit_peak_reservations::{
    evaluate_peak_resource_reservations, MissionTimeWindow, PeakQuantityInterval,
    PeakResourceKind, PeakResourcePool, PeakResourceReservation, PeakResourceReservationPlan,
    PeakResourceReservationReceipt, PeakResourceReservationReport, PeakResourceReservationStatus,
};
use symthaea_exoskeleton::suit_peak_usage_measurements::{
    evaluate_peak_resource_usage, verify_peak_resource_usage_receipt,
    PeakResourceUsageMeasurementPlan, PeakResourceUsageMeasurementReceipt,
    PeakResourceUsageMeasurementReport, PeakResourceUsageMeasurementStatus,
    PeakResourceUsageObservation, PeakUsageFinding, PeakUsageFindingCode,
    PeakUsageFindingSeverity, PeakUsageObservationStatus, ResourceMeasurementMethod,
};
use symthaea_exoskeleton::suit_profile::{
    SuitCapability, SuitCapabilityClaim, SuitEvidenceLevel, SuitMissionProfile,
    SuitVariantDeclaration,
};

fn fixture() -> (
    MissionConfiguration,
    Vec<MissionTaskRequirement>,
    MissionTaskReceipt,
    PeakResourceReservationPlan,
    PeakResourceReservationReport,
    PeakResourceReservationReceipt,
    PeakResourceUsageMeasurementPlan,
) {
    let variant_id = "replay-measurement-suit";
    let suit_revision = "suit-rm1";
    let vehicle_revision = "vehicle-rm1";
    let environments =
        BTreeSet::from([SuitOperatingEnvironment::LunarShadowOrPolarRegion]);
    let manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
        SuitInterfaceManifest {
            component_id: component_id.into(),
            hardware_revision: revision.into(),
            endpoints: vec![SuitInterfaceEndpoint {
                interface_id: "power.v1".into(),
                endpoint_id: format!("{component_id}.power"),
                kind: SuitInterfaceKind::ElectricalPower,
                role,
                contract_revision: "1.0.0".into(),
                schema_digest_sha256: format!("sha256:{}", "8".repeat(64)),
                connector_profile_id: "synthetic-power-v1".into(),
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
            evidence_reference: format!("synthetic-evidence:{capability:?}"),
        })
        .collect();
    let configuration = MissionConfiguration {
        schema_version: 1,
        mission_id: "synthetic-usage-replay".into(),
        configuration_revision: "cfg-usage-r1".into(),
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
    let tasks = vec![MissionTaskRequirement {
        task_id: "required-life-support".into(),
        required: true,
        required_capabilities: BTreeSet::from([SuitCapability::ThermalManagement]),
        required_component_ids: BTreeSet::from(["vehicle".into()]),
        required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
        required_environments: environments.clone(),
    }];
    let (task_report, task_receipt) =
        evaluate_mission_tasks(&configuration, &tasks).unwrap();
    let reservation_plan = PeakResourceReservationPlan {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
        expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
        minimum_evidence_level: SuitEvidenceLevel::BenchTested,
        reserve_basis_points: 1_000,
        pools: vec![PeakResourcePool {
            pool_id: "vehicle-power".into(),
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
            evidence_reference: "synthetic-capacity:power".into(),
            applicable_environments: environments,
        }],
        reservations: vec![PeakResourceReservation {
            reservation_id: "required-power".into(),
            task_id: "required-life-support".into(),
            pool_id: "vehicle-power".into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            window: MissionTimeWindow {
                start_ms: 0,
                end_ms: 100,
            },
            demand: PeakQuantityInterval {
                lower: 300,
                nominal: 400,
                upper: 500,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "synthetic-forecast:power".into(),
        }],
    };
    let (reservation_report, reservation_receipt) = evaluate_peak_resource_reservations(
        &configuration,
        &tasks,
        &task_receipt,
        &reservation_plan,
    )
    .unwrap();
    let measurement_plan = measurement_plan(
        &configuration,
        &reservation_report,
        &reservation_receipt,
        vec![observation(
            "usage-1",
            PeakQuantityInterval {
                lower: 400,
                nominal: 420,
                upper: 450,
            },
        )],
    );
    (
        configuration,
        tasks,
        task_receipt,
        reservation_plan,
        reservation_report,
        reservation_receipt,
        measurement_plan,
    )
}

fn observation(
    id: &str,
    observed_peak: PeakQuantityInterval,
) -> PeakResourceUsageObservation {
    PeakResourceUsageObservation {
        observation_id: id.into(),
        pool_id: "vehicle-power".into(),
        resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
        provider_component_id: "vehicle".into(),
        provider_hardware_revision: "vehicle-rm1".into(),
        window: MissionTimeWindow {
            start_ms: 25,
            end_ms: 75,
        },
        observed_peak,
        sample_count: 100,
        sampling_interval_ms: 10,
        measurement_method: ResourceMeasurementMethod::BenchInstrumented,
        evidence_level: SuitEvidenceLevel::BenchTested,
        evidence_reference: format!("synthetic-observation:{id}"),
    }
}

fn measurement_plan(
    configuration: &MissionConfiguration,
    report: &PeakResourceReservationReport,
    receipt: &PeakResourceReservationReceipt,
    observations: Vec<PeakResourceUsageObservation>,
) -> PeakResourceUsageMeasurementPlan {
    PeakResourceUsageMeasurementPlan {
        schema_version: 1,
        mission_id: configuration.mission_id.clone(),
        configuration_revision: configuration.configuration_revision.clone(),
        expected_configuration_digest_sha256: report.configuration_digest_sha256.clone(),
        expected_task_plan_digest_sha256: report.task_plan_digest_sha256.clone(),
        expected_reservation_plan_digest_sha256: report.reservation_plan_digest_sha256.clone(),
        expected_reservation_report_digest_sha256: receipt.report_digest_sha256.clone(),
        expected_reservation_receipt_digest_sha256: receipt.receipt_digest_sha256.clone(),
        minimum_evidence_level: SuitEvidenceLevel::BenchTested,
        minimum_sample_count: 10,
        maximum_sampling_interval_ms: 50,
        observations,
    }
}

#[test]
fn independent_replay_checks_plan_report_and_receipt_digests() {
    let (
        config,
        tasks,
        task_receipt,
        reservation_plan,
        reservation_report,
        reservation_receipt,
        measurement_plan,
    ) = fixture();
    let (report, receipt) = evaluate_peak_resource_usage(
        &config,
        &tasks,
        &task_receipt,
        &reservation_plan,
        &reservation_report,
        &reservation_receipt,
        &measurement_plan,
    )
    .unwrap();
    assert_eq!(
        report.status,
        PeakResourceUsageMeasurementStatus::MeetsDeclaredMeasurementPolicy
    );
    assert_eq!(report.observations[0].forecast_peak_upper, Some(500));
    assert_eq!(
        receipt.measurement_plan_digest_sha256,
        independent_plan_digest(&measurement_plan)
    );
    assert_eq!(receipt.report_digest_sha256, independent_report_digest(&report));
    assert_eq!(
        receipt.receipt_digest_sha256,
        independent_receipt_digest(&receipt)
    );
    assert!(verify_peak_resource_usage_receipt(
        &config,
        &tasks,
        &task_receipt,
        &reservation_plan,
        &reservation_report,
        &reservation_receipt,
        &measurement_plan,
        &report,
        &receipt,
    ));
}

#[test]
fn independent_replay_distinguishes_drift_and_budget_exceedance() {
    let (
        config,
        tasks,
        task_receipt,
        reservation_plan,
        reservation_report,
        reservation_receipt,
        mut measurement_plan,
    ) = fixture();
    measurement_plan.observations[0].observed_peak = PeakQuantityInterval {
        lower: 600,
        nominal: 650,
        upper: 700,
    };
    let (drift, drift_receipt) = evaluate_peak_resource_usage(
        &config,
        &tasks,
        &task_receipt,
        &reservation_plan,
        &reservation_report,
        &reservation_receipt,
        &measurement_plan,
    )
    .unwrap();
    assert_eq!(drift.status, PeakResourceUsageMeasurementStatus::Degraded);
    assert!(drift.findings.iter().any(|finding| {
        finding.code == PeakUsageFindingCode::ForecastDefinitelyUnderestimated
    }));
    assert_eq!(drift_receipt.report_digest_sha256, independent_report_digest(&drift));

    measurement_plan.observations[0].observed_peak = PeakQuantityInterval {
        lower: 950,
        nominal: 1_000,
        upper: 1_050,
    };
    let (blocked, blocked_receipt) = evaluate_peak_resource_usage(
        &config,
        &tasks,
        &task_receipt,
        &reservation_plan,
        &reservation_report,
        &reservation_receipt,
        &measurement_plan,
    )
    .unwrap();
    assert_eq!(blocked.status, PeakResourceUsageMeasurementStatus::Blocked);
    assert!(blocked.findings.iter().any(|finding| {
        finding.code == PeakUsageFindingCode::DefiniteBudgetLimitExceeded
            && finding.severity == PeakUsageFindingSeverity::Blocker
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

fn independent_plan_digest(plan: &PeakResourceUsageMeasurementPlan) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx035.measurement-plan.v1");
    h.u64(plan.schema_version as u64);
    h.string(&plan.mission_id);
    h.string(&plan.configuration_revision);
    h.string(&plan.expected_configuration_digest_sha256);
    h.string(&plan.expected_task_plan_digest_sha256);
    h.string(&plan.expected_reservation_plan_digest_sha256);
    h.string(&plan.expected_reservation_report_digest_sha256);
    h.string(&plan.expected_reservation_receipt_digest_sha256);
    h.string(evidence_tag(plan.minimum_evidence_level));
    h.u64(plan.minimum_sample_count);
    h.u64(plan.maximum_sampling_interval_ms);
    let mut observations: Vec<_> = plan.observations.iter().collect();
    observations.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    h.usize(observations.len());
    for observation in observations {
        h.string(&observation.observation_id);
        h.string(&observation.pool_id);
        h.string(resource_tag(observation.resource_kind));
        h.string(&observation.provider_component_id);
        h.string(&observation.provider_hardware_revision);
        h.u64(observation.window.start_ms);
        h.u64(observation.window.end_ms);
        interval(&mut h, observation.observed_peak);
        h.u64(observation.sample_count);
        h.u64(observation.sampling_interval_ms);
        h.string(method_tag(observation.measurement_method));
        h.string(evidence_tag(observation.evidence_level));
        h.string(&observation.evidence_reference);
    }
    h.finish()
}

fn independent_report_digest(report: &PeakResourceUsageMeasurementReport) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx035.measurement-report.v1");
    h.u64(report.schema_version as u64);
    h.string(&report.mission_id);
    h.string(&report.configuration_revision);
    h.string(&report.configuration_digest_sha256);
    h.string(&report.task_plan_digest_sha256);
    h.string(&report.reservation_plan_digest_sha256);
    h.string(&report.reservation_report_digest_sha256);
    h.string(&report.measurement_plan_digest_sha256);
    h.string(reservation_status_tag(report.baseline_reservation_status));
    h.string(measurement_status_tag(report.status));
    h.boolean(report.represents_physical_qualification);
    h.usize(report.observation_count);
    h.usize(report.forecasted_observation_count);
    h.usize(report.unmodeled_observation_count);
    h.usize(report.forecast_drift_count);
    h.usize(report.possible_budget_limit_exceedance_count);
    h.usize(report.definite_budget_limit_exceedance_count);

    h.usize(report.observations.len());
    for observation in &report.observations {
        h.string(&observation.observation_id);
        h.string(&observation.pool_id);
        h.string(resource_tag(observation.resource_kind));
        h.string(&observation.provider_component_id);
        h.string(&observation.provider_hardware_revision);
        h.u64(observation.window.start_ms);
        h.u64(observation.window.end_ms);
        interval(&mut h, observation.observed_peak);
        h.u64(observation.sample_count);
        h.u64(observation.sampling_interval_ms);
        h.string(method_tag(observation.measurement_method));
        h.string(evidence_tag(observation.evidence_level));
        h.string(&observation.evidence_reference);
        h.usize(observation.forecast_window_count);
        h.option_u64(observation.forecast_peak_upper);
        h.u64(observation.usable_capacity);
        h.string(observation_status_tag(observation.status));
        h.usize(observation.findings.len());
        for finding in &observation.findings {
            write_finding(&mut h, finding);
        }
    }
    h.usize(report.findings.len());
    for finding in &report.findings {
        write_finding(&mut h, finding);
    }
    h.finish()
}

fn write_finding(h: &mut IndependentHashWriter, finding: &PeakUsageFinding) {
    h.string(&finding.observation_id);
    h.string(&finding.pool_id);
    h.string(finding_code_tag(finding.code));
    h.string(match finding.severity {
        PeakUsageFindingSeverity::Degradation => "degradation",
        PeakUsageFindingSeverity::Blocker => "blocker",
    });
    h.option_u64(finding.forecast_peak_upper);
    h.u64(finding.usable_capacity);
    interval(&mut h, finding.observed_peak);
}

fn independent_receipt_digest(receipt: &PeakResourceUsageMeasurementReceipt) -> String {
    let mut h = IndependentHashWriter::new("symthaea.sx035.measurement-receipt.v1");
    h.string(&receipt.configuration_digest_sha256);
    h.string(&receipt.task_plan_digest_sha256);
    h.string(&receipt.task_receipt_digest_sha256);
    h.string(&receipt.reservation_plan_digest_sha256);
    h.string(&receipt.reservation_receipt_digest_sha256);
    h.string(&receipt.measurement_plan_digest_sha256);
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
fn evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}
fn method_tag(value: ResourceMeasurementMethod) -> &'static str {
    match value {
        ResourceMeasurementMethod::SimulatedTrace => "simulated_trace",
        ResourceMeasurementMethod::BenchInstrumented => "bench_instrumented",
        ResourceMeasurementMethod::HumanInLoopInstrumented => "human_in_loop_instrumented",
        ResourceMeasurementMethod::OperationalTelemetry => "operational_telemetry",
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
fn measurement_status_tag(value: PeakResourceUsageMeasurementStatus) -> &'static str {
    match value {
        PeakResourceUsageMeasurementStatus::Blocked => "blocked",
        PeakResourceUsageMeasurementStatus::Degraded => "degraded",
        PeakResourceUsageMeasurementStatus::MeetsDeclaredMeasurementPolicy => {
            "meets_declared_measurement_policy"
        }
    }
}
fn observation_status_tag(value: PeakUsageObservationStatus) -> &'static str {
    match value {
        PeakUsageObservationStatus::WithinDeclaredForecast => "within_declared_forecast",
        PeakUsageObservationStatus::ForecastMayBeUnderestimated => "forecast_may_be_underestimated",
        PeakUsageObservationStatus::ForecastDefinitelyUnderestimated => {
            "forecast_definitely_underestimated"
        }
        PeakUsageObservationStatus::PossibleBudgetLimitExceeded => "possible_budget_limit_exceeded",
        PeakUsageObservationStatus::DefiniteBudgetLimitExceeded => "definite_budget_limit_exceeded",
        PeakUsageObservationStatus::UnmodeledUsage => "unmodeled_usage",
        PeakUsageObservationStatus::EvidenceInsufficient => "evidence_insufficient",
    }
}
fn finding_code_tag(value: PeakUsageFindingCode) -> &'static str {
    match value {
        PeakUsageFindingCode::ProviderComponentMismatch => "provider_component_mismatch",
        PeakUsageFindingCode::ProviderHardwareRevisionMismatch => {
            "provider_hardware_revision_mismatch"
        }
        PeakUsageFindingCode::MeasurementEvidenceBelowPolicy => "measurement_evidence_below_policy",
        PeakUsageFindingCode::SampleCountBelowPolicy => "sample_count_below_policy",
        PeakUsageFindingCode::SamplingIntervalAbovePolicy => "sampling_interval_above_policy",
        PeakUsageFindingCode::NoDeclaredForecast => "no_declared_forecast",
        PeakUsageFindingCode::ForecastMayBeUnderestimated => "forecast_may_be_underestimated",
        PeakUsageFindingCode::ForecastDefinitelyUnderestimated => {
            "forecast_definitely_underestimated"
        }
        PeakUsageFindingCode::PossibleBudgetLimitExceeded => "possible_budget_limit_exceeded",
        PeakUsageFindingCode::DefiniteBudgetLimitExceeded => "definite_budget_limit_exceeded",
    }
}
