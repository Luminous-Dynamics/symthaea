// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compute Commons failure-containment laboratory.
//!
//! The goal is not high-fidelity plant simulation. It is to prove that the
//! multiscale contracts fail in the right direction when common infrastructure
//! assumptions disappear: parent coordination can vanish, the grid can disappear,
//! a heat sink can be lost, and thermal grade cannot be upgraded for free.

use chrono::{Duration, TimeZone, Utc};
use std::collections::BTreeSet;
use std::error::Error;
use symthaea_grid_physics::battery::Battery;
use symthaea_operating_authority::{LeaseAdmission, OperatingAuthorityRegistry};
use symthaea_operating_autonomy::{
    CoordinationMode, LocalSafetyEnvelope, evaluate_effective_operation,
};
use symthaea_operating_envelope::{
    BoundaryMetric, ObservedResourceMetric, OperatingEnvelope, OperatingMode,
    OperatingPoint, ResourceConstraint,
};
use symthaea_resource_hierarchy::{NodeRole, NodeScale, ResourceHierarchy, ResourceNode};
use symthaea_resource_model::{
    PortDirection, ResourceAmount, ResourceBalance, ResourceEdge, ResourceEnvelope,
    ResourceError, ResourceKind, ResourcePort, ResourceUnit,
};
use symthaea_resource_quality::{
    NumericQualityConstraint, PortQualityContract, QualifiedGraphError, QualifiedPort,
    QualifiedResourceGraph, QualityMetric, ResourceQualityProfile, ResourceQualityRequirement,
};
use symthaea_thermofluids::heat_pump::HeatPump;

#[derive(Debug)]
struct FailureLabReport {
    grid_loss_shed_fraction: f64,
    battery_reserve_after_kwh: f64,
    thermal_reduction_without_recovery_kw: f64,
    thermal_reduction_with_heat_pump_kw: f64,
    heat_pump_delivered_kw: f64,
    heat_pump_electrical_kw: f64,
    parent_expiry_local_only: bool,
    high_temperature_direct_route_blocked: bool,
}

fn watts(kind: ResourceKind, value: f64) -> ResourceAmount {
    ResourceAmount::new(kind, ResourceUnit::Watt, value).unwrap()
}

fn port(id: &str, direction: PortDirection, capacity: ResourceAmount) -> ResourcePort {
    ResourcePort {
        id: id.into(),
        direction,
        capacity,
    }
}

fn qualified_port(
    id: &str,
    direction: PortDirection,
    capacity: ResourceAmount,
    quality: PortQualityContract,
) -> QualifiedPort {
    QualifiedPort::new(port(id, direction, capacity), quality)
}

fn heat_profile(temp_c: f64) -> ResourceQualityProfile {
    let key = watts(ResourceKind::ThermalEnergy, 0.0).key;
    let mut profile = ResourceQualityProfile::new(key);
    profile
        .set_numeric(QualityMetric::TemperatureCelsius, temp_c)
        .unwrap();
    profile
}

fn heat_requirement(minimum_c: f64, maximum_c: Option<f64>) -> ResourceQualityRequirement {
    let key = watts(ResourceKind::ThermalEnergy, 0.0).key;
    let mut requirement = ResourceQualityRequirement::new(key);
    requirement
        .require_numeric(NumericQualityConstraint {
            metric: QualityMetric::TemperatureCelsius,
            minimum: Some(minimum_c),
            maximum: maximum_c,
        })
        .unwrap();
    requirement
}

fn hierarchy() -> ResourceHierarchy {
    let mut hierarchy = ResourceHierarchy::default();
    hierarchy
        .insert_root(
            ResourceNode::new(
                "commons",
                "Community Compute Commons",
                NodeScale::Community,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::CommunityService),
        )
        .unwrap();
    hierarchy
        .insert_child(
            "commons",
            ResourceNode::new(
                "compute-campus",
                "Compute campus",
                NodeScale::Facility,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::Compute),
        )
        .unwrap();
    hierarchy
}

fn local_survival_envelope() -> LocalSafetyEnvelope {
    let mut allowed_modes = BTreeSet::new();
    allowed_modes.insert(OperatingMode::Normal);
    allowed_modes.insert(OperatingMode::Islanded);
    allowed_modes.insert(OperatingMode::Emergency);
    LocalSafetyEnvelope {
        id: "compute-campus-local-survival-v1".into(),
        subject_node_id: "compute-campus".into(),
        allowed_modes,
        resource_constraints: vec![ResourceConstraint {
            key: watts(ResourceKind::Electricity, 0.0).key,
            metric: BoundaryMetric::Import,
            minimum: None,
            maximum: Some(250_000.0),
        }],
        critical_service_floor: 0.80,
        max_shed_fraction: 0.20,
    }
}

fn parent_envelope(t0: chrono::DateTime<Utc>) -> OperatingEnvelope {
    let mut allowed_modes = BTreeSet::new();
    allowed_modes.insert(OperatingMode::Normal);
    allowed_modes.insert(OperatingMode::Islanded);
    OperatingEnvelope {
        id: "commons-compute-g1".into(),
        issuer_node_id: "commons".into(),
        subject_node_id: "compute-campus".into(),
        generation: 1,
        valid_from: t0,
        valid_until: t0 + Duration::hours(1),
        allowed_modes,
        resource_constraints: vec![ResourceConstraint {
            key: watts(ResourceKind::Electricity, 0.0).key,
            metric: BoundaryMetric::Import,
            minimum: None,
            maximum: Some(180_000.0),
        }],
        critical_service_floor: 0.85,
        max_shed_fraction: 0.15,
    }
}

fn point(
    mode: OperatingMode,
    import_w: f64,
    critical_service_fraction: f64,
    shed_fraction: f64,
) -> OperatingPoint {
    OperatingPoint {
        mode,
        critical_service_fraction,
        shed_fraction,
        resources: vec![ObservedResourceMetric {
            key: watts(ResourceKind::Electricity, 0.0).key,
            metric: BoundaryMetric::Import,
            value: import_w,
        }],
    }
}

fn run_failure_lab() -> Result<FailureLabReport, Box<dyn Error>> {
    let hierarchy = hierarchy();
    let local = local_survival_envelope();
    let t0 = Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap();

    // --- Parent coordination exists and is intentionally stricter than local. ---
    let mut authority = OperatingAuthorityRegistry::new();
    assert_eq!(
        authority.admit(
            &hierarchy,
            parent_envelope(t0),
            t0 + Duration::minutes(1),
        )?,
        LeaseAdmission::Accepted
    );

    let coordinated = evaluate_effective_operation(
        &local,
        &hierarchy,
        &authority,
        &point(OperatingMode::Normal, 170_000.0, 0.95, 0.05),
        t0 + Duration::minutes(10),
    )?;
    assert!(coordinated.compliant);
    assert!(matches!(
        coordinated.mode,
        CoordinationMode::ParentCoordinated { .. }
    ));

    let parent_blocked = evaluate_effective_operation(
        &local,
        &hierarchy,
        &authority,
        &point(OperatingMode::Normal, 190_000.0, 0.95, 0.05),
        t0 + Duration::minutes(10),
    )?;
    assert!(!parent_blocked.compliant);
    assert!(parent_blocked.local_violations.is_empty());
    assert!(!parent_blocked.parent_violations.is_empty());

    // --- Grid loss after parent connectivity/lease expiry. -----------------------
    let mut battery = Battery::new(500.0, 150.0, 0.90).with_soc(0.75);
    let delivered_battery_kwh = battery
        .discharge(50.0, 1.0)
        .expect("reference battery discharge must stay inside rating/energy limits");
    let solar_kw = 120.0;
    let demand_kw = 200.0;
    let supplied_kw = solar_kw + delivered_battery_kwh;
    let shed_fraction = ((demand_kw - supplied_kw) / demand_kw).max(0.0);
    assert!((shed_fraction - 0.15).abs() < 1e-9);

    let islanded = evaluate_effective_operation(
        &local,
        &hierarchy,
        &authority,
        &point(
            OperatingMode::Islanded,
            0.0,
            1.0 - shed_fraction,
            shed_fraction,
        ),
        t0 + Duration::hours(2),
    )?;
    assert!(islanded.compliant);
    assert_eq!(islanded.mode, CoordinationMode::LocalOnly);

    let unsafe_islanding = evaluate_effective_operation(
        &local,
        &hierarchy,
        &authority,
        &point(OperatingMode::Islanded, 0.0, 0.75, 0.25),
        t0 + Duration::hours(2),
    )?;
    assert!(!unsafe_islanding.compliant);
    assert!(!unsafe_islanding.local_violations.is_empty());
    assert!(unsafe_islanding.parent_violations.is_empty());

    // --- Loss of the greenhouse heat sink. ---------------------------------------
    let building_sent_w = 70_000.0;
    let building_loss_fraction = 0.05;
    let building_delivered_w = building_sent_w * (1.0 - building_loss_fraction);
    let building_loss_w = building_sent_w - building_delivered_w;

    let failed_thermal_balance = ResourceBalance {
        key: watts(ResourceKind::ThermalEnergy, 0.0).key,
        produced: 180_000.0,
        consumed: building_delivered_w,
        imported: 0.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: building_loss_w,
    };
    assert!(!failed_thermal_balance.conserved_within(1e-9)?);
    assert!((failed_thermal_balance.residual()? - 110_000.0).abs() < 1e-9);

    // Capacity remains an independent failure boundary even when thermal grade is
    // otherwise acceptable.
    let mut failed_route_graph = QualifiedResourceGraph::default();
    failed_route_graph.add_node(
        "compute",
        [qualified_port(
            "heat-out",
            PortDirection::Output,
            watts(ResourceKind::ThermalEnergy, 190_000.0),
            PortQualityContract::Provide(heat_profile(42.0)),
        )],
    )?;
    failed_route_graph.add_node(
        "building",
        [qualified_port(
            "heat-in",
            PortDirection::Input,
            watts(ResourceKind::ThermalEnergy, 80_000.0),
            PortQualityContract::Require(heat_requirement(35.0, None)),
        )],
    )?;
    let impossible_reroute = failed_route_graph.connect(ResourceEdge {
        id: "impossible-reroute".into(),
        from_node: "compute".into(),
        from_port: "heat-out".into(),
        to_node: "building".into(),
        to_port: "heat-in".into(),
        amount: watts(ResourceKind::ThermalEnergy, 180_000.0),
        loss_fraction: 0.0,
    });
    assert!(matches!(
        impossible_reroute,
        Err(QualifiedGraphError::Resource(
            ResourceError::DestinationContractViolation(_)
        ))
    ));

    // Grade is enforced by routing itself. Sufficient thermal power at 42 C cannot
    // directly satisfy a 60 C minimum consumer.
    let mut grade_graph = QualifiedResourceGraph::default();
    grade_graph.add_node(
        "compute",
        [qualified_port(
            "heat-out",
            PortDirection::Output,
            watts(ResourceKind::ThermalEnergy, 100_000.0),
            PortQualityContract::Provide(heat_profile(42.0)),
        )],
    )?;
    grade_graph.add_node(
        "high-temp-service",
        [qualified_port(
            "heat-in",
            PortDirection::Input,
            watts(ResourceKind::ThermalEnergy, 100_000.0),
            PortQualityContract::Require(heat_requirement(60.0, None)),
        )],
    )?;
    let direct_high_temp_route = grade_graph.connect(ResourceEdge {
        id: "direct-low-grade-route".into(),
        from_node: "compute".into(),
        from_port: "heat-out".into(),
        to_node: "high-temp-service".into(),
        to_port: "heat-in".into(),
        amount: watts(ResourceKind::ThermalEnergy, 60_000.0),
        loss_fraction: 0.0,
    });
    let high_temperature_direct_route_blocked = matches!(
        &direct_high_temp_route,
        Err(QualifiedGraphError::QualityMismatch { .. })
    );
    assert!(high_temperature_direct_route_blocked);
    assert!(grade_graph.graph().edges().is_empty());

    // --- Bounded thermal-grade recovery. -----------------------------------------
    // Instead of inventing 65 C heat, a heat pump consumes 20 kW electrical work
    // and 60 kW of the stranded 42 C source heat to deliver 80 kW at 65 C.
    let heat_pump = HeatPump::new(25_000.0, 4.0)?;
    let upgraded = heat_pump.heating_step(20_000.0, 42.0, 65.0)?;
    assert!(upgraded.conserved_within(1e-9));
    assert!((upgraded.source_heat_w - 60_000.0).abs() < 1e-9);
    assert!((upgraded.delivered_heat_w - 80_000.0).abs() < 1e-9);

    let mut recovery_graph = QualifiedResourceGraph::default();
    recovery_graph.add_node(
        "compute",
        [qualified_port(
            "heat-out",
            PortDirection::Output,
            watts(ResourceKind::ThermalEnergy, upgraded.source_heat_w),
            PortQualityContract::Provide(heat_profile(42.0)),
        )],
    )?;
    recovery_graph.add_node(
        "power-bus",
        [qualified_port(
            "power-out",
            PortDirection::Output,
            watts(ResourceKind::Electricity, 25_000.0),
            PortQualityContract::NotApplicable,
        )],
    )?;
    recovery_graph.add_node(
        "heat-pump",
        [
            qualified_port(
                "source-heat-in",
                PortDirection::Input,
                watts(ResourceKind::ThermalEnergy, upgraded.source_heat_w),
                PortQualityContract::Require(heat_requirement(30.0, Some(50.0))),
            ),
            qualified_port(
                "power-in",
                PortDirection::Input,
                watts(ResourceKind::Electricity, 25_000.0),
                PortQualityContract::NotApplicable,
            ),
            qualified_port(
                "upgraded-heat-out",
                PortDirection::Output,
                watts(ResourceKind::ThermalEnergy, upgraded.delivered_heat_w),
                PortQualityContract::Provide(heat_profile(65.0)),
            ),
        ],
    )?;
    recovery_graph.add_node(
        "high-temp-service",
        [qualified_port(
            "heat-in",
            PortDirection::Input,
            watts(ResourceKind::ThermalEnergy, upgraded.delivered_heat_w),
            PortQualityContract::Require(heat_requirement(60.0, None)),
        )],
    )?;
    recovery_graph.connect(ResourceEdge {
        id: "source-heat".into(),
        from_node: "compute".into(),
        from_port: "heat-out".into(),
        to_node: "heat-pump".into(),
        to_port: "source-heat-in".into(),
        amount: watts(ResourceKind::ThermalEnergy, upgraded.source_heat_w),
        loss_fraction: 0.0,
    })?;
    recovery_graph.connect(ResourceEdge {
        id: "heat-pump-power".into(),
        from_node: "power-bus".into(),
        from_port: "power-out".into(),
        to_node: "heat-pump".into(),
        to_port: "power-in".into(),
        amount: watts(ResourceKind::Electricity, upgraded.electrical_input_w),
        loss_fraction: 0.0,
    })?;
    recovery_graph.connect(ResourceEdge {
        id: "upgraded-heat".into(),
        from_node: "heat-pump".into(),
        from_port: "upgraded-heat-out".into(),
        to_node: "high-temp-service".into(),
        to_port: "heat-in".into(),
        amount: watts(ResourceKind::ThermalEnergy, upgraded.delivered_heat_w),
        loss_fraction: 0.0,
    })?;
    assert_eq!(recovery_graph.graph().edges().len(), 3);

    // Recovery consumes 60 kW of the 110 kW stranded low-grade heat, reducing the
    // required compute/thermal curtailment from 110 kW to 50 kW. At a degraded
    // 130 kW compute-heat operating point, low-grade conservation closes again.
    let after_recovery_balance = ResourceBalance {
        key: watts(ResourceKind::ThermalEnergy, 0.0).key,
        produced: 180_000.0,
        consumed: building_delivered_w + upgraded.source_heat_w,
        imported: 0.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: building_loss_w,
    };
    assert!((after_recovery_balance.residual()? - 50_000.0).abs() < 1e-9);

    let degraded_with_recovery = ResourceBalance {
        key: watts(ResourceKind::ThermalEnergy, 0.0).key,
        produced: 130_000.0,
        consumed: building_delivered_w + upgraded.source_heat_w,
        imported: 0.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: building_loss_w,
    };
    assert!(degraded_with_recovery.conserved_within(1e-9)?);

    Ok(FailureLabReport {
        grid_loss_shed_fraction: shed_fraction,
        battery_reserve_after_kwh: battery.stored_energy_kwh(),
        thermal_reduction_without_recovery_kw: failed_thermal_balance.residual()? / 1_000.0,
        thermal_reduction_with_heat_pump_kw: after_recovery_balance.residual()? / 1_000.0,
        heat_pump_delivered_kw: upgraded.delivered_heat_w / 1_000.0,
        heat_pump_electrical_kw: upgraded.electrical_input_w / 1_000.0,
        parent_expiry_local_only: matches!(islanded.mode, CoordinationMode::LocalOnly),
        high_temperature_direct_route_blocked,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let report = run_failure_lab()?;
    println!("Compute Commons failure-containment report: {report:#?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_commons_failure_lab_executes_all_invariants() {
        let report = run_failure_lab().unwrap();
        assert!((report.grid_loss_shed_fraction - 0.15).abs() < 1e-9);
        assert!(report.battery_reserve_after_kwh > 0.0);
        assert!((report.thermal_reduction_without_recovery_kw - 110.0).abs() < 1e-9);
        assert!((report.thermal_reduction_with_heat_pump_kw - 50.0).abs() < 1e-9);
        assert!((report.heat_pump_delivered_kw - 80.0).abs() < 1e-9);
        assert!((report.heat_pump_electrical_kw - 20.0).abs() < 1e-9);
        assert!(report.parent_expiry_local_only);
        assert!(report.high_temperature_direct_route_blocked);
    }
}
