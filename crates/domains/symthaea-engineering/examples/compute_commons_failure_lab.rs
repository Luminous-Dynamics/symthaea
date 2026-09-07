// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compute Commons failure-containment laboratory.
//!
//! The goal is not high-fidelity plant simulation. It is to prove that the
//! multiscale contracts fail in the right direction when common infrastructure
//! assumptions disappear: parent coordination can vanish, the grid can disappear,
//! a heat sink can be lost, and an invalid thermal reroute can be attempted.

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
    ResourceError, ResourceGraph, ResourceKind, ResourcePort, ResourceUnit,
};
use symthaea_resource_quality::{
    NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
    ResourceQualityRequirement,
};

#[derive(Debug)]
struct FailureLabReport {
    grid_loss_shed_fraction: f64,
    battery_reserve_after_kwh: f64,
    thermal_reduction_kw: f64,
    parent_expiry_local_only: bool,
    high_temperature_route_blocked: bool,
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

    // Parent is stricter: a point still inside the local 250 kW ceiling but above
    // the parent's 180 kW ceiling must be rejected while that lease is active.
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
    // One-hour average: 120 kW solar + 50 kW battery against a 200 kW demand.
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

    // Local autonomy is bounded rather than permissive: excessive shedding remains
    // illegal even though no parent lease is active.
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
    // In the normal habitat, compute exports 180 kW of heat. If only the 70 kW
    // building branch remains, the unconsumed thermal residual must stay visible.
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

    // A degraded workload that emits only what the surviving branch can accept is
    // physically account-balanced again.
    let degraded_thermal_balance = ResourceBalance {
        key: watts(ResourceKind::ThermalEnergy, 0.0).key,
        produced: building_sent_w,
        consumed: building_delivered_w,
        imported: 0.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: building_loss_w,
    };
    assert!(degraded_thermal_balance.conserved_within(1e-9)?);

    // The topology layer independently refuses the tempting but impossible response
    // of dumping all 180 kW into an 80 kW building heat interface.
    let mut failed_route_graph = ResourceGraph::default();
    failed_route_graph.add_node(
        "compute",
        [port(
            "heat-out",
            PortDirection::Output,
            watts(ResourceKind::ThermalEnergy, 190_000.0),
        )],
    )?;
    failed_route_graph.add_node(
        "building",
        [port(
            "heat-in",
            PortDirection::Input,
            watts(ResourceKind::ThermalEnergy, 80_000.0),
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
        Err(ResourceError::DestinationContractViolation(_))
    ));

    // Grade is another independent failure boundary: enough thermal power is still
    // not a valid route when the remaining consumer needs higher-temperature heat.
    let thermal_key = watts(ResourceKind::ThermalEnergy, 0.0).key;
    let mut low_grade_heat = ResourceQualityProfile::new(thermal_key);
    low_grade_heat.set_numeric(QualityMetric::TemperatureCelsius, 42.0)?;
    let mut high_temperature_requirement = ResourceQualityRequirement::new(thermal_key);
    high_temperature_requirement.require_numeric(NumericQualityConstraint {
        metric: QualityMetric::TemperatureCelsius,
        minimum: Some(60.0),
        maximum: None,
    })?;
    let quality = high_temperature_requirement.evaluate(&low_grade_heat)?;
    assert!(!quality.compatible);

    Ok(FailureLabReport {
        grid_loss_shed_fraction: shed_fraction,
        battery_reserve_after_kwh: battery.stored_energy_kwh(),
        thermal_reduction_kw: failed_thermal_balance.residual()? / 1_000.0,
        parent_expiry_local_only: matches!(islanded.mode, CoordinationMode::LocalOnly),
        high_temperature_route_blocked: !quality.compatible,
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
        assert!((report.thermal_reduction_kw - 110.0).abs() < 1e-9);
        assert!(report.parent_expiry_local_only);
        assert!(report.high_temperature_route_blocked);
    }
}
