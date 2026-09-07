// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reference Compute Commons habitat composition.
//!
//! This example intentionally stays at the engineering-contract layer. It proves
//! that mixed civic assets can share one resource hierarchy while preserving
//! conservation, port capacity, thermal grade, local operating authority, and
//! multi-objective trade-offs. Detailed CFD, transient grid studies, and building
//! simulation remain the responsibility of the existing solver/adaptor layers.

use chrono::{Duration, TimeZone, Utc};
use std::collections::BTreeSet;
use std::error::Error;
use symthaea_grid_physics::battery::Battery;
use symthaea_operating_authority::{LeaseAdmission, OperatingAuthorityRegistry};
use symthaea_operating_envelope::{
    BoundaryMetric, ObservedResourceMetric, OperatingEnvelope, OperatingMode,
    OperatingPoint, ResourceConstraint,
};
use symthaea_operations_research::{
    ObjectiveDirection, ParetoCandidate, ParetoObjective, rank_pareto,
};
use symthaea_resource_hierarchy::{NodeRole, NodeScale, ResourceHierarchy, ResourceNode};
use symthaea_resource_model::{
    PortDirection, ResourceAmount, ResourceBalance, ResourceEdge, ResourceEnvelope,
    ResourceGraph, ResourceKind, ResourcePort, ResourceUnit,
};
use symthaea_resource_quality::{
    NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
    ResourceQualityRequirement, ideal_heat_exergy_fraction,
};
use symthaea_thermofluids::thermal::convection_heat_rate;

fn watts(kind: ResourceKind, value: f64) -> ResourceAmount {
    ResourceAmount::new(kind, ResourceUnit::Watt, value).unwrap()
}

fn gpu_seconds(value: f64) -> ResourceAmount {
    ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, value).unwrap()
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

    for node in [
        ResourceNode::new(
            "compute-campus",
            "Liquid-cooled compute campus",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::Compute),
        ResourceNode::new(
            "energy-yard",
            "Solar and storage yard",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::EnergyGeneration),
        ResourceNode::new(
            "thermal-loop",
            "Recovered-heat loop",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::ThermalSource),
        ResourceNode::new(
            "greenhouse",
            "Food commons greenhouse",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::Greenhouse),
        ResourceNode::new(
            "community-building",
            "Low-temperature community building",
            NodeScale::Facility,
            ResourceEnvelope::default(),
        )
        .with_role(NodeRole::Building),
    ] {
        hierarchy.insert_child("commons", node).unwrap();
    }

    hierarchy
        .insert_child(
            "compute-campus",
            ResourceNode::new(
                "compute-pod-a",
                "Compute pod A",
                NodeScale::Assembly,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::Compute),
        )
        .unwrap();
    hierarchy
        .insert_child(
            "energy-yard",
            ResourceNode::new(
                "battery-a",
                "Battery A",
                NodeScale::Component,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::EnergyStorage),
        )
        .unwrap();
    hierarchy
        .insert_child(
            "energy-yard",
            ResourceNode::new(
                "solar-a",
                "Solar array A",
                NodeScale::Component,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::EnergyGeneration),
        )
        .unwrap();

    hierarchy
}

fn main() -> Result<(), Box<dyn Error>> {
    let hierarchy = hierarchy();
    assert_eq!(
        hierarchy.node("greenhouse").unwrap().role,
        NodeRole::Greenhouse
    );
    assert_eq!(
        hierarchy.node("battery-a").unwrap().role,
        NodeRole::EnergyStorage
    );

    // --- Static resource topology -------------------------------------------------
    let electricity = ResourceKind::Electricity;
    let heat = ResourceKind::ThermalEnergy;
    let mut graph = ResourceGraph::default();
    graph.add_node(
        "solar",
        [port(
            "power-out",
            PortDirection::Output,
            watts(electricity, 150_000.0),
        )],
    )?;
    graph.add_node(
        "grid",
        [port(
            "power-out",
            PortDirection::Output,
            watts(electricity, 120_000.0),
        )],
    )?;
    graph.add_node(
        "compute-campus",
        [
            port(
                "power-in",
                PortDirection::Input,
                watts(electricity, 250_000.0),
            ),
            port(
                "heat-out",
                PortDirection::Output,
                watts(heat, 190_000.0),
            ),
        ],
    )?;
    graph.add_node(
        "greenhouse",
        [port(
            "heat-in",
            PortDirection::Input,
            watts(heat, 120_000.0),
        )],
    )?;
    graph.add_node(
        "community-building",
        [port(
            "heat-in",
            PortDirection::Input,
            watts(heat, 80_000.0),
        )],
    )?;

    graph.connect(ResourceEdge {
        id: "solar-to-compute".into(),
        from_node: "solar".into(),
        from_port: "power-out".into(),
        to_node: "compute-campus".into(),
        to_port: "power-in".into(),
        amount: watts(electricity, 120_000.0),
        loss_fraction: 0.0,
    })?;
    graph.connect(ResourceEdge {
        id: "grid-to-compute".into(),
        from_node: "grid".into(),
        from_port: "power-out".into(),
        to_node: "compute-campus".into(),
        to_port: "power-in".into(),
        amount: watts(electricity, 80_000.0),
        loss_fraction: 0.0,
    })?;

    let greenhouse_heat = ResourceEdge {
        id: "compute-heat-to-greenhouse".into(),
        from_node: "compute-campus".into(),
        from_port: "heat-out".into(),
        to_node: "greenhouse".into(),
        to_port: "heat-in".into(),
        amount: watts(heat, 110_000.0),
        loss_fraction: 0.02,
    };
    let building_heat = ResourceEdge {
        id: "compute-heat-to-building".into(),
        from_node: "compute-campus".into(),
        from_port: "heat-out".into(),
        to_node: "community-building".into(),
        to_port: "heat-in".into(),
        amount: watts(heat, 70_000.0),
        loss_fraction: 0.05,
    };

    let greenhouse_delivered = greenhouse_heat.delivered()?.value;
    let building_delivered = building_heat.delivered()?.value;
    let thermal_losses = greenhouse_heat.lost()?.value + building_heat.lost()?.value;
    graph.connect(greenhouse_heat)?;
    graph.connect(building_heat)?;

    assert_eq!(
        graph.port_utilization("compute-campus", "power-in")?,
        200_000.0
    );
    assert_eq!(
        graph.port_utilization("compute-campus", "heat-out")?,
        180_000.0
    );

    // --- Conservation at the commons boundary -----------------------------------
    let electrical_balance = ResourceBalance {
        key: watts(electricity, 0.0).key,
        produced: 120_000.0,
        consumed: 200_000.0,
        imported: 80_000.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: 0.0,
    };
    assert!(electrical_balance.conserved_within(1e-9)?);

    let thermal_balance = ResourceBalance {
        key: watts(heat, 0.0).key,
        produced: 180_000.0,
        consumed: greenhouse_delivered + building_delivered,
        imported: 0.0,
        exported: 0.0,
        storage_delta: 0.0,
        modeled_losses: thermal_losses,
    };
    assert!(thermal_balance.conserved_within(1e-9)?);

    let compute_balance = ResourceBalance {
        key: gpu_seconds(0.0).key,
        produced: 360_000.0,
        consumed: 240_000.0,
        imported: 0.0,
        exported: 120_000.0,
        storage_delta: 0.0,
        modeled_losses: 0.0,
    };
    assert!(compute_balance.conserved_within(1e-9)?);

    // --- Thermodynamic feasibility and grade -------------------------------------
    // Two deliberately simple heat-exchanger checks. These are screening equations,
    // not CFD qualification.
    let greenhouse_hx_w = convection_heat_rate(250.0, 30.0, 15.0);
    let building_hx_w = convection_heat_rate(200.0, 30.0, 15.0);
    assert!(greenhouse_hx_w >= 110_000.0);
    assert!(building_hx_w >= 70_000.0);

    let thermal_key = watts(heat, 0.0).key;
    let mut recovered_heat = ResourceQualityProfile::new(thermal_key);
    recovered_heat.set_numeric(QualityMetric::TemperatureCelsius, 42.0)?;

    let mut greenhouse_requirement = ResourceQualityRequirement::new(thermal_key);
    greenhouse_requirement.require_numeric(NumericQualityConstraint {
        metric: QualityMetric::TemperatureCelsius,
        minimum: Some(30.0),
        maximum: Some(50.0),
    })?;
    assert!(greenhouse_requirement.evaluate(&recovered_heat)?.compatible);

    let mut low_temp_building_requirement = ResourceQualityRequirement::new(thermal_key);
    low_temp_building_requirement.require_numeric(NumericQualityConstraint {
        metric: QualityMetric::TemperatureCelsius,
        minimum: Some(35.0),
        maximum: None,
    })?;
    assert!(
        low_temp_building_requirement
            .evaluate(&recovered_heat)?
            .compatible
    );

    let mut high_temp_process = ResourceQualityRequirement::new(thermal_key);
    high_temp_process.require_numeric(NumericQualityConstraint {
        metric: QualityMetric::TemperatureCelsius,
        minimum: Some(60.0),
        maximum: None,
    })?;
    assert!(!high_temp_process.evaluate(&recovered_heat)?.compatible);

    let exergy_fraction = ideal_heat_exergy_fraction(42.0, 20.0)?;
    assert!(exergy_fraction > 0.0 && exergy_fraction < 1.0);

    // --- Existing battery physics: one-hour islanding reserve check --------------
    let mut battery = Battery::new(500.0, 150.0, 0.90).with_soc(0.75);
    let reserve_before_kwh = battery.stored_energy_kwh();
    let delivered_kwh = battery
        .discharge(50.0, 1.0)
        .expect("fixed reference discharge must remain inside battery limits");
    let reserve_after_kwh = battery.stored_energy_kwh();
    assert!((delivered_kwh - 50.0).abs() < 1e-9);
    assert!(reserve_after_kwh < reserve_before_kwh);

    // --- Immediate-parent runtime authority --------------------------------------
    let t0 = Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap();
    let now = t0 + Duration::minutes(10);
    let mut allowed_modes = BTreeSet::new();
    allowed_modes.insert(OperatingMode::Normal);
    allowed_modes.insert(OperatingMode::Islanded);
    let compute_envelope = OperatingEnvelope {
        id: "commons-compute-g1".into(),
        issuer_node_id: "commons".into(),
        subject_node_id: "compute-campus".into(),
        generation: 1,
        valid_from: t0,
        valid_until: t0 + Duration::hours(1),
        allowed_modes,
        resource_constraints: vec![ResourceConstraint {
            key: watts(electricity, 0.0).key,
            metric: BoundaryMetric::Import,
            minimum: None,
            maximum: Some(250_000.0),
        }],
        critical_service_floor: 0.80,
        max_shed_fraction: 0.20,
    };

    let mut authority = OperatingAuthorityRegistry::new();
    assert_eq!(
        authority.admit(&hierarchy, compute_envelope.clone(), now)?,
        LeaseAdmission::Accepted
    );
    assert!(authority.active(&hierarchy, "compute-campus", now).is_some());

    let operating_point = OperatingPoint {
        mode: OperatingMode::Normal,
        critical_service_fraction: 0.95,
        shed_fraction: 0.05,
        resources: vec![ObservedResourceMetric {
            key: watts(electricity, 0.0).key,
            metric: BoundaryMetric::Import,
            value: 200_000.0,
        }],
    };
    let evaluation = compute_envelope.evaluate(&hierarchy, &operating_point, now)?;
    assert!(evaluation.compliant);

    // --- Pareto alternatives, without collapsing them into one hidden score -------
    let useful_heat_kw = (greenhouse_delivered + building_delivered) / 1_000.0;
    let objectives = vec![
        ParetoObjective::new("grid_import_kw", ObjectiveDirection::Minimize),
        ParetoObjective::new("useful_heat_kw", ObjectiveDirection::Maximize),
        ParetoObjective::new("battery_reserve_kwh", ObjectiveDirection::Maximize),
    ];
    let candidates = vec![
        ParetoCandidate::new(
            "balanced",
            vec![80.0, useful_heat_kw, reserve_after_kwh],
        ),
        ParetoCandidate::new("renewable-first", vec![40.0, 160.0, 280.0]),
        ParetoCandidate::new("resilience-first", vec![100.0, 170.0, 360.0]),
        ParetoCandidate::new("dominated", vec![120.0, 150.0, 250.0]),
    ];
    let ranking = rank_pareto(&objectives, &candidates)?;
    assert!(ranking.rank_of("dominated").is_some_and(|rank| rank > 0));
    assert_eq!(ranking.first_front().len(), 3);

    let frontier: Vec<&str> = ranking
        .first_front()
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    println!("Compute Commons resource graph edges: {}", graph.edges().len());
    println!("Useful recovered heat: {useful_heat_kw:.1} kW");
    println!("Recovered-heat ideal exergy fraction: {exergy_fraction:.3}");
    println!("Battery reserve after islanding check: {reserve_after_kwh:.1} kWh");
    println!("Pareto operating frontier: {frontier:?}");

    Ok(())
}
