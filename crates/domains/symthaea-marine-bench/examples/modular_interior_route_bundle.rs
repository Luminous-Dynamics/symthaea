// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded joint route-bundle example over synthetic explicit anchors.
//! The emitted report remains a 2D path proxy; it is not an installation design.

use std::error::Error;

use symthaea_marine_bench::modular_interior::{Configuration, PositionMm};
use symthaea_marine_bench::modular_interior_route_bundle::{
    optimize_route_bundle, verify_route_bundle_report, RouteBundleLimits,
};
use symthaea_marine_bench::modular_interior_service_routing::{
    AnchorEvidenceClass, AnchorKind, AnchorSide, RoutingAnchor, RoutingLimits,
    ServiceRouteRequest,
};

fn module_port(
    instance_id: &str,
    port_id: &str,
    x_mm: i64,
    y_mm: i64,
    deck_id: &str,
) -> RoutingAnchor {
    RoutingAnchor {
        kind: AnchorKind::ModulePort,
        instance_id: Some(instance_id.to_owned()),
        port_id: Some(port_id.to_owned()),
        service_id: None,
        side: Some(AnchorSide::East),
        deck_id: deck_id.to_owned(),
        position_mm: PositionMm { x_mm, y_mm, z_mm: 0 },
        evidence_class: AnchorEvidenceClass::DesignAssumption,
        evidence_refs: vec!["SYNTHETIC-ROUTE-BUNDLE-DEMO".to_owned()],
    }
}

fn service(
    service_id: &str,
    x_mm: i64,
    y_mm: i64,
    deck_id: &str,
) -> RoutingAnchor {
    RoutingAnchor {
        kind: AnchorKind::PlatformService,
        instance_id: None,
        port_id: None,
        service_id: Some(service_id.to_owned()),
        side: None,
        deck_id: deck_id.to_owned(),
        position_mm: PositionMm { x_mm, y_mm, z_mm: 0 },
        evidence_class: AnchorEvidenceClass::DesignAssumption,
        evidence_refs: vec!["SYNTHETIC-SERVICE-TAP".to_owned()],
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let source = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let configuration: Configuration = serde_json::from_str(source)?;

    let requests = vec![
        ServiceRouteRequest {
            route_id: "BUNDLE-CABIN-AC".to_owned(),
            connection_id: "CONN-CABIN-01-AC_IN".to_owned(),
            start: module_port("CABIN-01", "AC_IN", 2800, 1800, "MAIN"),
            end: service("AC-BUS", 4700, 1800, "MAIN"),
            clearance_mm: 100,
            bend_penalty_mm: 100,
        },
        ServiceRouteRequest {
            route_id: "BUNDLE-WETROOM-AC".to_owned(),
            connection_id: "CONN-WETROOM-01-AC_IN".to_owned(),
            start: module_port("WETROOM-01", "AC_IN", 4600, 1800, "MAIN"),
            // A deliberately different explicit tap coordinate tests overlap semantics.
            end: service("AC-BUS", 4700, 1600, "MAIN"),
            clearance_mm: 100,
            bend_penalty_mm: 100,
        },
    ];
    let routing_limits = RoutingLimits {
        grid_pitch_mm: 100,
        max_route_requests: 16,
        max_grid_cells: 20_000,
        max_expanded_states: 200_000,
    };
    let bundle_limits = RouteBundleLimits::default();
    let report = optimize_route_bundle(
        &configuration,
        &requests,
        routing_limits.clone(),
        bundle_limits.clone(),
    )?;
    let verifies = verify_route_bundle_report(
        &configuration, &requests, &routing_limits, &bundle_limits, &report,
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !verifies {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "route-bundle recomputation verifier rejected the report",
        ).into());
    }
    Ok(())
}
