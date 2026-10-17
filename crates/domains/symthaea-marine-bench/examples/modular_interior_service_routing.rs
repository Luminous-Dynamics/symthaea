// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Demonstrates a 2D grid route using explicitly declared synthetic anchors.
//! The route is only a geometry proxy, not a qualified cable/pipe installation.

use std::error::Error;

use serde::Serialize;
use symthaea_marine_bench::modular_interior::{Configuration, PositionMm};
use symthaea_marine_bench::modular_interior_service_routing::{
    route_services, verify_service_routing_report, AnchorEvidenceClass, AnchorKind,
    AnchorSide, RoutingAnchor, RoutingLimits, ServiceRouteRequest,
};

#[derive(Serialize)]
struct RouteSummary {
    route_id: String,
    connection_id: String,
    disposition: String,
    path_length_mm_proxy: Option<u64>,
    bend_count: Option<usize>,
    expanded_states: usize,
    conflicts_with_route_ids: Vec<String>,
    path_xy_mm: Vec<(i64, i64)>,
}

#[derive(Serialize)]
struct ExampleSummary {
    router_version: String,
    fixture_values_synthetic: bool,
    grid_pitch_mm: i64,
    receipt_verifies_by_recomputation: bool,
    report_sha256: String,
    routes: Vec<RouteSummary>,
    claim_ceiling: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let source = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    let configuration: Configuration = serde_json::from_str(source)?;
    let request = ServiceRouteRequest {
        route_id: "AC-TRUNK-01".to_owned(),
        connection_id: "CONN-CABIN-01-AC_IN".to_owned(),
        start: RoutingAnchor {
            kind: AnchorKind::ModulePort,
            instance_id: Some("CABIN-01".to_owned()),
            port_id: Some("AC_IN".to_owned()),
            service_id: None,
            side: Some(AnchorSide::East),
            deck_id: "MAIN".to_owned(),
            position_mm: PositionMm { x_mm: 2800, y_mm: 1800, z_mm: 0 },
            evidence_class: AnchorEvidenceClass::DesignAssumption,
            evidence_refs: vec!["SYNTHETIC-PORT-ANCHOR".to_owned()],
        },
        end: RoutingAnchor {
            kind: AnchorKind::PlatformService,
            instance_id: None,
            port_id: None,
            service_id: Some("AC-BUS".to_owned()),
            side: None,
            deck_id: "MAIN".to_owned(),
            position_mm: PositionMm { x_mm: 4700, y_mm: 1800, z_mm: 0 },
            evidence_class: AnchorEvidenceClass::DesignAssumption,
            evidence_refs: vec!["SYNTHETIC-SERVICE-ANCHOR".to_owned()],
        },
        clearance_mm: 100,
        bend_penalty_mm: 100,
    };
    let limits = RoutingLimits {
        grid_pitch_mm: 100,
        max_route_requests: 16,
        max_grid_cells: 20_000,
        max_expanded_states: 200_000,
    };
    let report = route_services(&configuration, &[request.clone()], limits.clone())?;
    let receipt_verifies_by_recomputation =
        verify_service_routing_report(&configuration, &[request], &limits, &report);

    let routes = report.body.routes.iter().map(|route| RouteSummary {
        route_id: route.route_id.clone(),
        connection_id: route.connection_id.clone(),
        disposition: format!("{:?}", route.disposition),
        path_length_mm_proxy: route.path_length_mm,
        bend_count: route.bend_count,
        expanded_states: route.expanded_states,
        conflicts_with_route_ids: route.conflicts_with_route_ids.clone(),
        path_xy_mm: route.path.iter().map(|point| (point.x_mm, point.y_mm)).collect(),
    }).collect();

    let summary = ExampleSummary {
        router_version: report.body.router_version.clone(),
        fixture_values_synthetic: true,
        grid_pitch_mm: report.body.grid_pitch_mm,
        receipt_verifies_by_recomputation,
        report_sha256: report.report_sha256.clone(),
        routes,
        claim_ceiling: report.body.claim_ceiling.clone(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !receipt_verifies_by_recomputation {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "recomputed service-routing report did not verify",
        ).into());
    }
    Ok(())
}
