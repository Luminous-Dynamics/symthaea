// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound orthogonal 2D route screening for explicit service anchors.
//!
//! This module deliberately does not infer port locations. The caller supplies
//! coordinate anchors with evidence references; the module checks their declared
//! topology against the connection endpoint IDs and requires module-port anchors
//! to lie on the referenced module envelope boundary.
//!
//! The result is a grid-routing proxy, not a 3D pipe/cable design or safety approval.

use crate::modular_interior::{Configuration, Endpoint, PositionMm};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;

pub const SERVICE_ROUTING_VERSION: &str = "marine-interior-service-routing-v1";
pub const HARD_MAX_ROUTE_REQUESTS: usize = 32;
pub const HARD_MAX_GRID_CELLS: usize = 250_000;
pub const HARD_MAX_EXPANDED_STATES: usize = 750_000;
pub const HARD_MAX_CLEARANCE_MM: i64 = 5_000;
pub const HARD_MAX_BEND_PENALTY_MM: i64 = 100_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoutingLimits {
    pub grid_pitch_mm: i64,
    pub max_route_requests: usize,
    pub max_grid_cells: usize,
    pub max_expanded_states: usize,
}

impl Default for RoutingLimits {
    fn default() -> Self {
        Self {
            grid_pitch_mm: 100,
            max_route_requests: 16,
            max_grid_cells: 100_000,
            max_expanded_states: 300_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingError {
    pub code: String,
    pub message: String,
}

impl RoutingError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_owned(), message: message.into() }
    }
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RoutingError {}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorKind {
    ModulePort,
    PlatformService,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorSide {
    North,
    South,
    East,
    West,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorEvidenceClass {
    DesignAssumption,
    Measured,
    SourceVerified,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingAnchor {
    pub kind: AnchorKind,
    pub instance_id: Option<String>,
    pub port_id: Option<String>,
    pub service_id: Option<String>,
    /// Required on a module-port anchor to define an explicit outward approach.
    /// Platform-service anchors must leave this unset.
    pub side: Option<AnchorSide>,
    pub deck_id: String,
    pub position_mm: PositionMm,
    pub evidence_class: AnchorEvidenceClass,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceRouteRequest {
    pub route_id: String,
    pub connection_id: String,
    pub start: RoutingAnchor,
    pub end: RoutingAnchor,
    pub clearance_mm: i64,
    pub bend_penalty_mm: i64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RouteDisposition {
    Routed,
    NoPath,
    EndpointBlocked,
    SearchBudgetExceeded,
    InvalidGeneratedPath,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GridPoint {
    pub x_mm: i64,
    pub y_mm: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceRouteResult {
    pub route_id: String,
    pub connection_id: String,
    pub medium: String,
    pub disposition: RouteDisposition,
    pub path: Vec<GridPoint>,
    /// 2D centreline grid distance only; not installed pipe/cable length.
    pub path_length_mm: Option<u64>,
    pub bend_count: Option<usize>,
    /// Path length plus caller-provided bend penalties; not a safety score.
    pub search_cost_mm: Option<u64>,
    pub expanded_states: usize,
    pub conflicts_with_route_ids: Vec<String>,
    pub anchor_evidence_classes: Vec<AnchorEvidenceClass>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceRoutingReportBody {
    pub router_version: String,
    pub configuration_id: String,
    pub configuration_sha256: String,
    pub input_manifest_sha256: String,
    pub grid_pitch_mm: i64,
    pub limits: RoutingLimits,
    pub routes: Vec<ServiceRouteResult>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceRoutingReport {
    pub body: ServiceRoutingReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct AnchorFingerprint {
    kind: AnchorKind,
    instance_id: Option<String>,
    port_id: Option<String>,
    service_id: Option<String>,
    side: Option<AnchorSide>,
    deck_id: String,
    x_mm: i64,
    y_mm: i64,
    z_mm: i64,
    evidence_class: AnchorEvidenceClass,
    evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RequestFingerprint {
    route_id: String,
    connection_id: String,
    start: AnchorFingerprint,
    end: AnchorFingerprint,
    clearance_mm: i64,
    bend_penalty_mm: i64,
}

#[derive(Clone, Debug, Serialize)]
struct RouteInputManifest {
    requests: Vec<RequestFingerprint>,
}

#[derive(Clone, Debug)]
struct Rect {
    x_min: i64,
    x_max: i64,
    y_min: i64,
    y_max: i64,
}

#[derive(Clone, Debug)]
struct Obstacle {
    owner: String,
    rect: Rect,
}

#[derive(Clone, Debug)]
struct Grid {
    x_min_mm: i64,
    y_min_mm: i64,
    pitch_mm: i64,
    width: usize,
    height: usize,
    blocked: Vec<bool>,
}

#[derive(Clone, Debug)]
struct PathOutcome {
    disposition: RouteDisposition,
    cells: Vec<usize>,
    length_mm: Option<u64>,
    bend_count: Option<usize>,
    search_cost_mm: Option<u64>,
    expanded_states: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn serialization_error(error: serde_json::Error) -> RoutingError {
    RoutingError::new("SERIALIZATION_FAILED", error.to_string())
}

fn stable_id(value: &str) -> bool {
    (3..=96).contains(&value.len())
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn anchor_fingerprint(anchor: &RoutingAnchor) -> AnchorFingerprint {
    let mut refs = anchor.evidence_refs.clone();
    refs.sort();
    AnchorFingerprint {
        kind: anchor.kind,
        instance_id: anchor.instance_id.clone(),
        port_id: anchor.port_id.clone(),
        service_id: anchor.service_id.clone(),
        side: anchor.side,
        deck_id: anchor.deck_id.clone(),
        x_mm: anchor.position_mm.x_mm,
        y_mm: anchor.position_mm.y_mm,
        z_mm: anchor.position_mm.z_mm,
        evidence_class: anchor.evidence_class,
        evidence_refs: refs,
    }
}

fn request_fingerprint(request: &ServiceRouteRequest) -> RequestFingerprint {
    RequestFingerprint {
        route_id: request.route_id.clone(),
        connection_id: request.connection_id.clone(),
        start: anchor_fingerprint(&request.start),
        end: anchor_fingerprint(&request.end),
        clearance_mm: request.clearance_mm,
        bend_penalty_mm: request.bend_penalty_mm,
    }
}

fn normalize_requests(
    configuration: &Configuration,
    requests: &[ServiceRouteRequest],
    limits: &RoutingLimits,
) -> Result<(Vec<ServiceRouteRequest>, String), RoutingError> {
    if limits.grid_pitch_mm <= 0 {
        return Err(RoutingError::new("INVALID_GRID_PITCH", "grid pitch must be positive"));
    }
    if limits.max_route_requests == 0 || limits.max_route_requests > HARD_MAX_ROUTE_REQUESTS {
        return Err(RoutingError::new(
            "INVALID_ROUTE_REQUEST_LIMIT",
            format!("max_route_requests must be in 1..={HARD_MAX_ROUTE_REQUESTS}"),
        ));
    }
    if requests.is_empty() || requests.len() > limits.max_route_requests {
        return Err(RoutingError::new(
            "ROUTE_REQUEST_COUNT_OUT_OF_RANGE",
            format!("received {} route requests; configured maximum is {}", requests.len(), limits.max_route_requests),
        ));
    }
    if limits.max_grid_cells == 0 || limits.max_grid_cells > HARD_MAX_GRID_CELLS {
        return Err(RoutingError::new(
            "INVALID_GRID_CELL_LIMIT",
            format!("max_grid_cells must be in 1..={HARD_MAX_GRID_CELLS}"),
        ));
    }
    if limits.max_expanded_states == 0 || limits.max_expanded_states > HARD_MAX_EXPANDED_STATES {
        return Err(RoutingError::new(
            "INVALID_EXPANSION_LIMIT",
            format!("max_expanded_states must be in 1..={HARD_MAX_EXPANDED_STATES}"),
        ));
    }

    let mut route_ids = BTreeSet::new();
    let mut connection_ids = BTreeSet::new();
    let mut sorted = requests.to_vec();
    for request in &sorted {
        if !stable_id(&request.route_id) || !stable_id(&request.connection_id) {
            return Err(RoutingError::new("INVALID_ROUTE_ID", "route and connection IDs must be stable identifiers"));
        }
        if !route_ids.insert(request.route_id.clone()) {
            return Err(RoutingError::new("DUPLICATE_ROUTE_ID", format!("route ID {:?} is duplicated", request.route_id)));
        }
        if !connection_ids.insert(request.connection_id.clone()) {
            return Err(RoutingError::new("DUPLICATE_CONNECTION_ROUTE", format!("connection {:?} has multiple route requests", request.connection_id)));
        }
        if request.clearance_mm < 0 || request.clearance_mm > HARD_MAX_CLEARANCE_MM {
            return Err(RoutingError::new("INVALID_ROUTE_CLEARANCE", format!("route {:?} clearance must be in 0..={HARD_MAX_CLEARANCE_MM} mm", request.route_id)));
        }
        if request.bend_penalty_mm < 0 || request.bend_penalty_mm > HARD_MAX_BEND_PENALTY_MM {
            return Err(RoutingError::new("INVALID_BEND_PENALTY", format!("route {:?} bend penalty must be in 0..={HARD_MAX_BEND_PENALTY_MM} mm", request.route_id)));
        }
        validate_anchor_shape(&request.start)?;
        validate_anchor_shape(&request.end)?;
        let connections: Vec<_> = configuration.connections.iter()
            .filter(|connection| connection.connection_id == request.connection_id).collect();
        if connections.len() != 1 {
            return Err(RoutingError::new(
                "CONNECTION_NOT_UNIQUE",
                format!("connection {:?} resolves to {} entries", request.connection_id, connections.len()),
            ));
        }
        let connection = connections[0];
        let forward = endpoint_matches_anchor(&connection.endpoint_a, &request.start)
            && endpoint_matches_anchor(&connection.endpoint_b, &request.end);
        let reverse = endpoint_matches_anchor(&connection.endpoint_a, &request.end)
            && endpoint_matches_anchor(&connection.endpoint_b, &request.start);
        if !forward && !reverse {
            return Err(RoutingError::new(
                "ANCHORS_DO_NOT_MATCH_CONNECTION",
                format!("route {:?} anchors do not match connection {:?} endpoint IDs", request.route_id, request.connection_id),
            ));
        }
        if connection.medium.trim().is_empty() {
            return Err(RoutingError::new("CONNECTION_MEDIUM_MISSING", "connection medium is empty"));
        }
        if request.start.position_mm.z_mm != request.end.position_mm.z_mm {
            return Err(RoutingError::new(
                "CROSS_ELEVATION_ROUTE_UNSUPPORTED",
                "a 2D route requires start and end anchors to declare the same Z plane",
            ));
        }
        for anchor in [&request.start, &request.end] {
            if anchor.evidence_refs.is_empty() {
                return Err(RoutingError::new(
                    "ANCHOR_EVIDENCE_REFERENCE_MISSING",
                    format!("anchor on route {:?} must carry at least one evidence reference", request.route_id),
                ));
            }
            if !configuration.platform.deck_bounds.iter().any(|deck| deck.deck_id == anchor.deck_id) {
                return Err(RoutingError::new("ANCHOR_DECK_NOT_FOUND", format!("anchor references unknown deck {:?}", anchor.deck_id)));
            }
            match anchor.kind {
                AnchorKind::ModulePort => {
                    let instance_id = anchor.instance_id.as_deref().expect("shape checked");
                    let port_id = anchor.port_id.as_deref().expect("shape checked");
                    let placements: Vec<_> = configuration.placements.iter()
                        .filter(|placement| placement.instance_id == instance_id).collect();
                    if placements.len() != 1 {
                        return Err(RoutingError::new("ANCHOR_INSTANCE_NOT_UNIQUE", format!("anchor instance {:?} resolves to {} placements", instance_id, placements.len())));
                    }
                    if placements[0].deck_id != anchor.deck_id {
                        return Err(RoutingError::new("ANCHOR_DECK_MISMATCH", format!("anchor deck differs from instance {:?} deck", instance_id)));
                    }
                    let modules: Vec<_> = configuration.module_catalog.iter()
                        .filter(|module| module.module_id == placements[0].module_id && module.revision == placements[0].module_revision).collect();
                    if modules.len() != 1 {
                        return Err(RoutingError::new("ANCHOR_MODULE_NOT_UNIQUE", format!("anchor instance {:?} module revision resolves to {} definitions", instance_id, modules.len())));
                    }
                    let ports: Vec<_> = modules[0].ports.iter().filter(|port| port.port_id == port_id).collect();
                    if ports.len() != 1 {
                        return Err(RoutingError::new("ANCHOR_PORT_NOT_UNIQUE", format!("anchor port {:?} resolves to {} definitions", port_id, ports.len())));
                    }
                    if ports[0].medium != connection.medium {
                        return Err(RoutingError::new("ANCHOR_MEDIUM_MISMATCH", format!("anchor port {:?} medium differs from connection {:?}", port_id, connection.connection_id)));
                    }
                }
                AnchorKind::PlatformService => {
                    let service_id = anchor.service_id.as_deref().expect("shape checked");
                    let services: Vec<_> = configuration.platform_services.iter()
                        .filter(|service| service.service_id == service_id).collect();
                    if services.len() != 1 {
                        return Err(RoutingError::new("ANCHOR_SERVICE_NOT_UNIQUE", format!("service anchor {:?} resolves to {} platform services", service_id, services.len())));
                    }
                    if services[0].medium != connection.medium {
                        return Err(RoutingError::new("ANCHOR_MEDIUM_MISMATCH", format!("service anchor {:?} medium differs from connection {:?}", service_id, connection.connection_id)));
                    }
                }
            }
        }
    }
    sorted.sort_by(|a, b| a.route_id.cmp(&b.route_id).then(a.connection_id.cmp(&b.connection_id)));
    let manifest = RouteInputManifest {
        requests: sorted.iter().map(request_fingerprint).collect(),
    };
    let manifest_sha256 = digest(&manifest).map_err(serialization_error)?;
    Ok((sorted, manifest_sha256))
}

fn validate_anchor_shape(anchor: &RoutingAnchor) -> Result<(), RoutingError> {
    match anchor.kind {
        AnchorKind::ModulePort => {
            if anchor.instance_id.as_deref().is_none_or(str::is_empty)
                || anchor.port_id.as_deref().is_none_or(str::is_empty)
                || anchor.service_id.is_some()
                || anchor.side.is_none()
            {
                return Err(RoutingError::new("INVALID_MODULE_PORT_ANCHOR", "module-port anchor must have instance_id/port_id, an outward side, and no service_id"));
            }
        }
        AnchorKind::PlatformService => {
            if anchor.service_id.as_deref().is_none_or(str::is_empty)
                || anchor.instance_id.is_some()
                || anchor.port_id.is_some()
                || anchor.side.is_some()
            {
                return Err(RoutingError::new("INVALID_PLATFORM_SERVICE_ANCHOR", "platform-service anchor must have service_id, no module instance/port IDs, and no port side"));
            }
        }
    }
    Ok(())
}

fn endpoint_matches_anchor(endpoint: &Endpoint, anchor: &RoutingAnchor) -> bool {
    match anchor.kind {
        AnchorKind::ModulePort => endpoint.kind == "module_port"
            && endpoint.instance_id.as_deref() == anchor.instance_id.as_deref()
            && endpoint.port_id.as_deref() == anchor.port_id.as_deref()
            && endpoint.service_id.is_none(),
        AnchorKind::PlatformService => endpoint.kind == "platform_service"
            && endpoint.service_id.as_deref() == anchor.service_id.as_deref()
            && endpoint.instance_id.is_none()
            && endpoint.port_id.is_none(),
    }
}

fn module_bounds(
    configuration: &Configuration,
    instance_id: &str,
) -> Result<(String, Rect), RoutingError> {
    let placements: Vec<_> = configuration.placements.iter()
        .filter(|placement| placement.instance_id == instance_id).collect();
    if placements.len() != 1 {
        return Err(RoutingError::new("ANCHOR_INSTANCE_NOT_UNIQUE", format!("instance {:?} resolves to {} placements", instance_id, placements.len())));
    }
    let placement = placements[0];
    let modules: Vec<_> = configuration.module_catalog.iter()
        .filter(|module| module.module_id == placement.module_id && module.revision == placement.module_revision).collect();
    if modules.len() != 1 {
        return Err(RoutingError::new("ANCHOR_MODULE_NOT_UNIQUE", format!("module {:?} revision {} resolves to {} definitions", placement.module_id, placement.module_revision, modules.len())));
    }
    let module = modules[0];
    let (length, width) = match placement.rotation_z_deg {
        0 | 180 => (module.envelope_mm.length_mm, module.envelope_mm.width_mm),
        90 | 270 => (module.envelope_mm.width_mm, module.envelope_mm.length_mm),
        _ => return Err(RoutingError::new("UNSUPPORTED_MODULE_ROTATION", format!("instance {:?} has unsupported rotation {}", instance_id, placement.rotation_z_deg))),
    };
    if length <= 0 || width <= 0 {
        return Err(RoutingError::new("INVALID_MODULE_ENVELOPE", format!("instance {:?} has non-positive envelope", instance_id)));
    }
    let x0 = placement.position_mm.x_mm;
    let y0 = placement.position_mm.y_mm;
    let x1 = x0.checked_add(length).ok_or_else(|| RoutingError::new("ENVELOPE_OVERFLOW", "module envelope X overflow"))?;
    let y1 = y0.checked_add(width).ok_or_else(|| RoutingError::new("ENVELOPE_OVERFLOW", "module envelope Y overflow"))?;
    Ok((placement.deck_id.clone(), Rect { x_min: x0, x_max: x1, y_min: y0, y_max: y1 }))
}

fn anchor_on_module_boundary(
    configuration: &Configuration,
    anchor: &RoutingAnchor,
) -> Result<(), RoutingError> {
    if anchor.kind != AnchorKind::ModulePort {
        return Ok(());
    }
    let instance_id = anchor.instance_id.as_deref().expect("shape checked");
    let (deck_id, rect) = module_bounds(configuration, instance_id)?;
    let x = anchor.position_mm.x_mm;
    let y = anchor.position_mm.y_mm;
    let on_side = match anchor.side.expect("shape checked") {
        AnchorSide::West => x == rect.x_min && y > rect.y_min && y < rect.y_max,
        AnchorSide::East => x == rect.x_max && y > rect.y_min && y < rect.y_max,
        AnchorSide::South => y == rect.y_min && x > rect.x_min && x < rect.x_max,
        AnchorSide::North => y == rect.y_max && x > rect.x_min && x < rect.x_max,
    };
    if !on_side {
        return Err(RoutingError::new(
            "ANCHOR_NOT_ON_DECLARED_ENVELOPE_SIDE",
            format!("anchor for instance {:?} must lie strictly on its declared module side, not at a corner", instance_id),
        ));
    }
    if deck_id != anchor.deck_id {
        return Err(RoutingError::new("ANCHOR_DECK_MISMATCH", "module-port anchor deck differs from its placement deck"));
    }
    Ok(())
}

fn build_obstacles(
    configuration: &Configuration,
    deck_id: &str,
    start: &RoutingAnchor,
    end: &RoutingAnchor,
) -> Result<Vec<Obstacle>, RoutingError> {
    let mut obstacles = Vec::new();
    for placement in configuration.placements.iter().filter(|p| p.deck_id == deck_id) {
        let (placed_deck, rect) = module_bounds(configuration, &placement.instance_id)?;
        if placed_deck == deck_id {
            obstacles.push(Obstacle { owner: format!("module:{}", placement.instance_id), rect });
        }
    }
    // Treat all reserved regions as blocked in this first-pass router. Even
    // "requires_review" regions are conservative obstacles, not traversable space.
    for region in configuration.reserved_regions.iter().filter(|r| r.deck_id == deck_id) {
        let b = &region.bounds_mm;
        if b.x_min_mm >= b.x_max_mm || b.y_min_mm >= b.y_max_mm {
            return Err(RoutingError::new("INVALID_RESERVED_REGION_BOUNDS", format!("region {:?} has invalid XY bounds", region.region_id)));
        }
        obstacles.push(Obstacle {
            owner: format!("reserved:{}", region.region_id),
            rect: Rect { x_min: b.x_min_mm, x_max: b.x_max_mm, y_min: b.y_min_mm, y_max: b.y_max_mm },
        });
    }
    let _ = (start, end);
    obstacles.sort_by(|a, b| a.owner.cmp(&b.owner)
        .then(a.rect.x_min.cmp(&b.rect.x_min)).then(a.rect.y_min.cmp(&b.rect.y_min)));
    Ok(obstacles)
}

fn contains(rect: &Rect, x: i64, y: i64, clearance_mm: i64) -> bool {
    let x = i128::from(x);
    let y = i128::from(y);
    x >= i128::from(rect.x_min) - i128::from(clearance_mm)
        && x < i128::from(rect.x_max) + i128::from(clearance_mm)
        && y >= i128::from(rect.y_min) - i128::from(clearance_mm)
        && y < i128::from(rect.y_max) + i128::from(clearance_mm)
}

fn rect_for_anchor_owner_from_anchor(anchor: &RoutingAnchor) -> Option<String> {
    (anchor.kind == AnchorKind::ModulePort)
        .then(|| format!("module:{}", anchor.instance_id.as_deref().unwrap_or_default()))
}

fn segment_intersects_obstacle(
    rect: &Rect,
    a: &GridPoint,
    b: &GridPoint,
    clearance_mm: i64,
) -> bool {
    let xmin = i128::from(rect.x_min) - i128::from(clearance_mm);
    let xmax = i128::from(rect.x_max) + i128::from(clearance_mm);
    let ymin = i128::from(rect.y_min) - i128::from(clearance_mm);
    let ymax = i128::from(rect.y_max) + i128::from(clearance_mm);
    let ax = i128::from(a.x_mm);
    let ay = i128::from(a.y_mm);
    let bx = i128::from(b.x_mm);
    let by = i128::from(b.y_mm);
    if ay == by {
        ay >= ymin && ay < ymax && ax.max(bx) > xmin && ax.min(bx) < xmax
    } else if ax == bx {
        ax >= xmin && ax < xmax && ay.max(by) > ymin && ay.min(by) < ymax
    } else {
        // Routed and approach segments are required to be orthogonal.
        true
    }
}

fn segment_blocked_by_foreign_obstacle(
    obstacles: &[Obstacle],
    anchor: &RoutingAnchor,
    a: &GridPoint,
    b: &GridPoint,
    clearance_mm: i64,
) -> bool {
    let own = rect_for_anchor_owner_from_anchor(anchor).unwrap_or_default();
    obstacles.iter().any(|obstacle| {
        obstacle.owner != own && segment_intersects_obstacle(&obstacle.rect, a, b, clearance_mm)
    })
}

fn edge_blocked(
    grid: &Grid,
    obstacles: &[Obstacle],
    clearance_mm: i64,
    from: usize,
    to: usize,
) -> bool {
    let Ok(a) = cell_to_point(grid, from) else { return true; };
    let Ok(b) = cell_to_point(grid, to) else { return true; };
    obstacles.iter().any(|obstacle| {
        segment_intersects_obstacle(&obstacle.rect, &a, &b, clearance_mm)
    })
}

fn anchor_approach_points(
    anchor: &RoutingAnchor,
    clearance_mm: i64,
    grid_pitch_mm: i64,
) -> Result<Vec<GridPoint>, RoutingError> {
    let origin = GridPoint {
        x_mm: anchor.position_mm.x_mm,
        y_mm: anchor.position_mm.y_mm,
    };
    if anchor.kind != AnchorKind::ModulePort {
        return Ok(vec![origin]);
    }
    let side = anchor.side.expect("module-port anchor shape checked");
    let steps_i = ((i128::from(clearance_mm) + i128::from(grid_pitch_mm) - 1)
        / i128::from(grid_pitch_mm)).max(1);
    let steps = usize::try_from(steps_i)
        .map_err(|_| RoutingError::new("APPROACH_STEP_OVERFLOW", "approach step count overflows usize"))?;
    let (dx, dy) = match side {
        AnchorSide::North => (0i128, 1i128),
        AnchorSide::South => (0, -1),
        AnchorSide::East => (1, 0),
        AnchorSide::West => (-1, 0),
    };
    let mut points = Vec::with_capacity(steps.saturating_add(1));
    for step in 0..=steps {
        let offset = i128::from(grid_pitch_mm) * step as i128;
        let x = i128::from(origin.x_mm) + dx * offset;
        let y = i128::from(origin.y_mm) + dy * offset;
        points.push(GridPoint {
            x_mm: i64::try_from(x).map_err(|_| RoutingError::new("APPROACH_COORDINATE_OVERFLOW", "approach X coordinate overflows i64"))?,
            y_mm: i64::try_from(y).map_err(|_| RoutingError::new("APPROACH_COORDINATE_OVERFLOW", "approach Y coordinate overflows i64"))?,
        });
    }
    Ok(points)
}

fn grid_for_route(
    configuration: &Configuration,
    request: &ServiceRouteRequest,
    limits: &RoutingLimits,
) -> Result<(Grid, usize, usize, Vec<usize>, Vec<usize>, Vec<Obstacle>, bool), RoutingError> {
    anchor_on_module_boundary(configuration, &request.start)?;
    anchor_on_module_boundary(configuration, &request.end)?;
    if request.start.deck_id != request.end.deck_id {
        return Err(RoutingError::new("CROSS_DECK_ROUTE_UNSUPPORTED", "cross-deck routing requires a separately evidenced riser/penetration model"));
    }
    let deck_matches: Vec<_> = configuration.platform.deck_bounds.iter()
        .filter(|deck| deck.deck_id == request.start.deck_id).collect();
    if deck_matches.len() != 1 {
        return Err(RoutingError::new("DECK_NOT_UNIQUE", format!("deck {:?} resolves to {} definitions", request.start.deck_id, deck_matches.len())));
    }
    let bounds = &deck_matches[0].bounds_mm;
    let span_x = i128::from(bounds.x_max_mm) - i128::from(bounds.x_min_mm);
    let span_y = i128::from(bounds.y_max_mm) - i128::from(bounds.y_min_mm);
    if span_x <= 0 || span_y <= 0
        || span_x % i128::from(limits.grid_pitch_mm) != 0
        || span_y % i128::from(limits.grid_pitch_mm) != 0
    {
        return Err(RoutingError::new("GRID_DOES_NOT_TILE_DECK", "grid pitch must exactly divide positive deck X/Y spans"));
    }
    let width_i = span_x / i128::from(limits.grid_pitch_mm) + 1;
    let height_i = span_y / i128::from(limits.grid_pitch_mm) + 1;
    let cell_count_i = width_i * height_i;
    if cell_count_i <= 0 || cell_count_i > limits.max_grid_cells as i128
        || cell_count_i > HARD_MAX_GRID_CELLS as i128
    {
        return Err(RoutingError::new(
            "GRID_CELL_LIMIT_EXCEEDED",
            format!("deck grid needs {cell_count_i} cells; configured maximum is {}", limits.max_grid_cells.min(HARD_MAX_GRID_CELLS)),
        ));
    }
    let width = usize::try_from(width_i).map_err(|_| RoutingError::new("GRID_DIMENSION_OVERFLOW", "grid width overflows usize"))?;
    let height = usize::try_from(height_i).map_err(|_| RoutingError::new("GRID_DIMENSION_OVERFLOW", "grid height overflows usize"))?;
    let cell_count = usize::try_from(cell_count_i).map_err(|_| RoutingError::new("GRID_DIMENSION_OVERFLOW", "grid cell count overflows usize"))?;

    let aligned_index = |value: i64, min: i64| -> Option<usize> {
        let delta = i128::from(value) - i128::from(min);
        if delta < 0 || delta % i128::from(limits.grid_pitch_mm) != 0 {
            return None;
        }
        usize::try_from(delta / i128::from(limits.grid_pitch_mm)).ok()
    };
    let cell_for_point = |point: &GridPoint| -> Result<usize, RoutingError> {
        let xi = aligned_index(point.x_mm, bounds.x_min_mm)
            .ok_or_else(|| RoutingError::new("ANCHOR_NOT_GRID_ALIGNED", "anchor/approach X is outside bounds or not aligned to grid pitch"))?;
        let yi = aligned_index(point.y_mm, bounds.y_min_mm)
            .ok_or_else(|| RoutingError::new("ANCHOR_NOT_GRID_ALIGNED", "anchor/approach Y is outside bounds or not aligned to grid pitch"))?;
        if xi >= width || yi >= height {
            return Err(RoutingError::new("ANCHOR_APPROACH_OUT_OF_BOUNDS", "anchor or port-clearance approach extends outside deck bounds"));
        }
        Ok(yi * width + xi)
    };

    let start_points = anchor_approach_points(&request.start, request.clearance_mm, limits.grid_pitch_mm)?;
    let end_points = anchor_approach_points(&request.end, request.clearance_mm, limits.grid_pitch_mm)?;
    let start_approach: Vec<usize> = start_points.iter().map(cell_for_point).collect::<Result<_, _>>()?;
    let end_approach: Vec<usize> = end_points.iter().map(cell_for_point).collect::<Result<_, _>>()?;
    let start_escape = *start_approach.last().expect("approach contains anchor");
    let end_escape = *end_approach.last().expect("approach contains anchor");

    let obstacles = build_obstacles(configuration, &request.start.deck_id, &request.start, &request.end)?;
    let point_is_blocked = |anchor: &RoutingAnchor, point: &GridPoint| -> Result<bool, RoutingError> {
        let owner = rect_for_anchor_owner_from_anchor(anchor).unwrap_or_default();
        Ok(obstacles.iter().any(|obstacle| {
            obstacle.owner != owner
                && contains(&obstacle.rect, point.x_mm, point.y_mm, request.clearance_mm)
        }))
    };
    let mut endpoint_blocked = false;
    for points in [(&request.start, &start_points), (&request.end, &end_points)] {
        let (anchor, route_points) = points;
        for point in route_points {
            // The port's own module is the only obstacle permitted inside its outward
            // approach clearance. A different module or reserved area blocks the approach.
            if point_is_blocked(anchor, point)? {
                endpoint_blocked = true;
            }
        }
        for segment in route_points.windows(2) {
            if segment_blocked_by_foreign_obstacle(
                &obstacles, anchor, &segment[0], &segment[1], request.clearance_mm,
            ) {
                endpoint_blocked = true;
            }
        }
    }

    let mut blocked = vec![false; cell_count];
    for yi in 0..height {
        let y = i128::from(bounds.y_min_mm) + (yi as i128) * i128::from(limits.grid_pitch_mm);
        for xi in 0..width {
            let x = i128::from(bounds.x_min_mm) + (xi as i128) * i128::from(limits.grid_pitch_mm);
            let point_x = i64::try_from(x).map_err(|_| RoutingError::new("GRID_COORDINATE_OVERFLOW", "grid X coordinate overflows"))?;
            let point_y = i64::try_from(y).map_err(|_| RoutingError::new("GRID_COORDINATE_OVERFLOW", "grid Y coordinate overflows"))?;
            let index = yi * width + xi;
            blocked[index] = obstacles.iter().any(|obstacle| {
                contains(&obstacle.rect, point_x, point_y, request.clearance_mm)
            });
        }
    }

    if !endpoint_blocked {
        // These cells form explicit, directional port escape corridors. This is
        // preferable to suppressing clearance around an entire terminal module.
        for cell in start_approach.iter().chain(end_approach.iter()) {
            blocked[*cell] = false;
        }
    } else {
        blocked.fill(true);
    }

    Ok((Grid {
        x_min_mm: bounds.x_min_mm, y_min_mm: bounds.y_min_mm,
        pitch_mm: limits.grid_pitch_mm, width, height, blocked,
    }, start_escape, end_escape, start_approach, end_approach, obstacles, endpoint_blocked))
}

fn path_metrics(
    grid: &Grid,
    cells: &[usize],
    bend_penalty_mm: i64,
) -> (Option<u64>, Option<usize>, Option<u64>) {
    if cells.is_empty() {
        return (None, None, None);
    }
    let steps = cells.len().saturating_sub(1);
    let length = u64::try_from(steps).ok()
        .and_then(|count| count.checked_mul(u64::try_from(grid.pitch_mm).ok()?));
    let mut prior_axis = None;
    let mut bends = 0usize;
    for pair in cells.windows(2) {
        let ax = pair[0] % grid.width;
        let ay = pair[0] / grid.width;
        let bx = pair[1] % grid.width;
        let by = pair[1] / grid.width;
        let axis = if ay == by && ax.abs_diff(bx) == 1 {
            Some(1u8)
        } else if ax == bx && ay.abs_diff(by) == 1 {
            Some(2u8)
        } else {
            return (None, None, None);
        };
        if let (Some(previous), Some(current)) = (prior_axis, axis) {
            if previous != current { bends = bends.saturating_add(1); }
        }
        prior_axis = axis;
    }
    let score = length.and_then(|distance| {
        let bend_total = u64::try_from(bends).ok()?
            .checked_mul(u64::try_from(bend_penalty_mm).ok()?)?;
        distance.checked_add(bend_total)
    });
    (length, Some(bends), score)
}

fn cell_to_point(grid: &Grid, cell: usize) -> Result<GridPoint, RoutingError> {
    let x_index = cell % grid.width;
    let y_index = cell / grid.width;
    let x = i128::from(grid.x_min_mm) + i128::from(x_index as u64) * i128::from(grid.pitch_mm);
    let y = i128::from(grid.y_min_mm) + i128::from(y_index as u64) * i128::from(grid.pitch_mm);
    Ok(GridPoint {
        x_mm: i64::try_from(x).map_err(|_| RoutingError::new("GRID_COORDINATE_OVERFLOW", "route X coordinate overflows"))?,
        y_mm: i64::try_from(y).map_err(|_| RoutingError::new("GRID_COORDINATE_OVERFLOW", "route Y coordinate overflows"))?,
    })
}

fn manhattan_mm(grid: &Grid, a: usize, b: usize) -> i64 {
    let ax = (a % grid.width) as i128;
    let ay = (a / grid.width) as i128;
    let bx = (b % grid.width) as i128;
    let by = (b / grid.width) as i128;
    let steps = (ax - bx).abs() + (ay - by).abs();
    i64::try_from(steps.saturating_mul(i128::from(grid.pitch_mm))).unwrap_or(i64::MAX)
}

fn neighbor_cells(grid: &Grid, cell: usize) -> [(Option<usize>, u8); 4] {
    let x = cell % grid.width;
    let y = cell / grid.width;
    [
        (if x > 0 { Some(cell - 1) } else { None }, 1),
        (if x + 1 < grid.width { Some(cell + 1) } else { None }, 1),
        (if y > 0 { Some(cell - grid.width) } else { None }, 2),
        (if y + 1 < grid.height { Some(cell + grid.width) } else { None }, 2),
    ]
}

fn anchor_axis(anchor: &RoutingAnchor) -> u8 {
    match anchor.side {
        Some(AnchorSide::East | AnchorSide::West) => 1,
        Some(AnchorSide::North | AnchorSide::South) => 2,
        None => 0,
    }
}

fn search_grid(
    grid: &Grid,
    start: usize,
    goal: usize,
    request: &ServiceRouteRequest,
    max_expanded: usize,
    initial_direction: u8,
    terminal_direction: u8,
    obstacles: &[Obstacle],
) -> PathOutcome {
    if grid.blocked[start] || grid.blocked[goal] {
        return PathOutcome { disposition: RouteDisposition::EndpointBlocked, cells: Vec::new(), length_mm: None, bend_count: None, search_cost_mm: None, expanded_states: 0 };
    }
    if start == goal {
        return PathOutcome { disposition: RouteDisposition::Routed, cells: vec![start], length_mm: Some(0), bend_count: Some(0), search_cost_mm: Some(0), expanded_states: 0 };
    }
    // direction 0 = none, 1 = horizontal, 2 = vertical. The source approach
    // direction seeds A*; the sink approach bend is included in terminal cost.
    let state_count = grid.blocked.len().saturating_mul(3);
    let mut distances = vec![i64::MAX; state_count];
    let mut bends = vec![usize::MAX; state_count];
    let mut parents = vec![usize::MAX; state_count];
    let start_state = start * 3 + usize::from(initial_direction);
    distances[start_state] = 0;
    bends[start_state] = 0;
    let mut heap: BinaryHeap<Reverse<(i64, i64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((manhattan_mm(grid, start, goal), 0, start_state)));
    let mut expanded = 0usize;
    let mut finish: Option<usize> = None;
    let mut finish_cost = i64::MAX;

    while let Some(Reverse((f, cost, state))) = heap.peek().copied() {
        // Manhattan distance is admissible because bend penalties are non-negative.
        if finish.is_some() && f > finish_cost { break; }
        let _ = heap.pop();
        if cost != distances[state] { continue; }
        let cell = state / 3;
        let direction = (state % 3) as u8;
        if cell == goal {
            let terminal_bend = terminal_direction != 0
                && direction != 0
                && direction != terminal_direction;
            let terminal_cost = cost.saturating_add(
                if terminal_bend { request.bend_penalty_mm } else { 0 }
            );
            if terminal_cost < finish_cost
                || (terminal_cost == finish_cost && finish.is_none_or(|prior| state < prior))
            {
                finish = Some(state);
                finish_cost = terminal_cost;
            }
            continue;
        }
        if expanded >= max_expanded {
            return PathOutcome { disposition: RouteDisposition::SearchBudgetExceeded, cells: Vec::new(), length_mm: None, bend_count: None, search_cost_mm: None, expanded_states: expanded };
        }
        expanded += 1;
        for (next_cell, next_dir) in neighbor_cells(grid, cell) {
            let Some(next_cell) = next_cell else { continue; };
            if grid.blocked[next_cell]
                || edge_blocked(grid, obstacles, request.clearance_mm, cell, next_cell)
            {
                continue;
            }
            let move_cost = grid.pitch_mm.saturating_add(
                if direction != 0 && direction != next_dir { request.bend_penalty_mm } else { 0 }
            );
            let next_cost = cost.saturating_add(move_cost);
            let next_state = next_cell * 3 + usize::from(next_dir);
            let next_bends = bends[state].saturating_add(if direction != 0 && direction != next_dir { 1 } else { 0 });
            if next_cost < distances[next_state]
                || (next_cost == distances[next_state] && next_bends < bends[next_state])
            {
                distances[next_state] = next_cost;
                bends[next_state] = next_bends;
                parents[next_state] = state;
                let estimate = next_cost.saturating_add(manhattan_mm(grid, next_cell, goal));
                heap.push(Reverse((estimate, next_cost, next_state)));
            }
        }
    }
    let Some(mut state) = finish else {
        return PathOutcome { disposition: RouteDisposition::NoPath, cells: Vec::new(), length_mm: None, bend_count: None, search_cost_mm: None, expanded_states: expanded };
    };
    let terminal_bend = terminal_direction != 0
        && state % 3 != 0
        && (state % 3) as u8 != terminal_direction;
    let final_bends = bends[state].saturating_add(if terminal_bend { 1 } else { 0 });
    let mut cells = Vec::new();
    loop {
        cells.push(state / 3);
        if state == start_state { break; }
        state = parents[state];
        if state == usize::MAX {
            return PathOutcome { disposition: RouteDisposition::NoPath, cells: Vec::new(), length_mm: None, bend_count: None, search_cost_mm: None, expanded_states: expanded };
        }
    }
    cells.reverse();
    let length = u64::try_from(cells.len().saturating_sub(1))
        .ok().and_then(|steps| steps.checked_mul(u64::try_from(grid.pitch_mm).ok()?));
    let cost = u64::try_from(finish_cost).ok();
    PathOutcome {
        disposition: RouteDisposition::Routed,
        cells,
        length_mm: length,
        bend_count: Some(final_bends),
        search_cost_mm: cost,
        expanded_states: expanded,
    }
}

/// Routes explicit connection requests independently and flags exact shared-cell
/// conflicts. This first version does not jointly optimize the route bundle.
pub fn route_services(
    configuration: &Configuration,
    input_requests: &[ServiceRouteRequest],
    limits: RoutingLimits,
) -> Result<ServiceRoutingReport, RoutingError> {
    let (requests, input_manifest_sha256) = normalize_requests(configuration, input_requests, &limits)?;
    let configuration_sha256 = digest(configuration).map_err(serialization_error)?;
    let mut results = Vec::with_capacity(requests.len());
    let mut route_cells = BTreeMap::<String, BTreeSet<(String, usize)>>::new();

    for request in &requests {
        let (grid, start_escape, end_escape, start_approach, end_approach, obstacles, endpoint_blocked) =
            grid_for_route(configuration, request, &limits)?;
        let mut path_outcome = if endpoint_blocked {
            PathOutcome { disposition: RouteDisposition::EndpointBlocked, cells: Vec::new(), length_mm: None, bend_count: None, search_cost_mm: None, expanded_states: 0 }
        } else {
            search_grid(&grid, start_escape, end_escape, request, limits.max_expanded_states, anchor_axis(&request.start), anchor_axis(&request.end), &obstacles)
        };
        let mut complete_cells = Vec::new();
        if path_outcome.disposition == RouteDisposition::Routed {
            complete_cells.extend(start_approach.iter().copied());
            complete_cells.extend(path_outcome.cells.iter().skip(1).copied());
            complete_cells.extend(end_approach.iter().rev().skip(1).copied());
            let (length, bends, cost) = path_metrics(&grid, &complete_cells, request.bend_penalty_mm);
            if let (Some(length), Some(bends), Some(cost)) = (length, bends, cost) {
                path_outcome.length_mm = Some(length);
                path_outcome.bend_count = Some(bends);
                path_outcome.search_cost_mm = Some(cost);
            } else {
                // Never emit a "routed" disposition without a valid orthogonal cell path.
                path_outcome.disposition = RouteDisposition::InvalidGeneratedPath;
                path_outcome.length_mm = None;
                path_outcome.bend_count = None;
                path_outcome.search_cost_mm = None;
                complete_cells.clear();
            }
        }
        let mut path = Vec::with_capacity(complete_cells.len());
        for cell in &complete_cells {
            path.push(cell_to_point(&grid, *cell)?);
        }
        let cell_keys: BTreeSet<(String, usize)> = complete_cells.iter()
            .map(|cell| (request.start.deck_id.clone(), *cell)).collect();
        route_cells.insert(request.route_id.clone(), cell_keys);
        results.push(ServiceRouteResult {
            route_id: request.route_id.clone(),
            connection_id: request.connection_id.clone(),
            medium: configuration.connections.iter()
                .find(|connection| connection.connection_id == request.connection_id)
                .map(|connection| connection.medium.clone()).unwrap_or_default(),
            disposition: path_outcome.disposition,
            path,
            path_length_mm: path_outcome.length_mm,
            bend_count: path_outcome.bend_count,
            search_cost_mm: path_outcome.search_cost_mm,
            expanded_states: path_outcome.expanded_states,
            conflicts_with_route_ids: Vec::new(),
            anchor_evidence_classes: vec![request.start.evidence_class, request.end.evidence_class],
            claim_ceiling: concat!(
                "2D orthogonal deck-grid path proxy only. Supplied anchors are not authenticated by this router. ",
                "No 3D risers, installation tolerances, pipe/cable physics, fire stopping, regulatory ",
                "compliance, maintainability or seaworthiness is established."
            ).to_owned(),
        });
    }

    let mut conflict_map = BTreeMap::<String, BTreeSet<String>>::new();
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let a = &results[i];
            let b = &results[j];
            if a.disposition != RouteDisposition::Routed || b.disposition != RouteDisposition::Routed {
                continue;
            }
            let Some(a_cells) = route_cells.get(&a.route_id) else { continue; };
            let Some(b_cells) = route_cells.get(&b.route_id) else { continue; };
            if !a_cells.is_disjoint(b_cells) {
                conflict_map.entry(a.route_id.clone()).or_default().insert(b.route_id.clone());
                conflict_map.entry(b.route_id.clone()).or_default().insert(a.route_id.clone());
            }
        }
    }
    for result in &mut results {
        result.conflicts_with_route_ids = conflict_map.remove(&result.route_id)
            .unwrap_or_default().into_iter().collect();
    }

    let body = ServiceRoutingReportBody {
        router_version: SERVICE_ROUTING_VERSION.to_owned(),
        configuration_id: configuration.configuration_id.clone(),
        configuration_sha256,
        input_manifest_sha256,
        grid_pitch_mm: limits.grid_pitch_mm,
        limits: limits.clone(),
        routes: results,
        claim_ceiling: concat!(
            "Explicit-anchor, 2D orthogonal deck-grid screening only. Routes are calculated independently ",
            "then checked for shared-cell conflicts; a conflict-free bundle is not globally optimal and ",
            "does not establish true installation length, 3D route feasibility, system capacity or safety."
        ).to_owned(),
    };
    let report_sha256 = digest(&body).map_err(serialization_error)?;
    Ok(ServiceRoutingReport { body, report_sha256 })
}

/// Independently audits an emitted route polyline without rerunning A*.
/// This checks path geometry and reported metrics, not the physical truth of
/// supplied port anchors or suitability of the 2D abstraction.
pub fn verify_service_route_geometry(
    configuration: &Configuration,
    request: &ServiceRouteRequest,
    grid_pitch_mm: i64,
    result: &ServiceRouteResult,
) -> bool {
    if result.route_id != request.route_id
        || result.connection_id != request.connection_id
        || grid_pitch_mm <= 0
    {
        return false;
    }
    let connection = match configuration.connections.iter()
        .find(|connection| connection.connection_id == request.connection_id)
    {
        Some(connection) => connection,
        None => return false,
    };
    let forward = endpoint_matches_anchor(&connection.endpoint_a, &request.start)
        && endpoint_matches_anchor(&connection.endpoint_b, &request.end);
    let reverse = endpoint_matches_anchor(&connection.endpoint_a, &request.end)
        && endpoint_matches_anchor(&connection.endpoint_b, &request.start);
    if !forward && !reverse {
        return false;
    }
    if request.start.deck_id != request.end.deck_id
        || request.start.position_mm.z_mm != request.end.position_mm.z_mm
    {
        return false;
    }
    let deck_matches: Vec<_> = configuration.platform.deck_bounds.iter()
        .filter(|deck| deck.deck_id == request.start.deck_id).collect();
    if deck_matches.len() != 1 {
        return false;
    }
    let bounds = &deck_matches[0].bounds_mm;
    if bounds.x_max_mm <= bounds.x_min_mm || bounds.y_max_mm <= bounds.y_min_mm {
        return false;
    }

    if result.disposition != RouteDisposition::Routed {
        return result.path.is_empty()
            && result.path_length_mm.is_none()
            && result.bend_count.is_none()
            && result.search_cost_mm.is_none()
            && result.conflicts_with_route_ids.is_empty();
    }
    if result.path.is_empty() {
        return false;
    }
    if result.path.first().is_none_or(|point| {
        point.x_mm != request.start.position_mm.x_mm
            || point.y_mm != request.start.position_mm.y_mm
    }) || result.path.last().is_none_or(|point| {
        point.x_mm != request.end.position_mm.x_mm
            || point.y_mm != request.end.position_mm.y_mm
    }) {
        return false;
    }
    let aligned = |value: i64, min: i64| -> bool {
        let delta = i128::from(value) - i128::from(min);
        delta >= 0 && delta % i128::from(grid_pitch_mm) == 0
    };
    for point in &result.path {
        if point.x_mm < bounds.x_min_mm || point.x_mm > bounds.x_max_mm
            || point.y_mm < bounds.y_min_mm || point.y_mm > bounds.y_max_mm
            || !aligned(point.x_mm, bounds.x_min_mm)
            || !aligned(point.y_mm, bounds.y_min_mm)
        {
            return false;
        }
    }

    let start_approach = match anchor_approach_points(&request.start, request.clearance_mm, grid_pitch_mm) {
        Ok(points) => points,
        Err(_) => return false,
    };
    let end_approach = match anchor_approach_points(&request.end, request.clearance_mm, grid_pitch_mm) {
        Ok(points) => points,
        Err(_) => return false,
    };
    if result.path.len() < start_approach.len() || result.path.len() < end_approach.len() {
        return false;
    }
    let starts_with_approach = result.path.iter().take(start_approach.len())
        .zip(start_approach.iter())
        .all(|(point, expected)| point.x_mm == expected.x_mm && point.y_mm == expected.y_mm);
    let expected_end: Vec<_> = end_approach.iter().rev().skip(1).collect();
    let suffix_start = result.path.len().saturating_sub(expected_end.len());
    let ends_with_approach = result.path.iter().skip(suffix_start)
        .zip(expected_end.iter())
        .all(|(point, expected)| point.x_mm == expected.x_mm && point.y_mm == expected.y_mm);
    if !starts_with_approach || !ends_with_approach {
        return false;
    }

    let obstacles = match build_obstacles(configuration, &request.start.deck_id, &request.start, &request.end) {
        Ok(obstacles) => obstacles,
        Err(_) => return false,
    };
    let start_owner = rect_for_anchor_owner_from_anchor(&request.start).unwrap_or_default();
    let end_owner = rect_for_anchor_owner_from_anchor(&request.end).unwrap_or_default();
    let start_segment_count = start_approach.len().saturating_sub(1);
    let end_segment_count = expected_end.len();
    for (index, pair) in result.path.windows(2).enumerate() {
        let dx = (i128::from(pair[0].x_mm) - i128::from(pair[1].x_mm)).abs();
        let dy = (i128::from(pair[0].y_mm) - i128::from(pair[1].y_mm)).abs();
        let one_grid_step = (dx == i128::from(grid_pitch_mm) && dy == 0)
            || (dy == i128::from(grid_pitch_mm) && dx == 0);
        if !one_grid_step {
            return false;
        }
        let source_escape_segment = index < start_segment_count;
        let sink_escape_segment = index >= result.path.len().saturating_sub(1 + end_segment_count);
        for obstacle in &obstacles {
            if source_escape_segment && obstacle.owner == start_owner {
                continue;
            }
            if sink_escape_segment && obstacle.owner == end_owner {
                continue;
            }
            if segment_intersects_obstacle(&obstacle.rect, &pair[0], &pair[1], request.clearance_mm) {
                return false;
            }
        }
    }

    let steps = result.path.len().saturating_sub(1);
    let Some(length) = u64::try_from(steps).ok()
        .and_then(|count| count.checked_mul(u64::try_from(grid_pitch_mm).ok()?))
    else { return false; };
    let mut previous_axis = None;
    let mut bends = 0usize;
    for pair in result.path.windows(2) {
        let axis = if pair[0].y_mm == pair[1].y_mm { 1u8 } else { 2u8 };
        if previous_axis.is_some_and(|prior| prior != axis) {
            bends = bends.saturating_add(1);
        }
        previous_axis = Some(axis);
    }
    let Some(cost) = u64::try_from(bends).ok()
        .and_then(|count| count.checked_mul(u64::try_from(request.bend_penalty_mm).ok()?))
        .and_then(|penalty| length.checked_add(penalty))
    else { return false; };
    result.path_length_mm == Some(length)
        && result.bend_count == Some(bends)
        && result.search_cost_mm == Some(cost)
}

/// Recomputes the same bounded route search over exact inputs. This is integrity checking,
/// not a signature or certification of anchor/source authenticity.
pub fn verify_service_routing_report(
    configuration: &Configuration,
    requests: &[ServiceRouteRequest],
    limits: &RoutingLimits,
    report: &ServiceRoutingReport,
) -> bool {
    if digest(&report.body).ok().as_deref() != Some(report.report_sha256.as_str()) {
        return false;
    }
    if report.body.configuration_id != configuration.configuration_id
        || report.body.configuration_sha256 != digest(configuration).ok().unwrap_or_default()
        || report.body.limits != *limits
        || report.body.grid_pitch_mm != limits.grid_pitch_mm
    {
        return false;
    }
    let Ok((normalized_requests, manifest_sha256)) = normalize_requests(configuration, requests, limits) else {
        return false;
    };
    if report.body.input_manifest_sha256 != manifest_sha256
        || report.body.routes.len() != normalized_requests.len()
    {
        return false;
    }
    let request_by_id: BTreeMap<&str, &ServiceRouteRequest> = normalized_requests.iter()
        .map(|request| (request.route_id.as_str(), request)).collect();
    if report.body.routes.iter().any(|route| {
        let Some(request) = request_by_id.get(route.route_id.as_str()) else { return true; };
        !verify_service_route_geometry(configuration, request, limits.grid_pitch_mm, route)
    }) {
        return false;
    }
    // Independently recompute pairwise exact shared-cell conflicts from emitted paths.
    let mut expected_conflicts = BTreeMap::<String, BTreeSet<String>>::new();
    for i in 0..report.body.routes.len() {
        for j in (i + 1)..report.body.routes.len() {
            let a = &report.body.routes[i];
            let b = &report.body.routes[j];
            let Some(a_request) = request_by_id.get(a.route_id.as_str()) else { return false; };
            let Some(b_request) = request_by_id.get(b.route_id.as_str()) else { return false; };
            if a_request.start.deck_id != b_request.start.deck_id
                || a.disposition != RouteDisposition::Routed
                || b.disposition != RouteDisposition::Routed
            {
                continue;
            }
            if a.path.iter().any(|point| b.path.contains(point)) {
                expected_conflicts.entry(a.route_id.clone()).or_default().insert(b.route_id.clone());
                expected_conflicts.entry(b.route_id.clone()).or_default().insert(a.route_id.clone());
            }
        }
    }
    for route in &report.body.routes {
        let expected: Vec<String> = expected_conflicts.get(&route.route_id)
            .cloned().unwrap_or_default().into_iter().collect();
        if route.conflicts_with_route_ids != expected {
            return false;
        }
    }
    let Ok(recomputed) = route_services(configuration, requests, limits.clone()) else {
        return false;
    };
    recomputed.report_sha256 == report.report_sha256
        && digest(&recomputed.body).ok() == digest(&report.body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const ORACLE: &str = include_str!("../fixtures/modular-interiors/service-routing-oracle-v1.json");

    #[derive(Deserialize)]
    struct RoutingOracle {
        cases: Vec<RoutingOracleCase>,
    }

    #[derive(Deserialize)]
    struct RoutingOracleCase {
        case_id: String,
        expected_disposition: String,
        expected_min_length_mm: Option<u64>,
        expected_conflict_count: usize,
    }

    fn live() -> Configuration {
        serde_json::from_str(LIVE).expect("live fixture parses")
    }

    fn module_anchor(instance: &str, port: &str, x: i64, y: i64) -> RoutingAnchor {
        RoutingAnchor {
            kind: AnchorKind::ModulePort,
            instance_id: Some(instance.to_owned()),
            port_id: Some(port.to_owned()),
            service_id: None,
            side: Some(AnchorSide::East),
            deck_id: "MAIN".to_owned(),
            position_mm: PositionMm { x_mm: x, y_mm: y, z_mm: 0 },
            evidence_class: AnchorEvidenceClass::DesignAssumption,
            evidence_refs: vec!["SYNTHETIC-PORT-ANCHOR".to_owned()],
        }
    }

    fn service_anchor(service: &str, x: i64, y: i64) -> RoutingAnchor {
        RoutingAnchor {
            kind: AnchorKind::PlatformService,
            instance_id: None,
            port_id: None,
            service_id: Some(service.to_owned()),
            side: None,
            deck_id: "MAIN".to_owned(),
            position_mm: PositionMm { x_mm: x, y_mm: y, z_mm: 0 },
            evidence_class: AnchorEvidenceClass::DesignAssumption,
            evidence_refs: vec!["SYNTHETIC-SERVICE-ANCHOR".to_owned()],
        }
    }

    fn request_for(route_id: &str, connection_id: &str, start: RoutingAnchor, end: RoutingAnchor) -> ServiceRouteRequest {
        ServiceRouteRequest {
            route_id: route_id.to_owned(),
            connection_id: connection_id.to_owned(),
            start, end,
            clearance_mm: 100,
            bend_penalty_mm: 100,
        }
    }

    fn limits() -> RoutingLimits {
        RoutingLimits {
            grid_pitch_mm: 100,
            max_route_requests: 16,
            max_grid_cells: 20_000,
            max_expanded_states: 200_000,
        }
    }

    #[test]
    fn detour_around_wetroom_is_deterministic_and_receipt_verifies() {
        let baseline = live();
        let route = request_for(
            "AC-TRUNK-01", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        let report = route_services(&baseline, &[route.clone()], limits()).unwrap();
        let result = &report.body.routes[0];
        assert_eq!(result.disposition, RouteDisposition::Routed);
        assert!(result.path_length_mm.expect("route length") > 1_900);
        assert!(result.bend_count.unwrap_or(0) >= 2);
        for pair in result.path.windows(2) {
            let dx = pair[0].x_mm.abs_diff(pair[1].x_mm);
            let dy = pair[0].y_mm.abs_diff(pair[1].y_mm);
            assert_eq!(dx + dy, 100, "every route step is one orthogonal grid edge");
        }
        assert_eq!(result.path_length_mm,
            Some((result.path.len().saturating_sub(1) as u64) * 100));
        assert!(verify_service_routing_report(&baseline, &[route.clone()], &limits(), &report));
        let mut permuted_request = route.clone();
        permuted_request.route_id = "AC-TRUNK-02".to_owned();
        let permuted_report = route_services(&baseline, &[permuted_request], limits()).unwrap();
        assert_eq!(report.body.routes[0].path, permuted_report.body.routes[0].path);
    }

    #[test]
    fn no_path_and_expansion_cap_are_explicit() {
        let baseline = live();
        let no_path = request_for(
            "NO-PATH-ROUTE", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 13000, 1800),
        );
        // The full-height service-access reserved region blocks the far endpoint.
        let report = route_services(&baseline, &[no_path], limits()).unwrap();
        assert_eq!(report.body.routes[0].disposition, RouteDisposition::EndpointBlocked);

        // Synthetic wall crossing the full deck height: a genuine no-path case
        // distinct from an endpoint lying inside a reserved area.
        let mut walled = baseline.clone();
        walled.reserved_regions.push(crate::modular_interior::ReservedRegion {
            region_id: "SYNTHETIC-FULL-WIDTH-BARRIER".to_owned(),
            deck_id: "MAIN".to_owned(),
            bounds_mm: crate::modular_interior::BoundsMm {
                x_min_mm: 3500, x_max_mm: 3600,
                y_min_mm: 0, y_max_mm: 5000,
                z_min_mm: 0, z_max_mm: 2400,
            },
            purpose: "synthetic_no_path_control".to_owned(),
            clearance_rule: "keep_clear".to_owned(),
            evidence_status: "design_assumption".to_owned(),
        });
        let impossible = request_for(
            "CLOSED-CORRIDOR-ROUTE", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        let report = route_services(&walled, &[impossible], limits()).unwrap();
        assert_eq!(report.body.routes[0].disposition, RouteDisposition::NoPath);

        let direct = request_for(
            "BUDGET-ROUTE", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        let mut small_limit = limits();
        small_limit.max_expanded_states = 1;
        let report = route_services(&baseline, &[direct], small_limit).unwrap();
        assert_eq!(report.body.routes[0].disposition, RouteDisposition::SearchBudgetExceeded);
    }

    #[test]
    fn route_inputs_bind_to_existing_connection_endpoints() {
        let baseline = live();
        let route = request_for(
            "BAD-ANCHOR-01", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2700, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        assert_eq!(route_services(&baseline, &[route], limits()).unwrap_err().code,
            "ANCHOR_NOT_ON_DECLARED_ENVELOPE_SIDE");
    }

    #[test]
    fn mixed_elevation_anchors_are_not_projected_into_a_fake_2d_route() {
        let baseline = live();
        let mut end = service_anchor("AC-BUS", 4700, 1800);
        end.position_mm.z_mm = 1_000;
        let route = request_for(
            "CROSS-ELEVATION-ROUTE",
            "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            end,
        );
        assert_eq!(
            route_services(&baseline, &[route], limits()).unwrap_err().code,
            "CROSS_ELEVATION_ROUTE_UNSUPPORTED"
        );
    }

    #[test]
    fn duplicate_routes_for_connection_are_rejected() {
        let baseline = live();
        let first = request_for(
            "AC-TRUNK-01", "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        let mut second = first.clone();
        second.route_id = "AC-TRUNK-02".to_owned();
        assert_eq!(route_services(&baseline, &[first, second], limits()).unwrap_err().code,
            "DUPLICATE_CONNECTION_ROUTE");
    }

    #[test]
    fn independent_geometry_audit_rejects_tampered_polyline() {
        let baseline = live();
        let request = request_for(
            "GEOMETRY-AUDIT-ROUTE",
            "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        let report = route_services(&baseline, &[request.clone()], limits()).unwrap();
        let result = &report.body.routes[0];
        assert!(verify_service_route_geometry(&baseline, &request, 100, result));

        let mut tampered = serde_json::from_value::<ServiceRouteResult>(
            serde_json::to_value(result).unwrap()
        ).unwrap();
        tampered.path[5].x_mm = tampered.path[5].x_mm.saturating_add(100);
        assert!(!verify_service_route_geometry(&baseline, &request, 100, &tampered));
    }

    #[test]
    fn thin_obstacle_between_grid_nodes_blocks_the_crossing_segment() {
        let mut baseline = live();
        baseline.reserved_regions.push(crate::modular_interior::ReservedRegion {
            region_id: "SYNTHETIC-SUBGRID-WALL".to_owned(),
            deck_id: "MAIN".to_owned(),
            bounds_mm: crate::modular_interior::BoundsMm {
                x_min_mm: 2840, x_max_mm: 2860,
                y_min_mm: 1700, y_max_mm: 1900,
                z_min_mm: 0, z_max_mm: 2400,
            },
            purpose: "sub-grid-edge-crossing-control".to_owned(),
            clearance_rule: "keep_clear".to_owned(),
            evidence_status: "design_assumption".to_owned(),
        });
        let request = request_for(
            "SUBGRID-WALL-ROUTE",
            "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        // Neither grid endpoint at X=2800 nor X=2900 is in this narrow raw wall,
        // but the 100 mm edge crosses it; clearance inflation also blocks its corridor.
        let report = route_services(&baseline, &[request], limits()).unwrap();
        assert_eq!(report.body.routes[0].disposition, RouteDisposition::EndpointBlocked);
    }

    #[test]
    fn open_grid_matches_hand_calculated_manhattan_cost_oracle() {
        let grid = Grid {
            x_min_mm: 0,
            y_min_mm: 0,
            pitch_mm: 100,
            width: 5,
            height: 5,
            blocked: vec![false; 25],
        };
        let request = request_for(
            "GRID-ROUTE-001",
            "CONN-CABIN-01-AC_IN",
            module_anchor("CABIN-01", "AC_IN", 2800, 1800),
            service_anchor("AC-BUS", 4700, 1800),
        );
        // Any shortest 4-connected path from (0,0) to (4,4) has 8 edges and
        // exactly one bend at minimum; the optimal cost is 800 + 100.
        let outcome = search_grid(&grid, 0, 24, &request, 1_000, 0, 0, &[]);
        assert_eq!(outcome.disposition, RouteDisposition::Routed);
        assert_eq!(outcome.length_mm, Some(800));
        assert_eq!(outcome.bend_count, Some(1));
        assert_eq!(outcome.search_cost_mm, Some(900));
    }

    #[test]
    fn frozen_oracle_covers_route_length_and_shared_cell_conflicts() {
        let oracle: RoutingOracle = serde_json::from_str(ORACLE).expect("route oracle parses");
        for case in oracle.cases {
            match case.case_id.as_str() {
                "WETROOM-DETOUR-001" => {
                    let report = route_services(&live(), &[request_for(
                        "AC-TRUNK-01", "CONN-CABIN-01-AC_IN",
                        module_anchor("CABIN-01", "AC_IN", 2800, 1800),
                        service_anchor("AC-BUS", 4700, 1800),
                    )], limits()).unwrap();
                    let route = &report.body.routes[0];
                    assert_eq!(format!("{:?}", route.disposition).to_lowercase(), case.expected_disposition);
                    assert!(route.path_length_mm.unwrap_or(0) >= case.expected_min_length_mm.unwrap_or(0));
                    assert_eq!(route.conflicts_with_route_ids.len(), case.expected_conflict_count);
                }
                "SHARED-CELL-CONFLICT-002" => {
                    let baseline = live();
                    let cabin_route = request_for(
                        "ROUTE-CABIN", "CONN-CABIN-01-AC_IN",
                        module_anchor("CABIN-01", "AC_IN", 2800, 1800),
                        service_anchor("AC-BUS", 4700, 1800),
                    );
                    let wetroom_route = request_for(
                        "ROUTE-WETROOM", "CONN-WETROOM-01-AC_IN",
                        module_anchor("WETROOM-01", "AC_IN", 4600, 1800),
                        service_anchor("AC-BUS", 4700, 1800),
                    );
                    let report = route_services(&baseline, &[cabin_route, wetroom_route], limits()).unwrap();
                    assert_eq!(report.body.routes.len(), 2);
                    for route in &report.body.routes {
                        assert_eq!(route.disposition, RouteDisposition::Routed);
                        assert_eq!(route.conflicts_with_route_ids.len(), case.expected_conflict_count);
                        assert!(route.path_length_mm.unwrap_or(0) >= case.expected_min_length_mm.unwrap_or(0));
                    }
                }
                other => panic!("unknown routing oracle case: {other}"),
            }
        }
    }
}
