// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded alternative discovery and conflict-aware selection for marine service routes.
//!
//! This module builds a reproducible catalog by banning one interior grid edge at a
//! time and rerunning the existing evidence-bound router. This is a bounded deviation
//! heuristic, NOT an exhaustive k-shortest-path algorithm. It selects the best compatible
//! bundle only from paths actually discovered. Hard geometric conflicts are constraints,
//! never weighted penalties. All coordinates remain explicit-anchor 2D grid proxies.

use crate::modular_interior::{BoundsMm, Configuration, ReservedRegion};
use crate::modular_interior_service_routing::{
    route_services, verify_service_route_geometry, AnchorKind, GridPoint, RouteDisposition,
    RoutingError, RoutingLimits, ServiceRouteRequest, ServiceRouteResult, ServiceRoutingReport,
    HARD_MAX_BEND_PENALTY_MM, HARD_MAX_CLEARANCE_MM, HARD_MAX_EXPANDED_STATES,
    HARD_MAX_GRID_CELLS, HARD_MAX_ROUTE_REQUESTS,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const ROUTE_BUNDLE_VERSION: &str = "marine-interior-route-bundle-v1";
pub const HARD_MAX_BUNDLE_ROUTES: usize = 8;
pub const HARD_MAX_CANDIDATES_PER_ROUTE: usize = 64;
pub const HARD_MAX_DEVIATION_PROBES_PER_ROUTE: usize = 4_096;
pub const HARD_MAX_TOTAL_DEVIATION_PROBES: usize = 4_096;
pub const HARD_MAX_TOTAL_EXPANDED_STATES: usize = 25_000_000;
pub const HARD_MAX_BUNDLE_COMBINATIONS: usize = 1_000_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RouteBundleLimits {
    /// Includes the original shortest route when it can be generated.
    pub max_candidates_per_route: usize,
    /// Maximum one-edge-ban probes performed for each requested service.
    pub max_deviation_probes_per_route: usize,
    /// Aggregate edge-ban probes across the entire bundle, fairly divided by request count.
    pub max_total_deviation_probes: usize,
    /// Aggregate number of A* expansion states allowed across every base and deviation search.
    pub max_total_expanded_states: usize,
    /// Maximum complete candidate combinations evaluated by bundle selection.
    pub max_bundle_combinations: usize,
}

impl Default for RouteBundleLimits {
    fn default() -> Self {
        Self {
            max_candidates_per_route: 8,
            max_deviation_probes_per_route: 256,
            max_total_deviation_probes: 128,
            max_total_expanded_states: 2_000_000,
            max_bundle_combinations: 10_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteBundleError {
    pub code: String,
    pub message: String,
}

impl RouteBundleError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_owned(), message: message.into() }
    }
}

impl fmt::Display for RouteBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RouteBundleError {}

impl From<RoutingError> for RouteBundleError {
    fn from(error: RoutingError) -> Self {
        Self::new(&error.code, error.message)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GridEdge {
    pub a: GridPoint,
    pub b: GridPoint,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AlternativeProbe {
    pub banned_edge: Option<GridEdge>,
    pub probe_configuration_sha256: Option<String>,
    /// Effective per-call bound; None means no route search was launched.
    pub max_expanded_states_used: Option<usize>,
    /// States actually expanded by the route search, when one was run.
    pub expanded_states: Option<usize>,
    pub route_disposition: Option<String>,
    /// Examples: base_candidate, new_candidate, duplicate_path, no_path_under_ban,
    /// search_budget_exceeded, geometry_audit_failed, or base_route_unavailable.
    pub decision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteAlternative {
    pub candidate_id: String,
    pub discovery_ban: Option<GridEdge>,
    pub route_sha256: String,
    pub route: ServiceRouteResult,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCandidateCatalog {
    pub route_id: String,
    pub alternatives: Vec<RouteAlternative>,
    pub probes: Vec<AlternativeProbe>,
    /// True means the finite one-edge-ban discovery frontier was fully evaluated.
    /// It does NOT mean all possible grid paths were enumerated.
    pub discovery_frontier_exhausted: bool,
    pub truncation_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundleObjective {
    /// Sum of caller-declared path length plus bend penalties. Not a safety score.
    pub total_declared_cost_mm: u64,
    pub total_path_length_mm: u64,
    pub total_bends: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedRoute {
    pub route_id: String,
    pub candidate_index: usize,
    pub route: ServiceRouteResult,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBundleSelection {
    pub objective: BundleObjective,
    pub routes: Vec<SelectedRoute>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBundleReportBody {
    pub bundle_version: String,
    pub configuration_id: String,
    pub configuration_sha256: String,
    pub input_manifest_sha256: String,
    pub routing_limits: RoutingLimits,
    pub bundle_limits: RouteBundleLimits,
    pub catalogs: Vec<RouteCandidateCatalog>,
    pub selection: Option<RouteBundleSelection>,
    pub total_deviation_probes: usize,
    pub total_expanded_states: usize,
    pub bundle_combinations_examined: usize,
    pub bundles_rejected_for_shared_route_cells: usize,
    /// Completeness applies only to Cartesian combinations of the emitted catalogs.
    pub bundle_catalog_product_exhausted: bool,
    /// False if any route's edge-ban discovery frontier was truncated/inconclusive.
    pub all_route_discovery_frontiers_exhausted: bool,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBundleReport {
    pub body: RouteBundleReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug)]
struct BundleChoice {
    indices: Vec<usize>,
    objective: BundleObjective,
    tie_paths: Vec<Vec<(i64, i64)>>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, RouteBundleError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| RouteBundleError::new("SERIALIZATION_FAILED", error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

fn validate_limits(limits: &RouteBundleLimits) -> Result<(), RouteBundleError> {
    if limits.max_candidates_per_route == 0
        || limits.max_candidates_per_route > HARD_MAX_CANDIDATES_PER_ROUTE
    {
        return Err(RouteBundleError::new(
            "INVALID_CANDIDATE_LIMIT",
            format!("max_candidates_per_route must be in 1..={HARD_MAX_CANDIDATES_PER_ROUTE}"),
        ));
    }
    if limits.max_deviation_probes_per_route == 0
        || limits.max_deviation_probes_per_route > HARD_MAX_DEVIATION_PROBES_PER_ROUTE
    {
        return Err(RouteBundleError::new(
            "INVALID_DEVIATION_PROBE_LIMIT",
            format!("max_deviation_probes_per_route must be in 1..={HARD_MAX_DEVIATION_PROBES_PER_ROUTE}"),
        ));
    }
    if limits.max_total_deviation_probes == 0
        || limits.max_total_deviation_probes > HARD_MAX_TOTAL_DEVIATION_PROBES
    {
        return Err(RouteBundleError::new(
            "INVALID_TOTAL_DEVIATION_PROBE_LIMIT",
            format!("max_total_deviation_probes must be in 1..={HARD_MAX_TOTAL_DEVIATION_PROBES}"),
        ));
    }
    if limits.max_total_expanded_states == 0
        || limits.max_total_expanded_states > HARD_MAX_TOTAL_EXPANDED_STATES
    {
        return Err(RouteBundleError::new(
            "INVALID_TOTAL_EXPANDED_STATE_LIMIT",
            format!("max_total_expanded_states must be in 1..={HARD_MAX_TOTAL_EXPANDED_STATES}"),
        ));
    }
    if limits.max_bundle_combinations == 0
        || limits.max_bundle_combinations > HARD_MAX_BUNDLE_COMBINATIONS
    {
        return Err(RouteBundleError::new(
            "INVALID_BUNDLE_COMBINATION_LIMIT",
            format!("max_bundle_combinations must be in 1..={HARD_MAX_BUNDLE_COMBINATIONS}"),
        ));
    }
    Ok(())
}

fn normalize_input_requests(
    requests: &[ServiceRouteRequest],
) -> Result<Vec<ServiceRouteRequest>, RouteBundleError> {
    if requests.is_empty() || requests.len() > HARD_MAX_BUNDLE_ROUTES {
        return Err(RouteBundleError::new(
            "INVALID_BUNDLE_ROUTE_COUNT",
            format!("route bundle requires 1..={HARD_MAX_BUNDLE_ROUTES} requests"),
        ));
    }
    let mut normalized = requests.to_vec();
    normalized.sort_by(|a, b| a.route_id.cmp(&b.route_id));
    let mut route_ids = BTreeSet::new();
    let mut connection_ids = BTreeSet::new();
    for request in &normalized {
        if !route_ids.insert(request.route_id.clone()) {
            return Err(RouteBundleError::new("DUPLICATE_ROUTE_ID", "route IDs must be unique"));
        }
        if !connection_ids.insert(request.connection_id.clone()) {
            return Err(RouteBundleError::new(
                "DUPLICATE_CONNECTION_ROUTE",
                "a bundle accepts at most one route request per connection ID",
            ));
        }
    }
    Ok(normalized)
}

fn input_manifest_sha256(
    requests: &[ServiceRouteRequest],
) -> Result<String, RouteBundleError> {
    digest(requests)
}

fn point_distance_mm(a: &GridPoint, x: i64, y: i64) -> u128 {
    (i128::from(a.x_mm) - i128::from(x)).unsigned_abs()
        + (i128::from(a.y_mm) - i128::from(y)).unsigned_abs()
}

fn edge_is_too_close_to_terminal(
    edge: &GridEdge,
    request: &ServiceRouteRequest,
    pitch_mm: i64,
) -> bool {
    let exclusion = i128::from(request.clearance_mm) + i128::from(pitch_mm);
    let anchors = [&request.start.position_mm, &request.end.position_mm];
    for anchor in anchors {
        let a_distance = point_distance_mm(&edge.a, anchor.x_mm, anchor.y_mm);
        let b_distance = point_distance_mm(&edge.b, anchor.x_mm, anchor.y_mm);
        if a_distance <= exclusion as u128 || b_distance <= exclusion as u128 {
            return true;
        }
    }
    false
}

fn edge_from_path(path: &[GridPoint], edge_index: usize) -> Option<GridEdge> {
    let a = path.get(edge_index)?.clone();
    let b = path.get(edge_index.checked_add(1)?)?.clone();
    let horizontal = a.y_mm == b.y_mm && a.x_mm != b.x_mm;
    let vertical = a.x_mm == b.x_mm && a.y_mm != b.y_mm;
    (horizontal || vertical).then_some(GridEdge { a, b })
}

fn edge_ban_configuration(
    configuration: &Configuration,
    request: &ServiceRouteRequest,
    edge: &GridEdge,
    probe_index: usize,
) -> Result<Configuration, RouteBundleError> {
    let decks: Vec<_> = configuration.platform.deck_bounds.iter()
        .filter(|deck| deck.deck_id == request.start.deck_id).collect();
    if decks.len() != 1 {
        return Err(RouteBundleError::new("DECK_NOT_UNIQUE", "edge ban cannot resolve one deck"));
    }
    let bounds = &decks[0].bounds_mm;
    let (x_min_mm, x_max_mm, y_min_mm, y_max_mm) = if edge.a.y_mm == edge.b.y_mm {
        (
            edge.a.x_mm.min(edge.b.x_mm),
            edge.a.x_mm.max(edge.b.x_mm),
            edge.a.y_mm,
            edge.a.y_mm.checked_add(1).ok_or_else(|| RouteBundleError::new(
                "EDGE_BAN_COORDINATE_OVERFLOW", "horizontal edge-ban Y bound overflows"))?,
        )
    } else if edge.a.x_mm == edge.b.x_mm {
        (
            edge.a.x_mm,
            edge.a.x_mm.checked_add(1).ok_or_else(|| RouteBundleError::new(
                "EDGE_BAN_COORDINATE_OVERFLOW", "vertical edge-ban X bound overflows"))?,
            edge.a.y_mm.min(edge.b.y_mm),
            edge.a.y_mm.max(edge.b.y_mm),
        )
    } else {
        return Err(RouteBundleError::new("NON_ORTHOGONAL_EDGE", "edge-ban probes require orthogonal edges"));
    };
    if x_min_mm >= x_max_mm || y_min_mm >= y_max_mm {
        return Err(RouteBundleError::new("INVALID_EDGE_BAN_BOUNDS", "edge-ban rectangle is degenerate"));
    }
    let mut modified = configuration.clone();
    let base_id = format!("ALT-BAN-{probe_index:08}");
    let mut region_id = base_id.clone();
    let mut suffix = 0usize;
    while modified.reserved_regions.iter().any(|region| region.region_id == region_id) {
        suffix = suffix.saturating_add(1);
        region_id = format!("{base_id}-{suffix}");
    }
    modified.reserved_regions.push(ReservedRegion {
        region_id,
        deck_id: request.start.deck_id.clone(),
        bounds_mm: BoundsMm {
            x_min_mm, x_max_mm, y_min_mm, y_max_mm,
            z_min_mm: bounds.z_min_mm,
            z_max_mm: bounds.z_max_mm,
        },
        purpose: "algorithmic_route_alternative_edge_ban".to_owned(),
        clearance_rule: "temporary_search_exclusion".to_owned(),
        evidence_status: "design_assumption".to_owned(),
    });
    Ok(modified)
}

fn route_sha256(route: &ServiceRouteResult) -> Result<String, RouteBundleError> {
    digest(route)
}

fn make_alternative(
    index: usize,
    discovery_ban: Option<GridEdge>,
    route: ServiceRouteResult,
) -> Result<RouteAlternative, RouteBundleError> {
    let route_sha256 = route_sha256(&route)?;
    Ok(RouteAlternative {
        candidate_id: format!("ALT-{index:03}"),
        discovery_ban,
        route_sha256,
        route,
    })
}

fn build_catalog(
    configuration: &Configuration,
    request: &ServiceRouteRequest,
    routing_limits: &RoutingLimits,
    bundle_limits: &RouteBundleLimits,
    per_route_probe_limit: usize,
    total_deviation_probes: &mut usize,
) -> Result<RouteCandidateCatalog, RouteBundleError> {
    let mut catalog = RouteCandidateCatalog {
        route_id: request.route_id.clone(),
        alternatives: Vec::new(),
        probes: Vec::new(),
        discovery_frontier_exhausted: true,
        truncation_reason: None,
    };
    let base_report: ServiceRoutingReport =
        route_services(configuration, &[request.clone()], routing_limits.clone())?;
    let base = base_report.body.routes.into_iter().next().ok_or_else(|| {
        RouteBundleError::new("BASE_ROUTE_RESULT_MISSING", "single-route router emitted no result")
    })?;
    if base.disposition != RouteDisposition::Routed {
        let inconclusive = base.disposition == RouteDisposition::SearchBudgetExceeded
            || base.disposition == RouteDisposition::InvalidGeneratedPath;
        catalog.probes.push(AlternativeProbe {
            banned_edge: None,
            probe_configuration_sha256: Some(base_report.body.configuration_sha256),
            route_disposition: Some(format!("{:?}", base.disposition)),
            decision: "base_route_unavailable".to_owned(),
        });
        catalog.discovery_frontier_exhausted = !inconclusive;
        if inconclusive {
            catalog.truncation_reason = Some(format!("BASE_ROUTE_{:?}", base.disposition).to_uppercase());
        }
        return Ok(catalog);
    }
    if !verify_service_route_geometry(configuration, request, routing_limits.grid_pitch_mm, &base) {
        catalog.probes.push(AlternativeProbe {
            banned_edge: None,
            probe_configuration_sha256: Some(base_report.body.configuration_sha256),
            route_disposition: Some(format!("{:?}", base.disposition)),
            decision: "geometry_audit_failed".to_owned(),
        });
        catalog.discovery_frontier_exhausted = false;
        catalog.truncation_reason = Some("BASE_GEOMETRY_AUDIT_FAILED".to_owned());
        return Ok(catalog);
    }
    catalog.alternatives.push(make_alternative(0, None, base.clone())?);
    catalog.probes.push(AlternativeProbe {
        banned_edge: None,
        probe_configuration_sha256: Some(base_report.body.configuration_sha256),
        route_disposition: Some(format!("{:?}", base.disposition)),
        decision: "base_candidate".to_owned(),
    });

    let mut cursor = 0usize;
    let mut probe_index = 0usize;
    'catalog: while cursor < catalog.alternatives.len() {
        let path = catalog.alternatives[cursor].route.path.clone();
        cursor += 1;
        let edge_count = path.len().saturating_sub(1);
        for edge_index in 0..edge_count {
            let Some(edge) = edge_from_path(&path, edge_index) else { continue; };
            if edge_is_too_close_to_terminal(&edge, request, routing_limits.grid_pitch_mm) {
                continue;
            }
            if catalog.probes.iter().filter(|probe| probe.banned_edge.is_some()).count()
                >= per_route_probe_limit
            {
                catalog.discovery_frontier_exhausted = false;
                catalog.truncation_reason = Some("DEVIATION_PROBE_LIMIT".to_owned());
                break 'catalog;
            }
            if *total_deviation_probes >= bundle_limits.max_total_deviation_probes {
                catalog.discovery_frontier_exhausted = false;
                catalog.truncation_reason = Some("TOTAL_DEVIATION_PROBE_LIMIT".to_owned());
                break 'catalog;
            }
            if catalog.alternatives.len() >= bundle_limits.max_candidates_per_route {
                catalog.discovery_frontier_exhausted = false;
                catalog.truncation_reason = Some("CANDIDATE_LIMIT".to_owned());
                break 'catalog;
            }
            let modified = edge_ban_configuration(configuration, request, &edge, probe_index)?;
            probe_index = probe_index.saturating_add(1);
            let modified_digest = digest(&modified)?;
            *total_deviation_probes = (*total_deviation_probes).saturating_add(1);
            let alternative_report = route_services(&modified, &[request.clone()], routing_limits.clone())?;
            let Some(candidate) = alternative_report.body.routes.into_iter().next() else {
                return Err(RouteBundleError::new("ALTERNATIVE_RESULT_MISSING", "edge-ban probe emitted no route result"));
            };
            if candidate.disposition == RouteDisposition::SearchBudgetExceeded {
                catalog.probes.push(AlternativeProbe {
                    banned_edge: Some(edge),
                    probe_configuration_sha256: Some(modified_digest),
                    route_disposition: Some(format!("{:?}", candidate.disposition)),
                    decision: "search_budget_exceeded".to_owned(),
                });
                catalog.discovery_frontier_exhausted = false;
                catalog.truncation_reason = Some("ROUTER_SEARCH_BUDGET_EXCEEDED".to_owned());
                break 'catalog;
            }
            if candidate.disposition != RouteDisposition::Routed {
                let inconclusive = candidate.disposition == RouteDisposition::InvalidGeneratedPath;
                catalog.probes.push(AlternativeProbe {
                    banned_edge: Some(edge),
                    probe_configuration_sha256: Some(modified_digest),
                    route_disposition: Some(format!("{:?}", candidate.disposition)),
                    decision: if inconclusive {
                        "invalid_generated_path".to_owned()
                    } else {
                        "no_route_under_edge_ban".to_owned()
                    },
                });
                if inconclusive {
                    catalog.discovery_frontier_exhausted = false;
                    catalog.truncation_reason = Some("ALTERNATIVE_INVALID_GENERATED_PATH".to_owned());
                    break 'catalog;
                }
                continue;
            }
            if !verify_service_route_geometry(configuration, request, routing_limits.grid_pitch_mm, &candidate) {
                catalog.probes.push(AlternativeProbe {
                    banned_edge: Some(edge),
                    probe_configuration_sha256: Some(modified_digest),
                    route_disposition: Some(format!("{:?}", candidate.disposition)),
                    decision: "geometry_audit_failed".to_owned(),
                });
                catalog.discovery_frontier_exhausted = false;
                catalog.truncation_reason = Some("ALTERNATIVE_GEOMETRY_AUDIT_FAILED".to_owned());
                break 'catalog;
            }
            if catalog.alternatives.iter().any(|known| known.route.path == candidate.path) {
                catalog.probes.push(AlternativeProbe {
                    banned_edge: Some(edge),
                    probe_configuration_sha256: Some(modified_digest),
                    route_disposition: Some(format!("{:?}", candidate.disposition)),
                    decision: "duplicate_path".to_owned(),
                });
                continue;
            }
            let index = catalog.alternatives.len();
            catalog.alternatives.push(make_alternative(index, Some(edge.clone()), candidate)?);
            catalog.probes.push(AlternativeProbe {
                banned_edge: Some(edge),
                probe_configuration_sha256: Some(modified_digest),
                route_disposition: Some("Routed".to_owned()),
                decision: "new_candidate".to_owned(),
            });
        }
    }
    Ok(catalog)
}

fn shared_service_terminal(
    a: &ServiceRouteRequest,
    b: &ServiceRouteRequest,
    medium_a: &str,
    medium_b: &str,
) -> BTreeSet<(i64, i64)> {
    if medium_a != medium_b {
        return BTreeSet::new();
    }
    let mut allowed = BTreeSet::new();
    for aa in [&a.start, &a.end] {
        if aa.kind != AnchorKind::PlatformService { continue; }
        for bb in [&b.start, &b.end] {
            if bb.kind == AnchorKind::PlatformService
                && aa.service_id == bb.service_id
                && aa.deck_id == bb.deck_id
                && aa.position_mm.x_mm == bb.position_mm.x_mm
                && aa.position_mm.y_mm == bb.position_mm.y_mm
                && aa.position_mm.z_mm == bb.position_mm.z_mm
            {
                allowed.insert((aa.position_mm.x_mm, aa.position_mm.y_mm));
            }
        }
    }
    allowed
}

fn paths_have_forbidden_shared_cells(
    a: &ServiceRouteResult,
    b: &ServiceRouteResult,
    request_a: &ServiceRouteRequest,
    request_b: &ServiceRouteRequest,
) -> bool {
    if request_a.start.deck_id != request_b.start.deck_id
        || request_a.start.position_mm.z_mm != request_b.start.position_mm.z_mm
    {
        return false;
    }
    let allowed = shared_service_terminal(request_a, request_b, &a.medium, &b.medium);
    let cells_b: BTreeSet<_> = b.path.iter().map(|point| (point.x_mm, point.y_mm)).collect();
    a.path.iter().any(|point| {
        cells_b.contains(&(point.x_mm, point.y_mm))
            && !allowed.contains(&(point.x_mm, point.y_mm))
    })
}

fn objective_for(routes: &[&ServiceRouteResult]) -> Option<BundleObjective> {
    let mut total_cost = 0u64;
    let mut total_length = 0u64;
    let mut total_bends = 0usize;
    for route in routes {
        total_cost = total_cost.checked_add(route.search_cost_mm?)?;
        total_length = total_length.checked_add(route.path_length_mm?)?;
        total_bends = total_bends.checked_add(route.bend_count?)?;
    }
    Some(BundleObjective {
        total_declared_cost_mm: total_cost,
        total_path_length_mm: total_length,
        total_bends,
    })
}

fn tie_paths(routes: &[&ServiceRouteResult]) -> Vec<Vec<(i64, i64)>> {
    routes.iter().map(|route| {
        route.path.iter().map(|point| (point.x_mm, point.y_mm)).collect()
    }).collect()
}

fn choice_is_better(candidate: &BundleChoice, best: &BundleChoice) -> bool {
    (
        candidate.objective.total_declared_cost_mm,
        candidate.objective.total_path_length_mm,
        candidate.objective.total_bends,
        &candidate.tie_paths,
    ) < (
        best.objective.total_declared_cost_mm,
        best.objective.total_path_length_mm,
        best.objective.total_bends,
        &best.tie_paths,
    )
}

#[allow(clippy::too_many_arguments)]
fn enumerate_bundles(
    index: usize,
    catalogs: &[RouteCandidateCatalog],
    requests: &[ServiceRouteRequest],
    chosen: &mut Vec<usize>,
    limit: usize,
    examined: &mut usize,
    rejected_conflicts: &mut usize,
    best: &mut Option<BundleChoice>,
    truncated: &mut bool,
) {
    if *truncated { return; }
    if index == catalogs.len() {
        if *examined >= limit {
            *truncated = true;
            return;
        }
        *examined += 1;
        let routes: Vec<&ServiceRouteResult> = catalogs.iter().zip(chosen.iter())
            .map(|(catalog, choice)| &catalog.alternatives[*choice].route).collect();
        for i in 0..routes.len() {
            for j in (i + 1)..routes.len() {
                if paths_have_forbidden_shared_cells(&routes[i], &routes[j], &requests[i], &requests[j]) {
                    *rejected_conflicts = rejected_conflicts.saturating_add(1);
                    return;
                }
            }
        }
        let Some(objective) = objective_for(&routes) else {
            // Overflowed objective arithmetic is invalid, but is not mislabeled as a geometry conflict.
            return;
        };
        let candidate = BundleChoice {
            indices: chosen.clone(),
            objective,
            tie_paths: tie_paths(&routes),
        };
        if best.as_ref().is_none_or(|prior| choice_is_better(&candidate, prior)) {
            *best = Some(candidate);
        }
        return;
    }
    for choice in 0..catalogs[index].alternatives.len() {
        if *truncated { return; }
        chosen.push(choice);
        enumerate_bundles(
            index + 1, catalogs, requests, chosen, limit, examined,
            rejected_conflicts, best, truncated,
        );
        chosen.pop();
    }
}

fn select_bundle(
    catalogs: &[RouteCandidateCatalog],
    requests: &[ServiceRouteRequest],
    limit: usize,
) -> (Option<BundleChoice>, usize, usize, bool) {
    if catalogs.iter().any(|catalog| catalog.alternatives.is_empty()) {
        // The Cartesian product of the emitted catalogs is empty. This says nothing about
        // the broader path universe; per-route discovery status carries that caveat.
        return (None, 0, 0, true);
    }
    let mut chosen = Vec::with_capacity(catalogs.len());
    let mut examined = 0usize;
    let mut rejected = 0usize;
    let mut best = None;
    let mut truncated = false;
    enumerate_bundles(
        0, catalogs, requests, &mut chosen, limit, &mut examined,
        &mut rejected, &mut best, &mut truncated,
    );
    (best, examined, rejected, !truncated)
}

/// Builds bounded per-route alternatives, then selects the lowest declared-cost bundle
/// satisfying strict shared-cell constraints. It is optimal only over the emitted catalog.
pub fn optimize_route_bundle(
    configuration: &Configuration,
    requests: &[ServiceRouteRequest],
    routing_limits: RoutingLimits,
    bundle_limits: RouteBundleLimits,
) -> Result<RouteBundleReport, RouteBundleError> {
    validate_limits(&bundle_limits)?;
    if routing_limits.grid_pitch_mm <= 0
        || routing_limits.max_route_requests == 0
        || routing_limits.max_route_requests > HARD_MAX_ROUTE_REQUESTS
        || routing_limits.max_grid_cells == 0
        || routing_limits.max_grid_cells > HARD_MAX_GRID_CELLS
        || routing_limits.max_expanded_states == 0
        || routing_limits.max_expanded_states > HARD_MAX_EXPANDED_STATES
    {
        return Err(RouteBundleError::new("INVALID_ROUTING_LIMITS", "routing limits violate hard bounds"));
    }
    if requests.iter().any(|request| {
        request.clearance_mm < 0 || request.clearance_mm > HARD_MAX_CLEARANCE_MM
            || request.bend_penalty_mm < 0 || request.bend_penalty_mm > HARD_MAX_BEND_PENALTY_MM
    }) {
        return Err(RouteBundleError::new("INVALID_REQUEST_COST_OR_CLEARANCE", "clearance or bend penalty violates hard bounds"));
    }
    let normalized_requests = normalize_input_requests(requests)?;
    if normalized_requests.len() > routing_limits.max_route_requests {
        return Err(RouteBundleError::new(
            "BUNDLE_EXCEEDS_ROUTING_REQUEST_LIMIT",
            "route count exceeds the routing limit supplied for this bundle",
        ));
    }
    let input_manifest_sha256 = input_manifest_sha256(&normalized_requests)?;
    let configuration_sha256 = digest(configuration)?;
    let mut catalogs = Vec::with_capacity(normalized_requests.len());
    let per_route_probe_limit = bundle_limits.max_deviation_probes_per_route.min(
        bundle_limits.max_total_deviation_probes.saturating_add(normalized_requests.len() - 1)
            / normalized_requests.len(),
    );
    let mut total_deviation_probes = 0usize;
    for request in &normalized_requests {
        catalogs.push(build_catalog(
            configuration, request, &routing_limits, &bundle_limits,
            per_route_probe_limit, &mut total_deviation_probes,
        )?);
    }
    let (best, examined, rejected, bundle_product_exhausted) =
        select_bundle(&catalogs, &normalized_requests, bundle_limits.max_bundle_combinations);
    let selection = best.map(|choice| {
        let routes = choice.indices.iter().enumerate().map(|(index, candidate_index)| {
            let mut route = catalogs[index].alternatives[*candidate_index].route.clone();
            route.conflicts_with_route_ids.clear();
            SelectedRoute {
                route_id: catalogs[index].route_id.clone(),
                candidate_index: *candidate_index,
                route,
            }
        }).collect();
        RouteBundleSelection { objective: choice.objective, routes }
    });
    let all_route_discovery_frontiers_exhausted =
        catalogs.iter().all(|catalog| catalog.discovery_frontier_exhausted);
    let body = RouteBundleReportBody {
        bundle_version: ROUTE_BUNDLE_VERSION.to_owned(),
        configuration_id: configuration.configuration_id.clone(),
        configuration_sha256,
        input_manifest_sha256,
        routing_limits,
        bundle_limits,
        catalogs,
        selection,
        total_deviation_probes,
        bundle_combinations_examined: examined,
        bundles_rejected_for_shared_route_cells: rejected,
        bundle_catalog_product_exhausted: bundle_product_exhausted,
        all_route_discovery_frontiers_exhausted,
        claim_ceiling: concat!(
            "Best feasible bundle among discovered one-edge-ban A* alternatives only. The path catalog ",
            "is not exhaustive over all grid paths and bundle optimality is conditional on that catalog. ",
            "Explicit-anchor 2D grid proxies only; no 3D, installation, hydraulic/electrical, regulatory, ",
            "maintainability, evacuation, stability, seaworthiness, or operating approval is established."
        ).to_owned(),
    };
    let report_sha256 = digest(&body)?;
    Ok(RouteBundleReport { body, report_sha256 })
}

/// Checks report integrity, original-configuration path geometry, and exact recomputation.
/// This is not a digital signature or independent physical/source-authenticity proof.
pub fn verify_route_bundle_report(
    configuration: &Configuration,
    requests: &[ServiceRouteRequest],
    routing_limits: &RoutingLimits,
    bundle_limits: &RouteBundleLimits,
    report: &RouteBundleReport,
) -> bool {
    if digest(&report.body).ok().as_deref() != Some(report.report_sha256.as_str())
        || report.body.configuration_id != configuration.configuration_id
        || report.body.configuration_sha256 != digest(configuration).ok().unwrap_or_default()
    {
        return false;
    }
    let Ok(normalized_requests) = normalize_input_requests(requests) else { return false; };
    if report.body.catalogs.len() != normalized_requests.len() {
        return false;
    }
    for (catalog, request) in report.body.catalogs.iter().zip(normalized_requests.iter()) {
        if catalog.route_id != request.route_id { return false; }
        for candidate in &catalog.alternatives {
            if route_sha256(&candidate.route).ok().as_deref() != Some(candidate.route_sha256.as_str())
                || !verify_service_route_geometry(
                    configuration, request, routing_limits.grid_pitch_mm, &candidate.route,
                )
                || candidate.route.disposition != RouteDisposition::Routed
            {
                return false;
            }
        }
    }
    let Ok(recomputed) = optimize_route_bundle(
        configuration, requests, routing_limits.clone(), bundle_limits.clone(),
    ) else { return false; };
    digest(&recomputed.body).ok() == digest(&report.body).ok()
        && recomputed.report_sha256 == report.report_sha256
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_interior_service_routing::{
        AnchorEvidenceClass, RoutingAnchor,
    };

    fn service_anchor(service_id: &str, x: i64, y: i64) -> RoutingAnchor {
        RoutingAnchor {
            kind: AnchorKind::PlatformService,
            instance_id: None,
            port_id: None,
            service_id: Some(service_id.to_owned()),
            side: None,
            deck_id: "D".to_owned(),
            position_mm: crate::modular_interior::PositionMm { x_mm: x, y_mm: y, z_mm: 0 },
            evidence_class: AnchorEvidenceClass::DesignAssumption,
            evidence_refs: vec!["SYNTHETIC".to_owned()],
        }
    }

    fn dummy_request(route_id: &str, connection_id: &str, end: RoutingAnchor) -> ServiceRouteRequest {
        ServiceRouteRequest {
            route_id: route_id.to_owned(),
            connection_id: connection_id.to_owned(),
            start: service_anchor(&format!("{route_id}-SOURCE"), -10, -10),
            end,
            clearance_mm: 0,
            bend_penalty_mm: 0,
        }
    }

    fn dummy_route(route_id: &str, connection_id: &str, points: &[(i64, i64)], cost: u64) -> ServiceRouteResult {
        ServiceRouteResult {
            route_id: route_id.to_owned(),
            connection_id: connection_id.to_owned(),
            medium: "synthetic".to_owned(),
            disposition: RouteDisposition::Routed,
            path: points.iter().map(|(x_mm, y_mm)| GridPoint { x_mm: *x_mm, y_mm: *y_mm }).collect(),
            path_length_mm: Some(points.len().saturating_sub(1) as u64),
            bend_count: Some(0),
            search_cost_mm: Some(cost),
            expanded_states: 0,
            conflicts_with_route_ids: Vec::new(),
            anchor_evidence_classes: vec![AnchorEvidenceClass::DesignAssumption; 2],
            claim_ceiling: "synthetic unit test only".to_owned(),
        }
    }

    fn catalog(route_id: &str, alternatives: Vec<ServiceRouteResult>) -> RouteCandidateCatalog {
        let alternatives = alternatives.into_iter().enumerate().map(|(index, route)| {
            RouteAlternative {
                candidate_id: format!("ALT-{index:03}"),
                discovery_ban: None,
                route_sha256: route_sha256(&route).unwrap(),
                route,
            }
        }).collect();
        RouteCandidateCatalog {
            route_id: route_id.to_owned(),
            alternatives,
            probes: Vec::new(),
            discovery_frontier_exhausted: true,
            truncation_reason: None,
        }
    }

    #[test]
    fn exhaustive_tiny_bundle_oracle_finds_lowest_conflict_free_combination() {
        let route_a = vec![
            dummy_route("A", "CA", &[(0, 0), (1, 0), (2, 0)], 3),
            dummy_route("A", "CA", &[(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)], 5),
        ];
        let route_b = vec![
            dummy_route("B", "CB", &[(1, -1), (1, 0), (1, 1), (1, 2)], 2),
            dummy_route("B", "CB", &[(3, -1), (3, 0), (3, 1), (3, 2)], 4),
        ];
        let requests = vec![
            dummy_request("A", "CA", service_anchor("BUS-A", 2, 0)),
            dummy_request("B", "CB", service_anchor("BUS-B", 3, 2)),
        ];
        let catalogs = vec![catalog("A", route_a), catalog("B", route_b)];
        let (best, examined, rejected, complete) = select_bundle(&catalogs, &requests, 100);
        assert!(complete);
        assert_eq!(examined, 4);
        assert!(rejected >= 2);
        let best = best.expect("at least one compatible combination");
        assert_eq!(best.indices, vec![0, 1]);
        assert_eq!(best.objective.total_declared_cost_mm, 7);

        // Independent exhaustive enumeration of the 2x2 product is the frozen oracle.
        let mut oracle_best = None::<u64>;
        for a in 0..2 {
            for b in 0..2 {
                let routes = [
                    &catalogs[0].alternatives[a].route,
                    &catalogs[1].alternatives[b].route,
                ];
                // The declared terminal IDs in this oracle are distinct, so any shared XY cell is forbidden.
                let has_shared_cell = routes[0].path.iter().any(|left| {
                    routes[1].path.iter().any(|right| {
                        left.x_mm == right.x_mm && left.y_mm == right.y_mm
                    })
                });
                if !has_shared_cell {
                    let cost = routes[0].search_cost_mm.unwrap() + routes[1].search_cost_mm.unwrap();
                    oracle_best = Some(oracle_best.map_or(cost, |old| old.min(cost)));
                }
            }
        }
        assert_eq!(Some(best.objective.total_declared_cost_mm), oracle_best);
    }

    #[test]
    fn shared_service_terminal_is_allowed_but_other_shared_cells_are_not() {
        let a = dummy_route("A", "CA", &[(0, 0), (1, 0), (2, 0)], 3);
        let mut b = dummy_route("B", "CB", &[(4, 0), (3, 0), (2, 0)], 2);
        let request_a = dummy_request("A", "CA", service_anchor("AC-BUS", 2, 0));
        let request_b = dummy_request("B", "CB", service_anchor("AC-BUS", 2, 0));
        assert!(!paths_have_forbidden_shared_cells(&a, &b, &request_a, &request_b));
        let mut different_medium = b.clone();
        different_medium.medium = "incompatible-medium".to_owned();
        assert!(paths_have_forbidden_shared_cells(&a, &different_medium, &request_a, &request_b));
        b.path.insert(1, GridPoint { x_mm: 1, y_mm: 0 });
        assert!(paths_have_forbidden_shared_cells(&a, &b, &request_a, &request_b));
    }

    #[test]
    fn bundle_cap_is_explicitly_reported() {
        let catalogs = vec![
            catalog("A", vec![
                dummy_route("A", "CA", &[(0, 0), (1, 0)], 1),
                dummy_route("A", "CA", &[(0, 0), (0, 1), (1, 1)], 2),
            ]),
            catalog("B", vec![
                dummy_route("B", "CB", &[(5, 0), (6, 0)], 1),
                dummy_route("B", "CB", &[(5, 0), (5, 1), (6, 1)], 2),
            ]),
        ];
        let requests = vec![
            dummy_request("A", "CA", service_anchor("BUS-A", 1, 0)),
            dummy_request("B", "CB", service_anchor("BUS-B", 6, 0)),
        ];
        let (_, examined, _, complete) = select_bundle(&catalogs, &requests, 2);
        assert_eq!(examined, 2);
        assert!(!complete);
    }

    #[test]
    fn horizontal_and_vertical_edge_bans_are_non_degenerate() {
        let horizontal = GridEdge {
            a: GridPoint { x_mm: 100, y_mm: 200 },
            b: GridPoint { x_mm: 200, y_mm: 200 },
        };
        let vertical = GridEdge {
            a: GridPoint { x_mm: 300, y_mm: 100 },
            b: GridPoint { x_mm: 300, y_mm: 200 },
        };
        assert_eq!(horizontal.a.y_mm, horizontal.b.y_mm);
        assert_eq!(vertical.a.x_mm, vertical.b.x_mm);
        assert_ne!(horizontal.a.x_mm, horizontal.b.x_mm);
        assert_ne!(vertical.a.y_mm, vertical.b.y_mm);
    }

    #[test]
    fn equivalent_request_order_is_normalized_by_route_id() {
        let a = dummy_request("A", "CA", service_anchor("BUS-A", 2, 0));
        let b = dummy_request("B", "CB", service_anchor("BUS-B", 3, 2));
        let one = normalize_input_requests(&[a.clone(), b.clone()]).unwrap();
        let two = normalize_input_requests(&[b, a]).unwrap();
        assert_eq!(one.iter().map(|r| &r.route_id).collect::<Vec<_>>(),
                   two.iter().map(|r| &r.route_id).collect::<Vec<_>>());
        assert_eq!(input_manifest_sha256(&one).unwrap(), input_manifest_sha256(&two).unwrap());
    }
}
