// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic facility-zone graph evaluation.
//!
//! This module evaluates declared adjacency, movement edges and flow demands.
//! It does not infer geometry, qualify machines, approve egress or certify a
//! building. Distances are explicit integer millimetres with source references.
//! Synthetic, measured and imported distance evidence remain distinguishable.

use crate::graph::{dijkstra_u64, DijkstraError};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlowMode {
    Person,
    NewMaterial,
    WorkInProgress,
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DistanceBasis {
    Synthetic,
    Measured,
    Imported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zone {
    pub zone_id: String,
    pub functions: Vec<String>,
    /// Return flows may not cross this zone.
    pub new_material_only: bool,
    /// New-material and WIP flows may not enter this zone.
    pub returns_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ZonePair {
    pub first_zone_id: String,
    pub second_zone_id: String,
}

impl ZonePair {
    pub fn new(first: impl Into<String>, second: impl Into<String>) -> Self {
        let first = first.into();
        let second = second.into();
        if first <= second {
            Self { first_zone_id: first, second_zone_id: second }
        } else {
            Self { first_zone_id: second, second_zone_id: first }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TravelEdge {
    pub edge_id: String,
    pub from_zone_id: String,
    pub to_zone_id: String,
    pub distance_mm: u64,
    pub distance_source_ref: String,
    pub distance_basis: DistanceBasis,
    pub permitted_modes: Vec<FlowMode>,
    /// Protected egress edges may only carry person movement.
    pub protected_egress: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowDemand {
    pub flow_id: String,
    pub from_zone_id: String,
    pub to_zone_id: String,
    pub mode: FlowMode,
    pub units_per_period: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInput {
    pub layout_id: String,
    pub zones: Vec<Zone>,
    pub travel_edges: Vec<TravelEdge>,
    pub flows: Vec<FlowDemand>,
    pub required_functions: Vec<String>,
    /// Declared adjacencies are model assertions, not surveyed physical truth.
    pub declared_adjacencies: Vec<ZonePair>,
    pub required_adjacencies: Vec<ZonePair>,
    pub forbidden_adjacencies: Vec<ZonePair>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathEdgeSource {
    pub edge_id: String,
    pub source_ref: String,
    pub basis: DistanceBasis,
    pub distance_mm: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowMetric {
    pub flow_id: String,
    pub mode: FlowMode,
    pub units_per_period: u64,
    pub path_zone_ids: Vec<String>,
    pub path_edges: Vec<PathEdgeSource>,
    pub path_distance_mm: u128,
    pub weighted_travel_mm_per_period: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutReport {
    pub layout_id: String,
    pub zone_count: usize,
    pub edge_count: usize,
    pub flow_count: usize,
    pub flow_metrics: Vec<FlowMetric>,
    pub weighted_travel_mm_by_mode: BTreeMap<FlowMode, u128>,
    pub total_weighted_travel_mm_per_period: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    EmptyLayoutId,
    NoZones,
    EmptyIdentifier { kind: &'static str },
    DuplicateZoneId { zone_id: String },
    DuplicateZoneFunction { zone_id: String, function: String },
    ConflictingZoneUse { zone_id: String },
    DuplicateRequiredFunction { function: String },
    MissingRequiredFunction { function: String },
    UnknownZoneReference { context: &'static str, zone_id: String },
    SelfAdjacency { zone_id: String },
    DuplicateAdjacency { kind: &'static str, first_zone_id: String, second_zone_id: String },
    ConflictingAdjacency { first_zone_id: String, second_zone_id: String },
    MissingRequiredAdjacency { first_zone_id: String, second_zone_id: String },
    ForbiddenAdjacencyPresent { first_zone_id: String, second_zone_id: String },
    DuplicateEdgeId { edge_id: String },
    InvalidEdgeDistance { edge_id: String },
    MissingDistanceSource { edge_id: String },
    NoPermittedModes { edge_id: String },
    DuplicatePermittedMode { edge_id: String, mode: FlowMode },
    ProtectedEgressAllowsNonPerson { edge_id: String },
    DuplicateFlowId { flow_id: String },
    ZeroFlowUnits { flow_id: String },
    FlowHasNoFeasiblePath { flow_id: String, from_zone_id: String, to_zone_id: String, mode: FlowMode },
    InvalidPathTree { flow_id: String },
    ArithmeticOverflow { flow_id: String },
    ShortestPath(DijkstraError),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LayoutError {}

/// Evaluate declared facility zones, adjacency and typed travel flows.
/// Input vector ordering does not affect the deterministic report.
pub fn evaluate_layout(input: &LayoutInput) -> Result<LayoutReport, LayoutError> {
    if input.layout_id.trim().is_empty() {
        return Err(LayoutError::EmptyLayoutId);
    }
    if input.zones.is_empty() {
        return Err(LayoutError::NoZones);
    }

    let mut zones: Vec<&Zone> = input.zones.iter().collect();
    zones.sort_by(|a, b| a.zone_id.cmp(&b.zone_id));
    let mut zone_index = BTreeMap::new();

    for (index, zone) in zones.iter().enumerate() {
        nonempty("zone id", &zone.zone_id)?;
        if zone_index.insert(zone.zone_id.as_str(), index).is_some() {
            return Err(LayoutError::DuplicateZoneId { zone_id: zone.zone_id.clone() });
        }
        if zone.new_material_only && zone.returns_only {
            return Err(LayoutError::ConflictingZoneUse { zone_id: zone.zone_id.clone() });
        }
        let mut functions = BTreeSet::new();
        for function in &zone.functions {
            nonempty("zone function", function)?;
            if !functions.insert(function.as_str()) {
                return Err(LayoutError::DuplicateZoneFunction {
                    zone_id: zone.zone_id.clone(),
                    function: function.clone(),
                });
            }
        }
    }

    validate_functions(input, &zones)?;
    validate_adjacencies(input, &zone_index)?;

    let mut edges: Vec<&TravelEdge> = input.travel_edges.iter().collect();
    edges.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));
    let mut edge_ids = BTreeSet::new();

    for edge in &edges {
        nonempty("edge id", &edge.edge_id)?;
        if !edge_ids.insert(edge.edge_id.as_str()) {
            return Err(LayoutError::DuplicateEdgeId { edge_id: edge.edge_id.clone() });
        }
        require_zone(&zone_index, "travel edge source", &edge.from_zone_id)?;
        require_zone(&zone_index, "travel edge target", &edge.to_zone_id)?;
        if edge.from_zone_id == edge.to_zone_id || edge.distance_mm == 0 {
            return Err(LayoutError::InvalidEdgeDistance { edge_id: edge.edge_id.clone() });
        }
        if edge.distance_source_ref.trim().is_empty() {
            return Err(LayoutError::MissingDistanceSource { edge_id: edge.edge_id.clone() });
        }
        if edge.permitted_modes.is_empty() {
            return Err(LayoutError::NoPermittedModes { edge_id: edge.edge_id.clone() });
        }
        let mut modes = BTreeSet::new();
        for mode in &edge.permitted_modes {
            if !modes.insert(*mode) {
                return Err(LayoutError::DuplicatePermittedMode {
                    edge_id: edge.edge_id.clone(),
                    mode: *mode,
                });
            }
            if edge.protected_egress && *mode != FlowMode::Person {
                return Err(LayoutError::ProtectedEgressAllowsNonPerson {
                    edge_id: edge.edge_id.clone(),
                });
            }
        }
    }

    let mut flows: Vec<&FlowDemand> = input.flows.iter().collect();
    flows.sort_by(|a, b| a.flow_id.cmp(&b.flow_id));
    let mut flow_ids = BTreeSet::new();
    for flow in &flows {
        nonempty("flow id", &flow.flow_id)?;
        if !flow_ids.insert(flow.flow_id.as_str()) {
            return Err(LayoutError::DuplicateFlowId { flow_id: flow.flow_id.clone() });
        }
        require_zone(&zone_index, "flow source", &flow.from_zone_id)?;
        require_zone(&zone_index, "flow target", &flow.to_zone_id)?;
        if flow.units_per_period == 0 {
            return Err(LayoutError::ZeroFlowUnits { flow_id: flow.flow_id.clone() });
        }
    }

    let mut flow_metrics = Vec::with_capacity(flows.len());
    let mut weighted_travel_mm_by_mode = BTreeMap::<FlowMode, u128>::new();
    let mut total_weighted_travel_mm_per_period = 0_u128;

    for flow in &flows {
        let (path_zone_ids, path_edges, path_distance_mm) =
            shortest_path_for_flow(flow, &zones, &zone_index, &edges)?;
        let weighted = path_distance_mm
            .checked_mul(u128::from(flow.units_per_period))
            .ok_or_else(|| LayoutError::ArithmeticOverflow { flow_id: flow.flow_id.clone() })?;
        let mode_total = weighted_travel_mm_by_mode.entry(flow.mode).or_insert(0);
        *mode_total = mode_total
            .checked_add(weighted)
            .ok_or_else(|| LayoutError::ArithmeticOverflow { flow_id: flow.flow_id.clone() })?;
        total_weighted_travel_mm_per_period = total_weighted_travel_mm_per_period
            .checked_add(weighted)
            .ok_or_else(|| LayoutError::ArithmeticOverflow { flow_id: flow.flow_id.clone() })?;

        flow_metrics.push(FlowMetric {
            flow_id: flow.flow_id.clone(),
            mode: flow.mode,
            units_per_period: flow.units_per_period,
            path_zone_ids,
            path_edges,
            path_distance_mm,
            weighted_travel_mm_per_period: weighted,
        });
    }

    Ok(LayoutReport {
        layout_id: input.layout_id.clone(),
        zone_count: zones.len(),
        edge_count: edges.len(),
        flow_count: flows.len(),
        flow_metrics,
        weighted_travel_mm_by_mode,
        total_weighted_travel_mm_per_period,
    })
}

fn nonempty(kind: &'static str, value: &str) -> Result<(), LayoutError> {
    if value.trim().is_empty() {
        Err(LayoutError::EmptyIdentifier { kind })
    } else {
        Ok(())
    }
}

fn require_zone(
    zone_index: &BTreeMap<&str, usize>,
    context: &'static str,
    zone_id: &str,
) -> Result<(), LayoutError> {
    if zone_index.contains_key(zone_id) {
        Ok(())
    } else {
        Err(LayoutError::UnknownZoneReference {
            context,
            zone_id: zone_id.to_owned(),
        })
    }
}

fn validate_functions(input: &LayoutInput, zones: &[&Zone]) -> Result<(), LayoutError> {
    let mut required = BTreeSet::new();
    let available: BTreeSet<&str> = zones
        .iter()
        .flat_map(|zone| zone.functions.iter().map(String::as_str))
        .collect();

    for function in &input.required_functions {
        nonempty("required function", function)?;
        if !required.insert(function.as_str()) {
            return Err(LayoutError::DuplicateRequiredFunction { function: function.clone() });
        }
        if !available.contains(function.as_str()) {
            return Err(LayoutError::MissingRequiredFunction { function: function.clone() });
        }
    }
    Ok(())
}

fn validate_adjacencies(
    input: &LayoutInput,
    zone_index: &BTreeMap<&str, usize>,
) -> Result<(), LayoutError> {
    fn collect(
        kind: &'static str,
        pairs: &[ZonePair],
        zone_index: &BTreeMap<&str, usize>,
    ) -> Result<BTreeSet<ZonePair>, LayoutError> {
        let mut result = BTreeSet::new();
        for pair in pairs {
            require_zone(zone_index, "adjacency first endpoint", &pair.first_zone_id)?;
            require_zone(zone_index, "adjacency second endpoint", &pair.second_zone_id)?;
            if pair.first_zone_id == pair.second_zone_id {
                return Err(LayoutError::SelfAdjacency { zone_id: pair.first_zone_id.clone() });
            }
            let pair = ZonePair::new(pair.first_zone_id.clone(), pair.second_zone_id.clone());
            if !result.insert(pair.clone()) {
                return Err(LayoutError::DuplicateAdjacency {
                    kind,
                    first_zone_id: pair.first_zone_id,
                    second_zone_id: pair.second_zone_id,
                });
            }
        }
        Ok(result)
    }

    let declared = collect("declared", &input.declared_adjacencies, zone_index)?;
    let required = collect("required", &input.required_adjacencies, zone_index)?;
    let forbidden = collect("forbidden", &input.forbidden_adjacencies, zone_index)?;

    for pair in required.intersection(&forbidden) {
        return Err(LayoutError::ConflictingAdjacency {
            first_zone_id: pair.first_zone_id.clone(),
            second_zone_id: pair.second_zone_id.clone(),
        });
    }
    for pair in required.difference(&declared) {
        return Err(LayoutError::MissingRequiredAdjacency {
            first_zone_id: pair.first_zone_id.clone(),
            second_zone_id: pair.second_zone_id.clone(),
        });
    }
    for pair in forbidden.intersection(&declared) {
        return Err(LayoutError::ForbiddenAdjacencyPresent {
            first_zone_id: pair.first_zone_id.clone(),
            second_zone_id: pair.second_zone_id.clone(),
        });
    }
    Ok(())
}

fn zone_allows(zone: &Zone, mode: FlowMode) -> bool {
    match mode {
        FlowMode::Person => true,
        FlowMode::Return => !zone.new_material_only,
        FlowMode::NewMaterial | FlowMode::WorkInProgress => !zone.returns_only,
    }
}

fn shortest_path_for_flow(
    flow: &FlowDemand,
    zones: &[&Zone],
    zone_index: &BTreeMap<&str, usize>,
    edges: &[&TravelEdge],
) -> Result<(Vec<String>, Vec<PathEdgeSource>, u128), LayoutError> {
    let source = *zone_index
        .get(flow.from_zone_id.as_str())
        .expect("validated flow source");
    let target = *zone_index
        .get(flow.to_zone_id.as_str())
        .expect("validated flow target");

    if !zone_allows(zones[source], flow.mode) || !zone_allows(zones[target], flow.mode) {
        return Err(no_path(flow));
    }
    if source == target {
        return Ok((vec![zones[source].zone_id.clone()], Vec::new(), 0));
    }

    // Zone indexes, edge indexes and adjacency lists are stable-ID ordered.
    let mut adjacency = vec![Vec::<(usize, u64, usize)>::new(); zones.len()];
    for (edge_index, edge) in edges.iter().enumerate() {
        if !edge.permitted_modes.contains(&flow.mode) {
            continue;
        }
        if edge.protected_egress && flow.mode != FlowMode::Person {
            continue;
        }
        let from = *zone_index
            .get(edge.from_zone_id.as_str())
            .expect("validated edge source");
        let to = *zone_index
            .get(edge.to_zone_id.as_str())
            .expect("validated edge target");
        if zone_allows(zones[from], flow.mode) && zone_allows(zones[to], flow.mode) {
            adjacency[from].push((to, edge.distance_mm, edge_index));
        }
    }

    let tree = dijkstra_u64(&adjacency, source).map_err(LayoutError::ShortestPath)?;
    let Some(distance) = tree.distances[target] else {
        return Err(no_path(flow));
    };

    let mut path_zone_ids = vec![zones[target].zone_id.clone()];
    let mut path_edges = Vec::new();
    let mut current = target;

    while current != source {
        let previous = tree.predecessor_nodes[current]
            .ok_or_else(|| LayoutError::InvalidPathTree { flow_id: flow.flow_id.clone() })?;
        let edge_index = tree.predecessor_edges[current]
            .ok_or_else(|| LayoutError::InvalidPathTree { flow_id: flow.flow_id.clone() })?;
        let edge = edges
            .get(edge_index)
            .ok_or_else(|| LayoutError::InvalidPathTree { flow_id: flow.flow_id.clone() })?;

        path_edges.push(PathEdgeSource {
            edge_id: edge.edge_id.clone(),
            source_ref: edge.distance_source_ref.clone(),
            basis: edge.distance_basis,
            distance_mm: edge.distance_mm,
        });
        current = previous;
        path_zone_ids.push(zones[current].zone_id.clone());
    }

    path_zone_ids.reverse();
    path_edges.reverse();
    Ok((path_zone_ids, path_edges, distance))
}

fn no_path(flow: &FlowDemand) -> LayoutError {
    LayoutError::FlowHasNoFeasiblePath {
        flow_id: flow.flow_id.clone(),
        from_zone_id: flow.from_zone_id.clone(),
        to_zone_id: flow.to_zone_id.clone(),
        mode: flow.mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(id: &str, from: &str, to: &str, mm: u64, modes: &[FlowMode], egress: bool) -> TravelEdge {
        TravelEdge {
            edge_id: id.into(),
            from_zone_id: from.into(),
            to_zone_id: to.into(),
            distance_mm: mm,
            distance_source_ref: format!("fixture:{id}"),
            distance_basis: DistanceBasis::Synthetic,
            permitted_modes: modes.to_vec(),
            protected_egress: egress,
        }
    }

    fn flow(id: &str, from: &str, to: &str, mode: FlowMode, units: u64) -> FlowDemand {
        FlowDemand {
            flow_id: id.into(),
            from_zone_id: from.into(),
            to_zone_id: to.into(),
            mode,
            units_per_period: units,
        }
    }

    fn pair(a: &str, b: &str) -> ZonePair {
        ZonePair::new(a, b)
    }

    fn sample_layout() -> LayoutInput {
        LayoutInput {
            layout_id: "synthetic-hybrid-v0".into(),
            zones: vec![
                Zone { zone_id: "ASSEMBLY".into(), functions: vec!["cutting".into(), "assembly".into()], new_material_only: false, returns_only: false },
                Zone { zone_id: "DISPATCH".into(), functions: vec!["final_inspection".into(), "packing".into()], new_material_only: false, returns_only: false },
                Zone { zone_id: "RECEIVING".into(), functions: vec!["receiving".into(), "quarantine".into()], new_material_only: false, returns_only: false },
                Zone { zone_id: "REPAIR".into(), functions: vec!["repair".into(), "recovery".into()], new_material_only: false, returns_only: true },
                Zone { zone_id: "RAW".into(), functions: vec!["accepted_storage".into()], new_material_only: true, returns_only: false },
            ],
            travel_edges: vec![
                edge("e01", "RECEIVING", "RAW", 3_000, &[FlowMode::Person, FlowMode::NewMaterial], false),
                edge("e02", "RAW", "ASSEMBLY", 5_000, &[FlowMode::Person, FlowMode::NewMaterial, FlowMode::WorkInProgress], false),
                edge("e03", "ASSEMBLY", "DISPATCH", 4_000, &[FlowMode::Person, FlowMode::WorkInProgress], false),
                edge("e04", "DISPATCH", "REPAIR", 7_000, &[FlowMode::Person, FlowMode::Return], false),
                edge("e05", "ASSEMBLY", "RAW", 1_000, &[FlowMode::Return], false),
                edge("e06", "RAW", "REPAIR", 1_000, &[FlowMode::Return], false),
                edge("e07", "REPAIR", "RECEIVING", 9_000, &[FlowMode::Person], true),
            ],
            flows: vec![
                flow("f01", "RECEIVING", "RAW", FlowMode::NewMaterial, 2),
                flow("f02", "RAW", "ASSEMBLY", FlowMode::WorkInProgress, 3),
                flow("f03", "ASSEMBLY", "DISPATCH", FlowMode::WorkInProgress, 1),
                flow("f04", "DISPATCH", "REPAIR", FlowMode::Return, 1),
                flow("f05", "REPAIR", "RECEIVING", FlowMode::Person, 2),
            ],
            required_functions: vec![
                "receiving".into(), "quarantine".into(), "accepted_storage".into(),
                "cutting".into(), "assembly".into(), "final_inspection".into(),
                "packing".into(), "repair".into(), "recovery".into(),
            ],
            declared_adjacencies: vec![
                pair("RECEIVING", "RAW"), pair("RAW", "ASSEMBLY"),
                pair("ASSEMBLY", "DISPATCH"), pair("DISPATCH", "REPAIR"),
            ],
            required_adjacencies: vec![pair("RECEIVING", "RAW")],
            forbidden_adjacencies: vec![pair("RAW", "REPAIR")],
        }
    }

    #[test]
    fn evaluates_typed_flows_with_exact_distance_and_provenance() {
        let report = evaluate_layout(&sample_layout()).expect("valid synthetic layout");
        assert_eq!(report.zone_count, 5);
        assert_eq!(report.edge_count, 7);
        assert_eq!(report.flow_count, 5);
        assert_eq!(report.flow_metrics[0].flow_id, "f01");
        assert_eq!(report.flow_metrics[0].path_distance_mm, 3_000);
        assert_eq!(report.flow_metrics[0].weighted_travel_mm_per_period, 6_000);
        assert_eq!(report.flow_metrics[0].path_edges[0].edge_id, "e01");
        assert_eq!(report.flow_metrics[0].path_edges[0].basis, DistanceBasis::Synthetic);
        assert_eq!(report.flow_metrics[3].path_zone_ids, vec!["DISPATCH".to_owned(), "REPAIR".to_owned()]);
        assert_eq!(report.flow_metrics[4].path_edges[0].edge_id, "e07");
        assert_eq!(report.total_weighted_travel_mm_per_period, 50_000);
    }

    #[test]
    fn returns_cannot_cross_new_material_only_storage() {
        let mut input = sample_layout();
        input.flows.push(flow("f06", "ASSEMBLY", "REPAIR", FlowMode::Return, 1));
        assert!(matches!(
            evaluate_layout(&input),
            Err(LayoutError::FlowHasNoFeasiblePath { flow_id, mode: FlowMode::Return, .. })
                if flow_id == "f06"
        ));
    }

    #[test]
    fn protected_egress_is_people_only() {
        let mut input = sample_layout();
        input.flows.push(flow("f06", "REPAIR", "RECEIVING", FlowMode::NewMaterial, 1));
        assert!(matches!(
            evaluate_layout(&input),
            Err(LayoutError::FlowHasNoFeasiblePath { flow_id, mode: FlowMode::NewMaterial, .. })
                if flow_id == "f06"
        ));
    }

    #[test]
    fn missing_functions_and_adjacencies_fail_closed() {
        let mut input = sample_layout();
        input.required_functions.push("dyeing".into());
        assert_eq!(
            evaluate_layout(&input).expect_err("missing function"),
            LayoutError::MissingRequiredFunction { function: "dyeing".into() }
        );

        let mut input = sample_layout();
        input.required_adjacencies.push(pair("RECEIVING", "REPAIR"));
        assert!(matches!(
            evaluate_layout(&input),
            Err(LayoutError::MissingRequiredAdjacency { .. })
        ));
    }

    #[test]
    fn forbidden_adjacency_and_zero_distance_are_rejected() {
        let mut input = sample_layout();
        input.declared_adjacencies.push(pair("RAW", "REPAIR"));
        assert!(matches!(evaluate_layout(&input), Err(LayoutError::ForbiddenAdjacencyPresent { .. })));

        let mut input = sample_layout();
        input.travel_edges[0].distance_mm = 0;
        assert!(matches!(evaluate_layout(&input), Err(LayoutError::InvalidEdgeDistance { .. })));
    }

    #[test]
    fn unknown_zone_and_duplicate_ids_are_rejected() {
        let mut input = sample_layout();
        input.flows.push(flow("f99", "UNKNOWN", "REPAIR", FlowMode::Return, 1));
        assert!(matches!(evaluate_layout(&input), Err(LayoutError::UnknownZoneReference { .. })));

        let mut input = sample_layout();
        input.zones.push(input.zones[0].clone());
        assert!(matches!(evaluate_layout(&input), Err(LayoutError::DuplicateZoneId { .. })));
    }

    #[test]
    fn shuffled_input_produces_identical_report() {
        let input = sample_layout();
        let expected = evaluate_layout(&input).expect("valid");
        let mut shuffled = input;
        shuffled.zones.reverse();
        shuffled.travel_edges.reverse();
        shuffled.flows.reverse();
        shuffled.declared_adjacencies.reverse();
        assert_eq!(evaluate_layout(&shuffled).expect("valid shuffled"), expected);
    }
}
