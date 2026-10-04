// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compile an embedded functional void graph into candidate void CSG.
//!
//! The semantic graph needs explicit spatial anchors before it can become
//! geometry. Directionality remains metadata; static geometry does not prove
//! one-way flow.

use std::collections::BTreeMap;
use symthaea_fabrication_kernel::csg::{CSGNode, Transform3D};
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId, RegionId, VoidRegionRole, VoidRelation};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionAnchor {
    pub center_mm: [f32; 3],
    pub radius_mm: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortAnchor {
    pub center_mm: [f32; 3],
    pub radius_mm: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GeometryEmbedding {
    pub regions: BTreeMap<RegionId, RegionAnchor>,
    pub ports: BTreeMap<PortId, PortAnchor>,
}

impl GeometryEmbedding {
    pub fn with_region(mut self, id: RegionId, anchor: RegionAnchor) -> Self {
        self.regions.insert(id, anchor);
        self
    }

    pub fn with_port(mut self, id: PortId, anchor: PortAnchor) -> Self {
        self.ports.insert(id, anchor);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    InvalidGraph,
    MissingRegionAnchor(RegionId),
    MissingPortAnchor(PortId),
    InvalidAnchor(String),
    EmptyGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilationSemantics {
    pub directionality_is_intent_only: bool,
    pub output_is_candidate_geometry: bool,
}

impl Default for CompilationSemantics {
    fn default() -> Self {
        Self {
            directionality_is_intent_only: true,
            output_is_candidate_geometry: true,
        }
    }
}

pub fn compile_flow_void_geometry(
    graph: &FunctionalVoidGraph,
    embedding: &GeometryEmbedding,
) -> Result<(CSGNode, CompilationSemantics), CompileError> {
    graph.validate().map_err(|_| CompileError::InvalidGraph)?;

    let mut geometry = None;
    for region in &graph.regions {
        if matches!(region.role, VoidRegionRole::Barrier) {
            continue;
        }
        let anchor = embedding
            .regions
            .get(&region.id)
            .ok_or(CompileError::MissingRegionAnchor(region.id))?;
        if !anchor.radius_mm.is_finite() || anchor.radius_mm <= 0.0 {
            return Err(CompileError::InvalidAnchor("region radius must be finite and > 0 mm".into()));
        }
        geometry = Some(match geometry {
            Some(existing) => existing.union(sphere_at(anchor.center_mm, anchor.radius_mm)),
            None => sphere_at(anchor.center_mm, anchor.radius_mm),
        });
    }

    for connection in &graph.connections {
        if connection.relation != VoidRelation::FlowPath {
            continue;
        }
        let from = embedding
            .ports
            .get(&connection.from)
            .ok_or(CompileError::MissingPortAnchor(connection.from))?;
        let to = embedding
            .ports
            .get(&connection.to)
            .ok_or(CompileError::MissingPortAnchor(connection.to))?;
        let radius = from.radius_mm.min(to.radius_mm);
        geometry = Some(match geometry {
            Some(existing) => existing.union(cylinder_between(from.center_mm, to.center_mm, radius)?),
            None => cylinder_between(from.center_mm, to.center_mm, radius)?,
        });
    }

    Ok((geometry.ok_or(CompileError::EmptyGeometry)?, CompilationSemantics::default()))
}

fn sphere_at(center_mm: [f32; 3], radius_mm: f32) -> CSGNode {
    CSGNode::sphere().with_transform(Transform3D {
        scale: [radius_mm * 2.0, radius_mm * 2.0, radius_mm * 2.0],
        rotate: [0.0, 0.0, 0.0],
        translate: center_mm,
    })
}

fn cylinder_between(
    from_mm: [f32; 3],
    to_mm: [f32; 3],
    radius_mm: f32,
) -> Result<CSGNode, CompileError> {
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return Err(CompileError::InvalidAnchor("channel radius must be finite and > 0 mm".into()));
    }
    let delta = [to_mm[0] - from_mm[0], to_mm[1] - from_mm[1], to_mm[2] - from_mm[2]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if !length.is_finite() || length <= 1.0e-6 {
        return Err(CompileError::InvalidAnchor("connection endpoints must be distinct".into()));
    }
    let radial = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    let pitch = radial.atan2(delta[2]);
    let yaw = delta[1].atan2(delta[0]);
    let midpoint = [
        (from_mm[0] + to_mm[0]) * 0.5,
        (from_mm[1] + to_mm[1]) * 0.5,
        (from_mm[2] + to_mm[2]) * 0.5,
    ];
    Ok(CSGNode::cylinder().with_transform(Transform3D {
        scale: [radius_mm * 2.0, radius_mm * 2.0, length],
        rotate: [0.0, pitch, yaw],
        translate: midpoint,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_passive_void_graph::{VoidConnection, VoidPort, VoidRegion};

    fn graph() -> FunctionalVoidGraph {
        let mut graph = FunctionalVoidGraph::new();
        graph.add_region(VoidRegion { id: RegionId(1), role: VoidRegionRole::Inlet }).unwrap();
        graph.add_region(VoidRegion { id: RegionId(2), role: VoidRegionRole::Outlet }).unwrap();
        graph.add_port(VoidPort { id: PortId(10), region: RegionId(1) }).unwrap();
        graph.add_port(VoidPort { id: PortId(20), region: RegionId(2) }).unwrap();
        graph.connect(VoidConnection {
            from: PortId(10),
            to: PortId(20),
            relation: VoidRelation::FlowPath,
            bidirectional: false,
        }).unwrap();
        graph
    }

    fn embedding() -> GeometryEmbedding {
        GeometryEmbedding::default()
            .with_region(RegionId(1), RegionAnchor { center_mm: [0.0, 0.0, 0.0], radius_mm: 5.0 })
            .with_region(RegionId(2), RegionAnchor { center_mm: [20.0, 0.0, 0.0], radius_mm: 5.0 })
            .with_port(PortId(10), PortAnchor { center_mm: [0.0, 0.0, 0.0], radius_mm: 2.0 })
            .with_port(PortId(20), PortAnchor { center_mm: [20.0, 0.0, 0.0], radius_mm: 2.0 })
    }

    #[test]
    fn flow_intent_compiles_to_candidate_geometry() {
        let (geometry, semantics) = compile_flow_void_geometry(&graph(), &embedding()).unwrap();
        assert!(matches!(geometry, CSGNode::Boolean { .. }));
        assert!(semantics.directionality_is_intent_only);
        assert!(semantics.output_is_candidate_geometry);
    }

    #[test]
    fn missing_anchor_is_hard_error() {
        let mut e = embedding();
        e.ports.remove(&PortId(20));
        assert_eq!(compile_flow_void_geometry(&graph(), &e), Err(CompileError::MissingPortAnchor(PortId(20))));
    }

    #[test]
    fn coincident_ports_are_rejected() {
        let mut e = embedding();
        e.ports.get_mut(&PortId(20)).unwrap().center_mm = [0.0, 0.0, 0.0];
        assert!(matches!(compile_flow_void_geometry(&graph(), &e), Err(CompileError::InvalidAnchor(_))));
    }

    #[test]
    fn directionality_does_not_change_static_geometry() {
        let mut a = graph();
        let mut b = graph();
        a.connections[0].bidirectional = false;
        b.connections[0].bidirectional = true;
        let ga = compile_flow_void_geometry(&a, &embedding()).unwrap().0;
        let gb = compile_flow_void_geometry(&b, &embedding()).unwrap().0;
        assert_eq!(format!("{ga:?}"), format!("{gb:?}"));
    }
}