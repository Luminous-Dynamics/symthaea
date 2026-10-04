// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compile functional void intent into a material device candidate.
//!
//! This is deliberately separate from `compile_flow_void_geometry`: the latter
//! produces the void/tool geometry used for functional reasoning, while this
//! module subtracts that void from an explicit material body and adds external
//! port tunnels.

use std::collections::BTreeSet;

use symthaea_fabrication_kernel::csg::{CSGNode, Transform3D};
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId};

use crate::{compile_flow_void_geometry, CompileError, GeometryEmbedding};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyEnvelope {
    pub min_mm: [f32; 3],
    pub max_mm: [f32; 3],
}

impl BodyEnvelope {
    pub fn validate(&self) -> Result<(), CompileError> {
        if !self
            .min_mm
            .iter()
            .chain(self.max_mm.iter())
            .all(|value| value.is_finite())
        {
            return Err(CompileError::InvalidAnchor(
                "body envelope coordinates must be finite".into(),
            ));
        }
        if self
            .min_mm
            .iter()
            .zip(self.max_mm.iter())
            .any(|(min, max)| min >= max)
        {
            return Err(CompileError::InvalidAnchor(
                "body envelope min must be strictly below max".into(),
            ));
        }
        Ok(())
    }

    pub fn as_csg(&self) -> CSGNode {
        let center = [
            (self.min_mm[0] + self.max_mm[0]) * 0.5,
            (self.min_mm[1] + self.max_mm[1]) * 0.5,
            (self.min_mm[2] + self.max_mm[2]) * 0.5,
        ];
        let scale = [
            self.max_mm[0] - self.min_mm[0],
            self.max_mm[1] - self.min_mm[1],
            self.max_mm[2] - self.min_mm[2],
        ];
        CSGNode::cube().with_transform(Transform3D {
            scale,
            rotate: [0.0, 0.0, 0.0],
            translate: center,
        })
    }

    fn contains_strict(&self, point: [f32; 3]) -> bool {
        point.iter().enumerate().all(|(axis, value)| {
            *value > self.min_mm[axis] && *value < self.max_mm[axis]
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalPortSpec {
    pub port: PortId,
    pub outward_unit: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCompilationSemantics {
    pub output_is_material_candidate: bool,
    pub external_port_tunnels_are_explicit: bool,
    pub transport_remains_unproven: bool,
}

impl Default for DeviceCompilationSemantics {
    fn default() -> Self {
        Self {
            output_is_material_candidate: true,
            external_port_tunnels_are_explicit: true,
            transport_remains_unproven: true,
        }
    }
}

/// Compile a material device from a functional void graph.
///
/// The output is a candidate solid body. External ports are modeled as
/// subtraction tunnels that extend beyond the body envelope. No solver or
/// transport property is inferred from this compilation.
pub fn compile_passive_device_geometry(
    graph: &FunctionalVoidGraph,
    embedding: &GeometryEmbedding,
    body: &BodyEnvelope,
    external_ports: &[ExternalPortSpec],
) -> Result<(CSGNode, DeviceCompilationSemantics), CompileError> {
    body.validate()?;
    graph.validate().map_err(|_| CompileError::InvalidGraph)?;

    let (void_geometry, _) = compile_flow_void_geometry(graph, embedding)?;
    let mut material = body.as_csg().subtract(void_geometry);
    let mut seen_ports = BTreeSet::new();

    for spec in external_ports {
        if !seen_ports.insert(spec.port) {
            return Err(CompileError::InvalidAnchor(
                "external port may appear only once".into(),
            ));
        }
        let anchor = embedding
            .ports
            .get(&spec.port)
            .ok_or(CompileError::MissingPortAnchor(spec.port))?;
        if !anchor.radius_mm.is_finite() || anchor.radius_mm <= 0.0 {
            return Err(CompileError::InvalidAnchor(
                "external port radius must be finite and > 0 mm".into(),
            ));
        }
        if !body.contains_strict(anchor.center_mm) {
            return Err(CompileError::InvalidAnchor(
                "external port anchor must lie strictly inside the body".into(),
            ));
        }

        let direction_length = (
            spec.outward_unit[0] * spec.outward_unit[0]
                + spec.outward_unit[1] * spec.outward_unit[1]
                + spec.outward_unit[2] * spec.outward_unit[2]
        )
        .sqrt();
        if !direction_length.is_finite() || direction_length <= 1.0e-6 {
            return Err(CompileError::InvalidAnchor(
                "external port direction must be a finite nonzero vector".into(),
            ));
        }
        let direction = [
            spec.outward_unit[0] / direction_length,
            spec.outward_unit[1] / direction_length,
            spec.outward_unit[2] / direction_length,
        ];
        let exit_distance = ray_box_exit_distance(anchor.center_mm, direction, body)?;
        let extra = anchor.radius_mm * 2.0;
        let tunnel_end = [
            anchor.center_mm[0] + direction[0] * (exit_distance + extra),
            anchor.center_mm[1] + direction[1] * (exit_distance + extra),
            anchor.center_mm[2] + direction[2] * (exit_distance + extra),
        ];
        let tunnel = cylinder_between(anchor.center_mm, tunnel_end, anchor.radius_mm)?;
        material = material.subtract(tunnel);
    }

    Ok((material, DeviceCompilationSemantics::default()))
}

fn ray_box_exit_distance(
    point: [f32; 3],
    direction: [f32; 3],
    body: &BodyEnvelope,
) -> Result<f32, CompileError> {
    let mut best = f32::INFINITY;
    for axis in 0..3 {
        let d = direction[axis];
        if d > 1.0e-8 {
            best = best.min((body.max_mm[axis] - point[axis]) / d);
        } else if d < -1.0e-8 {
            best = best.min((body.min_mm[axis] - point[axis]) / d);
        }
    }
    if !best.is_finite() || best <= 0.0 {
        return Err(CompileError::InvalidAnchor(
            "external port direction does not exit the body envelope".into(),
        ));
    }
    Ok(best)
}

fn cylinder_between(
    from_mm: [f32; 3],
    to_mm: [f32; 3],
    radius_mm: f32,
) -> Result<CSGNode, CompileError> {
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return Err(CompileError::InvalidAnchor(
            "channel radius must be finite and > 0 mm".into(),
        ));
    }
    let delta = [
        to_mm[0] - from_mm[0],
        to_mm[1] - from_mm[1],
        to_mm[2] - from_mm[2],
    ];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if !length.is_finite() || length <= 1.0e-6 {
        return Err(CompileError::InvalidAnchor(
            "connection endpoints must be distinct".into(),
        ));
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
    use crate::compile_flow_void_geometry;
    use symthaea_fabrication_kernel::csg::CSGNode;
    use symthaea_passive_void_graph::{VoidConnection, VoidPort, VoidRegion, VoidRegionRole, VoidRelation, RegionId};

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
            .with_region(RegionId(1), RegionAnchor { center_mm: [-8.0, 0.0, 0.0], radius_mm: 4.0 })
            .with_region(RegionId(2), RegionAnchor { center_mm: [8.0, 0.0, 0.0], radius_mm: 4.0 })
            .with_port(PortId(10), PortAnchor { center_mm: [-8.0, 0.0, 0.0], radius_mm: 2.0 })
            .with_port(PortId(20), PortAnchor { center_mm: [8.0, 0.0, 0.0], radius_mm: 2.0 })
    }

    #[test]
    fn body_realization_produces_material_candidate() {
        let body = BodyEnvelope { min_mm: [-12.0, -10.0, -10.0], max_mm: [12.0, 10.0, 10.0] };
        let (material, semantics) = compile_passive_device_geometry(
            &graph(),
            &embedding(),
            &body,
            &[ExternalPortSpec { port: PortId(10), outward_unit: [-1.0, 0.0, 0.0] }, ExternalPortSpec { port: PortId(20), outward_unit: [1.0, 0.0, 0.0] }],
        ).unwrap();
        assert!(matches!(material, CSGNode::Boolean { .. }));
        assert!(semantics.output_is_material_candidate);
        assert!(semantics.external_port_tunnels_are_explicit);
        assert!(semantics.transport_remains_unproven);

        let mesh = symthaea_fabrication_kernel::mesh::resolve_to_mesh(&material);
        let report = symthaea_fabrication_kernel::validate::validate_mesh(&mesh);
        assert!(report.is_valid());
        assert!(report.is_watertight);
        assert!(report.signed_volume > 0.0);
    }

    #[test]
    fn invalid_external_direction_is_rejected() {
        let body = BodyEnvelope { min_mm: [-12.0, -10.0, -10.0], max_mm: [12.0, 10.0, 10.0] };
        let result = compile_passive_device_geometry(
            &graph(),
            &embedding(),
            &body,
            &[ExternalPortSpec { port: PortId(10), outward_unit: [0.0, 0.0, 0.0] }],
        );
        assert!(matches!(result, Err(CompileError::InvalidAnchor(_))));
    }
}