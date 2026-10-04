// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compile functional void intent into a material device candidate.
//!
//! This is deliberately separate from `compile_flow_void_geometry`: the latter
//! produces the void/tool geometry used for functional reasoning, while this
//! module subtracts that void from an explicit material body and adds typed
//! external port tunnels.

use std::collections::BTreeSet;

use symthaea_fabrication_kernel::csg::{CSGNode, Transform3D};
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId};

use crate::{
    compile_flow_void_geometry, CompileError, GeometryEmbedding, PortInterface,
};

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

/// A material-device external interface is now a complete typed geometric object.
///
/// The legacy port id + direction split is intentionally removed from the
/// device compiler surface so aperture, plane, normal, and solver identity
/// cannot silently drift apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalPortSpec {
    pub interface: PortInterface,
}

impl ExternalPortSpec {
    pub fn new(interface: PortInterface) -> Self {
        Self { interface }
    }

    pub fn port(&self) -> PortId {
        self.interface.port
    }
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
/// The output is a candidate solid body. Each external port is represented by
/// a tunnel whose center, aperture radius, and direction come from the same
/// validated `PortInterface`. No solver or transport property is inferred.
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
        let interface = spec.interface;
        if !seen_ports.insert(interface.port) {
            return Err(CompileError::InvalidAnchor(
                "external port may appear only once".into(),
            ));
        }

        interface.validate(0.001).map_err(|error| {
            CompileError::InvalidAnchor(format!("invalid typed port interface: {error:?}"))
        })?;

        let anchor = embedding
            .ports
            .get(&interface.port)
            .ok_or(CompileError::MissingPortAnchor(interface.port))?;
        if !same_position_and_radius(
            anchor.center_mm,
            anchor.radius_mm,
            interface.position_mm,
            interface.radius_mm(),
            0.001,
        ) {
            return Err(CompileError::InvalidAnchor(
                "typed port interface does not match the embedding anchor".into(),
            ));
        }
        if !body.contains_strict(interface.position_mm) {
            return Err(CompileError::InvalidAnchor(
                "external port interface position must lie strictly inside the body"
                    .into(),
            ));
        }

        let direction = interface.outward_normal_unit;
        let exit_distance = ray_box_exit_distance(interface.position_mm, direction, body)?;
        let extra = interface.radius_mm() * 2.0;
        let tunnel_end = [
            interface.position_mm[0] + direction[0] * (exit_distance + extra),
            interface.position_mm[1] + direction[1] * (exit_distance + extra),
            interface.position_mm[2] + direction[2] * (exit_distance + extra),
        ];
        let tunnel = cylinder_between(
            interface.position_mm,
            tunnel_end,
            interface.radius_mm(),
        )?;
        material = material.subtract(tunnel);
    }

    Ok((material, DeviceCompilationSemantics::default()))
}

fn same_position_and_radius(
    a_center: [f32; 3],
    a_radius: f32,
    b_center: [f32; 3],
    b_radius: f32,
    tolerance_mm: f32,
) -> bool {
    let center_distance = ((a_center[0] - b_center[0]).powi(2)
        + (a_center[1] - b_center[1]).powi(2)
        + (a_center[2] - b_center[2]).powi(2))
    .sqrt();
    center_distance <= tolerance_mm
        && (a_radius - b_radius).abs() <= tolerance_mm
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
            "external port interface normal does not exit the body envelope".into(),
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
    let length =
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
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
    use crate::{
        BoundaryConditionDomain, InterfacePlane, PortAperture, SolverBoundaryIdentity,
    };
    use symthaea_fabrication_kernel::csg::CSGNode;
    use symthaea_passive_void_graph::{
        RegionId, VoidConnection, VoidPort, VoidRegion, VoidRegionRole, VoidRelation,
    };

    fn graph() -> FunctionalVoidGraph {
        let mut graph = FunctionalVoidGraph::new();
        graph
            .add_region(VoidRegion {
                id: RegionId(1),
                role: VoidRegionRole::Inlet,
            })
            .unwrap();
        graph
            .add_region(VoidRegion {
                id: RegionId(2),
                role: VoidRegionRole::Outlet,
            })
            .unwrap();
        graph
            .add_port(VoidPort {
                id: PortId(10),
                region: RegionId(1),
            })
            .unwrap();
        graph
            .add_port(VoidPort {
                id: PortId(20),
                region: RegionId(2),
            })
            .unwrap();
        graph
            .connect(VoidConnection {
                from: PortId(10),
                to: PortId(20),
                relation: VoidRelation::FlowPath,
                bidirectional: false,
            })
            .unwrap();
        graph
    }

    fn embedding() -> GeometryEmbedding {
        GeometryEmbedding::default()
            .with_region(
                RegionId(1),
                RegionAnchor {
                    center_mm: [-8.0, 0.0, 0.0],
                    radius_mm: 4.0,
                },
            )
            .with_region(
                RegionId(2),
                RegionAnchor {
                    center_mm: [8.0, 0.0, 0.0],
                    radius_mm: 4.0,
                },
            )
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-8.0, 0.0, 0.0],
                    radius_mm: 2.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [8.0, 0.0, 0.0],
                    radius_mm: 2.0,
                },
            )
    }

    fn external_interface(port: PortId, position: [f32; 3], normal: [f32; 3]) -> ExternalPortSpec {
        ExternalPortSpec::new(
            PortInterface::new(
                port,
                position,
                PortAperture::Circular { radius_mm: 2.0 },
                normal,
                InterfacePlane::new(position, normal).unwrap(),
                SolverBoundaryIdentity {
                    domain: BoundaryConditionDomain::Fluidic,
                    id: port.0,
                },
            )
            .unwrap(),
        )
    }

    #[test]
    fn body_realization_produces_material_candidate() {
        let body = BodyEnvelope {
            min_mm: [-12.0, -10.0, -10.0],
            max_mm: [12.0, 10.0, 10.0],
        };
        let (material, semantics) = compile_passive_device_geometry(
            &graph(),
            &embedding(),
            &body,
            &[
                external_interface(PortId(10), [-8.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                external_interface(PortId(20), [8.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ],
        )
        .unwrap();
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
    fn invalid_external_interface_normal_is_rejected() {
        let body = BodyEnvelope {
            min_mm: [-12.0, -10.0, -10.0],
            max_mm: [12.0, 10.0, 10.0],
        };
        let result = compile_passive_device_geometry(
            &graph(),
            &embedding(),
            &body,
            &[ExternalPortSpec::new(
                PortInterface::new(
                    PortId(10),
                    [-8.0, 0.0, 0.0],
                    PortAperture::Circular { radius_mm: 2.0 },
                    [1.0, 0.0, 0.0],
                    InterfacePlane::new([-8.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).unwrap(),
                    SolverBoundaryIdentity {
                        domain: BoundaryConditionDomain::Fluidic,
                        id: 10,
                    },
                )
                .unwrap_err()
                .into_result()
            )],
        );
        assert!(result.is_err());
    }
}
