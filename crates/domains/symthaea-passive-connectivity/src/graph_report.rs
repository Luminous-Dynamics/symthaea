// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Whole-graph realization comparison for passive void candidates.
//!
//! This layer compares declared FlowPath intent edges against the candidate
//! mesh. A connected result means only that the two port anchors land in the
//! same mesh component. It does not establish pressure-flow capability,
//! directionality, hydraulic performance, or any other transport property.

use crate::{
    boundary::{PortBoundaryEvidence, PortBoundaryPolicy},
    evaluate_port_path, evaluate_port_path_with_boundary_policy, resolve_port_component,
    triangle_components, AnchorResolution, PortPathEvidence, PortPathStatus,
};
use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_passive_void_compiler::GeometryEmbedding;
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId, VoidRelation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRealizationStatus {
    InvalidGraph,
    NoFlowPathsDeclared,
    AllDeclaredPathsConnected,
    TopologyDivergence,
    PartialRealization,
    NoDeclaredPathsConnected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionRealization {
    pub from: PortId,
    pub to: PortId,
    pub evidence: PortPathEvidence,
}

impl ConnectionRealization {
    pub fn is_connected(&self) -> bool {
        matches!(self.evidence.status, PortPathStatus::Connected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionalGraphRealizationReport {
    pub status: GraphRealizationStatus,
    pub declared_flow_paths: usize,
    pub connected_paths: usize,
    pub disconnected_paths: usize,
    pub invalid_paths: usize,
    pub path_results: Vec<ConnectionRealization>,
    /// Port pairs that the realized mesh connects even though the functional graph
    /// declares no FlowPath in either direction.
    pub unexpected_flow_connectivity: Vec<UnexpectedFlowConnectivity>,
    /// Geometry/topology evidence only; transport remains unproven.
    pub physical_transport_unproven: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnexpectedFlowConnectivity {
    pub from: PortId,
    pub to: PortId,
    pub component: usize,
}

impl FunctionalGraphRealizationReport {
    pub fn evaluate(
        graph: &FunctionalVoidGraph,
        embedding: &GeometryEmbedding,
        candidate: &TriangleMesh,
    ) -> Self {
        Self::evaluate_inner(graph, embedding, candidate, None)
    }

    /// Evaluate the complete flow graph while permitting only explicitly allowed
    /// boundary openings.
    pub fn evaluate_with_boundary_policy(
        graph: &FunctionalVoidGraph,
        embedding: &GeometryEmbedding,
        candidate: &TriangleMesh,
        boundary_policy: &PortBoundaryPolicy,
    ) -> Self {
        Self::evaluate_inner(graph, embedding, candidate, Some(boundary_policy))
    }

    fn evaluate_inner(
        graph: &FunctionalVoidGraph,
        embedding: &GeometryEmbedding,
        candidate: &TriangleMesh,
        boundary_policy: Option<&PortBoundaryPolicy>,
    ) -> Self {
        if graph.validate().is_err() {
            return Self {
                status: GraphRealizationStatus::InvalidGraph,
                declared_flow_paths: 0,
                connected_paths: 0,
                disconnected_paths: 0,
                invalid_paths: 0,
                path_results: Vec::new(),
                unexpected_flow_connectivity: Vec::new(),
                physical_transport_unproven: true,
            };
        }

        let flow_connections: Vec<_> = graph
            .connections
            .iter()
            .filter(|connection| connection.relation == VoidRelation::FlowPath)
            .copied()
            .collect();

        if flow_connections.is_empty() {
            return Self {
                status: GraphRealizationStatus::NoFlowPathsDeclared,
                declared_flow_paths: 0,
                connected_paths: 0,
                disconnected_paths: 0,
                invalid_paths: 0,
                path_results: Vec::new(),
                unexpected_flow_connectivity: Self::find_unexpected_flow_connectivity(
                    graph, embedding, candidate, boundary_policy,
                ),
                physical_transport_unproven: true,
            };
        }

        let mut connected_paths = 0;
        let mut disconnected_paths = 0;
        let mut invalid_paths = 0;
        let mut path_results = Vec::with_capacity(flow_connections.len());

        for connection in flow_connections {
            let evidence = match boundary_policy {
                Some(policy) => evaluate_port_path_with_boundary_policy(
                    graph, embedding, candidate, connection.from, connection.to, policy,
                ),
                None => evaluate_port_path(
                    graph, embedding, candidate, connection.from, connection.to,
                ),
            };

            if matches!(evidence.status, PortPathStatus::Connected) {
                connected_paths += 1;
            } else if matches!(evidence.status, PortPathStatus::Disconnected) {
                disconnected_paths += 1;
            } else {
                invalid_paths += 1;
            }

            path_results.push(ConnectionRealization {
                from: connection.from,
                to: connection.to,
                evidence,
            });
        }

        let declared_flow_paths = path_results.len();
        let unexpected_flow_connectivity = Self::find_unexpected_flow_connectivity(
            graph, embedding, candidate, boundary_policy,
        );
        let status = if connected_paths == declared_flow_paths {
            if unexpected_flow_connectivity.is_empty() {
                GraphRealizationStatus::AllDeclaredPathsConnected
            } else {
                GraphRealizationStatus::TopologyDivergence
            }
        } else if connected_paths == 0 {
            GraphRealizationStatus::NoDeclaredPathsConnected
        } else {
            GraphRealizationStatus::PartialRealization
        };

        Self {
            status,
            declared_flow_paths,
            connected_paths,
            disconnected_paths,
            invalid_paths,
            path_results,
            unexpected_flow_connectivity,
            physical_transport_unproven: true,
        }
    }

    fn find_unexpected_flow_connectivity(
        graph: &FunctionalVoidGraph,
        embedding: &GeometryEmbedding,
        candidate: &TriangleMesh,
        boundary_policy: Option<&PortBoundaryPolicy>,
    ) -> Vec<UnexpectedFlowConnectivity> {
        let report = symthaea_fabrication_kernel::validate::validate_mesh(candidate);
        if !report.is_valid() {
            return Vec::new();
        }
        if let Some(policy) = boundary_policy {
            if !PortBoundaryEvidence::evaluate(candidate, embedding, policy).is_admissible() {
                return Vec::new();
            }
        } else if !report.is_watertight {
            return Vec::new();
        }

        let labels = triangle_components(candidate);
        let mut realized = Vec::new();
        for port in &graph.ports {
            if let AnchorResolution::Found(component) =
                resolve_port_component(candidate, embedding, &labels, port.id)
            {
                realized.push((port.id, component));
            }
        }

        let mut unexpected = Vec::new();
        for (index, (from, from_component)) in realized.iter().enumerate() {
            for (to, to_component) in realized.iter().skip(index + 1) {
                if from_component != to_component {
                    continue;
                }
                let declared_forward =
                    graph.declares_path(*from, *to, VoidRelation::FlowPath);
                let declared_reverse =
                    graph.declares_path(*to, *from, VoidRelation::FlowPath);
                if !declared_forward && !declared_reverse {
                    unexpected.push(UnexpectedFlowConnectivity {
                        from: *from,
                        to: *to,
                        component: *from_component,
                    });
                }
            }
        }
        unexpected
    }

    pub fn has_topology_leakage(&self) -> bool {
        !self.unexpected_flow_connectivity.is_empty()
    }

    pub fn is_geometrically_complete(&self) -> bool {
        matches!(self.status, GraphRealizationStatus::AllDeclaredPathsConnected)
            && self.invalid_paths == 0
            && self.disconnected_paths == 0
            && !self.has_topology_leakage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::{
        csg::{CSGNode, Transform3D},
        mesh::resolve_to_mesh,
    };
    use symthaea_passive_void_compiler::{GeometryEmbedding, PortAnchor};
    use symthaea_passive_void_graph::{
        RegionId, VoidConnection, VoidPort, VoidRegion, VoidRegionRole,
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
            .add_region(VoidRegion {
                id: RegionId(3),
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
            .add_port(VoidPort {
                id: PortId(30),
                region: RegionId(3),
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
            .connect(VoidConnection {
                from: PortId(20),
                to: PortId(30),
                relation: VoidRelation::FlowPath,
                bidirectional: false,
            })
            .unwrap();
        graph
    }

    fn embedding() -> GeometryEmbedding {
        GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-0.5, -0.5, -0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [0.0, 0.0, 0.0],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(30),
                PortAnchor {
                    center_mm: [0.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            )
    }

    #[test]
    fn all_declared_flow_paths_are_reported_connected() {
        let graph = graph();
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = FunctionalGraphRealizationReport::evaluate(&graph, &embedding(), &mesh);

        assert_eq!(report.status, GraphRealizationStatus::AllDeclaredPathsConnected);
        assert_eq!(report.declared_flow_paths, 2);
        assert_eq!(report.connected_paths, 2);
        assert_eq!(report.disconnected_paths, 0);
        assert_eq!(report.invalid_paths, 0);
        assert!(report.unexpected_flow_connectivity.is_empty());
        assert!(report.is_geometrically_complete());
        assert!(report.physical_transport_unproven);
    }

    #[test]
    fn realized_mesh_can_expose_unexpected_flow_connectivity() {
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
            .add_region(VoidRegion {
                id: RegionId(3),
                role: VoidRegionRole::Cavity,
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
            .add_port(VoidPort {
                id: PortId(30),
                region: RegionId(3),
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

        let embedding = GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-0.5, -0.5, -0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [0.0, 0.0, 0.0],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(30),
                PortAnchor {
                    center_mm: [0.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            );

        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = FunctionalGraphRealizationReport::evaluate(&graph, &embedding, &mesh);

        assert_eq!(report.status, GraphRealizationStatus::TopologyDivergence);
        assert!(report.has_topology_leakage());
        assert!(!report.is_geometrically_complete());
        assert!(report
            .unexpected_flow_connectivity
            .iter()
            .any(|pair| pair.from == PortId(10) && pair.to == PortId(30)));
    }

    #[test]
    fn disconnected_candidate_produces_partial_realization() {
        let graph = graph();
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [3.0, 0.0, 0.0],
            ..Default::default()
        }));
        left.merge(&right);

        let embedding = GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-0.5, -0.5, -0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [0.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(30),
                PortAnchor {
                    center_mm: [2.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            );

        let report = FunctionalGraphRealizationReport::evaluate(&graph, &embedding, &left);
        assert_eq!(report.status, GraphRealizationStatus::PartialRealization);
        assert_eq!(report.declared_flow_paths, 2);
        assert_eq!(report.connected_paths, 1);
        assert_eq!(report.disconnected_paths, 1);
        assert_eq!(report.invalid_paths, 0);
        assert!(!report.is_geometrically_complete());
    }

    #[test]
    fn no_flow_intent_is_explicitly_not_a_realization() {
        let mut graph = graph();
        graph
            .connections
            .retain(|connection| connection.relation != VoidRelation::FlowPath);
        let mesh = resolve_to_mesh(&CSGNode::cube());

        let report = FunctionalGraphRealizationReport::evaluate(&graph, &embedding(), &mesh);
        assert_eq!(report.status, GraphRealizationStatus::NoFlowPathsDeclared);
        assert_eq!(report.declared_flow_paths, 0);
        assert!(!report.is_geometrically_complete());
    }
}
