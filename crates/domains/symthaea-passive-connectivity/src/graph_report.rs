// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Whole-graph realization comparison for passive void candidates.
//!
//! This layer compares declared FlowPath intent edges against the candidate
//! mesh. A connected result means only that the two port anchors land in the
//! same mesh component. It does not establish pressure-flow capability,
//! directionality, hydraulic performance, or any other transport property.

use crate::{evaluate_port_path, PortPathEvidence, PortPathStatus};
use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_passive_void_compiler::GeometryEmbedding;
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId, VoidRelation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRealizationStatus {
    InvalidGraph,
    NoFlowPathsDeclared,
    AllDeclaredPathsConnected,
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
    /// Geometry/topology evidence only; transport remains unproven.
    pub physical_transport_unproven: bool,
}

impl FunctionalGraphRealizationReport {
    pub fn evaluate(
        graph: &FunctionalVoidGraph,
        embedding: &GeometryEmbedding,
        candidate: &TriangleMesh,
    ) -> Self {
        if graph.validate().is_err() {
            return Self {
                status: GraphRealizationStatus::InvalidGraph,
                declared_flow_paths: 0,
                connected_paths: 0,
                disconnected_paths: 0,
                invalid_paths: 0,
                path_results: Vec::new(),
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
                physical_transport_unproven: true,
            };
        }

        let mut connected_paths = 0;
        let mut disconnected_paths = 0;
        let mut invalid_paths = 0;
        let mut path_results = Vec::with_capacity(flow_connections.len());

        for connection in flow_connections {
            let evidence = evaluate_port_path(
                graph,
                embedding,
                candidate,
                connection.from,
                connection.to,
            );

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
        let status = if connected_paths == declared_flow_paths {
            GraphRealizationStatus::AllDeclaredPathsConnected
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
            physical_transport_unproven: true,
        }
    }

    pub fn is_geometrically_complete(&self) -> bool {
        matches!(self.status, GraphRealizationStatus::AllDeclaredPathsConnected)
            && self.invalid_paths == 0
            && self.disconnected_paths == 0
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
        assert!(report.is_geometrically_complete());
        assert!(report.physical_transport_unproven);
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
