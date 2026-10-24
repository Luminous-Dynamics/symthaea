// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed functional-void graphs for passive physical design.
//!
//! A void graph is an *intent model*: it says which regions/ports a designer
//! claims should be functionally related. It does not infer those relations
//! from a mesh and does not certify geometric connectivity.

/// Stable identifier for a void region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub u32);

/// Stable identifier for a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PortId(pub u32);

/// Intended semantic role of a void region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VoidRegionRole {
    Inlet,
    Outlet,
    Junction,
    Cavity,
    Reservoir,
    Barrier,
    Sink,
    Unknown,
}

/// Intended physical relation between ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VoidRelation {
    FlowPath,
    PressureCoupling,
    ThermalPath,
    AcousticCoupling,
    ElectromagneticCoupling,
}

/// Explicit description of a port on a void region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoidPort {
    pub id: PortId,
    pub region: RegionId,
}

/// Explicit functional region in the void graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoidRegion {
    pub id: RegionId,
    pub role: VoidRegionRole,
}

/// A typed edge asserting an intended physical relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoidConnection {
    pub from: PortId,
    pub to: PortId,
    pub relation: VoidRelation,
    pub bidirectional: bool,
}

/// Structural validation errors for the intent graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoidGraphError {
    DuplicateRegion(RegionId),
    DuplicatePort(PortId),
    MissingRegionForPort(PortId),
    MissingPortForConnection(PortId),
    SelfConnection(PortId),
}

/// Deterministic, serializable functional-void intent graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FunctionalVoidGraph {
    pub regions: Vec<VoidRegion>,
    pub ports: Vec<VoidPort>,
    pub connections: Vec<VoidConnection>,
}

impl FunctionalVoidGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_region(&mut self, region: VoidRegion) -> Result<(), VoidGraphError> {
        if self.regions.iter().any(|existing| existing.id == region.id) {
            return Err(VoidGraphError::DuplicateRegion(region.id));
        }
        self.regions.push(region);
        self.regions.sort_unstable_by_key(|item| item.id);
        Ok(())
    }

    pub fn add_port(&mut self, port: VoidPort) -> Result<(), VoidGraphError> {
        if self.ports.iter().any(|existing| existing.id == port.id) {
            return Err(VoidGraphError::DuplicatePort(port.id));
        }
        if !self.regions.iter().any(|region| region.id == port.region) {
            return Err(VoidGraphError::MissingRegionForPort(port.id));
        }
        self.ports.push(port);
        self.ports.sort_unstable_by_key(|item| item.id);
        Ok(())
    }

    pub fn connect(&mut self, connection: VoidConnection) -> Result<(), VoidGraphError> {
        if connection.from == connection.to {
            return Err(VoidGraphError::SelfConnection(connection.from));
        }
        if !self.ports.iter().any(|port| port.id == connection.from) {
            return Err(VoidGraphError::MissingPortForConnection(connection.from));
        }
        if !self.ports.iter().any(|port| port.id == connection.to) {
            return Err(VoidGraphError::MissingPortForConnection(connection.to));
        }
        self.connections.push(connection);
        self.connections.sort_unstable_by_key(|item| {
            (item.from, item.to, item.relation, item.bidirectional)
        });
        Ok(())
    }

    /// Validate all graph references and uniqueness constraints.
    pub fn validate(&self) -> Result<(), VoidGraphError> {
        let mut region_ids = Vec::new();
        for region in &self.regions {
            if region_ids.contains(&region.id) {
                return Err(VoidGraphError::DuplicateRegion(region.id));
            }
            region_ids.push(region.id);
        }

        let mut port_ids = Vec::new();
        for port in &self.ports {
            if port_ids.contains(&port.id) {
                return Err(VoidGraphError::DuplicatePort(port.id));
            }
            if !region_ids.contains(&port.region) {
                return Err(VoidGraphError::MissingRegionForPort(port.id));
            }
            port_ids.push(port.id);
        }

        for connection in &self.connections {
            if connection.from == connection.to {
                return Err(VoidGraphError::SelfConnection(connection.from));
            }
            if !port_ids.contains(&connection.from) {
                return Err(VoidGraphError::MissingPortForConnection(connection.from));
            }
            if !port_ids.contains(&connection.to) {
                return Err(VoidGraphError::MissingPortForConnection(connection.to));
            }
        }

        Ok(())
    }

    /// Determine whether the *declared graph* contains a path between ports.
    ///
    /// This is graph reachability only. It must not be presented as a physical
    /// connectivity result until downstream geometry/physics evidence confirms it.
    pub fn declares_path(
        &self,
        from: PortId,
        to: PortId,
        relation: VoidRelation,
    ) -> bool {
        if self.validate().is_err() {
            return false;
        }

        let mut frontier = vec![from];
        let mut seen = vec![from];

        while let Some(current) = frontier.pop() {
            if current == to {
                return true;
            }

            for connection in &self.connections {
                if connection.relation != relation {
                    continue;
                }

                let next = if connection.from == current {
                    Some(connection.to)
                } else if connection.bidirectional && connection.to == current {
                    Some(connection.from)
                } else {
                    None
                };

                if let Some(next) = next {
                    if !seen.contains(&next) {
                        seen.push(next);
                        frontier.push(next);
                    }
                }
            }
        }

        false
    }

    /// Stable digest of the complete graph, independent of insertion order.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"functional-void-graph:v1");

        for region in &self.regions {
            hasher.update(&region.id.0.to_le_bytes());
            hasher.update(&[region_role_byte(region.role)]);
        }
        for port in &self.ports {
            hasher.update(&port.id.0.to_le_bytes());
            hasher.update(&port.region.0.to_le_bytes());
        }
        for connection in &self.connections {
            hasher.update(&connection.from.0.to_le_bytes());
            hasher.update(&connection.to.0.to_le_bytes());
            hasher.update(&[relation_byte(connection.relation)]);
            hasher.update(&[u8::from(connection.bidirectional)]);
        }

        *hasher.finalize().as_bytes()
    }
}

const fn region_role_byte(role: VoidRegionRole) -> u8 {
    match role {
        VoidRegionRole::Inlet => 0,
        VoidRegionRole::Outlet => 1,
        VoidRegionRole::Junction => 2,
        VoidRegionRole::Cavity => 3,
        VoidRegionRole::Reservoir => 4,
        VoidRegionRole::Barrier => 5,
        VoidRegionRole::Sink => 6,
        VoidRegionRole::Unknown => 7,
    }
}

const fn relation_byte(relation: VoidRelation) -> u8 {
    match relation {
        VoidRelation::FlowPath => 0,
        VoidRelation::PressureCoupling => 1,
        VoidRelation::ThermalPath => 2,
        VoidRelation::AcousticCoupling => 3,
        VoidRelation::ElectromagneticCoupling => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_flow_graph() -> FunctionalVoidGraph {
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

    #[test]
    fn declared_path_is_distinct_from_physical_proof() {
        let graph = simple_flow_graph();
        assert!(graph.declares_path(
            PortId(10),
            PortId(20),
            VoidRelation::FlowPath
        ));
    }

    #[test]
    fn reverse_flow_requires_bidirectional_intent() {
        let graph = simple_flow_graph();
        assert!(!graph.declares_path(
            PortId(20),
            PortId(10),
            VoidRelation::FlowPath
        ));
    }

    #[test]
    fn graph_is_insertion_order_independent() {
        let a = simple_flow_graph();

        let mut b = FunctionalVoidGraph::new();
        b.add_region(VoidRegion {
            id: RegionId(2),
            role: VoidRegionRole::Outlet,
        })
        .unwrap();
        b.add_region(VoidRegion {
            id: RegionId(1),
            role: VoidRegionRole::Inlet,
        })
        .unwrap();
        b.add_port(VoidPort {
            id: PortId(20),
            region: RegionId(2),
        })
        .unwrap();
        b.add_port(VoidPort {
            id: PortId(10),
            region: RegionId(1),
        })
        .unwrap();
        b.connect(VoidConnection {
            from: PortId(10),
            to: PortId(20),
            relation: VoidRelation::FlowPath,
            bidirectional: false,
        })
        .unwrap();

        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn invalid_reference_is_rejected() {
        let mut graph = FunctionalVoidGraph::new();
        assert_eq!(
            graph.add_port(VoidPort {
                id: PortId(1),
                region: RegionId(999),
            }),
            Err(VoidGraphError::MissingRegionForPort(PortId(1)))
        );
    }

    #[test]
    fn multiple_physical_relations_can_share_ports() {
        let mut graph = simple_flow_graph();
        graph
            .connect(VoidConnection {
                from: PortId(10),
                to: PortId(20),
                relation: VoidRelation::ThermalPath,
                bidirectional: true,
            })
            .unwrap();
        assert!(graph.declares_path(
            PortId(10),
            PortId(20),
            VoidRelation::ThermalPath
        ));
    }
}
