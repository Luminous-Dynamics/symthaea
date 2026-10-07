//! Deterministic semantic schedule derived from an execution dependency graph.
//!
//! The ordinal assigned to a node is a semantic execution position only.
//! Backend lowering is responsible for translating dependencies into concrete
//! synchronization primitives without changing the graph semantics.

use std::collections::{BTreeMap, HashSet};

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{DependencyKind, ExecutionGraph, GraphError, ResourceId, MAX_GRAPH_EDGES, MAX_GRAPH_NODES};

pub const EXECUTION_SCHEDULE_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScheduledNode {
    pub id: u32,
    /// Deterministic semantic position. This is not a Vulkan semaphore value.
    pub ordinal: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ScheduledDependency {
    pub from: u32,
    pub to: u32,
    pub from_ordinal: u32,
    pub to_ordinal: u32,
    pub resource: ResourceId,
    pub kind: DependencyKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionSchedule {
    pub version: u16,
    pub graph_digest: String,
    pub nodes: Vec<ScheduledNode>,
    pub dependencies: Vec<ScheduledDependency>,
}

impl ExecutionSchedule {
    pub fn from_graph(graph: &ExecutionGraph) -> Result<Self, ScheduleError> {
        graph.validate().map_err(ScheduleError::Graph)?;
        let order = graph.topological_order().map_err(ScheduleError::Graph)?;
        let ordinal_by_id = order.iter().copied().enumerate().map(|(n, id)| (id, n as u32)).collect::<BTreeMap<_, _>>();
        let nodes = order.iter().copied().enumerate().map(|(n, id)| ScheduledNode { id, ordinal: n as u32 }).collect::<Vec<_>>();
        let mut dependencies = graph.dependencies.iter().map(|edge| {
            Ok(ScheduledDependency {
                from: edge.from,
                to: edge.to,
                from_ordinal: *ordinal_by_id.get(&edge.from).ok_or(ScheduleError::UnknownNode(edge.from))?,
                to_ordinal: *ordinal_by_id.get(&edge.to).ok_or(ScheduleError::UnknownNode(edge.to))?,
                resource: edge.resource.clone(),
                kind: edge.kind,
            })
        }).collect::<Result<Vec<_>, ScheduleError>>()?;
        dependencies.sort();
        let schedule = Self {
            version: EXECUTION_SCHEDULE_VERSION,
            graph_digest: graph.digest_hex().map_err(ScheduleError::Graph)?,
            nodes,
            dependencies,
        };
        schedule.verify_against(graph)?;
        Ok(schedule)
    }

    pub fn verify_against(&self, graph: &ExecutionGraph) -> Result<(), ScheduleError> {
        graph.validate().map_err(ScheduleError::Graph)?;
        if self.version != EXECUTION_SCHEDULE_VERSION {
            return Err(ScheduleError::UnsupportedVersion(self.version));
        }
        let expected_digest = graph.digest_hex().map_err(ScheduleError::Graph)?;
        if self.graph_digest != expected_digest {
            return Err(ScheduleError::GraphDigestMismatch);
        }
        let expected = Self::canonical_from_graph(graph)?;
        if self.nodes != expected.nodes {
            return Err(ScheduleError::NodeScheduleMismatch);
        }
        if self.dependencies != expected.dependencies {
            return Err(ScheduleError::DependencyScheduleMismatch);
        }
        self.validate_internal()
    }

    pub fn digest(&self) -> Result<[u8; 32], ScheduleError> {
        self.validate_internal()?;
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-gpu-fabric.execution-schedule.v1\0");
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&(self.graph_digest.len() as u32).to_le_bytes());
        hasher.update(self.graph_digest.as_bytes());
        hasher.update(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            hasher.update(&node.id.to_le_bytes());
            hasher.update(&node.ordinal.to_le_bytes());
        }
        hasher.update(&(self.dependencies.len() as u32).to_le_bytes());
        for dep in &self.dependencies {
            hasher.update(&dep.from.to_le_bytes());
            hasher.update(&dep.to.to_le_bytes());
            hasher.update(&dep.from_ordinal.to_le_bytes());
            hasher.update(&dep.to_ordinal.to_le_bytes());
            hasher.update(&(dep.resource.as_str().len() as u32).to_le_bytes());
            hasher.update(dep.resource.as_str().as_bytes());
            hasher.update(&[match dep.kind {
                DependencyKind::ReadAfterWrite => 1,
                DependencyKind::WriteAfterRead => 2,
                DependencyKind::WriteAfterWrite => 3,
            }]);
        }
        Ok(*hasher.finalize().as_bytes())
    }

    pub fn digest_hex(&self) -> Result<String, ScheduleError> {
        Ok(self.digest()?.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    pub fn direct_prerequisites(&self, node_id: u32) -> Result<Vec<u32>, ScheduleError> {
        self.validate_internal()?;
        if !self.nodes.iter().any(|node| node.id == node_id) {
            return Err(ScheduleError::UnknownNode(node_id));
        }
        let mut prerequisites = self.dependencies.iter().filter(|dep| dep.to == node_id).map(|dep| dep.from).collect::<Vec<_>>();
        prerequisites.sort_unstable();
        prerequisites.dedup();
        Ok(prerequisites)
    }

    fn canonical_from_graph(graph: &ExecutionGraph) -> Result<Self, ScheduleError> {
        let order = graph.topological_order().map_err(ScheduleError::Graph)?;
        let ordinal_by_id = order.iter().copied().enumerate().map(|(n, id)| (id, n as u32)).collect::<BTreeMap<_, _>>();
        let nodes = order.iter().copied().enumerate().map(|(n, id)| ScheduledNode { id, ordinal: n as u32 }).collect::<Vec<_>>();
        let mut dependencies = graph.dependencies.iter().map(|edge| ScheduledDependency {
            from: edge.from,
            to: edge.to,
            from_ordinal: ordinal_by_id[&edge.from],
            to_ordinal: ordinal_by_id[&edge.to],
            resource: edge.resource.clone(),
            kind: edge.kind,
        }).collect::<Vec<_>>();
        dependencies.sort();
        Ok(Self {
            version: EXECUTION_SCHEDULE_VERSION,
            graph_digest: graph.digest_hex().map_err(ScheduleError::Graph)?,
            nodes,
            dependencies,
        })
    }

    fn validate_internal(&self) -> Result<(), ScheduleError> {
        if self.version != EXECUTION_SCHEDULE_VERSION {
            return Err(ScheduleError::UnsupportedVersion(self.version));
        }
        if self.nodes.len() > MAX_GRAPH_NODES {
            return Err(ScheduleError::NodeLimitExceeded(self.nodes.len()));
        }
        if self.dependencies.len() > MAX_GRAPH_EDGES {
            return Err(ScheduleError::EdgeLimitExceeded(self.dependencies.len()));
        }
        let mut ids = HashSet::with_capacity(self.nodes.len());
        let mut ordinals = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if !ids.insert(node.id) {
                return Err(ScheduleError::DuplicateNode(node.id));
            }
            if !ordinals.insert(node.ordinal) {
                return Err(ScheduleError::DuplicateOrdinal);
            }
        }
        let mut sorted = ordinals.into_iter().collect::<Vec<_>>();
        sorted.sort_unstable();
        if sorted.iter().copied().enumerate().any(|(expected, ordinal)| ordinal != expected as u32) {
            return Err(ScheduleError::NonContiguousOrdinals);
        }
        let node_by_id = self.nodes.iter().map(|node| (node.id, node)).collect::<BTreeMap<_, _>>();
        let mut dependencies = HashSet::with_capacity(self.dependencies.len());
        for dep in &self.dependencies {
            let from = node_by_id.get(&dep.from).ok_or(ScheduleError::UnknownNode(dep.from))?;
            let to = node_by_id.get(&dep.to).ok_or(ScheduleError::UnknownNode(dep.to))?;
            if dep.from_ordinal != from.ordinal || dep.to_ordinal != to.ordinal {
                return Err(ScheduleError::OrdinalMismatch);
            }
            if dep.from_ordinal >= dep.to_ordinal {
                return Err(ScheduleError::NonForwardDependency { from: dep.from, to: dep.to });
            }
            if !dependencies.insert(dep.clone()) {
                return Err(ScheduleError::DuplicateDependency);
            }
        }
        Ok(())
    }
}

impl TryFrom<&ExecutionGraph> for ExecutionSchedule {
    type Error = ScheduleError;

    fn try_from(graph: &ExecutionGraph) -> Result<Self, Self::Error> {
        Self::from_graph(graph)
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("invalid execution graph: {0}")]
    Graph(GraphError),
    #[error("unsupported execution schedule version {0}")]
    UnsupportedVersion(u16),
    #[error("execution graph digest does not match the schedule")]
    GraphDigestMismatch,
    #[error("schedule node order does not match the canonical graph order")]
    NodeScheduleMismatch,
    #[error("schedule dependency mapping does not match the canonical graph dependencies")]
    DependencyScheduleMismatch,
    #[error("unknown execution node {0}")]
    UnknownNode(u32),
    #[error("execution schedule node count {0} exceeds the graph bound")]
    NodeLimitExceeded(usize),
    #[error("execution schedule dependency count {0} exceeds the graph bound")]
    EdgeLimitExceeded(usize),
    #[error("execution schedule contains duplicate node {0}")]
    DuplicateNode(u32),
    #[error("execution schedule contains duplicate ordinals")]
    DuplicateOrdinal,
    #[error("execution schedule ordinals are not contiguous from zero")]
    NonContiguousOrdinals,
    #[error("execution schedule dependency ordinal mismatch")]
    OrdinalMismatch,
    #[error("execution schedule dependency {from}->{to} is not forward")]
    NonForwardDependency { from: u32, to: u32 },
    #[error("execution schedule contains a duplicate dependency")]
    DuplicateDependency,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessKind, DependencyEdge, ExecutionNode, GpuOperation, ResourceUse};

    fn resource(name: &str) -> ResourceId { ResourceId::new(name).unwrap() }

    fn node(id: u32, r: &ResourceId, access: AccessKind) -> ExecutionNode {
        ExecutionNode::new(id, GpuOperation::HdcBindXor { dimensions: 8 }, vec![ResourceUse::new(r.clone(), access)])
    }

    #[test]
    fn schedule_is_deterministic() {
        let r = resource("hv");
        let graph = ExecutionGraph::new(
            vec![node(20, &r, AccessKind::Read), node(10, &r, AccessKind::Write)],
            vec![DependencyEdge::new(10, 20, r, DependencyKind::ReadAfterWrite)],
        ).unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        assert_eq!(schedule.nodes, vec![ScheduledNode { id: 10, ordinal: 0 }, ScheduledNode { id: 20, ordinal: 1 }]);
        assert_eq!(schedule.direct_prerequisites(20).unwrap(), vec![10]);
        assert!(!schedule.digest_hex().unwrap().is_empty());
    }

    #[test]
    fn insertion_order_does_not_change_schedule_digest() {
        let r = resource("hv");
        let graph_a = ExecutionGraph::new(
            vec![node(2, &r, AccessKind::Read), node(1, &r, AccessKind::Write)],
            vec![DependencyEdge::new(1, 2, r.clone(), DependencyKind::ReadAfterWrite)],
        ).unwrap();
        let graph_b = ExecutionGraph::new(
            vec![node(1, &r, AccessKind::Write), node(2, &r, AccessKind::Read)],
            vec![DependencyEdge::new(1, 2, r, DependencyKind::ReadAfterWrite)],
        ).unwrap();
        let a = ExecutionSchedule::from_graph(&graph_a).unwrap();
        let b = ExecutionSchedule::from_graph(&graph_b).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn transitive_dependency_is_not_fabricated_as_direct_wait() {
        let r = resource("hv");
        let graph = ExecutionGraph::new(
            vec![node(1, &r, AccessKind::Write), node(2, &r, AccessKind::ReadWrite), node(3, &r, AccessKind::Read)],
            vec![
                DependencyEdge::new(1, 2, r.clone(), DependencyKind::ReadAfterWrite),
                DependencyEdge::new(2, 3, r, DependencyKind::ReadAfterWrite),
            ],
        ).unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        assert_eq!(schedule.direct_prerequisites(3).unwrap(), vec![2]);
    }

    #[test]
    fn tampering_schedule_ordinals_is_rejected() {
        let r = resource("hv");
        let graph = ExecutionGraph::new(
            vec![node(1, &r, AccessKind::Write), node(2, &r, AccessKind::Read)],
            vec![DependencyEdge::new(1, 2, r, DependencyKind::ReadAfterWrite)],
        ).unwrap();
        let mut schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        schedule.dependencies[0].from_ordinal = schedule.dependencies[0].to_ordinal;
        assert!(matches!(schedule.digest(), Err(ScheduleError::NonForwardDependency { .. })));
    }
}