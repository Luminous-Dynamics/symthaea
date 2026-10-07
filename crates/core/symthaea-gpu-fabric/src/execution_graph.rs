//! Backend-neutral execution dependency graph for Symthaea GPU Fabric.
//!
//! The graph is deliberately independent of Vulkan/WebGPU. It expresses semantic
//! resource hazards and a deterministic partial order. Backends may lower that
//! order to their own synchronization primitives.

use std::collections::{BTreeSet, HashMap, HashSet};

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::GpuOperation;

pub const EXECUTION_GRAPH_VERSION: u16 = 1;
pub const MAX_GRAPH_NODES: usize = 4096;
pub const MAX_GRAPH_EDGES: usize = 16_384;
pub const MAX_RESOURCE_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceId(String);

impl ResourceId {
    pub fn new(value: impl Into<String>) -> Result<Self, GraphError> {
        let value = value.into();
        if value.is_empty() {
            return Err(GraphError::EmptyResourceId);
        }
        if value.len() > MAX_RESOURCE_ID_BYTES {
            return Err(GraphError::ResourceIdTooLong {
                bytes: value.len(),
                max: MAX_RESOURCE_ID_BYTES,
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

impl AccessKind {
    const fn is_read(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    const fn is_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUse {
    pub resource: ResourceId,
    pub access: AccessKind,
}

impl ResourceUse {
    pub fn new(resource: ResourceId, access: AccessKind) -> Self {
        Self { resource, access }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionNode {
    pub id: u32,
    pub operation: GpuOperation,
    pub resources: Vec<ResourceUse>,
}

impl ExecutionNode {
    pub fn new(id: u32, operation: GpuOperation, resources: Vec<ResourceUse>) -> Self {
        Self {
            id,
            operation,
            resources,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DependencyKind {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub from: u32,
    pub to: u32,
    pub resource: ResourceId,
    pub kind: DependencyKind,
}

impl DependencyEdge {
    pub fn new(from: u32, to: u32, resource: ResourceId, kind: DependencyKind) -> Self {
        Self {
            from,
            to,
            resource,
            kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub version: u16,
    pub nodes: Vec<ExecutionNode>,
    pub dependencies: Vec<DependencyEdge>,
}

impl ExecutionGraph {
    pub fn new(
        nodes: Vec<ExecutionNode>,
        dependencies: Vec<DependencyEdge>,
    ) -> Result<Self, GraphError> {
        let graph = Self {
            version: EXECUTION_GRAPH_VERSION,
            nodes,
            dependencies,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        if self.version != EXECUTION_GRAPH_VERSION {
            return Err(GraphError::UnsupportedVersion(self.version));
        }
        if self.nodes.len() > MAX_GRAPH_NODES {
            return Err(GraphError::NodeLimitExceeded {
                nodes: self.nodes.len(),
                max: MAX_GRAPH_NODES,
            });
        }
        if self.dependencies.len() > MAX_GRAPH_EDGES {
            return Err(GraphError::EdgeLimitExceeded {
                edges: self.dependencies.len(),
                max: MAX_GRAPH_EDGES,
            });
        }

        let mut nodes = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if nodes.insert(node.id, node).is_some() {
                return Err(GraphError::DuplicateNode(node.id));
            }

            let mut resources = HashSet::with_capacity(node.resources.len());
            for use_ in &node.resources {
                if !resources.insert(use_.resource.clone()) {
                    return Err(GraphError::DuplicateResourceUse {
                        node: node.id,
                        resource: use_.resource.clone(),
                    });
                }
            }
        }

        let mut edges = HashSet::with_capacity(self.dependencies.len());
        for edge in &self.dependencies {
            if edge.from == edge.to {
                return Err(GraphError::SelfDependency(edge.from));
            }

            let from = nodes
                .get(&edge.from)
                .ok_or(GraphError::UnknownNode(edge.from))?;
            let to = nodes
                .get(&edge.to)
                .ok_or(GraphError::UnknownNode(edge.to))?;

            if !from
                .resources
                .iter()
                .any(|use_| use_.resource == edge.resource)
                || !to
                    .resources
                    .iter()
                    .any(|use_| use_.resource == edge.resource)
            {
                return Err(GraphError::EdgeResourceNotShared {
                    from: edge.from,
                    to: edge.to,
                    resource: edge.resource.clone(),
                });
            }

            if !edge_kind_is_valid(from, to, &edge.resource, edge.kind) {
                return Err(GraphError::InvalidDependencyKind {
                    from: edge.from,
                    to: edge.to,
                    resource: edge.resource.clone(),
                    kind: edge.kind,
                });
            }

            if !edges.insert(edge.clone()) {
                return Err(GraphError::DuplicateDependency(edge.clone()));
            }
        }

        let topological_order = self.deterministic_topological_order()?;
        let node_ids = topological_order.clone();
        let mut index_by_id = HashMap::with_capacity(node_ids.len());
        for (index, id) in node_ids.iter().copied().enumerate() {
            index_by_id.insert(id, index);
        }

        let word_count = node_ids.len().div_ceil(64);
        let mut reachability = vec![vec![0_u64; word_count]; node_ids.len()];
        let mut adjacency = vec![Vec::<usize>::new(); node_ids.len()];

        for edge in &self.dependencies {
            let from = *index_by_id
                .get(&edge.from)
                .ok_or(GraphError::UnknownNode(edge.from))?;
            let to = *index_by_id
                .get(&edge.to)
                .ok_or(GraphError::UnknownNode(edge.to))?;
            adjacency[from].push(to);
        }

        for &node_id in topological_order.iter().rev() {
            let node_index = *index_by_id
                .get(&node_id)
                .ok_or(GraphError::UnknownNode(node_id))?;

            for &successor in &adjacency[node_index] {
                let (before, after) = reachability.split_at_mut(successor);
                let current = &mut before[node_index];
                let successor_row = &after[0];

                current[successor / 64] |= 1_u64 << (successor % 64);
                for word in 0..word_count {
                    current[word] |= successor_row[word];
                }
            }
        }

        for left_index in 0..self.nodes.len() {
            for right_index in left_index + 1..self.nodes.len() {
                let left = &self.nodes[left_index];
                let right = &self.nodes[right_index];
                let left_graph_index = *index_by_id
                    .get(&left.id)
                    .ok_or(GraphError::UnknownNode(left.id))?;
                let right_graph_index = *index_by_id
                    .get(&right.id)
                    .ok_or(GraphError::UnknownNode(right.id))?;

                for resource in shared_write_resources(left, right) {
                    let ordered = (reachability[left_graph_index][right_graph_index / 64]
                        & (1_u64 << (right_graph_index % 64)))
                        != 0
                        || (reachability[right_graph_index][left_graph_index / 64]
                            & (1_u64 << (left_graph_index % 64)))
                            != 0;

                    if !ordered {
                        return Err(GraphError::UnorderedResourceConflict {
                            left: left.id,
                            right: right.id,
                            resource,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    pub fn topological_order(&self) -> Result<Vec<u32>, GraphError> {
        self.validate()?;
        self.deterministic_topological_order()
    }

    fn deterministic_topological_order(&self) -> Result<Vec<u32>, GraphError> {
        let mut indegree = self
            .nodes
            .iter()
            .map(|node| (node.id, 0_u32))
            .collect::<HashMap<_, _>>();

        let mut adjacency = self
            .nodes
            .iter()
            .map(|node| (node.id, Vec::<u32>::new()))
            .collect::<HashMap<_, _>>();

        for edge in &self.dependencies {
            *indegree
                .get_mut(&edge.to)
                .ok_or(GraphError::UnknownNode(edge.to))? += 1;
            adjacency
                .get_mut(&edge.from)
                .ok_or(GraphError::UnknownNode(edge.from))?
                .push(edge.to);
        }

        for successors in adjacency.values_mut() {
            successors.sort_unstable();
        }

        let mut ready = BTreeSet::new();
        for (&id, &degree) in &indegree {
            if degree == 0 {
                ready.insert(id);
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);

            for &successor in adjacency
                .get(&id)
                .ok_or(GraphError::UnknownNode(id))?
            {
                let degree = indegree
                    .get_mut(&successor)
                    .ok_or(GraphError::UnknownNode(successor))?;
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(successor);
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected(
                self.nodes
                    .iter()
                    .map(|node| node.id)
                    .find(|id| !order.contains(id))
                    .unwrap_or_default(),
            ));
        }

        Ok(order)
    }

    /// Digest of the exact semantic graph, independent of insertion order.
    pub fn digest(&self) -> Result<[u8; 32], GraphError> {
        self.validate()?;

        let mut nodes = self.nodes.clone();
        nodes.sort_by_key(|node| node.id);
        for node in &mut nodes {
            node.resources.sort_by(|left, right| {
                left.resource
                    .cmp(&right.resource)
                    .then(left.access.cmp(&right.access))
            });
        }

        let mut dependencies = self.dependencies.clone();
        dependencies.sort();

        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-gpu-fabric.execution-graph.v1\0");
        hasher.update(&self.version.to_le_bytes());

        hasher.update(&(nodes.len() as u32).to_le_bytes());
        for node in nodes {
            hasher.update(&node.id.to_le_bytes());
            match node.operation {
                GpuOperation::HdcBindXor { dimensions } => {
                    hasher.update(&[1]);
                    hasher.update(&dimensions.to_le_bytes());
                }
            }

            hasher.update(&(node.resources.len() as u32).to_le_bytes());
            for use_ in node.resources {
                hasher.update(&(use_.resource.0.len() as u32).to_le_bytes());
                hasher.update(use_.resource.0.as_bytes());
                hasher.update(&[match use_.access {
                    AccessKind::Read => 1,
                    AccessKind::Write => 2,
                    AccessKind::ReadWrite => 3,
                }]);
            }
        }

        hasher.update(&(dependencies.len() as u32).to_le_bytes());
        for edge in dependencies {
            hasher.update(&edge.from.to_le_bytes());
            hasher.update(&edge.to.to_le_bytes());
            hasher.update(&(edge.resource.0.len() as u32).to_le_bytes());
            hasher.update(edge.resource.0.as_bytes());
            hasher.update(&[match edge.kind {
                DependencyKind::ReadAfterWrite => 1,
                DependencyKind::WriteAfterRead => 2,
                DependencyKind::WriteAfterWrite => 3,
            }]);
        }

        Ok(*hasher.finalize().as_bytes())
    }

    pub fn digest_hex(&self) -> Result<String, GraphError> {
        Ok(self
            .digest()?
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

fn edge_kind_is_valid(
    from: &ExecutionNode,
    to: &ExecutionNode,
    resource: &ResourceId,
    kind: DependencyKind,
) -> bool {
    let Some(from_access) = from
        .resources
        .iter()
        .find(|use_| &use_.resource == resource)
        .map(|use_| use_.access)
    else {
        return false;
    };

    let Some(to_access) = to
        .resources
        .iter()
        .find(|use_| &use_.resource == resource)
        .map(|use_| use_.access)
    else {
        return false;
    };

    match kind {
        DependencyKind::ReadAfterWrite => from_access.is_write() && to_access.is_read(),
        DependencyKind::WriteAfterRead => from_access.is_read() && to_access.is_write(),
        DependencyKind::WriteAfterWrite => from_access.is_write() && to_access.is_write(),
    }
}

fn shared_write_resources(
    left: &ExecutionNode,
    right: &ExecutionNode,
) -> BTreeSet<ResourceId> {
    let mut resources = BTreeSet::new();

    for left_use in &left.resources {
        if !left_use.access.is_write() {
            continue;
        }

        if right
            .resources
            .iter()
            .any(|use_| use_.resource == left_use.resource)
        {
            resources.insert(left_use.resource.clone());
        }
    }

    for right_use in &right.resources {
        if !right_use.access.is_write() {
            continue;
        }

        if left
            .resources
            .iter()
            .any(|use_| use_.resource == right_use.resource)
        {
            resources.insert(right_use.resource.clone());
        }
    }

    resources
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GraphError {
    #[error("unsupported execution graph version {0}")]
    UnsupportedVersion(u16),
    #[error("execution graph node count {nodes} exceeds maximum {max}")]
    NodeLimitExceeded { nodes: usize, max: usize },
    #[error("execution graph edge count {edges} exceeds maximum {max}")]
    EdgeLimitExceeded { edges: usize, max: usize },
    #[error("resource id must not be empty")]
    EmptyResourceId,
    #[error("resource id is {bytes} bytes; maximum is {max}")]
    ResourceIdTooLong { bytes: usize, max: usize },
    #[error("duplicate execution node {0}")]
    DuplicateNode(u32),
    #[error("duplicate resource use on node {node}: {resource}")]
    DuplicateResourceUse { node: u32, resource: ResourceId },
    #[error("self dependency on node {0}")]
    SelfDependency(u32),
    #[error("unknown execution node {0}")]
    UnknownNode(u32),
    #[error("dependency resource {resource} is not declared by both nodes {from} and {to}")]
    EdgeResourceNotShared {
        from: u32,
        to: u32,
        resource: ResourceId,
    },
    #[error("dependency kind is invalid for nodes {from}->{to} and resource {resource}: {kind:?}")]
    InvalidDependencyKind {
        from: u32,
        to: u32,
        resource: ResourceId,
        kind: DependencyKind,
    },
    #[error("duplicate dependency {0:?}")]
    DuplicateDependency(DependencyEdge),
    #[error("execution graph contains a cycle involving node {0}")]
    CycleDetected(u32),
    #[error("unordered write/read conflict between nodes {left} and {right} on resource {resource}")]
    UnorderedResourceConflict {
        left: u32,
        right: u32,
        resource: ResourceId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(name: &str) -> ResourceId {
        ResourceId::new(name).unwrap()
    }

    fn op(dimensions: u32) -> GpuOperation {
        GpuOperation::HdcBindXor { dimensions }
    }

    #[test]
    fn read_after_write_dependency_is_valid() {
        let r = resource("hv");
        let graph = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Read)]),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                r,
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();

        assert_eq!(graph.topological_order().unwrap(), vec![1, 2]);
    }

    #[test]
    fn missing_hazard_dependency_is_rejected() {
        let r = resource("hv");
        let error = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r, AccessKind::Read)]),
            ],
            Vec::new(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            GraphError::UnorderedResourceConflict { .. }
        ));
    }

    #[test]
    fn read_read_does_not_require_ordering() {
        let r = resource("hv");
        let graph = ExecutionGraph::new(
            vec![
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Read)]),
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r, AccessKind::Read)]),
            ],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(graph.topological_order().unwrap(), vec![1, 2]);
    }

    #[test]
    fn dependency_kind_is_checked_against_accesses() {
        let r = resource("hv");
        let error = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Read)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r, AccessKind::Read)]),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                resource("hv"),
                DependencyKind::WriteAfterRead,
            )],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            GraphError::InvalidDependencyKind { .. }
        ));
    }

    #[test]
    fn cycles_are_rejected() {
        let shared = resource("shared");
        let graph = ExecutionGraph {
            version: EXECUTION_GRAPH_VERSION,
            nodes: vec![
                ExecutionNode::new(
                    1,
                    op(8),
                    vec![ResourceUse::new(shared.clone(), AccessKind::Write)],
                ),
                ExecutionNode::new(
                    2,
                    op(8),
                    vec![ResourceUse::new(shared.clone(), AccessKind::Write)],
                ),
            ],
            dependencies: vec![
                DependencyEdge::new(
                    1,
                    2,
                    shared.clone(),
                    DependencyKind::WriteAfterWrite,
                ),
                DependencyEdge::new(
                    2,
                    1,
                    shared,
                    DependencyKind::WriteAfterWrite,
                ),
            ],
        };

        assert!(matches!(
            graph.validate(),
            Err(GraphError::CycleDetected(_))
        ));
    }

    #[test]
    fn one_node_order_can_protect_multiple_shared_resources() {
        let first = resource("first");
        let second = resource("second");
        let graph = ExecutionGraph::new(
            vec![
                ExecutionNode::new(
                    1,
                    op(8),
                    vec![
                        ResourceUse::new(first.clone(), AccessKind::Write),
                        ResourceUse::new(second.clone(), AccessKind::Write),
                    ],
                ),
                ExecutionNode::new(
                    2,
                    op(8),
                    vec![
                        ResourceUse::new(first.clone(), AccessKind::Read),
                        ResourceUse::new(second.clone(), AccessKind::Read),
                    ],
                ),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                first,
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();

        assert_eq!(graph.topological_order().unwrap(), vec![1, 2]);
    }

    #[test]
    fn digest_and_order_are_insertion_order_independent() {
        let r = resource("hv");
        let first = ExecutionGraph::new(
            vec![
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Read)]),
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                r.clone(),
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();

        let second = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r, AccessKind::Read)]),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                resource("hv"),
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();

        assert_eq!(first.topological_order().unwrap(), vec![1, 2]);
        assert_eq!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn changing_dependency_changes_digest() {
        let r = resource("hv");
        let base = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r.clone(), AccessKind::ReadWrite)]),
            ],
            vec![DependencyEdge::new(
                1,
                2,
                r.clone(),
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();

        let different = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, op(8), vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, op(8), vec![ResourceUse::new(r.clone(), AccessKind::ReadWrite)]),
            ],
            vec![DependencyEdge::new(
                2,
                1,
                r,
                DependencyKind::WriteAfterWrite,
            )],
        )
        .unwrap();

        assert_ne!(base.digest_hex().unwrap(), different.digest_hex().unwrap());
    }

    #[test]
    fn resource_ids_are_bounded() {
        assert!(matches!(
            ResourceId::new(""),
            Err(GraphError::EmptyResourceId)
        ));
        assert!(matches!(
            ResourceId::new("x".repeat(MAX_RESOURCE_ID_BYTES + 1)),
            Err(GraphError::ResourceIdTooLong { .. })
        ));
    }
}
