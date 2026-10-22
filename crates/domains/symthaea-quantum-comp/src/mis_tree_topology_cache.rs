//! Exact cache identities for tree-shaped rooted local QAOA neighborhoods.
//!
//! Only induced neighborhoods proven to be trees are canonicalized here. Cyclic
//! neighborhoods return `None` rather than risking unsafe reuse. The resulting
//! structural identity is not, by itself, a complete score-cache key: callers must
//! also bind the exact QAOA angle manifest, observable, evaluator/precision, and
//! execution profile with `MisTreeCacheContext`.

use core::fmt;
use std::collections::VecDeque;

/// Error while validating a graph used to extract a rooted neighborhood.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisTreeTopologyError {
    /// Root or edge endpoint is outside the declared graph.
    VertexOutOfRange {
        /// Invalid vertex index.
        vertex: usize,
        /// Declared vertex count.
        vertex_count: usize,
    },
    /// A simple undirected graph cannot contain a self-loop.
    SelfLoop {
        /// Vertex with the self-loop.
        vertex: usize,
    },
}

impl fmt::Display for MisTreeTopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange { vertex, vertex_count } => write!(
                f,
                "vertex {vertex} is outside graph with {vertex_count} vertices"
            ),
            Self::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex} is not a simple graph edge"),
        }
    }
}

impl std::error::Error for MisTreeTopologyError {}

/// Label-invariant canonical form for an induced rooted tree neighborhood.
///
/// `canonical_form` is a versioned-shape-independent parenthesized tree encoding:
/// each node is represented by parentheses around its lexicographically sorted child
/// encodings. The caller should persist the format version alongside long-lived cache
/// artifacts; a future encoding change must invalidate old keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootedTreeSignature {
    radius: usize,
    vertex_count: usize,
    edge_count: usize,
    canonical_form: String,
}

impl RootedTreeSignature {
    /// Graph-distance radius used to extract the neighborhood.
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Number of vertices in the induced neighborhood.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Number of edges in the induced neighborhood.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Exact rooted-tree canonical form, independent of input vertex labels.
    pub fn canonical_form(&self) -> &str {
        &self.canonical_form
    }

    /// Versioned key payload suitable for storage or hashing by a caller.
    pub fn versioned_form(&self) -> String {
        format!(
            "symthaea-rooted-tree-v1|radius={}|vertices={}|edges={}|{}",
            self.radius, self.vertex_count, self.edge_count, self.canonical_form
        )
    }
}

/// Extract a canonical rooted neighborhood if its induced graph is a tree.
///
/// The graph is interpreted as simple and undirected. Duplicate/reversed edge records
/// are normalized away; self-loops and out-of-range endpoints are rejected. The
/// neighborhood contains every vertex within graph distance `radius` of `root` and
/// every original graph edge whose endpoints are both in that set. If the induced
/// neighborhood has a cycle, the function returns `Ok(None)`: no canonical form or
/// cache reuse is authorized by this tree-only implementation.
///
/// The signature is exact for rooted-tree isomorphism, but it does not decide whether
/// that topology fully captures a particular QAOA score. The caller must bind it to
/// the precise circuit light-cone/evaluator semantics and other cache-context fields.
pub fn rooted_tree_neighborhood_signature(
    vertex_count: usize,
    edges: &[(usize, usize)],
    root: usize,
    radius: usize,
) -> Result<Option<RootedTreeSignature>, MisTreeTopologyError> {
    if root >= vertex_count {
        return Err(MisTreeTopologyError::VertexOutOfRange {
            vertex: root,
            vertex_count,
        });
    }

    let mut adjacency = vec![Vec::<usize>::new(); vertex_count];
    for &(left, right) in edges {
        if left >= vertex_count {
            return Err(MisTreeTopologyError::VertexOutOfRange {
                vertex: left,
                vertex_count,
            });
        }
        if right >= vertex_count {
            return Err(MisTreeTopologyError::VertexOutOfRange {
                vertex: right,
                vertex_count,
            });
        }
        if left == right {
            return Err(MisTreeTopologyError::SelfLoop { vertex: left });
        }
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut distance = vec![usize::MAX; vertex_count];
    let mut parent = vec![None::<usize>; vertex_count];
    let mut queue = VecDeque::new();
    let mut traversal_order = Vec::new();
    distance[root] = 0;
    queue.push_back(root);

    while let Some(vertex) = queue.pop_front() {
        traversal_order.push(vertex);
        if distance[vertex] >= radius {
            continue;
        }
        for &neighbor in &adjacency[vertex] {
            if distance[neighbor] == usize::MAX {
                distance[neighbor] = distance[vertex] + 1;
                parent[neighbor] = Some(vertex);
                queue.push_back(neighbor);
            }
        }
    }

    let mut in_neighborhood = vec![false; vertex_count];
    for &vertex in &traversal_order {
        in_neighborhood[vertex] = true;
    }

    let mut doubled_edge_count = 0_usize;
    for &vertex in &traversal_order {
        doubled_edge_count = doubled_edge_count
            .checked_add(adjacency[vertex].iter().filter(|&&neighbor| in_neighborhood[neighbor]).count())
            .expect("edge count cannot exceed the allocated adjacency capacity");
    }
    let edge_count = doubled_edge_count / 2;
    if edge_count != traversal_order.len().saturating_sub(1) {
        return Ok(None);
    }

    // The BFS order places every parent before its children. Reversing it gives an
    // iterative post-order, avoiding recursion depth proportional to neighborhood size.
    let mut encodings = vec![None::<String>; vertex_count];
    for &vertex in traversal_order.iter().rev() {
        let mut child_forms = adjacency[vertex]
            .iter()
            .copied()
            .filter(|&neighbor| in_neighborhood[neighbor] && parent[neighbor] == Some(vertex))
            .map(|child| {
                encodings[child]
                    .as_ref()
                    .expect("child encoding must precede parent in reverse traversal")
                    .clone()
            })
            .collect::<Vec<_>>();
        child_forms.sort_unstable();

        let capacity = 2 + child_forms.iter().map(String::len).sum::<usize>();
        let mut encoded = String::with_capacity(capacity);
        encoded.push('(');
        for child in child_forms {
            encoded.push_str(&child);
        }
        encoded.push(')');
        encodings[vertex] = Some(encoded);
    }

    Ok(Some(RootedTreeSignature {
        radius,
        vertex_count: traversal_order.len(),
        edge_count,
        canonical_form: encodings[root]
            .take()
            .expect("root encoding must exist after post-order"),
    }))
}

/// Full evaluation identity that must accompany a rooted topology cache key.
///
/// Digests and identifiers are caller-supplied labels; this struct compares them
/// exactly but does not independently verify their provenance. For physical hardware,
/// `execution_profile_id` must distinguish device/calibration window, qubit layout,
/// transpiler/compiler settings, gate orientation, and readout mitigation configuration
/// whenever those can change the result distribution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MisTreeCacheContext {
    qaoa_depth: usize,
    angle_manifest_sha256: [u8; 32],
    observable_id: String,
    evaluator_version: String,
    numerical_precision: String,
    execution_profile_id: String,
}

impl MisTreeCacheContext {
    /// Construct a complete evaluation context, rejecting missing identifiers.
    pub fn new(
        qaoa_depth: usize,
        angle_manifest_sha256: [u8; 32],
        observable_id: impl Into<String>,
        evaluator_version: impl Into<String>,
        numerical_precision: impl Into<String>,
        execution_profile_id: impl Into<String>,
    ) -> Result<Self, MisTreeCacheContextError> {
        let observable_id = observable_id.into();
        let evaluator_version = evaluator_version.into();
        let numerical_precision = numerical_precision.into();
        let execution_profile_id = execution_profile_id.into();
        for (field, value) in [
            ("observable_id", observable_id.as_str()),
            ("evaluator_version", evaluator_version.as_str()),
            ("numerical_precision", numerical_precision.as_str()),
            ("execution_profile_id", execution_profile_id.as_str()),
        ] {
            if value.trim().is_empty() || value.chars().any(char::is_control) {
                return Err(MisTreeCacheContextError::InvalidField { field });
            }
        }
        Ok(Self {
            qaoa_depth,
            angle_manifest_sha256,
            observable_id,
            evaluator_version,
            numerical_precision,
            execution_profile_id,
        })
    }

    /// Configured QAOA depth.
    pub fn qaoa_depth(&self) -> usize {
        self.qaoa_depth
    }

    /// Caller-supplied digest of the exact angle manifest.
    pub fn angle_manifest_sha256(&self) -> [u8; 32] {
        self.angle_manifest_sha256
    }

    /// Identifier for the scored observable and sign/normalization convention.
    pub fn observable_id(&self) -> &str {
        &self.observable_id
    }

    /// Evaluator implementation/version label.
    pub fn evaluator_version(&self) -> &str {
        &self.evaluator_version
    }

    /// Numeric precision and relevant arithmetic mode.
    pub fn numerical_precision(&self) -> &str {
        &self.numerical_precision
    }

    /// Caller-defined execution context (ideal simulator or frozen hardware profile).
    pub fn execution_profile_id(&self) -> &str {
        &self.execution_profile_id
    }
}

/// Validation error for the non-empty fields in a score-cache context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisTreeCacheContextError {
    /// Field must be non-empty and contain no control characters.
    InvalidField {
        /// Field label rejected by validation.
        field: &'static str,
    },
}

impl fmt::Display for MisTreeCacheContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField { field } => write!(f, "{field} must be non-empty and contain no control characters"),
        }
    }
}

impl std::error::Error for MisTreeCacheContextError {}

/// Exact cache key for a rooted tree score under one fully specified evaluator context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MisTreeCacheKey {
    topology: RootedTreeSignature,
    context: MisTreeCacheContext,
}

impl MisTreeCacheKey {
    /// Combine a verified tree-shaped neighborhood signature and evaluation context.
    pub fn new(topology: RootedTreeSignature, context: MisTreeCacheContext) -> Self {
        Self { topology, context }
    }

    /// Rooted topology component.
    pub fn topology(&self) -> &RootedTreeSignature {
        &self.topology
    }

    /// Evaluation context component.
    pub fn context(&self) -> &MisTreeCacheContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn context(
        angle_digest: [u8; 32],
        evaluator: &str,
        precision: &str,
        profile: &str,
    ) -> MisTreeCacheContext {
        MisTreeCacheContext::new(
            3,
            angle_digest,
            "root-inclusion/v1",
            evaluator,
            precision,
            profile,
        )
        .expect("valid cache context")
    }

    #[test]
    fn canonical_signature_is_invariant_to_tree_relabeling_edge_order_and_duplicates() {
        let first = rooted_tree_neighborhood_signature(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            0,
            3,
        ).expect("valid path").expect("path neighborhood is a tree");

        // Relabel old vertices 0->2, 1->0, 2->3, 3->1 and reverse/duplicate edges.
        let second = rooted_tree_neighborhood_signature(
            4,
            &[(3, 1), (0, 3), (2, 0), (1, 3), (0, 2)],
            2,
            3,
        ).expect("valid relabeling").expect("same rooted path is a tree");

        assert_eq!(first, second);
        assert_eq!(first.vertex_count(), 4);
        assert_eq!(first.edge_count(), 3);
        assert_eq!(first.versioned_form().split('|').next(), Some("symthaea-rooted-tree-v1"));
    }

    #[test]
    fn rooted_signature_distinguishes_path_from_star() {
        let path = rooted_tree_neighborhood_signature(
            4, &[(0, 1), (1, 2), (2, 3)], 1, 3
        ).expect("valid path").expect("tree");
        let star = rooted_tree_neighborhood_signature(
            4, &[(0, 1), (0, 2), (0, 3)], 0, 3
        ).expect("valid star").expect("tree");
        assert_ne!(path.canonical_form(), star.canonical_form());
    }

    #[test]
    fn radius_is_part_of_neighborhood_identity() {
        let graph = [(0, 1), (1, 2), (2, 3)];
        let radius_zero = rooted_tree_neighborhood_signature(4, &graph, 1, 0)
            .expect("valid graph").expect("radius-zero neighborhood is tree");
        let radius_one = rooted_tree_neighborhood_signature(4, &graph, 1, 1)
            .expect("valid graph").expect("radius-one neighborhood is tree");
        assert_ne!(radius_zero, radius_one);
        assert_eq!(radius_zero.vertex_count(), 1);
        assert_eq!(radius_zero.edge_count(), 0);
    }

    #[test]
    fn cyclic_induced_neighborhood_is_explicitly_uncacheable() {
        let triangle = [(0, 1), (1, 2), (2, 0)];
        assert_eq!(
            rooted_tree_neighborhood_signature(3, &triangle, 0, 1)
                .expect("valid graph with a cycle"),
            None
        );
    }

    #[test]
    fn rejects_invalid_roots_endpoints_and_self_loops() {
        assert_eq!(
            rooted_tree_neighborhood_signature(2, &[], 2, 1)
                .expect_err("root is out of range"),
            MisTreeTopologyError::VertexOutOfRange { vertex: 2, vertex_count: 2 }
        );
        assert_eq!(
            rooted_tree_neighborhood_signature(2, &[(0, 2)], 0, 1)
                .expect_err("endpoint is out of range"),
            MisTreeTopologyError::VertexOutOfRange { vertex: 2, vertex_count: 2 }
        );
        assert_eq!(
            rooted_tree_neighborhood_signature(2, &[(1, 1)], 0, 1)
                .expect_err("self loop is invalid"),
            MisTreeTopologyError::SelfLoop { vertex: 1 }
        );
    }

    #[test]
    fn same_topology_shares_only_when_all_context_fields_match() {
        let tree = rooted_tree_neighborhood_signature(3, &[(0, 1), (1, 2)], 1, 1)
            .expect("valid path").expect("tree");
        let key = MisTreeCacheKey::new(
            tree.clone(),
            context([7; 32], "tensor-network/v1", "f64", "ideal-exact-v1"),
        );
        let same = MisTreeCacheKey::new(
            tree.clone(),
            context([7; 32], "tensor-network/v1", "f64", "ideal-exact-v1"),
        );
        let changed_angles = MisTreeCacheKey::new(
            tree.clone(),
            context([8; 32], "tensor-network/v1", "f64", "ideal-exact-v1"),
        );
        let changed_profile = MisTreeCacheKey::new(
            tree,
            context([7; 32], "tensor-network/v1", "f64", "device-calibration-layout-A"),
        );
        assert_eq!(key, same);
        assert_ne!(key, changed_angles);
        assert_ne!(key, changed_profile);

        let mut cache = HashSet::new();
        assert!(cache.insert(key));
        assert!(!cache.insert(same));
        assert!(cache.insert(changed_angles));
        assert!(cache.insert(changed_profile));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn refuses_empty_or_control_character_context_identifiers() {
        assert_eq!(
            MisTreeCacheContext::new(1, [0; 32], " ", "eval/v1", "f64", "sim")
                .expect_err("blank observable is invalid"),
            MisTreeCacheContextError::InvalidField { field: "observable_id" }
        );
        assert_eq!(
            MisTreeCacheContext::new(1, [0; 32], "root\nvalue", "eval/v1", "f64", "sim")
                .expect_err("control chars are invalid"),
            MisTreeCacheContextError::InvalidField { field: "observable_id" }
        );
    }

    #[test]
    fn cache_keys_distinguish_observable_evaluator_precision_and_depth() {
        let tree = rooted_tree_neighborhood_signature(2, &[(0, 1)], 0, 1)
            .expect("valid edge").expect("tree");
        let base = MisTreeCacheKey::new(
            tree.clone(), context([1; 32], "eval/v1", "f64", "profile-A")
        );
        let different_evaluator = MisTreeCacheKey::new(
            tree.clone(),
            MisTreeCacheContext::new(3, [1; 32], "root-inclusion/v1", "eval/v2", "f64", "profile-A")
                .expect("valid context"),
        );
        let different_precision = MisTreeCacheKey::new(
            tree.clone(),
            MisTreeCacheContext::new(3, [1; 32], "root-inclusion/v1", "eval/v1", "f32", "profile-A")
                .expect("valid context"),
        );
        let different_observable = MisTreeCacheKey::new(
            tree.clone(),
            MisTreeCacheContext::new(3, [1; 32], "root-cost/v1", "eval/v1", "f64", "profile-A")
                .expect("valid context"),
        );
        let different_depth = MisTreeCacheKey::new(
            tree.clone(),
            MisTreeCacheContext::new(4, [1; 32], "root-inclusion/v1", "eval/v1", "f64", "profile-A")
                .expect("valid context"),
        );
        assert_ne!(base, different_evaluator);
        assert_ne!(base, different_precision);
        assert_ne!(base, different_observable);
        assert_ne!(base, different_depth);
    }
}
