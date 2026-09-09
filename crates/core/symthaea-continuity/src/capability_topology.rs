// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic structural topology derived from a validated capability graph.
//!
//! This module answers only graph-theoretic questions: which stable capability
//! identities participate in the same strongly connected component, and which
//! components reference which prerequisite components.
//!
//! It deliberately does **not** decide whether an AND/OR prerequisite is
//! satisfied, whether a capability is realized or currently available, what is
//! cheapest to restore, or what anyone is authorized to do.
//!
//! Core theorem:
//!
//! `CapabilityTopologyV1 != RecoveryState != CurrentAvailability != Authority`.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{CapabilityGraphSnapshotId, CapabilityId, ValidatedCapabilityGraphV1};

/// Snapshot-local deterministic index of one strongly connected component.
///
/// Component indices are derived from a specific graph snapshot and are not
/// stable identities across graph revisions. Components are ordered by their
/// smallest stable [`CapabilityId`], making the numbering deterministic for one
/// exact snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityComponentIndex(usize);

impl CapabilityComponentIndex {
    /// Zero-based snapshot-local component index.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// One strongly connected component in the dependency-reference graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityComponentV1 {
    index: CapabilityComponentIndex,
    members: Vec<CapabilityId>,
    cyclic: bool,
}

impl CapabilityComponentV1 {
    /// Snapshot-local deterministic component index.
    pub fn index(&self) -> CapabilityComponentIndex {
        self.index
    }

    /// Stable capability identities in canonical order.
    pub fn members(&self) -> &[CapabilityId] {
        &self.members
    }

    /// Whether this component contains a dependency-reference cycle.
    ///
    /// A multi-member SCC is cyclic by definition. A singleton is cyclic only
    /// when its definition directly references itself.
    pub fn is_cyclic(&self) -> bool {
        self.cyclic
    }
}

/// One edge in the SCC condensation graph.
///
/// Direction preserves the capability-reference direction:
/// `dependent -> prerequisite`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityCondensationEdgeV1 {
    dependent: CapabilityComponentIndex,
    prerequisite: CapabilityComponentIndex,
}

impl CapabilityCondensationEdgeV1 {
    pub fn dependent(&self) -> CapabilityComponentIndex {
        self.dependent
    }

    pub fn prerequisite(&self) -> CapabilityComponentIndex {
        self.prerequisite
    }
}

/// Deterministic SCC decomposition and condensation DAG for one validated graph.
///
/// This is a derived, non-Serde analysis object. It does not become a new source
/// of truth and does not carry execution or governance authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityTopologyV1 {
    source_snapshot_id: CapabilityGraphSnapshotId,
    components: Vec<CapabilityComponentV1>,
    component_of: BTreeMap<CapabilityId, CapabilityComponentIndex>,
    condensation_edges: Vec<CapabilityCondensationEdgeV1>,
}

impl CapabilityTopologyV1 {
    /// Exact graph snapshot from which this topology was derived.
    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    /// Canonically ordered SCCs.
    pub fn components(&self) -> &[CapabilityComponentV1] {
        &self.components
    }

    /// Canonically ordered, duplicate-free cross-component dependency edges.
    pub fn condensation_edges(&self) -> &[CapabilityCondensationEdgeV1] {
        &self.condensation_edges
    }

    /// Resolve a capability into its snapshot-local SCC.
    pub fn component_of(&self, capability_id: CapabilityId) -> Option<CapabilityComponentIndex> {
        self.component_of.get(&capability_id).copied()
    }

    /// Resolve a component by its snapshot-local index.
    pub fn component(
        &self,
        index: CapabilityComponentIndex,
    ) -> Option<&CapabilityComponentV1> {
        self.components.get(index.0)
    }

    /// True when the dependency-reference graph contains no directed cycle.
    pub fn is_acyclic(&self) -> bool {
        self.components.iter().all(|component| !component.cyclic)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityTopologyError {
    #[error(
        "validated capability graph unexpectedly references missing capability {missing:?} from {source:?}"
    )]
    MissingIndexedCapability {
        source: CapabilityId,
        missing: CapabilityId,
    },
}

/// Derive deterministic SCC and condensation topology from a validated graph.
///
/// Every `Leaf(CapabilityId)` contributes a structural reference edge whether it
/// appears below `AllOf` or `AnyOf`. That is intentional: SCC topology records
/// possible dependency relationships but does not interpret prerequisite
/// satisfaction. Assumption-scoped reachability belongs in a later layer.
pub fn analyze_capability_topology(
    graph: &ValidatedCapabilityGraphV1,
) -> Result<CapabilityTopologyV1, CapabilityTopologyError> {
    let definitions = graph.definitions();
    let capability_ids: Vec<CapabilityId> = definitions.iter().map(|definition| definition.id()).collect();
    let index: BTreeMap<CapabilityId, usize> = capability_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, capability_id)| (capability_id, index))
        .collect();

    let mut adjacency = vec![Vec::<usize>::new(); definitions.len()];
    for (source_index, definition) in definitions.iter().enumerate() {
        for referenced in definition.referenced_capabilities() {
            let Some(&target_index) = index.get(&referenced) else {
                // This should be unreachable because the input wrapper is
                // constructible only after closed-world graph validation. Keep
                // it fallible anyway so this analysis layer never relies on a
                // panic for a trust-boundary invariant.
                return Err(CapabilityTopologyError::MissingIndexedCapability {
                    source: definition.id(),
                    missing: referenced,
                });
            };
            adjacency[source_index].push(target_index);
        }
        adjacency[source_index].sort_unstable();
        adjacency[source_index].dedup();
    }

    let mut reverse = vec![Vec::<usize>::new(); adjacency.len()];
    for (source, targets) in adjacency.iter().enumerate() {
        for &target in targets {
            reverse[target].push(source);
        }
    }
    for incoming in &mut reverse {
        incoming.sort_unstable();
        incoming.dedup();
    }

    // Iterative Kosaraju avoids recursion-depth dependence for large capability
    // graphs while remaining deterministic because node and adjacency order are
    // canonical.
    let finish_order = deterministic_finish_order(&adjacency);
    let mut assigned = vec![false; adjacency.len()];
    let mut raw_components: Vec<(Vec<usize>, bool)> = Vec::new();

    for &start in finish_order.iter().rev() {
        if assigned[start] {
            continue;
        }

        let mut members = Vec::new();
        let mut stack = vec![start];
        assigned[start] = true;
        while let Some(node) = stack.pop() {
            members.push(node);
            // Reverse iteration preserves ascending traversal after LIFO push.
            for &incoming in reverse[node].iter().rev() {
                if !assigned[incoming] {
                    assigned[incoming] = true;
                    stack.push(incoming);
                }
            }
        }
        members.sort_unstable();

        let cyclic = members.len() > 1
            || members
                .first()
                .is_some_and(|&node| adjacency[node].binary_search(&node).is_ok());
        raw_components.push((members, cyclic));
    }

    // Canonicalize component numbering by smallest capability id. Definitions
    // are already in CapabilityId order, so the minimum member index has the
    // same ordering.
    raw_components.sort_by_key(|(members, _)| members.first().copied().unwrap_or(usize::MAX));

    let mut node_component = vec![CapabilityComponentIndex(0); adjacency.len()];
    let mut components = Vec::with_capacity(raw_components.len());
    let mut component_of = BTreeMap::new();

    for (component_number, (member_indices, cyclic)) in raw_components.into_iter().enumerate() {
        let component_index = CapabilityComponentIndex(component_number);
        let members: Vec<CapabilityId> = member_indices
            .iter()
            .map(|&member| capability_ids[member])
            .collect();

        for (&member_index, &capability_id) in member_indices.iter().zip(members.iter()) {
            node_component[member_index] = component_index;
            component_of.insert(capability_id, component_index);
        }

        components.push(CapabilityComponentV1 {
            index: component_index,
            members,
            cyclic,
        });
    }

    let mut edge_set = BTreeSet::new();
    for (source, targets) in adjacency.iter().enumerate() {
        let source_component = node_component[source];
        for &target in targets {
            let target_component = node_component[target];
            if source_component != target_component {
                edge_set.insert((source_component, target_component));
            }
        }
    }

    let condensation_edges = edge_set
        .into_iter()
        .map(|(dependent, prerequisite)| CapabilityCondensationEdgeV1 {
            dependent,
            prerequisite,
        })
        .collect();

    Ok(CapabilityTopologyV1 {
        source_snapshot_id: graph.id(),
        components,
        component_of,
        condensation_edges,
    })
}

fn deterministic_finish_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let mut visited = vec![false; adjacency.len()];
    let mut order = Vec::with_capacity(adjacency.len());

    for start in 0..adjacency.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];

        while !stack.is_empty() {
            let last = stack.len() - 1;
            let node = stack[last].0;
            let next_child = stack[last].1;

            if next_child < adjacency[node].len() {
                let child = adjacency[node][next_child];
                stack[last].1 += 1;
                if !visited[child] {
                    visited[child] = true;
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilityDefinitionV1, CapabilityGraphSnapshotV1, CapabilityRequirementV1,
    };

    fn id(name: &str) -> CapabilityId {
        CapabilityId::new("org.example", name).unwrap()
    }

    fn leaf(name: &str) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, None).unwrap()
    }

    fn validated(definitions: Vec<CapabilityDefinitionV1>) -> ValidatedCapabilityGraphV1 {
        CapabilityGraphSnapshotV1::new(definitions)
            .unwrap()
            .validate()
            .unwrap()
    }

    #[test]
    fn empty_graph_has_empty_acyclic_topology() {
        let graph = validated(vec![]);
        let topology = analyze_capability_topology(&graph).unwrap();

        assert_eq!(topology.source_snapshot_id(), graph.id());
        assert!(topology.components().is_empty());
        assert!(topology.condensation_edges().is_empty());
        assert!(topology.is_acyclic());
    }

    #[test]
    fn singleton_without_self_reference_is_not_cyclic() {
        let a = leaf("a");
        let a_id = a.id();
        let topology = analyze_capability_topology(&validated(vec![a])).unwrap();

        let component_index = topology.component_of(a_id).unwrap();
        let component = topology.component(component_index).unwrap();
        assert_eq!(component.members(), &[a_id]);
        assert!(!component.is_cyclic());
    }

    #[test]
    fn direct_self_reference_is_a_cyclic_singleton() {
        let a_id = id("a");
        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();
        let topology = analyze_capability_topology(&validated(vec![a])).unwrap();

        let component = topology.component(topology.component_of(a_id).unwrap()).unwrap();
        assert!(component.is_cyclic());
        assert_eq!(component.members(), &[a_id]);
        assert!(!topology.is_acyclic());
    }

    #[test]
    fn mutual_cycle_collapses_into_one_component() {
        let a_id = id("a");
        let b_id = id("b");
        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b_id)),
        )
        .unwrap();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();

        let topology = analyze_capability_topology(&validated(vec![a, b])).unwrap();
        let a_component = topology.component_of(a_id).unwrap();
        let b_component = topology.component_of(b_id).unwrap();
        assert_eq!(a_component, b_component);
        assert!(topology.component(a_component).unwrap().is_cyclic());
        assert!(topology.condensation_edges().is_empty());
    }

    #[test]
    fn condensation_preserves_dependent_to_prerequisite_direction() {
        let a = leaf("a");
        let a_id = a.id();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();
        let b_id = b.id();

        let topology = analyze_capability_topology(&validated(vec![b, a])).unwrap();
        let expected = CapabilityCondensationEdgeV1 {
            dependent: topology.component_of(b_id).unwrap(),
            prerequisite: topology.component_of(a_id).unwrap(),
        };
        assert_eq!(topology.condensation_edges(), &[expected]);
    }

    #[test]
    fn condensation_deduplicates_edges_between_components() {
        let a_id = id("a");
        let b_id = id("b");
        let c_id = id("c");

        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b_id)),
        )
        .unwrap();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();
        let c = CapabilityDefinitionV1::new(
            "org.example",
            "c",
            Some(
                CapabilityRequirementV1::all_of(vec![
                    CapabilityRequirementV1::leaf(a_id),
                    CapabilityRequirementV1::leaf(b_id),
                ])
                .unwrap(),
            ),
        )
        .unwrap();

        let topology = analyze_capability_topology(&validated(vec![c, b, a])).unwrap();
        let cycle = topology.component_of(a_id).unwrap();
        assert_eq!(cycle, topology.component_of(b_id).unwrap());
        let c_component = topology.component_of(c_id).unwrap();
        assert_eq!(
            topology.condensation_edges(),
            &[CapabilityCondensationEdgeV1 {
                dependent: c_component,
                prerequisite: cycle,
            }]
        );
    }

    #[test]
    fn any_of_contributes_all_reference_edges_without_claiming_satisfaction() {
        let a = leaf("a");
        let b = leaf("b");
        let a_id = a.id();
        let b_id = b.id();
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::any_of(vec![
                    CapabilityRequirementV1::leaf(a_id),
                    CapabilityRequirementV1::leaf(b_id),
                ])
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();

        let topology = analyze_capability_topology(&validated(vec![target, b, a])).unwrap();
        let target_component = topology.component_of(target_id).unwrap();
        let edges: BTreeSet<_> = topology
            .condensation_edges()
            .iter()
            .map(|edge| (edge.dependent(), edge.prerequisite()))
            .collect();

        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(target_component, topology.component_of(a_id).unwrap())));
        assert!(edges.contains(&(target_component, topology.component_of(b_id).unwrap())));
    }

    #[test]
    fn component_numbering_is_snapshot_deterministic() {
        let a = leaf("a");
        let b = leaf("b");
        let c = leaf("c");
        let graph = validated(vec![c, a, b]);

        let left = analyze_capability_topology(&graph).unwrap();
        let right = analyze_capability_topology(&graph).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.components()
                .iter()
                .map(CapabilityComponentV1::index)
                .collect::<Vec<_>>(),
            (0..left.components().len())
                .map(CapabilityComponentIndex)
                .collect::<Vec<_>>()
        );
    }
}