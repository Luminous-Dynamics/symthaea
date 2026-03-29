// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Reasoning Bridge
//!
//! Exports knowledge engine causal edges into the full do-calculus
//! CausalDAG from symthaea-causal-reasoning, enabling counterfactual
//! queries, backdoor adjustment, and intervention analysis.
//!
//! Science: Pearl (2009) Causality, Spirtes et al. (2000) Causation

use super::causal_bridge::CausalKnowledgeBridge;
use std::collections::HashMap;
use symthaea_causal_reasoning::causal_calculus::{CausalDAG, StructuralCausalModel};

/// Bridge from knowledge engine causal edges to full CausalDAG.
///
/// Maintains a mapping between text entity names and DAG node IDs,
/// enabling bidirectional translation between the knowledge engine's
/// string-based causal graph and the reasoning crate's index-based DAG.
pub struct CausalReasoningBridge {
    /// Maps entity name → CausalDAG node ID
    node_map: HashMap<String, usize>,
    /// Reverse map: node ID → entity name
    reverse_map: HashMap<usize, String>,
    /// The full causal DAG for do-calculus reasoning
    dag: CausalDAG,
    /// SCM wrapping the DAG for backdoor/frontdoor/causal_effect queries
    scm: StructuralCausalModel,
    /// Total edges synced
    total_synced: u64,
}

impl Default for CausalReasoningBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalReasoningBridge {
    pub fn new() -> Self {
        let dag = CausalDAG::new();
        let scm = StructuralCausalModel::new(dag.clone());
        Self {
            node_map: HashMap::new(),
            reverse_map: HashMap::new(),
            dag,
            scm,
            total_synced: 0,
        }
    }

    /// Ensure a node exists in the DAG for the given entity name.
    /// Returns the node ID (existing or newly created).
    fn ensure_node(&mut self, name: &str) -> usize {
        if let Some(&id) = self.node_map.get(name) {
            return id;
        }
        // Add node with binary values (present/absent is simplest model)
        let id = self
            .dag
            .add_node(name, vec!["absent".into(), "present".into()], true);
        self.node_map.insert(name.to_string(), id);
        self.reverse_map.insert(id, name.to_string());
        id
    }

    /// Rebuild the SCM from the current DAG state.
    ///
    /// Must be called after any structural changes (add_node, add_edge)
    /// to keep the SCM in sync for backdoor/causal_effect queries.
    fn rebuild_scm(&mut self) {
        self.scm = StructuralCausalModel::new(self.dag.clone());
    }

    /// Sync all edges from the CausalKnowledgeBridge into the CausalDAG.
    ///
    /// Call periodically (e.g., every 100 cycles) to keep the DAG current.
    /// Idempotent: duplicate edges are ignored by CausalDAG::add_edge.
    pub fn sync_from_bridge(&mut self, bridge: &CausalKnowledgeBridge) {
        let initial_synced = self.total_synced;

        for (cause, effect, _strength) in bridge.export_edges() {
            let cause_id = self.ensure_node(&cause);
            let effect_id = self.ensure_node(&effect);

            // CausalDAG::add_edge is idempotent (checks adjacency matrix)
            // but we track whether the edge was actually new
            if !self.dag.adjacency[cause_id][effect_id] {
                self.dag.add_edge(cause_id, effect_id);
                self.total_synced += 1;
            }
        }

        // Rebuild SCM only if structure changed
        if self.total_synced > initial_synced {
            self.rebuild_scm();
        }
    }

    /// Query: are X and Y d-separated given conditioning set Z?
    ///
    /// Returns None if any entity name is unknown.
    pub fn d_separated(&self, x: &str, y: &str, given: &[&str]) -> Option<bool> {
        let x_id = *self.node_map.get(x)?;
        let y_id = *self.node_map.get(y)?;
        let z: std::collections::HashSet<usize> = given
            .iter()
            .filter_map(|name| self.node_map.get(*name).copied())
            .collect();
        Some(self.dag.d_separated(x_id, y_id, &z).separated)
    }

    /// Find all valid backdoor adjustment sets for estimating
    /// the causal effect of X on Y.
    pub fn backdoor_sets(&self, x: &str, y: &str) -> Option<Vec<Vec<String>>> {
        let x_id = *self.node_map.get(x)?;
        let y_id = *self.node_map.get(y)?;
        let sets = self.scm.find_backdoor_sets(x_id, y_id);
        Some(
            sets.into_iter()
                .map(|set| {
                    set.into_iter()
                        .filter_map(|id| self.reverse_map.get(&id).cloned())
                        .collect()
                })
                .collect(),
        )
    }

    /// Estimate causal effect of X on Y (if identifiable).
    ///
    /// Requires conditional probability tables to be set on the SCM.
    /// Returns None if the effect is not identifiable or tables are missing.
    pub fn causal_effect(&self, x: &str, y: &str) -> Option<f64> {
        let x_id = *self.node_map.get(x)?;
        let y_id = *self.node_map.get(y)?;
        self.scm.causal_effect(x_id, y_id)
    }

    /// Get all ancestors of a node (transitive causes).
    pub fn ancestors(&self, name: &str) -> Option<Vec<String>> {
        let id = *self.node_map.get(name)?;
        Some(
            self.dag
                .ancestors(id)
                .into_iter()
                .filter_map(|aid| self.reverse_map.get(&aid).cloned())
                .collect(),
        )
    }

    /// Get all descendants of a node (transitive effects).
    pub fn descendants(&self, name: &str) -> Option<Vec<String>> {
        let id = *self.node_map.get(name)?;
        Some(
            self.dag
                .descendants(id)
                .into_iter()
                .filter_map(|did| self.reverse_map.get(&did).cloned())
                .collect(),
        )
    }

    /// Number of nodes in the full causal DAG.
    pub fn node_count(&self) -> usize {
        self.node_map.len()
    }

    /// Total edges synced from the knowledge bridge.
    pub fn total_synced(&self) -> u64 {
        self.total_synced
    }

    /// Access the underlying DAG (for advanced queries).
    pub fn dag(&self) -> &CausalDAG {
        &self.dag
    }

    /// Access the underlying SCM (for intervention/counterfactual queries).
    pub fn scm(&self) -> &StructuralCausalModel {
        &self.scm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::extraction::{ExtractedRelation, SemanticRole};

    fn make_bridge_with_chain() -> CausalKnowledgeBridge {
        let mut bridge = CausalKnowledgeBridge::new(100);

        // Build: sanctions → oil_shortage → price_spike → inflation
        let r1 = ExtractedRelation {
            subject: "sanctions".into(),
            predicate: "caused".into(),
            object: "oil shortage".into(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Result,
            is_causal: true,
            is_negated: false,
            confidence: 0.8,
        };
        let r2 = ExtractedRelation {
            subject: "oil shortage".into(),
            predicate: "caused".into(),
            object: "price spike".into(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Result,
            is_causal: true,
            is_negated: false,
            confidence: 0.8,
        };
        let r3 = ExtractedRelation {
            subject: "price spike".into(),
            predicate: "caused".into(),
            object: "inflation".into(),
            subject_role: SemanticRole::Agent,
            object_role: SemanticRole::Result,
            is_causal: true,
            is_negated: false,
            confidence: 0.8,
        };

        bridge.process_relation(&r1, "test", 1);
        bridge.process_relation(&r2, "test", 2);
        bridge.process_relation(&r3, "test", 3);
        bridge
    }

    #[test]
    fn test_sync_from_bridge() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        assert_eq!(crb.node_count(), 4); // sanctions, oil shortage, price spike, inflation
        assert!(crb.total_synced() >= 3);
    }

    #[test]
    fn test_sync_idempotent() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);
        let first_count = crb.total_synced();
        crb.sync_from_bridge(&bridge);
        assert_eq!(crb.total_synced(), first_count); // No new edges
    }

    #[test]
    fn test_ancestors_descendants() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        let ancestors = crb.ancestors("inflation").unwrap();
        assert!(ancestors.contains(&"sanctions".to_string()));
        assert!(ancestors.contains(&"price spike".to_string()));

        let descendants = crb.descendants("sanctions").unwrap();
        assert!(descendants.contains(&"inflation".to_string()));
    }

    #[test]
    fn test_d_separation() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        // sanctions → oil shortage → price spike: conditioning on oil shortage
        // should d-separate sanctions from price spike
        let separated = crb.d_separated("sanctions", "price spike", &["oil shortage"]);
        assert!(separated.is_some());
        // In a chain, conditioning on the middle node d-separates the endpoints
        assert!(separated.unwrap());
    }

    #[test]
    fn test_unknown_entities() {
        let crb = CausalReasoningBridge::new();
        assert!(crb.d_separated("foo", "bar", &[]).is_none());
        assert!(crb.ancestors("nonexistent").is_none());
    }

    #[test]
    fn test_backdoor_sets() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        // In a simple chain, backdoor sets should exist
        let sets = crb.backdoor_sets("sanctions", "inflation");
        assert!(sets.is_some());
    }
}
