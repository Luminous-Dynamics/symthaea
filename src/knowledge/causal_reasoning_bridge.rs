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
// `symthaea-causal-reasoning` has TWO structurally different `CausalDAG` types:
// this module's `causal_calculus::CausalDAG` (id/name/values/is_observed nodes +
// an adjacency matrix, built via add_node/add_edge) and
// `counterfactual::CausalDAG` (bare `Vec<String>` node names + edge pairs, used
// by `ConsciousReasoningEngine::analyze_counterfactual`). They do not interoperate
// without conversion — see `to_identification_dag()` below (AGW plan Phase 5.3,
// found 2026-07-10 when the un-islanding attempt discovered this mismatch).
//
// Gated: `symthaea-causal-reasoning` is declared `default-features = false`, and its
// `counterfactual` module only exists when that crate's `counterfactual` feature is on —
// which only `reasoning_engine` turns on (`Cargo.toml:825`). Without this gate the import
// is unresolved in any build lacking `reasoning_engine`, e.g. the CI leg
// `--no-default-features --features safety-agents,ssm_language`. Found 2026-07-31.
#[cfg(feature = "reasoning_engine")]
use symthaea_causal_reasoning::counterfactual::CausalDAG as IdentificationCausalDAG;

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

    /// Convert to the `identification::CausalDAG` shape
    /// `ConsciousReasoningEngine::analyze_counterfactual()` expects (AGW plan
    /// Phase 5.3). Node order is preserved 1:1: `causal_calculus::CausalDAG`
    /// assigns each node's `id` as its insertion-order index into `nodes`
    /// (see `add_node`), which is exactly the implicit index
    /// `identification::CausalDAG` uses — so edge index pairs carry over
    /// unchanged and only node names need projecting out.
    ///
    /// Gated to match its return type's import (see the `use` at the top of this file).
    #[cfg(feature = "reasoning_engine")]
    pub fn to_identification_dag(&self) -> IdentificationCausalDAG {
        let names: Vec<String> = self.dag.nodes.iter().map(|n| n.name.clone()).collect();
        IdentificationCausalDAG::new(names, self.dag.edges.clone())
    }

    /// Resolve a synced entity name to its node index in the
    /// `identification::CausalDAG` returned by `to_identification_dag()`
    /// (same indices as `dag()`'s own `causal_calculus::CausalDAG`).
    pub fn identification_node_index(&self, entity: &str) -> Option<usize> {
        self.node_map.get(entity).copied()
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

    // ── AGW plan Phase 5.3: does the un-islanded pipeline actually work? ──────
    // Deliberately test-only, not wired into the live ~31Hz loop (that wiring
    // is a separate, riskier step scoped for a dedicated session — see
    // AGW_PLAN_2026-07-09.md). This proves the mechanism end-to-end BEFORE any
    // live wiring is attempted: real extracted relations -> CausalKnowledgeBridge
    // -> CausalReasoningBridge::sync_from_bridge -> to_identification_dag() ->
    // the exact type ConsciousReasoningEngine::analyze_counterfactual() expects.

    // Gated for the same reason as the test below: `to_identification_dag()` only exists
    // when `reasoning_engine` is on.
    #[cfg(feature = "reasoning_engine")]
    #[test]
    fn identification_dag_conversion_preserves_topology() {
        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        let idag = crb.to_identification_dag();
        assert_eq!(idag.num_nodes(), 4);

        let s = crb.identification_node_index("sanctions").unwrap();
        let o = crb.identification_node_index("oil shortage").unwrap();
        let p = crb.identification_node_index("price spike").unwrap();
        let i = crb.identification_node_index("inflation").unwrap();

        assert_eq!(idag.nodes[s], "sanctions");
        assert_eq!(idag.nodes[i], "inflation");
        assert!(
            idag.has_path(s, i),
            "sanctions must reach inflation transitively"
        );
        assert!(
            idag.parents(o).contains(&s),
            "oil shortage's parent must be sanctions in the converted DAG"
        );
    }

    // Gated to match its imports: `consciousness::counterfactual` and
    // `consciousness::reasoning_engine` are both `#[cfg(feature = "reasoning_engine")]`
    // (`consciousness/mod.rs:237,240`), but this test used them unconditionally. That made
    // `cargo test` fail to compile the whole lib-test target for any configuration without
    // that feature — e.g. `--no-default-features --features profile-voice,creative,ssm_language`,
    // the documented poetry build. It went unnoticed because the combo was only ever
    // exercised with `cargo check`, which does not build test targets.
    // Found 2026-07-28 while verifying unrelated Broca work.
    #[cfg(feature = "reasoning_engine")]
    #[test]
    fn end_to_end_counterfactual_query_through_the_unislanded_pipeline() {
        use crate::consciousness::counterfactual::{CausalQuery, CausalQueryOutcome};
        use crate::consciousness::reasoning_engine::ConsciousReasoningEngine;

        let bridge = make_bridge_with_chain();
        let mut crb = CausalReasoningBridge::new();
        crb.sync_from_bridge(&bridge);

        let idag = crb.to_identification_dag();
        let treatment = crb.identification_node_index("sanctions").unwrap();
        let outcome_node = crb.identification_node_index("inflation").unwrap();

        let query = CausalQuery {
            treatment,
            outcome: outcome_node,
            conditioning: vec![],
        };

        let engine = ConsciousReasoningEngine::new();
        let result = engine.analyze_counterfactual(&idag, &query);

        // The claim under test is narrow and honest: a real knowledge-derived
        // chain, converted through the new bridge method, produces a
        // meaningful (non-panicking, decidable) outcome from the reasoning
        // engine's real entry point -- proving the "island" pieces genuinely
        // interoperate. It does NOT prove this is worth wiring into the live
        // loop, nor what threshold should gate it there.
        match result {
            CausalQueryOutcome::Identified { .. } => {} // sanctions -> inflation is a simple chain: identifiable
            other => panic!(
                "expected a simple 3-hop chain to be identifiable, got {:?}",
                other
            ),
        }
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
