// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Queryable, fail-closed source derivation graphs.
//!
//! A flat origin label cannot explain how a source was copied, quoted, revised,
//! or synthesized from multiple upstream sources. This module represents these
//! relationships explicitly and resolves attributed terminal origins for a
//! selected evidence set.
//!
//! The graph uses the PROV-DM convention that a derivation points from a derived
//! entity to the entity it was derived from. This is a deliberately small subset
//! inspired by W3C PROV-O, not a claim of standards conformance.
//!
//! A primary-origin node is an explicit provenance assertion, not proof that the
//! source is independent, authoritative, honest, or true. An unattributed terminal
//! stays unresolved. The module never infers provenance from URL differences,
//! text similarity, publisher identity, or HDC proximity.

use std::collections::{HashMap, HashSet, VecDeque};

/// Role asserted for an entity in the lineage graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProvenanceNodeKind {
    /// An explicitly attributed origin for this domain/query.
    ///
    /// This label must come from upstream provenance or adjudication. It is not
    /// automatically inferred merely because a node has no recorded parents.
    PrimaryOrigin,
    /// A record known to have been derived from one or more upstream entities.
    DerivedArtifact,
    /// A terminal record whose upstream/origin identity is not established.
    UnknownOrigin,
}

/// Typed derivation relation. Each relation means the derived entity points back
/// to an upstream source entity; semantics are inspired by W3C PROV-O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProvenanceDerivationKind {
    /// General derivation / transformation.
    DerivedFrom,
    /// Copies some or all of another source, e.g. quotation or syndication.
    QuotedFrom,
    /// Revises an earlier entity.
    RevisionOf,
    /// Declared primary source for the derived entity.
    HadPrimarySource,
}

/// One entity/version in the provenance graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceGraphNode {
    /// Stable, non-empty entity/revision ID.
    pub entity_id: String,
    pub kind: ProvenanceNodeKind,
}

/// An assertion that one artifact depends on an upstream entity.
///
/// derived_entity_id is the newer/derived item. source_entity_id is its parent
/// (the item it was derived from), not its child.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProvenanceDerivationEdge {
    pub derived_entity_id: String,
    pub source_entity_id: String,
    pub kind: ProvenanceDerivationKind,
}

/// In-memory provenance graph. This value type performs no I/O or signature
/// verification; callers provide the graph built from auditable source records.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EvidenceProvenanceGraph {
    pub nodes: Vec<ProvenanceGraphNode>,
    pub edges: Vec<ProvenanceDerivationEdge>,
}

/// Policy for how many distinct explicitly attributed upstream roots are needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceGraphPolicy {
    pub minimum_distinct_origins: usize,
    /// If true, any unattributed terminal in the selected ancestry causes abstention.
    pub require_all_lineage_resolved: bool,
}

impl ProvenanceGraphPolicy {
    pub fn new(
        minimum_distinct_origins: usize,
        require_all_lineage_resolved: bool,
    ) -> Result<Self, String> {
        if minimum_distinct_origins == 0 {
            return Err("minimum_distinct_origins must be at least 1".to_owned());
        }
        Ok(Self {
            minimum_distinct_origins,
            require_all_lineage_resolved,
        })
    }

    /// Conservative default for multi-origin corroboration: two explicit roots,
    /// with no unknown terminal lineage in the selected ancestry.
    pub fn strict() -> Self {
        Self {
            minimum_distinct_origins: 2,
            require_all_lineage_resolved: true,
        }
    }
}

impl Default for ProvenanceGraphPolicy {
    fn default() -> Self {
        Self::strict()
    }
}

/// Machine-readable graph or policy failure reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProvenanceGraphAbstentionReason {
    EmptyEvidenceSet,
    DuplicateEntityId,
    EmptyEntityId,
    MissingDerivedEntity,
    MissingSourceEntity,
    OriginNodeHasParents,
    UnknownOriginNodeHasParents,
    DerivedArtifactHasNoParents,
    CycleDetected,
    UnknownEvidenceEntity,
    UnknownOriginInSelectedLineage,
    InsufficientDistinctOrigins,
}

/// Outcome is an eligibility gate, never a truth or causal-validity verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceGraphDecision {
    EligibleForReview,
    Abstain,
}

/// Auditable summary of a selected set of evidence entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceGraphAssessment {
    pub decision: ProvenanceGraphDecision,
    pub reasons: Vec<ProvenanceGraphAbstentionReason>,
    /// Count of requested evidence IDs, including repeats in the input.
    pub requested_evidence_count: usize,
    /// Count of distinct requested evidence IDs.
    pub unique_evidence_count: usize,
    /// Sorted, deduplicated explicitly attributed terminal primary origins.
    pub root_origin_ids: Vec<String>,
    /// Sorted, deduplicated unattributed terminal entities in the selected ancestry.
    pub unresolved_terminal_ids: Vec<String>,
}

impl ProvenanceGraphAssessment {
    pub fn is_eligible_for_review(&self) -> bool {
        self.decision == ProvenanceGraphDecision::EligibleForReview
    }
}

fn push_reason(
    reasons: &mut Vec<ProvenanceGraphAbstentionReason>,
    reason: ProvenanceGraphAbstentionReason,
) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

impl EvidenceProvenanceGraph {
    /// Resolve upstream lineage for a selected evidence set.
    ///
    /// Each distinct primary-origin entity counts at most once across the entire
    /// requested evidence set. Multiple derived reports sharing an origin do not
    /// increase the count. A derived synthesis can have multiple parents and thus
    /// multiple roots, but this does not assert those roots are independent.
    ///
    /// Structural graph validation is global: duplicate IDs, dangling edges,
    /// invalid node roles, or any directed cycle make the graph unusable. This
    /// prevents callers from quietly ignoring an invalid provenance document.
    pub fn assess(
        &self,
        evidence_entity_ids: &[String],
        policy: &ProvenanceGraphPolicy,
    ) -> ProvenanceGraphAssessment {
        use ProvenanceGraphAbstentionReason as R;

        let mut reasons = Vec::new();
        let mut root_origin_ids = HashSet::new();
        let mut unresolved_terminal_ids = HashSet::new();

        if evidence_entity_ids.is_empty() {
            reasons.push(R::EmptyEvidenceSet);
            return ProvenanceGraphAssessment {
                decision: ProvenanceGraphDecision::Abstain,
                reasons,
                requested_evidence_count: 0,
                unique_evidence_count: 0,
                root_origin_ids: Vec::new(),
                unresolved_terminal_ids: Vec::new(),
            };
        }

        let mut node_index: HashMap<&str, usize> = HashMap::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            if node.entity_id.trim().is_empty() {
                push_reason(&mut reasons, R::EmptyEntityId);
                continue;
            }
            if node_index.insert(node.entity_id.as_str(), index).is_some() {
                push_reason(&mut reasons, R::DuplicateEntityId);
            }
        }

        let mut parent_indexes: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        let mut children_indexes: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for edge in &self.edges {
            let derived_index = match node_index.get(edge.derived_entity_id.as_str()).copied() {
                Some(index) => index,
                None => {
                    push_reason(&mut reasons, R::MissingDerivedEntity);
                    continue;
                }
            };
            let source_index = match node_index.get(edge.source_entity_id.as_str()).copied() {
                Some(index) => index,
                None => {
                    push_reason(&mut reasons, R::MissingSourceEntity);
                    continue;
                }
            };
            parent_indexes[derived_index].push(source_index);
            children_indexes[source_index].push(derived_index);
        }

        for (index, node) in self.nodes.iter().enumerate() {
            match node.kind {
                ProvenanceNodeKind::PrimaryOrigin if !parent_indexes[index].is_empty() => {
                    push_reason(&mut reasons, R::OriginNodeHasParents);
                }
                ProvenanceNodeKind::UnknownOrigin if !parent_indexes[index].is_empty() => {
                    push_reason(&mut reasons, R::UnknownOriginNodeHasParents);
                }
                ProvenanceNodeKind::DerivedArtifact if parent_indexes[index].is_empty() => {
                    push_reason(&mut reasons, R::DerivedArtifactHasNoParents);
                }
                _ => {}
            }
        }

        // Kahn's algorithm validates the full directed graph without recursive
        // call-stack growth. Sorting direction is parent -> derived child.
        if reasons.is_empty() {
            let mut remaining_parents: Vec<usize> =
                parent_indexes.iter().map(Vec::len).collect();
            let mut ready = VecDeque::new();
            for (index, count) in remaining_parents.iter().enumerate() {
                if *count == 0 {
                    ready.push_back(index);
                }
            }
            let mut processed = 0usize;
            while let Some(parent) = ready.pop_front() {
                processed += 1;
                for child in &children_indexes[parent] {
                    remaining_parents[*child] -= 1;
                    if remaining_parents[*child] == 0 {
                        ready.push_back(*child);
                    }
                }
            }
            if processed != self.nodes.len() {
                push_reason(&mut reasons, R::CycleDetected);
            }
        }

        let unique_evidence: HashSet<&str> =
            evidence_entity_ids.iter().map(String::as_str).collect();

        if !reasons.is_empty() {
            return ProvenanceGraphAssessment {
                decision: ProvenanceGraphDecision::Abstain,
                reasons,
                requested_evidence_count: evidence_entity_ids.len(),
                unique_evidence_count: unique_evidence.len(),
                root_origin_ids: Vec::new(),
                unresolved_terminal_ids: Vec::new(),
            };
        }

        let mut visited = HashSet::new();
        let mut pending: Vec<usize> = Vec::with_capacity(unique_evidence.len());
        for evidence_id in &unique_evidence {
            match node_index.get(evidence_id).copied() {
                Some(index) => pending.push(index),
                None => push_reason(&mut reasons, R::UnknownEvidenceEntity),
            }
        }

        while let Some(index) = pending.pop() {
            if !visited.insert(index) {
                continue;
            }
            let node = &self.nodes[index];
            match node.kind {
                ProvenanceNodeKind::PrimaryOrigin => {
                    root_origin_ids.insert(node.entity_id.clone());
                }
                ProvenanceNodeKind::UnknownOrigin => {
                    unresolved_terminal_ids.insert(node.entity_id.clone());
                }
                ProvenanceNodeKind::DerivedArtifact => {}
            }
            pending.extend(parent_indexes[index].iter().copied());
        }

        if policy.require_all_lineage_resolved && !unresolved_terminal_ids.is_empty() {
            push_reason(&mut reasons, R::UnknownOriginInSelectedLineage);
        }
        if root_origin_ids.len() < policy.minimum_distinct_origins {
            push_reason(&mut reasons, R::InsufficientDistinctOrigins);
        }

        let decision = if reasons.is_empty() {
            ProvenanceGraphDecision::EligibleForReview
        } else {
            ProvenanceGraphDecision::Abstain
        };

        ProvenanceGraphAssessment {
            decision,
            reasons,
            requested_evidence_count: evidence_entity_ids.len(),
            unique_evidence_count: unique_evidence.len(),
            root_origin_ids: {
                let mut values: Vec<_> = root_origin_ids.into_iter().collect();
                values.sort();
                values
            },
            unresolved_terminal_ids: {
                let mut values: Vec<_> = unresolved_terminal_ids.into_iter().collect();
                values.sort();
                values
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: ProvenanceNodeKind) -> ProvenanceGraphNode {
        ProvenanceGraphNode {
            entity_id: id.to_owned(),
            kind,
        }
    }

    fn edge(
        derived: &str,
        source: &str,
        kind: ProvenanceDerivationKind,
    ) -> ProvenanceDerivationEdge {
        ProvenanceDerivationEdge {
            derived_entity_id: derived.to_owned(),
            source_entity_id: source.to_owned(),
            kind,
        }
    }

    fn two_origin_graph() -> EvidenceProvenanceGraph {
        EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("origin-B", ProvenanceNodeKind::PrimaryOrigin),
                node("article-A", ProvenanceNodeKind::DerivedArtifact),
                node("article-B", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("article-A", "origin-A", ProvenanceDerivationKind::QuotedFrom),
                edge("article-B", "origin-B", ProvenanceDerivationKind::RevisionOf),
            ],
        }
    }

    fn assess(graph: &EvidenceProvenanceGraph, evidence: &[&str]) -> ProvenanceGraphAssessment {
        let ids: Vec<String> = evidence.iter().map(|id| (*id).to_owned()).collect();
        graph.assess(&ids, &ProvenanceGraphPolicy::strict())
    }

    #[test]
    fn distinct_articles_with_one_primary_origin_count_as_one_root() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("primary", ProvenanceNodeKind::PrimaryOrigin),
                node("article-A", ProvenanceNodeKind::DerivedArtifact),
                node("article-B", ProvenanceNodeKind::DerivedArtifact),
                node("article-C", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("article-A", "primary", ProvenanceDerivationKind::QuotedFrom),
                edge("article-B", "primary", ProvenanceDerivationKind::DerivedFrom),
                edge("article-C", "article-B", ProvenanceDerivationKind::RevisionOf),
            ],
        };

        let result = assess(&graph, &["article-A", "article-B", "article-C"]);
        assert_eq!(result.root_origin_ids, vec!["primary"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::InsufficientDistinctOrigins));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn multi_parent_synthesis_retains_both_upstream_roots() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("origin-B", ProvenanceNodeKind::PrimaryOrigin),
                node("synthesis", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("synthesis", "origin-A", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "origin-B", ProvenanceDerivationKind::HadPrimarySource),
            ],
        };

        let result = assess(&graph, &["synthesis"]);
        assert_eq!(result.root_origin_ids, vec!["origin-A", "origin-B"]);
        assert_eq!(result.decision, ProvenanceGraphDecision::EligibleForReview);
    }

    #[test]
    fn multiple_evidence_items_deduplicate_shared_ancestors() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("origin-B", ProvenanceNodeKind::PrimaryOrigin),
                node("article-A", ProvenanceNodeKind::DerivedArtifact),
                node("article-B", ProvenanceNodeKind::DerivedArtifact),
                node("article-C", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("article-A", "origin-A", ProvenanceDerivationKind::DerivedFrom),
                edge("article-B", "origin-A", ProvenanceDerivationKind::QuotedFrom),
                edge("article-C", "origin-B", ProvenanceDerivationKind::DerivedFrom),
            ],
        };

        let result = assess(&graph, &["article-A", "article-B", "article-C"]);
        assert_eq!(result.requested_evidence_count, 3);
        assert_eq!(result.root_origin_ids, vec!["origin-A", "origin-B"]);
        assert_eq!(result.decision, ProvenanceGraphDecision::EligibleForReview);
    }

    #[test]
    fn repeated_requested_evidence_ids_do_not_inflate_item_or_origin_count() {
        let graph = two_origin_graph();
        let result = assess(&graph, &["article-A", "article-A", "article-B"]);
        assert_eq!(result.requested_evidence_count, 3);
        assert_eq!(result.unique_evidence_count, 2);
        assert_eq!(result.root_origin_ids, vec!["origin-A", "origin-B"]);
        assert_eq!(result.decision, ProvenanceGraphDecision::EligibleForReview);
    }

    #[test]
    fn unknown_terminal_lineage_abstains_under_strict_policy() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("unknown", ProvenanceNodeKind::UnknownOrigin),
                node("synthesis", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("synthesis", "origin-A", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "unknown", ProvenanceDerivationKind::DerivedFrom),
            ],
        };

        let result = assess(&graph, &["synthesis"]);
        assert_eq!(result.root_origin_ids, vec!["origin-A"]);
        assert_eq!(result.unresolved_terminal_ids, vec!["unknown"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::UnknownOriginInSelectedLineage));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn permissive_resolution_policy_does_not_credit_unknown_as_a_root() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("origin-B", ProvenanceNodeKind::PrimaryOrigin),
                node("unknown", ProvenanceNodeKind::UnknownOrigin),
                node("synthesis", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("synthesis", "origin-A", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "origin-B", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "unknown", ProvenanceDerivationKind::DerivedFrom),
            ],
        };
        let policy = ProvenanceGraphPolicy::new(2, false).unwrap();
        let result = graph.assess(&["synthesis".to_owned()], &policy);
        assert_eq!(result.root_origin_ids, vec!["origin-A", "origin-B"]);
        assert_eq!(result.unresolved_terminal_ids, vec!["unknown"]);
        assert_eq!(result.decision, ProvenanceGraphDecision::EligibleForReview);
    }

    #[test]
    fn dangling_source_edge_abstains() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![node("article", ProvenanceNodeKind::DerivedArtifact)],
            edges: vec![edge("article", "missing-source", ProvenanceDerivationKind::QuotedFrom)],
        };
        let result = assess(&graph, &["article"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::MissingSourceEntity));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn cycle_abstains_even_if_each_node_has_a_parent() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("article-A", ProvenanceNodeKind::DerivedArtifact),
                node("article-B", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("article-A", "article-B", ProvenanceDerivationKind::DerivedFrom),
                edge("article-B", "article-A", ProvenanceDerivationKind::RevisionOf),
            ],
        };
        let result = assess(&graph, &["article-A"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::CycleDetected));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn role_conflicts_and_parentless_derived_nodes_abstain() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-with-parent", ProvenanceNodeKind::PrimaryOrigin),
                node("child", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![edge(
                "origin-with-parent",
                "child",
                ProvenanceDerivationKind::DerivedFrom,
            )],
        };
        let result = assess(&graph, &["child"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::OriginNodeHasParents));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn duplicate_ids_empty_input_and_missing_requested_id_fail_closed() {
        let duplicate = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin", ProvenanceNodeKind::PrimaryOrigin),
                node("origin", ProvenanceNodeKind::PrimaryOrigin),
            ],
            edges: vec![],
        };
        assert!(assess(&duplicate, &["origin"])
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::DuplicateEntityId));

        let graph = EvidenceProvenanceGraph {
            nodes: vec![node("origin", ProvenanceNodeKind::PrimaryOrigin)],
            edges: vec![],
        };
        assert!(graph
            .assess(&[], &ProvenanceGraphPolicy::strict())
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::EmptyEvidenceSet));
        assert!(assess(&graph, &["missing"])
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::UnknownEvidenceEntity));
    }

    #[test]
    fn graph_with_multiple_paths_counts_each_terminal_origin_once() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin-A", ProvenanceNodeKind::PrimaryOrigin),
                node("origin-B", ProvenanceNodeKind::PrimaryOrigin),
                node("intermediate", ProvenanceNodeKind::DerivedArtifact),
                node("synthesis", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("intermediate", "origin-A", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "origin-A", ProvenanceDerivationKind::QuotedFrom),
                edge("synthesis", "intermediate", ProvenanceDerivationKind::DerivedFrom),
                edge("synthesis", "origin-B", ProvenanceDerivationKind::DerivedFrom),
            ],
        };

        let result = assess(&graph, &["synthesis"]);
        assert_eq!(result.root_origin_ids, vec!["origin-A", "origin-B"]);
        assert_eq!(result.decision, ProvenanceGraphDecision::EligibleForReview);
    }

    #[test]
    fn basic_graph_resolves_two_roots() {
        let result = assess(&two_origin_graph(), &["article-A", "article-B"]);
        assert_eq!(result.root_origin_ids.len(), 2);
    }

    #[test]
    fn disconnected_cycle_invalidates_the_graph_snapshot() {
        let graph = EvidenceProvenanceGraph {
            nodes: vec![
                node("origin", ProvenanceNodeKind::PrimaryOrigin),
                node("evidence", ProvenanceNodeKind::DerivedArtifact),
                node("cycle-A", ProvenanceNodeKind::DerivedArtifact),
                node("cycle-B", ProvenanceNodeKind::DerivedArtifact),
            ],
            edges: vec![
                edge("evidence", "origin", ProvenanceDerivationKind::DerivedFrom),
                edge("cycle-A", "cycle-B", ProvenanceDerivationKind::DerivedFrom),
                edge("cycle-B", "cycle-A", ProvenanceDerivationKind::RevisionOf),
            ],
        };

        let result = assess(&graph, &["evidence"]);
        assert!(result
            .reasons
            .contains(&ProvenanceGraphAbstentionReason::CycleDetected));
        assert_eq!(result.decision, ProvenanceGraphDecision::Abstain);
    }

    #[test]
    fn graph_policy_rejects_zero_required_origins() {
        assert!(ProvenanceGraphPolicy::new(0, true).is_err());
    }

    /// Opt-in release-mode scaling benchmark of graph validation and root resolution.
    #[test]
    #[ignore = "manual release-mode scaling benchmark; not a CI correctness gate"]
    fn benchmark_lineage_graph_scaling() {
        use std::time::Instant;

        const REPEATS: usize = 7;
        let policy = ProvenanceGraphPolicy::strict();

        for size in [1_000usize, 10_000, 100_000] {
            let mut graph = EvidenceProvenanceGraph::default();
            let mut selected = Vec::with_capacity(size);
            graph.nodes.reserve(size * 2);
            graph.edges.reserve(size);

            for index in 0..size {
                let origin_id = format!("origin-{index}");
                let evidence_id = format!("evidence-{index}");
                graph.nodes.push(node(&origin_id, ProvenanceNodeKind::PrimaryOrigin));
                graph.nodes.push(node(&evidence_id, ProvenanceNodeKind::DerivedArtifact));
                graph.edges.push(edge(
                    &evidence_id,
                    &origin_id,
                    ProvenanceDerivationKind::DerivedFrom,
                ));
                selected.push(evidence_id);
            }

            let expected = graph.assess(&selected, &policy);
            assert_eq!(expected.decision, ProvenanceGraphDecision::EligibleForReview);
            assert_eq!(expected.root_origin_ids.len(), size);
            let mut samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                let assessment = graph.assess(&selected, &policy);
                std::hint::black_box((
                    assessment.decision,
                    assessment.root_origin_ids.len(),
                    assessment.unresolved_terminal_ids.len(),
                ));
                samples.push(start.elapsed().as_nanos());
            }
            samples.sort_unstable();
            let p50 = samples[REPEATS / 2];
            let p95 = samples[(REPEATS * 95 + 99) / 100 - 1];
            println!(
                "nodes={} edges={} selected={} reps={} p50={} p95={} ns/node={}",
                graph.nodes.len(),
                graph.edges.len(),
                selected.len(),
                REPEATS,
                p50,
                p95,
                p50 / graph.nodes.len() as u128
            );
        }
    }
}
