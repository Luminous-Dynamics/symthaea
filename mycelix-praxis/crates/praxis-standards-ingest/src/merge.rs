// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Merge multiple curriculum documents into a unified knowledge graph.
//!
//! Takes any number of curriculum documents (K-12, undergraduate, graduate,
//! doctoral) plus optional bridge documents and combines them into a single
//! `CurriculumDocument` with all nodes and edges from all sources.
//!
//! This produces the **Lifelong Epistemic Path** — a single connected graph
//! from PreK through PhD.

use crate::bridge::BridgeDocument;
use crate::converter::{CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode};
use serde::Serialize;
use std::collections::HashSet;

/// Statistics about the merged graph.
#[derive(Debug, Serialize)]
pub struct MergeStatistics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub sources_merged: usize,
    pub bridge_edges_added: usize,
    pub duplicate_nodes_skipped: usize,
    pub levels: Vec<String>,
    pub subjects: Vec<String>,
}

/// Merge multiple curriculum documents and bridge edges into a unified graph.
pub fn merge_documents(
    documents: &[CurriculumDocument],
    bridges: &[BridgeDocument],
) -> (CurriculumDocument, MergeStatistics) {
    let mut all_nodes: Vec<CurriculumNode> = Vec::new();
    let mut all_edges: Vec<CurriculumEdge> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut levels: HashSet<String> = HashSet::new();
    let mut subjects: HashSet<String> = HashSet::new();
    let mut duplicate_count = 0;

    // Collect all nodes and intra-document edges
    for doc in documents {
        levels.insert(doc.metadata.grade_level.clone());
        subjects.insert(doc.metadata.subject_area.clone());

        for node in &doc.nodes {
            if seen_ids.insert(node.id.clone()) {
                all_nodes.push(clone_node(node));
            } else {
                duplicate_count += 1;
            }
        }

        all_edges.extend(doc.edges.iter().cloned());
    }

    // Collect bridge edges
    let mut bridge_edge_count = 0;
    for bridge in bridges {
        bridge_edge_count += bridge.edges.len();
        all_edges.extend(bridge.edges.iter().cloned());
    }

    // Deduplicate edges
    let unique_edges = dedup_edges(all_edges);

    // Collect all domains across all documents
    let domains: Vec<String> = documents
        .iter()
        .flat_map(|d| d.metadata.domains.iter().cloned())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let stats = MergeStatistics {
        total_nodes: all_nodes.len(),
        total_edges: unique_edges.len(),
        sources_merged: documents.len(),
        bridge_edges_added: bridge_edge_count,
        duplicate_nodes_skipped: duplicate_count,
        levels: levels.into_iter().collect(),
        subjects: subjects.into_iter().collect(),
    };

    let merged = CurriculumDocument {
        metadata: CurriculumMetadata {
            title: "Unified Lifelong Learning Graph".to_string(),
            framework: "Multi-Framework".to_string(),
            source: String::new(),
            grade_level: "PreK-Doctoral".to_string(),
            subject_area: "Multi-Subject".to_string(),
            domain: "Multi-Subject".to_string(),
            version: "2024".to_string(),
            total_standards: all_nodes.len(),
            domains,
            created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
            notes: format!(
                "Merged {} curriculum documents with {} bridge edges. \
                 {} unique nodes, {} edges.",
                documents.len(),
                bridge_edge_count,
                all_nodes.len(),
                unique_edges.len(),
            ),
            academic_level: None,
            institution: None,
            cip_code: None,
            total_credits: None,
            duration_semesters: None,
        },
        nodes: all_nodes,
        edges: unique_edges,
    };

    (merged, stats)
}

/// Clone a curriculum node (manual since we own the struct).
fn clone_node(n: &CurriculumNode) -> CurriculumNode {
    CurriculumNode {
        id: n.id.clone(),
        title: n.title.clone(),
        description: n.description.clone(),
        node_type: n.node_type.clone(),
        difficulty: n.difficulty.clone(),
        domain: n.domain.clone(),
        subdomain: n.subdomain.clone(),
        tags: n.tags.clone(),
        estimated_hours: n.estimated_hours,
        grade_levels: n.grade_levels.clone(),
        bloom_level: n.bloom_level.clone(),
        subject_area: n.subject_area.clone(),
        academic_standards: n.academic_standards.clone(),
        credit_hours: n.credit_hours,
        course_level: n.course_level.clone(),
        cip_code: n.cip_code.clone(),
        program_id: n.program_id.clone(),
        corequisites: n.corequisites.clone(), supplementary_resources: n.supplementary_resources.clone(),
        exam_weight: n.exam_weight.clone(),
    }
}

/// Deduplicate edges by (from, to, edge_type).
fn dedup_edges(edges: Vec<CurriculumEdge>) -> Vec<CurriculumEdge> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    edges
        .into_iter()
        .filter(|e| seen.insert((e.from.clone(), e.to.clone(), e.edge_type.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::{AcademicStandardRef, CurriculumMetadata};

    fn make_meta(grade: &str, subject: &str) -> CurriculumMetadata {
        CurriculumMetadata {
            title: format!("{subject} ({grade})"),
            framework: "Test".into(),
            source: String::new(),
            grade_level: grade.into(),
            subject_area: subject.into(),
            domain: subject.into(),
            version: "2024".into(),
            total_standards: 0,
            domains: vec![],
            created_at: "2026-01-01".into(),
            notes: String::new(),
            academic_level: None,
            institution: None,
            cip_code: None,
            total_credits: None,
            duration_semesters: None,
        }
    }

    fn make_node(id: &str) -> CurriculumNode {
        CurriculumNode {
            id: id.into(),
            title: id.into(),
            description: String::new(),
            node_type: "Concept".into(),
            difficulty: "Intermediate".into(),
            domain: "Test".into(),
            subdomain: String::new(),
            tags: vec![],
            estimated_hours: 10,
            grade_levels: vec![],
            bloom_level: "Apply".into(),
            subject_area: "Test".into(),
            academic_standards: vec![],
            credit_hours: None,
            course_level: None,
            cip_code: None,
            program_id: None,
            corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        }
    }

    #[test]
    fn test_merge_two_documents() {
        let doc_a = CurriculumDocument {
            metadata: make_meta("Grade5", "Mathematics"),
            nodes: vec![make_node("A1"), make_node("A2")],
            edges: vec![CurriculumEdge {
                from: "A1".into(), to: "A2".into(),
                edge_type: "Requires".into(), strength_permille: 800,
                rationale: "test".into(),
            }],
        };
        let doc_b = CurriculumDocument {
            metadata: make_meta("Undergraduate", "CS"),
            nodes: vec![make_node("B1")],
            edges: vec![],
        };

        let (merged, stats) = merge_documents(&[doc_a, doc_b], &[]);
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 1);
        assert_eq!(stats.sources_merged, 2);
    }

    #[test]
    fn test_merge_with_bridge() {
        let doc_a = CurriculumDocument {
            metadata: make_meta("Grade5", "Math"),
            nodes: vec![make_node("A1")],
            edges: vec![],
        };
        let doc_b = CurriculumDocument {
            metadata: make_meta("Undergraduate", "CS"),
            nodes: vec![make_node("B1")],
            edges: vec![],
        };
        let bridge = BridgeDocument {
            title: "Test".into(),
            description: "Test".into(),
            edges: vec![CurriculumEdge {
                from: "A1".into(), to: "B1".into(),
                edge_type: "LeadsTo".into(), strength_permille: 700,
                rationale: "bridge".into(),
            }],
            statistics: crate::bridge::BridgeStatistics {
                total_edges: 1, k12_to_undergrad: 1,
                undergrad_to_grad: 0, grad_to_phd: 0,
                levels_bridged: vec![], subjects_bridged: vec![],
            },
        };

        let (merged, stats) = merge_documents(&[doc_a, doc_b], &[bridge]);
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
        assert_eq!(stats.bridge_edges_added, 1);
    }

    #[test]
    fn test_duplicate_nodes_skipped() {
        let doc_a = CurriculumDocument {
            metadata: make_meta("Grade5", "Math"),
            nodes: vec![make_node("SHARED"), make_node("A1")],
            edges: vec![],
        };
        let doc_b = CurriculumDocument {
            metadata: make_meta("Grade6", "Math"),
            nodes: vec![make_node("SHARED"), make_node("B1")],
            edges: vec![],
        };

        let (merged, stats) = merge_documents(&[doc_a, doc_b], &[]);
        assert_eq!(stats.total_nodes, 3); // SHARED + A1 + B1
        assert_eq!(stats.duplicate_nodes_skipped, 1);
    }

    #[test]
    fn test_duplicate_edges_deduped() {
        let edge = CurriculumEdge {
            from: "A".into(), to: "B".into(),
            edge_type: "Requires".into(), strength_permille: 800,
            rationale: "test".into(),
        };
        let deduped = dedup_edges(vec![edge.clone(), edge.clone(), edge]);
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn test_merge_empty() {
        let (merged, stats) = merge_documents(&[], &[]);
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
    }
}
