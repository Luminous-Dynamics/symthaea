// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Graph statistics and analysis for curriculum documents.

use crate::converter::CurriculumDocument;
use crate::taxonomy::{BroadField, IscedLevel};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

/// Comprehensive statistics about a curriculum graph.
#[derive(Debug, Serialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub nodes_by_level: Vec<(String, usize)>,
    pub nodes_by_subject: Vec<(String, usize)>,
    pub nodes_by_isced_field: Vec<(String, usize)>,
    pub nodes_by_type: Vec<(String, usize)>,
    pub nodes_by_bloom: Vec<(String, usize)>,
    pub edges_by_type: Vec<(String, usize)>,
    pub connectivity: ConnectivityStats,
    pub coverage: CoverageStats,
    pub total_estimated_hours: u32,
}

/// Connectivity metrics.
#[derive(Debug, Serialize)]
pub struct ConnectivityStats {
    pub orphan_nodes: usize,
    pub root_nodes: usize,
    pub leaf_nodes: usize,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub avg_degree: f32,
}

/// Coverage analysis — what's present and what's missing.
#[derive(Debug, Serialize)]
pub struct CoverageStats {
    pub isced_levels_present: Vec<String>,
    pub isced_levels_missing: Vec<String>,
    pub isced_fields_present: Vec<String>,
    pub isced_fields_missing: Vec<String>,
    pub has_k12: bool,
    pub has_undergraduate: bool,
    pub has_graduate: bool,
    pub has_doctoral: bool,
    pub has_career_pathways: bool,
}

/// Analyze a curriculum document and return comprehensive statistics.
pub fn analyze(doc: &CurriculumDocument) -> GraphStats {
    let mut level_counts: HashMap<String, usize> = HashMap::new();
    let mut subject_counts: HashMap<String, usize> = HashMap::new();
    let mut field_counts: HashMap<String, usize> = HashMap::new();
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut bloom_counts: HashMap<String, usize> = HashMap::new();
    let mut edge_type_counts: HashMap<String, usize> = HashMap::new();
    let mut total_hours = 0u32;

    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut out_degree: HashMap<String, usize> = HashMap::new();
    let node_ids: HashSet<String> = doc.nodes.iter().map(|n| n.id.clone()).collect();

    for node in &doc.nodes {
        for g in &node.grade_levels {
            *level_counts.entry(g.clone()).or_default() += 1;
        }
        *subject_counts.entry(node.subject_area.clone()).or_default() += 1;

        let field = BroadField::from_subject(&node.subject_area);
        *field_counts.entry(field.label().to_string()).or_default() += 1;

        *type_counts.entry(node.node_type.clone()).or_default() += 1;
        *bloom_counts.entry(node.bloom_level.clone()).or_default() += 1;
        total_hours += node.estimated_hours;
    }

    for edge in &doc.edges {
        *edge_type_counts.entry(edge.edge_type.clone()).or_default() += 1;
        *out_degree.entry(edge.from.clone()).or_default() += 1;
        *in_degree.entry(edge.to.clone()).or_default() += 1;
    }

    let orphans = node_ids
        .iter()
        .filter(|id| !in_degree.contains_key(*id) && !out_degree.contains_key(*id))
        .count();
    let roots = node_ids.iter().filter(|id| !in_degree.contains_key(*id)).count();
    let leaves = node_ids.iter().filter(|id| !out_degree.contains_key(*id)).count();
    let max_in = in_degree.values().copied().max().unwrap_or(0);
    let max_out = out_degree.values().copied().max().unwrap_or(0);
    let total_degree: usize = in_degree.values().sum::<usize>() + out_degree.values().sum::<usize>();
    let avg_degree = if doc.nodes.is_empty() {
        0.0
    } else {
        total_degree as f32 / doc.nodes.len() as f32
    };

    // Coverage analysis
    let grade_levels: HashSet<String> = doc
        .nodes
        .iter()
        .flat_map(|n| n.grade_levels.iter().cloned())
        .collect();

    let all_isced_levels = [
        "EarlyChildhood", "Primary", "LowerSecondary", "UpperSecondary",
        "PostSecondaryNonTertiary", "ShortCycleTertiary", "Bachelors", "Masters", "Doctoral",
    ];
    let present_isced: HashSet<String> = grade_levels
        .iter()
        .map(|g| format!("{:?}", IscedLevel::from_grade_level(g)))
        .collect();
    let missing_isced: Vec<String> = all_isced_levels
        .iter()
        .filter(|l| !present_isced.contains(**l))
        .map(|l| l.to_string())
        .collect();

    let all_fields: Vec<&str> = vec![
        "Education", "Arts & Humanities", "Social Sciences", "Business & Law",
        "Natural Sciences & Mathematics", "Information & Communication Technologies",
        "Engineering & Manufacturing", "Agriculture & Environment", "Health & Welfare",
        "Services & Recreation", "Meta-Learning (Consciousness Computing)",
    ];
    let present_fields: HashSet<String> = field_counts.keys().cloned().collect();
    let missing_fields: Vec<String> = all_fields
        .iter()
        .filter(|f| !present_fields.contains(**f))
        .map(|f| f.to_string())
        .collect();

    let has_k12 = grade_levels.iter().any(|g| g.starts_with("Grade") || g == "Kindergarten");
    let has_undergrad = grade_levels.contains("Undergraduate");
    let has_grad = grade_levels.contains("Graduate");
    let has_doctoral = grade_levels.contains("Doctoral");
    let has_career = doc.nodes.iter().any(|n| n.tags.contains(&"esco".to_string()) || n.tags.contains(&"career".to_string()));

    let mut sorted_levels: Vec<_> = level_counts.into_iter().collect();
    sorted_levels.sort_by(|a, b| b.1.cmp(&a.1));
    let mut sorted_subjects: Vec<_> = subject_counts.into_iter().collect();
    sorted_subjects.sort_by(|a, b| b.1.cmp(&a.1));
    let mut sorted_fields: Vec<_> = field_counts.into_iter().collect();
    sorted_fields.sort_by(|a, b| b.1.cmp(&a.1));
    let mut sorted_types: Vec<_> = type_counts.into_iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(&a.1));

    let bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
    let sorted_bloom: Vec<_> = bloom_order
        .iter()
        .map(|b| (b.to_string(), *bloom_counts.get(*b).unwrap_or(&0)))
        .collect();

    let mut sorted_edge_types: Vec<_> = edge_type_counts.into_iter().collect();
    sorted_edge_types.sort_by(|a, b| b.1.cmp(&a.1));

    GraphStats {
        total_nodes: doc.nodes.len(),
        total_edges: doc.edges.len(),
        nodes_by_level: sorted_levels,
        nodes_by_subject: sorted_subjects,
        nodes_by_isced_field: sorted_fields,
        nodes_by_type: sorted_types,
        nodes_by_bloom: sorted_bloom,
        edges_by_type: sorted_edge_types,
        connectivity: ConnectivityStats {
            orphan_nodes: orphans,
            root_nodes: roots,
            leaf_nodes: leaves,
            max_in_degree: max_in,
            max_out_degree: max_out,
            avg_degree,
        },
        coverage: CoverageStats {
            isced_levels_present: present_isced.into_iter().collect(),
            isced_levels_missing: missing_isced,
            isced_fields_present: present_fields.into_iter().collect(),
            isced_fields_missing: missing_fields,
            has_k12,
            has_undergraduate: has_undergrad,
            has_graduate: has_grad,
            has_doctoral,
            has_career_pathways: has_career,
        },
        total_estimated_hours: total_hours,
    }
}

/// Print statistics to stderr in a human-readable format.
pub fn print_stats(stats: &GraphStats) {
    eprintln!("=== UNIFIED GRAPH STATISTICS ===\n");
    eprintln!("Nodes: {}", stats.total_nodes);
    eprintln!("Edges: {}", stats.total_edges);
    eprintln!("Total estimated hours: {}", stats.total_estimated_hours);

    eprintln!("\n--- By Academic Level ---");
    for (level, count) in &stats.nodes_by_level {
        eprintln!("  {:<20} {:>5}", level, count);
    }

    eprintln!("\n--- By ISCED-F Field ---");
    for (field, count) in &stats.nodes_by_isced_field {
        eprintln!("  {:<45} {:>5}", field, count);
    }

    eprintln!("\n--- By Subject ---");
    for (subject, count) in &stats.nodes_by_subject {
        eprintln!("  {:<50} {:>5}", subject, count);
    }

    eprintln!("\n--- By Node Type ---");
    for (t, count) in &stats.nodes_by_type {
        eprintln!("  {:<15} {:>5}", t, count);
    }

    eprintln!("\n--- Bloom's Taxonomy ---");
    for (bloom, count) in &stats.nodes_by_bloom {
        let pct = if stats.total_nodes > 0 { *count as f32 / stats.total_nodes as f32 * 100.0 } else { 0.0 };
        let bar: String = "#".repeat((pct / 2.0) as usize);
        eprintln!("  {:<12} {:>5} ({:>5.1}%) {}", bloom, count, pct, bar);
    }

    eprintln!("\n--- Edge Types ---");
    for (t, count) in &stats.edges_by_type {
        eprintln!("  {:<15} {:>5}", t, count);
    }

    eprintln!("\n--- Connectivity ---");
    eprintln!("  Orphan nodes:   {}", stats.connectivity.orphan_nodes);
    eprintln!("  Root nodes:     {}", stats.connectivity.root_nodes);
    eprintln!("  Leaf nodes:     {}", stats.connectivity.leaf_nodes);
    eprintln!("  Max in-degree:  {}", stats.connectivity.max_in_degree);
    eprintln!("  Max out-degree: {}", stats.connectivity.max_out_degree);
    eprintln!("  Avg degree:     {:.1}", stats.connectivity.avg_degree);

    eprintln!("\n--- Coverage ---");
    eprintln!("  K-12:          {}", if stats.coverage.has_k12 { "YES" } else { "MISSING" });
    eprintln!("  Undergraduate: {}", if stats.coverage.has_undergraduate { "YES" } else { "MISSING" });
    eprintln!("  Graduate:      {}", if stats.coverage.has_graduate { "YES" } else { "MISSING" });
    eprintln!("  Doctoral:      {}", if stats.coverage.has_doctoral { "YES" } else { "MISSING" });
    eprintln!("  Career paths:  {}", if stats.coverage.has_career_pathways { "YES" } else { "MISSING" });

    if !stats.coverage.isced_fields_missing.is_empty() {
        eprintln!("\n  Missing ISCED-F fields:");
        for f in &stats.coverage.isced_fields_missing {
            eprintln!("    - {}", f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::*;

    fn make_doc(nodes: Vec<CurriculumNode>, edges: Vec<CurriculumEdge>) -> CurriculumDocument {
        CurriculumDocument {
            metadata: CurriculumMetadata {
                title: "Test".into(), framework: "Test".into(), source: String::new(),
                grade_level: "Grade3".into(), subject_area: "Math".into(), domain: "Math".into(),
                version: "1".into(), total_standards: nodes.len(), domains: vec![],
                created_at: "2026-01-01".into(), notes: String::new(),
                academic_level: None, institution: None, cip_code: None,
                total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        }
    }

    fn make_node(id: &str, subject: &str, grade: &str, bloom: &str) -> CurriculumNode {
        CurriculumNode {
            id: id.into(), title: id.into(), description: String::new(),
            node_type: "Concept".into(), difficulty: "Intermediate".into(),
            domain: "Test".into(), subdomain: String::new(), tags: vec![],
            estimated_hours: 10, grade_levels: vec![grade.into()],
            bloom_level: bloom.into(), subject_area: subject.into(),
            academic_standards: vec![], credit_hours: None, course_level: None,
            cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        }
    }

    #[test]
    fn test_analyze_basic() {
        let doc = make_doc(
            vec![
                make_node("A", "Mathematics", "Grade3", "Apply"),
                make_node("B", "Mathematics", "Grade3", "Analyze"),
            ],
            vec![CurriculumEdge {
                from: "A".into(), to: "B".into(),
                edge_type: "Requires".into(), strength_permille: 800,
                rationale: "test".into(),
            }],
        );

        let stats = analyze(&doc);
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
        assert_eq!(stats.connectivity.orphan_nodes, 0);
        assert_eq!(stats.connectivity.root_nodes, 1);
        assert_eq!(stats.connectivity.leaf_nodes, 1);
        assert!(stats.coverage.has_k12);
        assert!(!stats.coverage.has_undergraduate);
    }

    #[test]
    fn test_analyze_empty() {
        let doc = make_doc(vec![], vec![]);
        let stats = analyze(&doc);
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.connectivity.avg_degree, 0.0);
    }

    #[test]
    fn test_isced_field_classification() {
        let doc = make_doc(
            vec![
                make_node("A", "Mathematics", "Grade3", "Apply"),
                make_node("B", "Computer Science", "Undergraduate", "Analyze"),
                make_node("C", "Literature", "Doctoral", "Create"),
            ],
            vec![],
        );

        let stats = analyze(&doc);
        assert!(stats.nodes_by_isced_field.iter().any(|(f, _)| f.contains("Natural Sciences")));
        assert!(stats.nodes_by_isced_field.iter().any(|(f, _)| f.contains("ICT") || f.contains("Information")));
        assert!(stats.nodes_by_isced_field.iter().any(|(f, _)| f.contains("Arts") || f.contains("Humanities")));
    }
}
