// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-level bridge edge generator.
//!
//! Creates prerequisite edges that span across academic levels:
//! K-12 → Undergraduate → Graduate → Doctoral.
//!
//! This is the core feature that enables a **continuous learning pathway**
//! from Grade 8 Algebra through a PhD in Mathematics — something that
//! doesn't exist in any current education system.
//!
//! # Algorithm
//!
//! 1. Load multiple curriculum documents at different levels
//! 2. Match nodes across levels by subject/domain similarity and Bloom progression
//! 3. Generate `LeadsTo` edges from terminal K-12 nodes to introductory college courses
//! 4. Generate `LeadsTo` edges from advanced undergrad to graduate courses
//! 5. Generate `LeadsTo` edges from graduate coursework to PhD milestones
//!
//! The matching uses subject-area alignment (exact or fuzzy) and Bloom-level
//! ordering to ensure edges flow from lower to higher cognitive complexity.

use crate::converter::{CurriculumDocument, CurriculumEdge, CurriculumNode};
use serde::{Deserialize, Serialize};

/// A cross-level bridge document containing only the connecting edges.
#[derive(Debug, Serialize, Deserialize)]
pub struct BridgeDocument {
    pub title: String,
    pub description: String,
    pub edges: Vec<CurriculumEdge>,
    pub statistics: BridgeStatistics,
}

/// Statistics about the generated bridge.
#[derive(Debug, Serialize, Deserialize)]
pub struct BridgeStatistics {
    pub total_edges: usize,
    pub k12_to_undergrad: usize,
    pub undergrad_to_grad: usize,
    pub grad_to_phd: usize,
    pub levels_bridged: Vec<String>,
    pub subjects_bridged: Vec<String>,
}

/// Academic level of a curriculum document (inferred from metadata).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    K12,
    Undergraduate,
    Graduate,
    Doctoral,
}

impl Level {
    /// Infer level from a curriculum document's metadata.
    pub fn from_document(doc: &CurriculumDocument) -> Self {
        match doc.metadata.grade_level.as_str() {
            "Doctoral" => Level::Doctoral,
            "Graduate" => Level::Graduate,
            "Undergraduate" => Level::Undergraduate,
            "PostDoctoral" => Level::Doctoral,
            "Professional" => Level::Graduate,
            _ => Level::K12,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Level::K12 => "K-12",
            Level::Undergraduate => "Undergraduate",
            Level::Graduate => "Graduate",
            Level::Doctoral => "Doctoral",
        }
    }
}

/// Generate cross-level bridge edges between multiple curriculum documents.
///
/// Documents are automatically sorted by level and matched by subject.
pub fn generate_bridge(documents: &[CurriculumDocument]) -> BridgeDocument {
    let mut classified: Vec<(Level, &CurriculumDocument)> = documents
        .iter()
        .map(|d| (Level::from_document(d), d))
        .collect();

    classified.sort_by(|a, b| a.0.cmp(&b.0));

    let mut all_edges = Vec::new();
    let mut k12_to_undergrad = 0;
    let mut undergrad_to_grad = 0;
    let mut grad_to_phd = 0;

    // Generate edges between adjacent levels
    for i in 0..classified.len() {
        for j in (i + 1)..classified.len() {
            let (ref level_a, doc_a) = classified[i];
            let (ref level_b, doc_b) = classified[j];

            // Only bridge adjacent levels
            if !are_adjacent(level_a, level_b) {
                continue;
            }

            // Only bridge matching subjects
            if !subjects_match(doc_a, doc_b) {
                continue;
            }

            let edges = bridge_two_levels(doc_a, level_a, doc_b, level_b);

            for _edge in &edges {
                match (level_a, level_b) {
                    (Level::K12, Level::Undergraduate) => k12_to_undergrad += 1,
                    (Level::K12, Level::Graduate) | (Level::K12, Level::Doctoral) => {
                        // Skip-level: count as K-12→undergrad for simplicity
                        k12_to_undergrad += 1;
                    }
                    (Level::Undergraduate, Level::Graduate) => undergrad_to_grad += 1,
                    (Level::Graduate, Level::Doctoral) | (Level::Undergraduate, Level::Doctoral) => {
                        grad_to_phd += 1
                    }
                    _ => {}
                }
            }
            all_edges.extend(edges);
        }
    }

    let levels_bridged: Vec<String> = classified
        .iter()
        .map(|(l, _)| l.label().to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let subjects_bridged: Vec<String> = classified
        .iter()
        .map(|(_, d)| d.metadata.subject_area.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    BridgeDocument {
        title: "Cross-Level Learning Pathway Bridge".to_string(),
        description: format!(
            "Bridges {} curriculum documents across {} academic levels.",
            documents.len(),
            levels_bridged.len()
        ),
        edges: all_edges.clone(),
        statistics: BridgeStatistics {
            total_edges: all_edges.len(),
            k12_to_undergrad,
            undergrad_to_grad,
            grad_to_phd,
            levels_bridged,
            subjects_bridged,
        },
    }
}

/// Check if two levels can be bridged (adjacent or skip-level).
fn are_adjacent(a: &Level, b: &Level) -> bool {
    matches!(
        (a, b),
        (Level::K12, Level::Undergraduate)
            | (Level::K12, Level::Graduate)
            | (Level::K12, Level::Doctoral)
            | (Level::Undergraduate, Level::Graduate)
            | (Level::Undergraduate, Level::Doctoral)
            | (Level::Graduate, Level::Doctoral)
    )
}

/// Check if two documents cover matching subjects.
fn subjects_match(a: &CurriculumDocument, b: &CurriculumDocument) -> bool {
    let subj_a = normalize_for_matching(&a.metadata.subject_area);
    let subj_b = normalize_for_matching(&b.metadata.subject_area);

    // Exact match
    if subj_a == subj_b {
        return true;
    }

    // Known cross-mappings
    let mappings: &[(&[&str], &[&str])] = &[
        (
            &["mathematics", "math"],
            &["discrete structures", "algorithms", "statistics"],
        ),
        (
            &["science", "physical sciences"],
            &["physics", "chemistry", "biology", "biological sciences"],
        ),
        (
            &["computer", "technology"],
            &[
                "computer science",
                "software",
                "information",
                "algorithms",
                "architecture",
                "operating systems",
                "networking",
                "security",
                "programming",
            ],
        ),
        (
            &["english", "language arts"],
            &["writing", "composition", "literature", "rhetoric"],
        ),
    ];

    for (group_a, group_b) in mappings {
        let a_matches = group_a.iter().any(|k| subj_a.contains(k))
            || group_b.iter().any(|k| subj_a.contains(k));
        let b_matches = group_a.iter().any(|k| subj_b.contains(k))
            || group_b.iter().any(|k| subj_b.contains(k));
        if a_matches && b_matches {
            return true;
        }
    }

    // CIP code prefix matching
    if let (Some(ref cip_a), Some(ref cip_b)) = (&a.metadata.cip_code, &b.metadata.cip_code) {
        let prefix_a = cip_a.split('.').next().unwrap_or("");
        let prefix_b = cip_b.split('.').next().unwrap_or("");
        if !prefix_a.is_empty() && prefix_a == prefix_b {
            return true;
        }
    }

    false
}

/// Generate bridge edges between two documents at adjacent levels.
fn bridge_two_levels(
    lower: &CurriculumDocument,
    lower_level: &Level,
    upper: &CurriculumDocument,
    upper_level: &Level,
) -> Vec<CurriculumEdge> {
    let mut edges = Vec::new();

    // Find "terminal" nodes in the lower level (highest Bloom, last in sequence)
    let terminal_nodes = find_terminal_nodes(&lower.nodes);

    // Find "entry" nodes in the upper level (lowest Bloom, first in sequence)
    let entry_nodes = find_entry_nodes(&upper.nodes);

    // For PhD: connect to the first milestone (coursework)
    let target_nodes = if *upper_level == Level::Doctoral {
        upper
            .nodes
            .iter()
            .filter(|n| n.node_type == "Course" || n.id.contains("coursework"))
            .collect::<Vec<_>>()
    } else {
        entry_nodes
    };

    // Connect terminals to entries by topic overlap
    for terminal in &terminal_nodes {
        for entry in &target_nodes {
            if nodes_topic_overlap(terminal, entry) {
                edges.push(CurriculumEdge {
                    from: terminal.id.clone(),
                    to: entry.id.clone(),
                    edge_type: "LeadsTo".to_string(),
                    strength_permille: 750,
                    rationale: format!(
                        "Cross-level bridge: {} ({}) → {} ({}).",
                        terminal.title,
                        lower_level.label(),
                        entry.title,
                        upper_level.label()
                    ),
                });
            }
        }
    }

    // If no topic matches, create a general bridge from the last terminal to the first entry
    if edges.is_empty() && !terminal_nodes.is_empty() && !target_nodes.is_empty() {
        let last_terminal = terminal_nodes.last().unwrap();
        let first_entry = target_nodes.first().unwrap();
        edges.push(CurriculumEdge {
            from: last_terminal.id.clone(),
            to: first_entry.id.clone(),
            edge_type: "LeadsTo".to_string(),
            strength_permille: 600,
            rationale: format!(
                "General bridge: {} ({}) → {} ({}).",
                lower.metadata.title,
                lower_level.label(),
                upper.metadata.title,
                upper_level.label()
            ),
        });
    }

    edges
}

/// Find terminal nodes — those with highest Bloom level or at the end of sequences.
fn find_terminal_nodes(nodes: &[CurriculumNode]) -> Vec<&CurriculumNode> {
    if nodes.is_empty() {
        return vec![];
    }

    let max_bloom = nodes
        .iter()
        .map(|n| bloom_ordinal(&n.bloom_level))
        .max()
        .unwrap_or(0);

    // All nodes at the highest Bloom level
    let top_bloom: Vec<&CurriculumNode> = nodes
        .iter()
        .filter(|n| bloom_ordinal(&n.bloom_level) >= max_bloom.saturating_sub(1))
        .collect();

    if top_bloom.len() <= 5 {
        top_bloom
    } else {
        // Take last 5 if too many
        top_bloom.into_iter().rev().take(5).collect()
    }
}

/// Find entry nodes — those with lowest Bloom level or at the start of sequences.
fn find_entry_nodes(nodes: &[CurriculumNode]) -> Vec<&CurriculumNode> {
    if nodes.is_empty() {
        return vec![];
    }

    let min_bloom = nodes
        .iter()
        .map(|n| bloom_ordinal(&n.bloom_level))
        .min()
        .unwrap_or(0);

    // All nodes at the lowest Bloom level
    let bottom_bloom: Vec<&CurriculumNode> = nodes
        .iter()
        .filter(|n| bloom_ordinal(&n.bloom_level) <= min_bloom + 1)
        .collect();

    if bottom_bloom.len() <= 5 {
        bottom_bloom
    } else {
        bottom_bloom.into_iter().take(5).collect()
    }
}

/// Check if two nodes have overlapping topics/tags.
fn nodes_topic_overlap(a: &CurriculumNode, b: &CurriculumNode) -> bool {
    // Direct tag overlap
    for tag_a in &a.tags {
        for tag_b in &b.tags {
            if tag_a == tag_b {
                return true;
            }
        }
    }

    // Subdomain match
    if !a.subdomain.is_empty() && !b.subdomain.is_empty() {
        let sub_a = normalize_for_matching(&a.subdomain);
        let sub_b = normalize_for_matching(&b.subdomain);
        if sub_a.contains(&sub_b) || sub_b.contains(&sub_a) {
            return true;
        }
    }

    // Description keyword overlap (simplified)
    let keywords_a = extract_keywords(&a.description);
    let keywords_b = extract_keywords(&b.description);
    let overlap = keywords_a
        .iter()
        .filter(|k| keywords_b.contains(k))
        .count();

    overlap >= 2
}

/// Extract significant keywords from a description.
fn extract_keywords(desc: &str) -> Vec<String> {
    let stopwords = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "this", "that", "these", "those", "it", "its", "not", "no", "as", "if",
        "such", "than", "each", "all", "any", "both", "few", "more", "most", "other",
        "some", "into", "through", "during", "before", "after", "above", "below", "between",
        "about", "their", "them", "they", "we", "our", "us", "you", "your", "he", "she",
        "using", "based", "how", "what", "when", "where", "which", "who", "use",
    ];

    desc.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !stopwords.contains(w))
        .map(|w| w.to_string())
        .collect()
}

fn bloom_ordinal(bloom: &str) -> u8 {
    match bloom {
        "Remember" => 0,
        "Understand" => 1,
        "Apply" => 2,
        "Analyze" => 3,
        "Evaluate" => 4,
        "Create" => 5,
        _ => 1,
    }
}

fn normalize_for_matching(s: &str) -> String {
    s.to_lowercase()
        .replace(['-', '_', '&'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::CurriculumMetadata;

    fn make_node(id: &str, bloom: &str, tags: &[&str], subdomain: &str) -> CurriculumNode {
        CurriculumNode {
            id: id.into(),
            title: id.into(),
            description: format!("Description for {id}"),
            node_type: "Concept".into(),
            difficulty: "Intermediate".into(),
            domain: "Test".into(),
            subdomain: subdomain.into(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            estimated_hours: 10,
            grade_levels: vec![],
            bloom_level: bloom.into(),
            subject_area: "Mathematics".into(),
            academic_standards: vec![],
            credit_hours: None,
            course_level: None,
            cip_code: None,
            program_id: None,
            corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        }
    }

    fn make_doc(grade_level: &str, subject: &str, nodes: Vec<CurriculumNode>) -> CurriculumDocument {
        CurriculumDocument {
            metadata: CurriculumMetadata {
                title: format!("{subject} ({grade_level})"),
                framework: "Test".into(),
                source: String::new(),
                grade_level: grade_level.into(),
                subject_area: subject.into(),
                domain: subject.into(),
                version: "2024".into(),
                total_standards: nodes.len(),
                domains: vec![],
                created_at: "2026-01-01".into(),
                notes: String::new(),
                academic_level: None,
                institution: None,
                cip_code: None,
                total_credits: None,
                duration_semesters: None,
            },
            nodes,
            edges: vec![],
        }
    }

    #[test]
    fn test_level_detection() {
        let k12 = make_doc("Grade5", "Math", vec![]);
        let undergrad = make_doc("Undergraduate", "Math", vec![]);
        let doctoral = make_doc("Doctoral", "CS", vec![]);

        assert_eq!(Level::from_document(&k12), Level::K12);
        assert_eq!(Level::from_document(&undergrad), Level::Undergraduate);
        assert_eq!(Level::from_document(&doctoral), Level::Doctoral);
    }

    #[test]
    fn test_subjects_match_exact() {
        let a = make_doc("Grade5", "Mathematics", vec![]);
        let b = make_doc("Undergraduate", "Mathematics", vec![]);
        assert!(subjects_match(&a, &b));
    }

    #[test]
    fn test_subjects_match_fuzzy() {
        let a = make_doc("Grade8", "Mathematics", vec![]);
        let b = make_doc("Undergraduate", "CS2013: Algorithms and Complexity", vec![]);
        // Math maps to algorithms via the cross-mapping
        // Actually, "algorithms" is in the CS group, "mathematics" is in the math group
        // The mapping says math group maps to algorithms group
        assert!(subjects_match(&a, &b));
    }

    #[test]
    fn test_subjects_dont_match() {
        let a = make_doc("Grade5", "EnglishLanguageArts", vec![]);
        let b = make_doc("Undergraduate", "Physics", vec![]);
        assert!(!subjects_match(&a, &b));
    }

    #[test]
    fn test_bridge_generates_edges() {
        let k12 = make_doc(
            "Grade8",
            "Mathematics",
            vec![
                make_node("8.EE.1", "Apply", &["algebra", "equations"], "Expressions"),
                make_node("8.EE.2", "Analyze", &["algebra", "functions"], "Expressions"),
            ],
        );

        let undergrad = make_doc(
            "Undergraduate",
            "Mathematics",
            vec![
                make_node("MATH101", "Understand", &["algebra", "calculus"], "Calculus"),
                make_node("MATH201", "Apply", &["linear algebra"], "Linear Algebra"),
            ],
        );

        let bridge = generate_bridge(&[k12, undergrad]);
        assert!(!bridge.edges.is_empty());
        assert!(bridge.statistics.k12_to_undergrad > 0);
        assert!(bridge.edges.iter().all(|e| e.edge_type == "LeadsTo"));
    }

    #[test]
    fn test_bridge_three_levels() {
        let k12 = make_doc(
            "Grade12",
            "Mathematics",
            vec![make_node("12.CALC", "Apply", &["calculus"], "Calculus")],
        );

        let undergrad = make_doc(
            "Undergraduate",
            "Mathematics",
            vec![make_node("MATH301", "Analyze", &["calculus", "analysis"], "Analysis")],
        );

        let doctoral = make_doc(
            "Doctoral",
            "Mathematics",
            vec![make_node("phd-math/coursework", "Analyze", &["mathematics"], "Year 1")],
        );

        let bridge = generate_bridge(&[k12, undergrad, doctoral]);
        assert!(bridge.statistics.k12_to_undergrad > 0);
        // Undergrad → Doctoral (skipping Graduate since none present)
        assert!(bridge.statistics.total_edges >= 2);
        assert!(bridge.statistics.levels_bridged.len() >= 2);
    }

    #[test]
    fn test_empty_documents() {
        let bridge = generate_bridge(&[]);
        assert!(bridge.edges.is_empty());
        assert_eq!(bridge.statistics.total_edges, 0);
    }

    #[test]
    fn test_single_document_no_bridge() {
        let k12 = make_doc("Grade5", "Mathematics", vec![
            make_node("5.NF.1", "Apply", &["fractions"], "Fractions"),
        ]);
        let bridge = generate_bridge(&[k12]);
        assert!(bridge.edges.is_empty());
    }

    #[test]
    fn test_terminal_nodes_selection() {
        let nodes = vec![
            make_node("A", "Remember", &[], ""),
            make_node("B", "Apply", &[], ""),
            make_node("C", "Create", &[], ""),
            make_node("D", "Create", &[], ""),
        ];
        let terminals = find_terminal_nodes(&nodes);
        // Should select nodes at Create level (highest)
        assert!(terminals.iter().any(|n| n.id == "C"));
        assert!(terminals.iter().any(|n| n.id == "D"));
    }

    #[test]
    fn test_entry_nodes_selection() {
        let nodes = vec![
            make_node("A", "Understand", &[], ""),
            make_node("B", "Understand", &[], ""),
            make_node("C", "Analyze", &[], ""),
        ];
        let entries = find_entry_nodes(&nodes);
        assert!(entries.iter().any(|n| n.id == "A"));
        assert!(entries.iter().any(|n| n.id == "B"));
    }

    #[test]
    fn test_topic_overlap() {
        let a = make_node("X", "Apply", &["algebra", "equations"], "Algebra");
        let b = make_node("Y", "Analyze", &["algebra", "proofs"], "Algebra");
        assert!(nodes_topic_overlap(&a, &b));
    }

    #[test]
    fn test_no_topic_overlap() {
        let a = make_node("X", "Apply", &["fractions"], "Fractions");
        let b = make_node("Y", "Apply", &["geometry"], "Geometry");
        assert!(!nodes_topic_overlap(&a, &b));
    }

    #[test]
    fn test_bloom_ordinal() {
        assert!(bloom_ordinal("Create") > bloom_ordinal("Apply"));
        assert!(bloom_ordinal("Apply") > bloom_ordinal("Understand"));
        assert!(bloom_ordinal("Understand") > bloom_ordinal("Remember"));
    }
}
