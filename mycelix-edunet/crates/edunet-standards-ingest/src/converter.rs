// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Converts Common Standards Project API data into EduNet curriculum JSON format.
//!
//! Output matches the schema in `examples/curriculum/grade3_math.json`.

use crate::api_types::{Standard, StandardSet};
use serde::Serialize;
use std::collections::HashMap;

/// Complete curriculum document matching the edunet examples format.
#[derive(Debug, Serialize)]
pub struct CurriculumDocument {
    pub metadata: CurriculumMetadata,
    pub nodes: Vec<CurriculumNode>,
    pub edges: Vec<CurriculumEdge>,
}

/// Curriculum metadata block.
#[derive(Debug, Serialize)]
pub struct CurriculumMetadata {
    pub title: String,
    pub framework: String,
    pub source: String,
    pub grade_level: String,
    pub subject_area: String,
    pub domain: String,
    pub version: String,
    pub total_standards: usize,
    pub domains: Vec<String>,
    pub created_at: String,
    pub notes: String,
}

/// A curriculum node (maps to KnowledgeNode in knowledge_zome).
#[derive(Debug, Serialize)]
pub struct CurriculumNode {
    pub id: String,
    pub title: String,
    pub description: String,
    pub node_type: String,
    pub difficulty: String,
    pub domain: String,
    pub subdomain: String,
    pub tags: Vec<String>,
    pub estimated_hours: u32,
    pub grade_levels: Vec<String>,
    pub bloom_level: String,
    pub subject_area: String,
    pub academic_standards: Vec<AcademicStandardRef>,
}

/// Academic standard reference within a node.
#[derive(Debug, Serialize)]
pub struct AcademicStandardRef {
    pub framework: String,
    pub code: String,
    pub description: String,
    pub grade_level: String,
}

/// A prerequisite edge between nodes.
#[derive(Debug, Serialize)]
pub struct CurriculumEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub strength_permille: u16,
    pub rationale: String,
}

/// Convert a CSP StandardSet into an EduNet CurriculumDocument.
pub fn convert_standard_set(set: &StandardSet) -> CurriculumDocument {
    let grade_level = infer_grade_level(&set.education_levels);
    let subject_area = normalize_subject(&set.subject);
    let framework = infer_framework(set);

    // Build domain lookup: standard_id -> domain name (from depth-0 ancestors)
    let domain_map = build_domain_map(&set.standards);

    // Only convert leaf standards (depth >= 2), not domain/cluster headings
    let leaf_standards: Vec<&Standard> = set
        .standards
        .iter()
        .filter(|s| s.is_leaf_standard())
        .collect();

    // Collect unique domains (from depth-0 entries)
    let domains: Vec<String> = set
        .standards
        .iter()
        .filter(|s| s.depth == 0)
        .map(|s| s.description.clone())
        .collect();

    let nodes: Vec<CurriculumNode> = leaf_standards
        .iter()
        .map(|s| standard_to_node(s, &grade_level, &subject_area, &framework, &domain_map))
        .collect();

    let edges = infer_edges(&leaf_standards, &domain_map);

    let source_url = set
        .document
        .as_ref()
        .and_then(|d| d.source_url.clone())
        .unwrap_or_default();

    let version = set
        .document
        .as_ref()
        .and_then(|d| d.valid.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let metadata = CurriculumMetadata {
        title: set.title.clone(),
        framework: framework.clone(),
        source: source_url,
        grade_level: grade_level.clone(),
        subject_area: subject_area.clone(),
        domain: subject_area.clone(),
        version,
        total_standards: nodes.len(),
        domains,
        created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
        notes: format!(
            "Auto-ingested from Common Standards Project (set {}). \
             Node structure matches KnowledgeNode in knowledge_zome/integrity/src/lib.rs.",
            set.id
        ),
    };

    CurriculumDocument {
        metadata,
        nodes,
        edges,
    }
}

/// Convert a single CSP standard into a curriculum node.
fn standard_to_node(
    standard: &Standard,
    grade_level: &str,
    subject_area: &str,
    framework: &str,
    domain_map: &HashMap<String, String>,
) -> CurriculumNode {
    let code = standard.code();
    let subdomain = domain_map
        .get(&standard.id)
        .cloned()
        .unwrap_or_else(|| subject_area.to_string());

    let title = make_title(&standard.description);
    let bloom_level = infer_bloom_level(&code, &standard.description);
    let tags = extract_tags(&standard.description);
    let estimated_hours = estimate_hours(&bloom_level);
    let node_type = infer_node_type(&bloom_level);
    let difficulty = infer_difficulty(grade_level);

    CurriculumNode {
        id: code.clone(),
        title,
        description: standard.description.clone(),
        node_type,
        difficulty,
        domain: subject_area.to_string(),
        subdomain,
        tags,
        estimated_hours,
        grade_levels: vec![grade_level.to_string()],
        bloom_level,
        subject_area: subject_area.to_string(),
        academic_standards: vec![AcademicStandardRef {
            framework: framework.to_string(),
            code,
            description: standard.description.clone(),
            grade_level: grade_level.to_string(),
        }],
    }
}

/// Build a map from standard ID to its domain name (depth-0 ancestor).
fn build_domain_map(standards: &[Standard]) -> HashMap<String, String> {
    let id_to_desc: HashMap<&str, &str> = standards
        .iter()
        .map(|s| (s.id.as_str(), s.description.as_str()))
        .collect();

    let mut map = HashMap::new();
    for s in standards {
        // Walk ancestor_ids to find the depth-0 domain
        if let Some(root_id) = s.ancestor_ids.last() {
            if let Some(desc) = id_to_desc.get(root_id.as_str()) {
                map.insert(s.id.clone(), desc.to_string());
            }
        }
    }
    map
}

/// Infer prerequisite edges within a standard set.
///
/// Heuristic: standards in the same subdomain with adjacent position likely
/// have a sequential dependency. Also, lower-Bloom standards feed higher ones.
fn infer_edges(standards: &[&Standard], domain_map: &HashMap<String, String>) -> Vec<CurriculumEdge> {
    let mut edges = Vec::new();

    // Group standards by subdomain
    let mut by_domain: HashMap<String, Vec<&Standard>> = HashMap::new();
    for s in standards {
        let domain = domain_map
            .get(&s.id)
            .cloned()
            .unwrap_or_default();
        by_domain.entry(domain).or_default().push(s);
    }

    for (_domain, mut group) in by_domain {
        // Sort by position within the domain
        group.sort_by_key(|s| s.position);

        // Sequential edges within domain
        for pair in group.windows(2) {
            let from_code = pair[0].code();
            let to_code = pair[1].code();
            edges.push(CurriculumEdge {
                from: from_code.clone(),
                to: to_code.clone(),
                edge_type: "Requires".to_string(),
                strength_permille: 800,
                rationale: format!(
                    "{from_code} is a prerequisite for {to_code} within the same domain."
                ),
            });
        }
    }

    edges
}

/// Map CSP education level codes to GradeLevel enum names.
fn infer_grade_level(levels: &[String]) -> String {
    if levels.is_empty() {
        return "Adult".to_string();
    }
    // Use the first level as representative
    match levels[0].as_str() {
        "Pre-K" => "PreK",
        "K" => "Kindergarten",
        "01" => "Grade1",
        "02" => "Grade2",
        "03" => "Grade3",
        "04" => "Grade4",
        "05" => "Grade5",
        "06" => "Grade6",
        "07" => "Grade7",
        "08" => "Grade8",
        "09" => "Grade9",
        "10" => "Grade10",
        "11" => "Grade11",
        "12" => "Grade12",
        "HigherEducation" | "Undergraduate-UpperDivision" | "Undergraduate-LowerDivision" => {
            "College"
        }
        _ => "Adult",
    }
    .to_string()
}

/// Normalize subject string.
fn normalize_subject(subject: &str) -> String {
    let lower = subject.to_lowercase();
    if lower.contains("math") {
        "Mathematics".to_string()
    } else if lower.contains("english") || lower.contains("ela") || lower.contains("language arts")
    {
        "EnglishLanguageArts".to_string()
    } else if lower.contains("science") {
        "Science".to_string()
    } else if lower.contains("social") || lower.contains("history") {
        "SocialStudies".to_string()
    } else if lower.contains("art") {
        "Arts".to_string()
    } else if lower.contains("tech") || lower.contains("computer") {
        "Technology".to_string()
    } else if lower.contains("physical") || lower.contains("health") {
        "PhysicalEducation".to_string()
    } else if subject.is_empty() {
        "General".to_string()
    } else {
        subject.to_string()
    }
}

/// Infer standards framework from the set metadata.
fn infer_framework(set: &StandardSet) -> String {
    let title_lower = set.title.to_lowercase();
    if title_lower.contains("common core") || title_lower.contains("ccss") {
        "CommonCore".to_string()
    } else if title_lower.contains("ngss") || title_lower.contains("next generation science") {
        "NGSS".to_string()
    } else {
        // Fall back to jurisdiction title if available
        set.jurisdiction
            .as_ref()
            .map(|j| j.title.clone())
            .unwrap_or_else(|| "Custom".to_string())
    }
}

/// Create a concise title from a standard description.
fn make_title(description: &str) -> String {
    // Take the first sentence, capped at 80 chars
    let first_sentence = description
        .split(|c: char| c == '.' || c == ';')
        .next()
        .unwrap_or(description)
        .trim();

    if first_sentence.len() <= 80 {
        first_sentence.to_string()
    } else {
        format!("{}...", &first_sentence[..77])
    }
}

/// Infer Bloom's taxonomy level from code and description.
fn infer_bloom_level(code: &str, description: &str) -> String {
    let desc = description.to_lowercase();
    if desc.contains("create") || desc.contains("design") || desc.contains("construct") {
        "Create"
    } else if desc.contains("evaluate") || desc.contains("judge") || desc.contains("justify") {
        "Evaluate"
    } else if desc.contains("analyze") || desc.contains("compare") || desc.contains("distinguish")
    {
        "Analyze"
    } else if desc.contains("solve")
        || desc.contains("apply")
        || desc.contains("use ")
        || desc.contains("determine")
        || desc.contains("multiply")
        || desc.contains("divide")
        || desc.contains("compute")
        || desc.contains("calculate")
    {
        "Apply"
    } else if desc.contains("interpret")
        || desc.contains("explain")
        || desc.contains("understand")
        || desc.contains("describe")
        || desc.contains("represent")
    {
        "Understand"
    } else if code.contains(".7") || desc.contains("fluently") || desc.contains("know") {
        "Remember"
    } else {
        "Understand"
    }
    .to_string()
}

/// Extract keyword tags from a description.
fn extract_tags(description: &str) -> Vec<String> {
    let keywords = [
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "fractions",
        "decimals",
        "geometry",
        "measurement",
        "data",
        "algebra",
        "equations",
        "patterns",
        "place-value",
        "area",
        "perimeter",
        "volume",
        "angles",
        "symmetry",
        "graphs",
        "probability",
        "ratios",
        "proportions",
        "functions",
        "expressions",
        "integers",
        "coordinates",
        "transformations",
        "congruence",
        "similarity",
        "polynomials",
        "reading",
        "writing",
        "vocabulary",
        "grammar",
        "comprehension",
        "inference",
        "argument",
        "evidence",
        "narrative",
        "informational",
        "research",
        "speaking",
        "listening",
        "phonics",
        "fluency",
        "scientific",
        "experiment",
        "hypothesis",
        "observation",
        "energy",
        "matter",
        "force",
        "motion",
        "ecosystem",
        "evolution",
        "genetics",
        "cells",
        "earth",
        "weather",
        "climate",
    ];

    let desc_lower = description.to_lowercase();
    keywords
        .iter()
        .filter(|kw| desc_lower.contains(*kw))
        .map(|kw| kw.to_string())
        .collect()
}

/// Estimate learning hours based on Bloom's level.
fn estimate_hours(bloom: &str) -> u32 {
    match bloom {
        "Remember" => 4,
        "Understand" => 6,
        "Apply" => 8,
        "Analyze" => 10,
        "Evaluate" => 10,
        "Create" => 12,
        _ => 8,
    }
}

/// Infer node type from Bloom's level.
fn infer_node_type(bloom: &str) -> String {
    match bloom {
        "Remember" | "Understand" => "Concept",
        "Apply" | "Analyze" => "Skill",
        "Evaluate" | "Create" => "Project",
        _ => "Concept",
    }
    .to_string()
}

/// Infer difficulty from grade level.
fn infer_difficulty(grade_level: &str) -> String {
    match grade_level {
        "PreK" | "Kindergarten" | "Grade1" | "Grade2" | "Grade3" => "Beginner",
        "Grade4" | "Grade5" | "Grade6" | "Grade7" | "Grade8" => "Intermediate",
        "Grade9" | "Grade10" | "Grade11" | "Grade12" => "Advanced",
        "College" | "Adult" => "Expert",
        _ => "Intermediate",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_types::{DocumentInfo, JurisdictionRef};

    fn sample_standard(id: &str, depth: u32, notation: &str, desc: &str) -> Standard {
        Standard {
            id: id.to_string(),
            asn_identifier: None,
            position: 0,
            depth,
            statement_notation: Some(notation.to_string()),
            statement_label: if depth < 2 {
                Some(if depth == 0 { "Domain" } else { "Cluster" }.to_string())
            } else {
                Some("Standard".to_string())
            },
            description: desc.to_string(),
            parent_id: None,
            ancestor_ids: vec![],
        }
    }

    fn sample_set() -> StandardSet {
        StandardSet {
            id: "TEST_SET".to_string(),
            title: "Common Core Grade 3 Math".to_string(),
            subject: "Math".to_string(),
            normalized_subject: Some("math".to_string()),
            education_levels: vec!["03".to_string()],
            document: Some(DocumentInfo {
                id: Some("D123".to_string()),
                valid: Some("2010".to_string()),
                title: Some("CCSS Math".to_string()),
                source_url: Some("https://example.com".to_string()),
                publication_status: None,
            }),
            jurisdiction: Some(JurisdictionRef {
                id: "J1".to_string(),
                title: "Common Core".to_string(),
            }),
            license: None,
            standards: vec![
                sample_standard("D1", 0, "", "Operations & Algebraic Thinking"),
                sample_standard("C1", 1, "", "Represent and solve problems"),
                sample_standard("S1", 2, "3.OA.A.1", "Interpret products of whole numbers"),
                sample_standard("S2", 2, "3.OA.A.2", "Interpret whole-number quotients"),
            ],
        }
    }

    #[test]
    fn test_convert_produces_valid_document() {
        let set = sample_set();
        let doc = convert_standard_set(&set);

        assert_eq!(doc.metadata.framework, "CommonCore");
        assert_eq!(doc.metadata.grade_level, "Grade3");
        assert_eq!(doc.metadata.subject_area, "Mathematics");
        assert_eq!(doc.metadata.total_standards, 2); // only leaf standards
        assert_eq!(doc.nodes.len(), 2);
        assert_eq!(doc.nodes[0].id, "3.OA.A.1");
        assert_eq!(doc.nodes[1].id, "3.OA.A.2");
    }

    #[test]
    fn test_grade_level_mapping() {
        assert_eq!(infer_grade_level(&["Pre-K".into()]), "PreK");
        assert_eq!(infer_grade_level(&["K".into()]), "Kindergarten");
        assert_eq!(infer_grade_level(&["03".into()]), "Grade3");
        assert_eq!(infer_grade_level(&["12".into()]), "Grade12");
        assert_eq!(infer_grade_level(&[]), "Adult");
    }

    #[test]
    fn test_normalize_subject() {
        assert_eq!(normalize_subject("Math"), "Mathematics");
        assert_eq!(normalize_subject("English Language Arts"), "EnglishLanguageArts");
        assert_eq!(normalize_subject("Science"), "Science");
        assert_eq!(normalize_subject(""), "General");
    }

    #[test]
    fn test_bloom_inference() {
        assert_eq!(infer_bloom_level("", "Interpret products"), "Understand");
        assert_eq!(infer_bloom_level("", "Solve word problems"), "Apply");
        assert_eq!(infer_bloom_level("", "Analyze the relationship"), "Analyze");
        assert_eq!(infer_bloom_level("", "Create a model"), "Create");
        assert_eq!(infer_bloom_level(".7", "Know from memory"), "Remember");
    }

    #[test]
    fn test_make_title() {
        assert_eq!(
            make_title("Interpret products of whole numbers"),
            "Interpret products of whole numbers"
        );
        let long = "A".repeat(100);
        let title = make_title(&long);
        assert!(title.len() <= 80);
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_extract_tags() {
        let tags = extract_tags("Use multiplication and division to solve problems");
        assert!(tags.contains(&"multiplication".to_string()));
        assert!(tags.contains(&"division".to_string()));
    }

    #[test]
    fn test_edges_inferred_within_domain() {
        let mut set = sample_set();
        // Give both standards the same ancestor so they share a domain
        set.standards[2].ancestor_ids = vec!["D1".to_string()];
        set.standards[2].position = 1;
        set.standards[3].ancestor_ids = vec!["D1".to_string()];
        set.standards[3].position = 2;

        let doc = convert_standard_set(&set);
        assert!(!doc.edges.is_empty());
        assert_eq!(doc.edges[0].from, "3.OA.A.1");
        assert_eq!(doc.edges[0].to, "3.OA.A.2");
    }

    #[test]
    fn test_difficulty_by_grade() {
        assert_eq!(infer_difficulty("Grade1"), "Beginner");
        assert_eq!(infer_difficulty("Grade5"), "Intermediate");
        assert_eq!(infer_difficulty("Grade10"), "Advanced");
        assert_eq!(infer_difficulty("College"), "Expert");
    }

    #[test]
    fn test_node_type_from_bloom() {
        assert_eq!(infer_node_type("Remember"), "Concept");
        assert_eq!(infer_node_type("Apply"), "Skill");
        assert_eq!(infer_node_type("Create"), "Project");
    }

    #[test]
    fn test_standard_is_leaf() {
        let leaf = sample_standard("S1", 2, "3.OA.A.1", "desc");
        assert!(leaf.is_leaf_standard());

        let domain = sample_standard("D1", 0, "", "domain");
        assert!(!domain.is_leaf_standard());
    }
}
