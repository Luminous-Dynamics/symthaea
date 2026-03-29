// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Converts Common Standards Project API data into EduNet curriculum JSON format.
//!
//! Output matches the schema in `examples/curriculum/grade3_math.json`.

use crate::api_types::{Standard, StandardSet};
use crate::higher_ed_types::{PhDMilestone, ProgramDescriptor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete curriculum document matching the edunet examples format.
#[derive(Debug, Serialize, Deserialize)]
pub struct CurriculumDocument {
    pub metadata: CurriculumMetadata,
    pub nodes: Vec<CurriculumNode>,
    pub edges: Vec<CurriculumEdge>,
}

/// Curriculum metadata block.
#[derive(Debug, Serialize, Deserialize)]
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
    // ---- Higher-ed extensions (optional, omitted from K-12 output) ----
    /// Academic level: "Undergraduate", "Graduate", "Doctoral", etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub academic_level: Option<String>,
    /// Institution name (None for framework-level curricula).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub institution: Option<String>,
    /// CIP code for this program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cip_code: Option<String>,
    /// Total credit hours for degree completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_credits: Option<u16>,
    /// Duration in semesters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_semesters: Option<u8>,
}

/// A curriculum node (maps to KnowledgeNode in knowledge_zome).
#[derive(Debug, Serialize, Deserialize)]
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
    // ---- Higher-ed extensions (optional, omitted from K-12 output) ----
    /// Credit hours (US semester credits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credit_hours: Option<u8>,
    /// Course level ("100", "200-300", "500-600", etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub course_level: Option<String>,
    /// CIP code classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cip_code: Option<String>,
    /// Parent program ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program_id: Option<String>,
    /// Corequisite node IDs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corequisites: Vec<String>,
    /// Supplementary external resources (for when generated content is insufficient).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supplementary_resources: Vec<SupplementaryResource>,
    /// Exam weight (marks allocated in formal assessments).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exam_weight: Option<ExamWeight>,
}

/// Exam mark allocation for a curriculum node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamWeight {
    /// Paper number (1 or 2 for NSC Matric).
    pub paper: u8,
    /// Marks allocated to this topic.
    pub marks: u16,
    /// Total marks for the paper.
    pub total_paper_marks: u16,
    /// Percentage weight (marks/total * 100).
    pub percentage: f32,
}

/// An external resource that supplements the generated content for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplementaryResource {
    /// Human-readable title.
    pub title: String,
    /// URL to the resource.
    pub url: String,
    /// Source platform.
    pub source: ResourceSource,
    /// Content type.
    pub content_type: ResourceType,
    /// How relevant this resource is to the node (0-100).
    pub relevance_score: u8,
    /// Standard code alignment (if the resource explicitly covers this standard).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aligned_standard: Option<String>,
}

/// Platform/source of a supplementary resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceSource {
    KhanAcademy,
    OpenStax,
    MitOcw,
    Wikipedia,
    YouTube,
    Brilliant,
    Coursera,
    EdX,
    Custom(String),
}

/// Content type of a supplementary resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Video,
    Textbook,
    Interactive,
    Article,
    Exercise,
    Course,
}

/// Academic standard reference within a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicStandardRef {
    pub framework: String,
    pub code: String,
    pub description: String,
    pub grade_level: String,
}

/// A prerequisite edge between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        academic_level: None,
        institution: None,
        cip_code: None,
        total_credits: None,
        duration_semesters: None,
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
        credit_hours: None,
        course_level: None,
        cip_code: None,
        program_id: None,
        corequisites: vec![], supplementary_resources: vec![],
        exam_weight: None,
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

// ============================================================
// Higher Education Converters
// ============================================================

/// Convert a higher-ed program descriptor into a CurriculumDocument.
pub fn convert_program(program: &ProgramDescriptor) -> CurriculumDocument {
    let grade_level = program.level.to_grade_level().to_string();
    let subject_area = program
        .subject_area
        .clone()
        .unwrap_or_else(|| program.title.clone());
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for course in &program.courses {
        let bloom = if course.outcomes.is_empty() {
            infer_bloom_level(&course.id, &course.description)
        } else {
            // Use the highest Bloom level among outcomes
            highest_bloom(&course.outcomes)
        };

        let difficulty = course.level.to_difficulty().to_string();

        let node = CurriculumNode {
            id: course.id.clone(),
            title: course.title.clone(),
            description: course.description.clone(),
            node_type: "Course".to_string(),
            difficulty,
            domain: subject_area.clone(),
            subdomain: course
                .topics
                .first()
                .cloned()
                .unwrap_or_default(),
            tags: course.topics.clone(),
            estimated_hours: course.total_hours(),
            grade_levels: vec![grade_level.clone()],
            bloom_level: bloom,
            subject_area: subject_area.clone(),
            academic_standards: vec![],
            credit_hours: course.credits,
            course_level: Some(course.level.number_range().to_string()),
            cip_code: program.cip_code.clone(),
            program_id: Some(program.id.clone()),
            corequisites: course.corequisites.clone(), supplementary_resources: vec![],
            exam_weight: None,
        };
        nodes.push(node);

        // Explicit prerequisite edges
        for prereq_id in &course.prerequisites {
            edges.push(CurriculumEdge {
                from: prereq_id.clone(),
                to: course.id.clone(),
                edge_type: "Requires".to_string(),
                strength_permille: 900,
                rationale: format!(
                    "{} is a prerequisite for {}.",
                    prereq_id, course.title
                ),
            });
        }
    }

    // Collect unique topic domains
    let domains: Vec<String> = program
        .courses
        .iter()
        .flat_map(|c| c.topics.first().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let metadata = CurriculumMetadata {
        title: program.title.clone(),
        framework: program
            .cip_code
            .as_ref()
            .map(|c| format!("CIP:{c}"))
            .unwrap_or_else(|| "Custom".to_string()),
        source: String::new(),
        grade_level,
        subject_area: program.title.clone(),
        domain: program.title.clone(),
        version: "2024".to_string(),
        total_standards: nodes.len(),
        domains,
        created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
        notes: format!(
            "Higher-education program: {}. Level: {}.",
            program.title,
            program.level.to_grade_level()
        ),
        academic_level: Some(program.level.to_grade_level().to_string()),
        institution: program.institution.clone(),
        cip_code: program.cip_code.clone(),
        total_credits: program.total_credits,
        duration_semesters: program.duration_semesters,
    };

    CurriculumDocument {
        metadata,
        nodes,
        edges,
    }
}

/// Convert a list of PhD milestones into a CurriculumDocument.
pub fn convert_phd_template(
    discipline: &str,
    cip_code: Option<&str>,
    milestones: &[PhDMilestone],
) -> CurriculumDocument {
    let nodes: Vec<CurriculumNode> = milestones
        .iter()
        .map(|m| CurriculumNode {
            id: m.id.clone(),
            title: m.title.clone(),
            description: m.description.clone(),
            node_type: m.milestone_type.to_node_type().to_string(),
            difficulty: "Expert".to_string(),
            domain: discipline.to_string(),
            subdomain: format!("Year {}", m.typical_year),
            tags: vec![
                discipline.to_lowercase(),
                "phd".to_string(),
                format!("year-{}", m.typical_year),
            ],
            estimated_hours: m.estimated_hours,
            grade_levels: vec!["Doctoral".to_string()],
            bloom_level: m.milestone_type.to_bloom_level().to_string(),
            subject_area: discipline.to_string(),
            academic_standards: vec![],
            credit_hours: None,
            course_level: Some("900".to_string()),
            cip_code: cip_code.map(|s| s.to_string()),
            program_id: None,
            corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        })
        .collect();

    let edges: Vec<CurriculumEdge> = milestones
        .iter()
        .flat_map(|m| {
            m.prerequisites.iter().map(move |prereq| CurriculumEdge {
                from: prereq.clone(),
                to: m.id.clone(),
                edge_type: "LeadsTo".to_string(),
                strength_permille: 950,
                rationale: format!(
                    "{} must be completed before {}.",
                    prereq, m.title
                ),
            })
        })
        .collect();

    let metadata = CurriculumMetadata {
        title: format!("PhD in {discipline}"),
        framework: cip_code
            .map(|c| format!("CIP:{c}"))
            .unwrap_or_else(|| "Custom".to_string()),
        source: String::new(),
        grade_level: "Doctoral".to_string(),
        subject_area: discipline.to_string(),
        domain: discipline.to_string(),
        version: "2024".to_string(),
        total_standards: nodes.len(),
        domains: vec![
            "Coursework".to_string(),
            "Examination".to_string(),
            "Research".to_string(),
        ],
        created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
        notes: format!(
            "PhD progression template for {discipline}. Milestones represent canonical \
             doctoral program stages."
        ),
        academic_level: Some("Doctoral".to_string()),
        institution: None,
        cip_code: cip_code.map(|s| s.to_string()),
        total_credits: None,
        duration_semesters: Some(10), // ~5 years typical
    };

    CurriculumDocument {
        metadata,
        nodes,
        edges,
    }
}

/// Find the highest Bloom level among a set of outcomes.
fn highest_bloom(outcomes: &[crate::higher_ed_types::OutcomeDescriptor]) -> String {
    let order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
    outcomes
        .iter()
        .filter_map(|o| order.iter().position(|&b| b == o.bloom_level))
        .max()
        .map(|idx| order[idx].to_string())
        .unwrap_or_else(|| "Understand".to_string())
}

// ============================================================
// K-12 Helpers
// ============================================================

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
        // Find a char boundary at or before position 77
        let truncate_at = first_sentence
            .char_indices()
            .take_while(|(i, _)| *i <= 77)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(77);
        format!("{}...", &first_sentence[..truncate_at])
    }
}

/// Infer Bloom's taxonomy level from code and description.
fn infer_bloom_level(code: &str, description: &str) -> String {
    let desc = description.to_lowercase();
    if desc.contains("create") || desc.contains("design") || desc.contains("construct")
        || desc.contains("synthesize") || desc.contains("formulate") || desc.contains("propose")
        || desc.contains("develop a") || desc.contains("produce") || desc.contains("compose")
        || desc.contains("theorize") || desc.contains("invent")
    {
        "Create"
    } else if desc.contains("evaluate") || desc.contains("judge") || desc.contains("justify")
        || desc.contains("critique") || desc.contains("defend") || desc.contains("assess")
        || desc.contains("appraise") || desc.contains("hypothesize")
    {
        "Evaluate"
    } else if desc.contains("analyze") || desc.contains("compare") || desc.contains("distinguish")
        || desc.contains("differentiate") || desc.contains("examine") || desc.contains("investigate")
        || desc.contains("derive") || desc.contains("prove")
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
        || desc.contains("implement")
        || desc.contains("demonstrate")
    {
        "Apply"
    } else if desc.contains("interpret")
        || desc.contains("explain")
        || desc.contains("understand")
        || desc.contains("describe")
        || desc.contains("represent")
        || desc.contains("classify")
        || desc.contains("summarize")
    {
        "Understand"
    } else if code.contains(".7") || desc.contains("fluently") || desc.contains("know")
        || desc.contains("recall") || desc.contains("identify") || desc.contains("define")
    {
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
        // Higher-ed CS/Engineering
        "algorithm",
        "data structure",
        "complexity",
        "compiler",
        "operating system",
        "database",
        "network",
        "security",
        "machine learning",
        "artificial intelligence",
        "optimization",
        "linear algebra",
        "calculus",
        "statistics",
        "differential equation",
        "discrete math",
        "proof",
        "theorem",
        "abstract",
        "modeling",
        "simulation",
        "software engineering",
        "architecture",
        // Higher-ed Sciences
        "biochemistry",
        "molecular",
        "thermodynamics",
        "quantum",
        "organic chemistry",
        "inorganic",
        "electromagnetism",
        "mechanics",
        "relativity",
        "statistical mechanics",
        "neuroscience",
        "pharmacology",
        "pathology",
        "epidemiology",
        // Higher-ed Humanities/Social
        "theory",
        "methodology",
        "qualitative",
        "quantitative",
        "ethnography",
        "historiography",
        "jurisprudence",
        "epistemology",
        "ontology",
        "pedagogy",
        // Research
        "dissertation",
        "thesis",
        "peer review",
        "literature review",
        "methodology",
        "empirical",
        "longitudinal",
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
