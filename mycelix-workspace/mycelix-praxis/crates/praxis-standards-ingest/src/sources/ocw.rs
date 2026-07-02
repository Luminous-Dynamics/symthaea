// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MIT OpenCourseWare source via the MIT Learn API.
//!
//! Fetches real university courses from `api.learn.mit.edu/api/v1/` and
//! converts them into EduNet curriculum documents with department mapping,
//! course levels, and topic-based tags.
//!
//! Prerequisites: The Learn API does not include structured prerequisite data.
//! We infer sequential dependencies from course numbering (lower numbers
//! before higher within the same department).

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

const LEARN_API: &str = "https://api.learn.mit.edu/api/v1";

/// MIT Learn API paginated response.
#[derive(Debug, Deserialize)]
pub(crate) struct LearnResponse {
    count: usize,
    results: Vec<OcwCourse>,
}

/// An OCW course from the MIT Learn API.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwCourse {
    id: i64,
    readable_id: Option<String>,
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    departments: Vec<OcwDepartment>,
    #[serde(default)]
    topics: Vec<OcwTopic>,
    #[serde(default)]
    runs: Vec<OcwRun>,
    #[serde(default)]
    course: Option<OcwCourseDetail>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwDepartment {
    department_id: String,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwTopic {
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwRun {
    #[serde(default)]
    level: Vec<OcwLevel>,
    #[serde(default)]
    semester: Option<String>,
    #[serde(default)]
    year: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwLevel {
    #[serde(default)]
    code: String,
    #[serde(default)]
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwCourseDetail {
    #[serde(default)]
    course_numbers: Vec<OcwCourseNumber>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OcwCourseNumber {
    value: String,
    #[serde(default)]
    listing_type: Option<String>,
}

/// MIT Learn API department listing.
#[derive(Debug, Deserialize)]
pub(crate) struct DeptResponse {
    results: Vec<DeptEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DeptEntry {
    department_id: String,
    name: String,
}

/// MIT OCW curriculum source.
pub struct OcwSource {
    http: Client,
}

impl OcwSource {
    pub fn new() -> Result<Self, SourceError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("edunet-standards-ingest/0.1.0")
            .build()?;
        Ok(Self { http })
    }

    /// Fetch a department's courses and convert to a CurriculumDocument.
    ///
    /// This is the primary public API — combines fetch + convert in one call.
    pub async fn fetch_and_convert(
        &self,
        dept_id: &str,
        limit: usize,
    ) -> Result<(CurriculumDocument, usize), SourceError> {
        let courses = self.fetch_department(dept_id, limit).await?;
        let count = courses.len();

        let dept_name = KNOWN_DEPARTMENTS
            .iter()
            .find(|(id, _)| *id == dept_id)
            .map(|(_, name)| name.to_string())
            .unwrap_or_else(|| format!("Department {dept_id}"));

        let doc = self.courses_to_document(&dept_name, dept_id, &courses);
        Ok((doc, count))
    }

    /// List all MIT departments.
    async fn list_departments(&self) -> Result<Vec<DeptEntry>, SourceError> {
        let url = format!("{LEARN_API}/departments/?limit=50");
        let resp: DeptResponse = self.http.get(&url).send().await?.json().await?;
        Ok(resp.results)
    }

    /// Fetch OCW courses for a department.
    async fn fetch_department(
        &self,
        dept_id: &str,
        limit: usize,
    ) -> Result<Vec<OcwCourse>, SourceError> {
        let url = format!(
            "{LEARN_API}/courses/?platform=ocw&department={dept_id}&limit={limit}&offset=0"
        );
        let resp: LearnResponse = self.http.get(&url).send().await?.json().await?;
        Ok(resp.results)
    }

    /// Convert fetched OCW courses into a CurriculumDocument.
    fn courses_to_document(
        &self,
        dept_name: &str,
        dept_id: &str,
        courses: &[OcwCourse],
    ) -> CurriculumDocument {
        let cip_code = dept_to_cip(dept_id);
        let subject_area = dept_to_subject(dept_name);

        let mut nodes: Vec<CurriculumNode> = courses
            .iter()
            .map(|c| course_to_node(c, &subject_area, cip_code))
            .collect();

        // Sort by course number for consistent ordering
        nodes.sort_by(|a, b| a.id.cmp(&b.id));

        // Infer prerequisite edges from course numbering
        let edges = infer_prereqs_from_numbering(&nodes, dept_id);

        let domains: Vec<String> = courses
            .iter()
            .flat_map(|c| c.topics.iter().map(|t| t.name.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(10)
            .collect();

        let metadata = CurriculumMetadata {
            title: format!("MIT OCW: {dept_name}"),
            framework: "MIT OpenCourseWare".to_string(),
            source: format!("https://ocw.mit.edu/search/?d={dept_id}&s=department_course_numbers.sort_coursenum"),
            grade_level: "Undergraduate".to_string(),
            subject_area: subject_area.clone(),
            domain: subject_area,
            version: "2024".to_string(),
            total_standards: nodes.len(),
            domains,
            created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
            notes: format!(
                "Auto-ingested from MIT Learn API (department {dept_id}). \
                 Prerequisites inferred from course numbering."
            ),
            academic_level: Some("Undergraduate".to_string()),
            institution: Some("Massachusetts Institute of Technology".to_string()),
            cip_code: cip_code.map(|c| c.to_string()),
            total_credits: None,
            duration_semesters: None,
        };

        CurriculumDocument {
            metadata,
            nodes,
            edges,
        }
    }
}

impl CurriculumSource for OcwSource {
    fn name(&self) -> &str {
        "MIT OpenCourseWare"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        // Synchronous — return embedded department list
        Ok(KNOWN_DEPARTMENTS
            .iter()
            .map(|(id, name)| SourceEntry {
                id: id.to_string(),
                title: format!("MIT Department {id}: {name}"),
                subject: name.to_string(),
                level: "Undergraduate".to_string(),
                description: "MIT OpenCourseWare courses".to_string(),
            })
            .collect())
    }

    fn fetch(&self, _id: &str) -> Result<CurriculumDocument, SourceError> {
        Err(SourceError::Other(
            "Use fetch_department() for async OCW queries".to_string(),
        ))
    }
}

/// Convert an OCW course to a CurriculumNode.
fn course_to_node(course: &OcwCourse, subject_area: &str, cip_code: Option<&str>) -> CurriculumNode {
    let course_num = extract_course_number(course);
    let level_str = extract_level(course);
    let bloom = infer_bloom_from_level(&level_str);
    let difficulty = infer_difficulty_from_level(&level_str);
    let topics: Vec<String> = course.topics.iter().map(|t| t.name.to_lowercase()).collect();

    // Estimate hours: 3 credits * 45 hours = 135 for a typical MIT course
    let estimated_hours = if level_str.contains("Graduate") { 180 } else { 135 };

    let grade_levels = if level_str.contains("Graduate") {
        vec!["Graduate".to_string()]
    } else {
        vec!["Undergraduate".to_string()]
    };

    CurriculumNode {
        id: course_num.clone(),
        title: course.title.clone(),
        description: truncate_desc(&course.description, 500),
        node_type: "Course".to_string(),
        difficulty,
        domain: subject_area.to_string(),
        subdomain: course
            .topics
            .first()
            .map(|t| t.name.clone())
            .unwrap_or_default(),
        tags: topics,
        estimated_hours,
        grade_levels,
        bloom_level: bloom,
        subject_area: subject_area.to_string(),
        academic_standards: vec![AcademicStandardRef {
            framework: "MIT OCW".to_string(),
            code: course_num,
            description: truncate_desc(&course.title, 200),
            grade_level: level_str,
        }],
        credit_hours: Some(if course.title.to_lowercase().contains("seminar") { 3 } else { 12 }),
        course_level: Some(infer_course_level_from_number(course)),
        cip_code: cip_code.map(|c| c.to_string()),
        program_id: None,
        corequisites: vec![], supplementary_resources: vec![],
        exam_weight: None,
    }
}

/// Extract the primary course number (e.g., "6.006").
fn extract_course_number(course: &OcwCourse) -> String {
    // Try course_numbers first
    if let Some(ref detail) = course.course {
        if let Some(primary) = detail
            .course_numbers
            .iter()
            .find(|cn| cn.listing_type.as_deref() == Some("primary"))
            .or_else(|| detail.course_numbers.first())
        {
            return primary.value.clone();
        }
    }

    // Fall back to readable_id
    if let Some(ref rid) = course.readable_id {
        // readable_id format: "6.006+spring_2020" → "6.006"
        if let Some(base) = rid.split('+').next() {
            return base.replace('_', ".").to_string();
        }
    }

    format!("OCW-{}", course.id)
}

/// Extract course level from runs data.
fn extract_level(course: &OcwCourse) -> String {
    for run in &course.runs {
        for level in &run.level {
            if !level.name.is_empty() {
                return level.name.clone();
            }
        }
    }
    "Undergraduate".to_string()
}

/// Infer Bloom level from course level.
fn infer_bloom_from_level(level: &str) -> String {
    if level.contains("Graduate") {
        "Analyze".to_string()
    } else {
        "Apply".to_string()
    }
}

fn infer_difficulty_from_level(level: &str) -> String {
    if level.contains("Graduate") {
        "Expert".to_string()
    } else {
        "Advanced".to_string()
    }
}

/// Infer course level range from the course number.
fn infer_course_level_from_number(course: &OcwCourse) -> String {
    let num = extract_course_number(course);
    // MIT course numbers: X.YYY where YYY determines level
    // < 100: intro, 100-400: undergrad, 500+: grad
    if let Some(dot_pos) = num.find('.') {
        let after_dot = &num[dot_pos + 1..];
        if let Ok(n) = after_dot.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<u32>() {
            return match n {
                0..=99 => "100",
                100..=299 => "200-300",
                300..=499 => "300-400",
                500..=699 => "500-600",
                _ => "700+",
            }.to_string();
        }
    }
    "200-300".to_string()
}

/// Infer prerequisite edges from course numbering within a department.
fn infer_prereqs_from_numbering(nodes: &[CurriculumNode], dept_id: &str) -> Vec<CurriculumEdge> {
    let mut edges = Vec::new();

    // Parse course numbers and group by department prefix
    let mut numbered: Vec<(&CurriculumNode, u32)> = nodes
        .iter()
        .filter_map(|n| {
            let num = parse_mit_number(&n.id, dept_id)?;
            Some((n, num))
        })
        .collect();

    numbered.sort_by_key(|(_, num)| *num);

    // Connect intro courses (< 100) to intermediate (100-299)
    let intros: Vec<_> = numbered.iter().filter(|(_, n)| *n < 100).collect();
    let intermediates: Vec<_> = numbered.iter().filter(|(_, n)| *n >= 100 && *n < 300).collect();
    let advanced: Vec<_> = numbered.iter().filter(|(_, n)| *n >= 300 && *n < 500).collect();
    let graduate: Vec<_> = numbered.iter().filter(|(_, n)| *n >= 500).collect();

    // Intro → first few intermediate courses
    for (intro_node, _) in &intros {
        for (inter_node, _) in intermediates.iter().take(3) {
            edges.push(CurriculumEdge {
                from: intro_node.id.clone(),
                to: inter_node.id.clone(),
                edge_type: "Requires".to_string(),
                strength_permille: 700,
                rationale: format!(
                    "Introductory course {} typically precedes {}.",
                    intro_node.id, inter_node.id
                ),
            });
        }
    }

    // Some intermediate → advanced connections
    if let Some((last_inter, _)) = intermediates.last() {
        for (adv_node, _) in advanced.iter().take(3) {
            edges.push(CurriculumEdge {
                from: last_inter.id.clone(),
                to: adv_node.id.clone(),
                edge_type: "Requires".to_string(),
                strength_permille: 700,
                rationale: format!(
                    "Intermediate course {} typically precedes {}.",
                    last_inter.id, adv_node.id
                ),
            });
        }
    }

    // Advanced → graduate connections
    if let Some((last_adv, _)) = advanced.last() {
        for (grad_node, _) in graduate.iter().take(2) {
            edges.push(CurriculumEdge {
                from: last_adv.id.clone(),
                to: grad_node.id.clone(),
                edge_type: "Requires".to_string(),
                strength_permille: 600,
                rationale: format!(
                    "Advanced course {} typically precedes graduate course {}.",
                    last_adv.id, grad_node.id
                ),
            });
        }
    }

    edges
}

/// Parse an MIT course number like "6.006" to its numeric part (6).
fn parse_mit_number(id: &str, _dept_id: &str) -> Option<u32> {
    if let Some(dot_pos) = id.find('.') {
        let after = &id[dot_pos + 1..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    } else {
        None
    }
}

fn truncate_desc(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let target = max.saturating_sub(3);
        let truncate_at = s
            .char_indices()
            .take_while(|(i, _)| *i <= target)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(target);
        format!("{}...", &s[..truncate_at])
    }
}

/// Map MIT department IDs to CIP codes.
fn dept_to_cip(dept_id: &str) -> Option<&'static str> {
    match dept_id {
        "1" => Some("14.08"),   // Civil Engineering
        "2" => Some("14.19"),   // Mechanical Engineering
        "3" => Some("14.20"),   // Materials Science
        "5" => Some("40.05"),   // Chemistry
        "6" => Some("11.0701"), // EECS → Computer Science
        "7" => Some("26.01"),   // Biology
        "8" => Some("40.08"),   // Physics
        "9" => Some("26.15"),   // Brain & Cognitive Sciences
        "10" => Some("14.07"),  // Chemical Engineering
        "11" => Some("04"),     // Urban Studies (Architecture)
        "12" => Some("40.06"),  // Earth/Atmospheric/Planetary Sciences
        "14" => Some("45.06"),  // Economics
        "15" => Some("52"),     // Management/Business
        "16" => Some("14.02"),  // Aero/Astro Engineering
        "17" => Some("45.10"),  // Political Science
        "18" => Some("27.01"),  // Mathematics
        "20" => Some("14.05"),  // Biological Engineering
        "21L" => Some("23"),    // Literature
        "22" => Some("14.23"),  // Nuclear Engineering
        "24" => Some("38"),     // Philosophy (Linguistics)
        _ => None,
    }
}

/// Map department name to subject area.
fn dept_to_subject(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("computer") || lower.contains("electrical") {
        "Computer Science".to_string()
    } else if lower.contains("math") {
        "Mathematics".to_string()
    } else if lower.contains("physics") {
        "Physics".to_string()
    } else if lower.contains("biology") || lower.contains("biological") {
        "Biological Sciences".to_string()
    } else if lower.contains("chemistry") || lower.contains("chemical") {
        "Chemistry".to_string()
    } else if lower.contains("economics") {
        "Economics".to_string()
    } else {
        name.to_string()
    }
}

/// Known MIT departments for offline listing.
const KNOWN_DEPARTMENTS: &[(&str, &str)] = &[
    ("1", "Civil and Environmental Engineering"),
    ("2", "Mechanical Engineering"),
    ("3", "Materials Science and Engineering"),
    ("5", "Chemistry"),
    ("6", "Electrical Engineering and Computer Science"),
    ("7", "Biology"),
    ("8", "Physics"),
    ("9", "Brain and Cognitive Sciences"),
    ("10", "Chemical Engineering"),
    ("11", "Urban Studies and Planning"),
    ("12", "Earth, Atmospheric, and Planetary Sciences"),
    ("14", "Economics"),
    ("15", "Management (Sloan)"),
    ("16", "Aeronautics and Astronautics"),
    ("17", "Political Science"),
    ("18", "Mathematics"),
    ("20", "Biological Engineering"),
    ("22", "Nuclear Science and Engineering"),
    ("24", "Linguistics and Philosophy"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocw_source_list_available() {
        let source = OcwSource::new().unwrap();
        let entries = source.list_available().unwrap();
        assert!(entries.len() >= 19);
        assert!(entries.iter().any(|e| e.id == "6"));
        assert!(entries.iter().any(|e| e.id == "18"));
    }

    #[test]
    fn test_dept_to_cip() {
        assert_eq!(dept_to_cip("6"), Some("11.0701"));
        assert_eq!(dept_to_cip("18"), Some("27.01"));
        assert_eq!(dept_to_cip("8"), Some("40.08"));
        assert_eq!(dept_to_cip("99"), None);
    }

    #[test]
    fn test_extract_course_number() {
        let course = OcwCourse {
            id: 1,
            readable_id: Some("6.006+spring_2020".into()),
            title: "Intro to Algorithms".into(),
            description: String::new(),
            url: String::new(),
            departments: vec![],
            topics: vec![],
            runs: vec![],
            course: Some(OcwCourseDetail {
                course_numbers: vec![OcwCourseNumber {
                    value: "6.006".into(),
                    listing_type: Some("primary".into()),
                }],
            }),
        };
        assert_eq!(extract_course_number(&course), "6.006");
    }

    #[test]
    fn test_course_level_from_number() {
        let make = |num: &str| OcwCourse {
            id: 1,
            readable_id: Some(format!("{num}+s2020")),
            title: "Test".into(),
            description: String::new(),
            url: String::new(),
            departments: vec![],
            topics: vec![],
            runs: vec![],
            course: Some(OcwCourseDetail {
                course_numbers: vec![OcwCourseNumber {
                    value: num.into(),
                    listing_type: Some("primary".into()),
                }],
            }),
        };

        assert_eq!(infer_course_level_from_number(&make("6.01")), "100");
        assert_eq!(infer_course_level_from_number(&make("6.006")), "100");
        assert_eq!(infer_course_level_from_number(&make("6.042")), "100");
        assert_eq!(infer_course_level_from_number(&make("6.170")), "200-300");
        assert_eq!(infer_course_level_from_number(&make("6.824")), "700+");
    }

    #[test]
    fn test_parse_mit_number() {
        assert_eq!(parse_mit_number("6.006", "6"), Some(6));
        assert_eq!(parse_mit_number("6.042", "6"), Some(42));
        assert_eq!(parse_mit_number("18.01", "18"), Some(1));
        assert_eq!(parse_mit_number("18.600", "18"), Some(600));
        assert_eq!(parse_mit_number("nope", "6"), None);
    }

    #[test]
    fn test_courses_to_document() {
        let source = OcwSource::new().unwrap();
        let courses = vec![
            OcwCourse {
                id: 1,
                readable_id: Some("6.001+fall_2020".into()),
                title: "Structure and Interpretation of Computer Programs".into(),
                description: "Intro to programming".into(),
                url: "https://ocw.mit.edu/courses/6-001".into(),
                departments: vec![OcwDepartment { department_id: "6".into(), name: "EECS".into() }],
                topics: vec![OcwTopic { name: "Programming".into() }],
                runs: vec![OcwRun { level: vec![OcwLevel { code: "undergraduate".into(), name: "Undergraduate".into() }], semester: Some("Fall".into()), year: Some(2020) }],
                course: Some(OcwCourseDetail { course_numbers: vec![OcwCourseNumber { value: "6.001".into(), listing_type: Some("primary".into()) }] }),
            },
            OcwCourse {
                id: 2,
                readable_id: Some("6.006+spring_2020".into()),
                title: "Introduction to Algorithms".into(),
                description: "Algorithms and data structures".into(),
                url: "https://ocw.mit.edu/courses/6-006".into(),
                departments: vec![OcwDepartment { department_id: "6".into(), name: "EECS".into() }],
                topics: vec![OcwTopic { name: "Algorithms".into() }],
                runs: vec![OcwRun { level: vec![OcwLevel { code: "undergraduate".into(), name: "Undergraduate".into() }], semester: Some("Spring".into()), year: Some(2020) }],
                course: Some(OcwCourseDetail { course_numbers: vec![OcwCourseNumber { value: "6.006".into(), listing_type: Some("primary".into()) }] }),
            },
        ];

        let doc = source.courses_to_document("Electrical Engineering and Computer Science", "6", &courses);
        assert_eq!(doc.nodes.len(), 2);
        assert_eq!(doc.metadata.institution.as_deref(), Some("Massachusetts Institute of Technology"));
        assert_eq!(doc.metadata.cip_code.as_deref(), Some("11.0701"));
        assert!(doc.nodes[0].credit_hours.is_some());
    }

    #[test]
    fn test_infer_prereqs_from_numbering() {
        let source = OcwSource::new().unwrap();
        let courses = vec![
            make_test_course("6.001", "Intro"),
            make_test_course("6.006", "Algorithms"),
            make_test_course("6.046", "Design of Algorithms"),
            make_test_course("6.172", "Performance Engineering"),
            make_test_course("6.824", "Distributed Systems"),
        ];
        let doc = source.courses_to_document("EECS", "6", &courses);
        // Should have edges: intro → intermediate, intermediate → advanced, etc.
        assert!(!doc.edges.is_empty());
    }

    fn make_test_course(num: &str, title: &str) -> OcwCourse {
        OcwCourse {
            id: 0,
            readable_id: Some(format!("{num}+s2020")),
            title: title.into(),
            description: String::new(),
            url: String::new(),
            departments: vec![],
            topics: vec![],
            runs: vec![OcwRun { level: vec![OcwLevel { code: "undergraduate".into(), name: "Undergraduate".into() }], semester: None, year: None }],
            course: Some(OcwCourseDetail {
                course_numbers: vec![OcwCourseNumber {
                    value: num.into(),
                    listing_type: Some("primary".into()),
                }],
            }),
        }
    }
}
