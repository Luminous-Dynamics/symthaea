// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! South African CAPS (Curriculum and Assessment Policy Statement) source.
//!
//! Covers Grade 10-12 (FET Phase) Mathematics and Physical Sciences, curated
//! from the Department of Basic Education's published CAPS documents.
//!
//! - **Mathematics**: Paper 1 (Algebra, Sequences, Finance, Functions, Calculus,
//!   Probability, Counting) + Paper 2 (Geometry, Analytical Geometry,
//!   Trigonometry, Statistics)
//! - **Physical Sciences**: Paper 1 (Mechanics, Waves, Electricity, Modern
//!   Physics) + Paper 2 (Organic Chemistry, Reaction Rates, Equilibrium,
//!   Acids/Bases, Electrochemistry)
//!
//! Embedded source — the CAPS PDFs are publicly available from
//! <https://www.education.gov.za/Curriculum/CurriculumAssessmentPolicyStatements.aspx>
//! but have no API.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
    ExamWeight, ResourceSource, ResourceType, SupplementaryResource,
};

/// South African CAPS (Curriculum and Assessment Policy Statement) source.
///
/// Grade 10-12 FET Phase Mathematics and Physical Sciences.
pub struct CapsSource;

impl CapsSource {
    pub fn new() -> Self {
        Self
    }
}

impl CurriculumSource for CapsSource {
    fn name(&self) -> &str {
        "CAPS (South Africa)"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            // Foundation Phase (Gr1-3)
            SourceEntry { id: "caps-math-1".into(), title: "CAPS Mathematics Grade 1".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Foundation Phase Grade 1 — counting, addition, subtraction to 20".into() },
            SourceEntry { id: "caps-math-2".into(), title: "CAPS Mathematics Grade 2".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Foundation Phase Grade 2 — place value, addition/subtraction to 99".into() },
            SourceEntry { id: "caps-math-3".into(), title: "CAPS Mathematics Grade 3".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Foundation Phase Grade 3 — multiplication, fractions, measurement".into() },
            // Intermediate Phase (Gr4-6)
            SourceEntry { id: "caps-math-4".into(), title: "CAPS Mathematics Grade 4".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Intermediate Phase Grade 4 — whole numbers to 10000, fractions, data".into() },
            SourceEntry { id: "caps-math-5".into(), title: "CAPS Mathematics Grade 5".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Intermediate Phase Grade 5 — decimals, percentages, geometry".into() },
            SourceEntry { id: "caps-math-6".into(), title: "CAPS Mathematics Grade 6".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Intermediate Phase Grade 6 — integers intro, ratio, transformation geometry".into() },
            // Senior Phase (Gr7-8)
            SourceEntry { id: "caps-math-7".into(), title: "CAPS Mathematics Grade 7".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Senior Phase Grade 7 — algebraic expressions intro, geometry constructions".into() },
            SourceEntry { id: "caps-math-8".into(), title: "CAPS Mathematics Grade 8".into(), subject: "Mathematics".into(), level: "K-12".into(), description: "Senior Phase Grade 8 — equations, functions intro, Pythagoras, data handling".into() },
            // Existing Gr9-12
            SourceEntry {
                id: "caps-math-9".into(),
                title: "CAPS Mathematics Grade 9".into(),
                subject: "Mathematics".into(),
                level: "K-12".into(),
                description: "Senior Phase Grade 9 Mathematics — algebra, geometry, data handling fundamentals".into(),
            },
            SourceEntry {
                id: "caps-natsci-9".into(),
                title: "CAPS Natural Sciences Grade 9".into(),
                subject: "Natural Sciences".into(),
                level: "K-12".into(),
                description: "Senior Phase Grade 9 Natural Sciences — physics and chemistry foundations for FET".into(),
            },
            SourceEntry {
                id: "caps-math-10".into(),
                title: "CAPS Mathematics Grade 10".into(),
                subject: "Mathematics".into(),
                level: "K-12".into(),
                description: "FET Phase Grade 10 Mathematics — foundations for Matric".into(),
            },
            SourceEntry {
                id: "caps-math-11".into(),
                title: "CAPS Mathematics Grade 11".into(),
                subject: "Mathematics".into(),
                level: "K-12".into(),
                description: "FET Phase Grade 11 Mathematics — intermediate".into(),
            },
            SourceEntry {
                id: "caps-math-12".into(),
                title: "CAPS Mathematics Grade 12 (Matric)".into(),
                subject: "Mathematics".into(),
                level: "K-12".into(),
                description: "NSC Matric Mathematics — Paper 1 & Paper 2".into(),
            },
            SourceEntry {
                id: "caps-physics-10".into(),
                title: "CAPS Physical Sciences Grade 10".into(),
                subject: "Physical Sciences".into(),
                level: "K-12".into(),
                description: "FET Phase Grade 10 Physical Sciences — foundations".into(),
            },
            SourceEntry {
                id: "caps-physics-11".into(),
                title: "CAPS Physical Sciences Grade 11".into(),
                subject: "Physical Sciences".into(),
                level: "K-12".into(),
                description: "FET Phase Grade 11 Physical Sciences — intermediate".into(),
            },
            SourceEntry {
                id: "caps-physics-12".into(),
                title: "CAPS Physical Sciences Grade 12 (Matric)".into(),
                subject: "Physical Sciences".into(),
                level: "K-12".into(),
                description: "NSC Matric Physical Sciences — Physics (Paper 1) & Chemistry (Paper 2)".into(),
            },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        match id {
            "caps-math-1" => Ok(build_math_grade(1)),
            "caps-math-2" => Ok(build_math_grade(2)),
            "caps-math-3" => Ok(build_math_grade(3)),
            "caps-math-4" => Ok(build_math_grade(4)),
            "caps-math-5" => Ok(build_math_grade(5)),
            "caps-math-6" => Ok(build_math_grade(6)),
            "caps-math-7" => Ok(build_math_grade(7)),
            "caps-math-8" => Ok(build_math_grade(8)),
            "caps-math-9" => Ok(build_math_grade_9()),
            "caps-natsci-9" => Ok(build_natsci_grade_9()),
            "caps-math-10" => Ok(build_math_grade_10()),
            "caps-math-11" => Ok(build_math_grade_11()),
            "caps-math-12" => Ok(build_math_grade_12()),
            "caps-physics-10" => Ok(build_physics_grade_10()),
            "caps-physics-11" => Ok(build_physics_grade_11()),
            "caps-physics-12" => Ok(build_physics_grade_12()),
            _ => Err(SourceError::NotFound(format!(
                "Unknown CAPS source: {id}. Options: caps-math-10, caps-math-11, caps-math-12, \
                 caps-physics-10, caps-physics-11, caps-physics-12"
            ))),
        }
    }
}

// ============================================================
// Helper: build a CAPS node
// ============================================================

struct TopicDef {
    code: &'static str,
    title: &'static str,
    description: &'static str,
    subdomain: &'static str,
    node_type: &'static str,
    bloom: &'static str,
    hours: u32,
    tags: &'static [&'static str],
    /// NSC exam mark allocation: (paper, marks, total_paper_marks).
    /// None for Gr10/11 topics (assessed internally, not via NSC).
    exam_marks: Option<(u8, u16, u16)>,
}

fn build_caps_resources(
    node_id: &str,
    title: &str,
    subject_area: &str,
    tags: &[String],
) -> Vec<SupplementaryResource> {
    let mut resources = Vec::new();
    let search_term = title.replace(' ', "+");
    let subject_lower = subject_area.to_lowercase();

    // Khan Academy — great for Math and Science
    if subject_lower.contains("math") || subject_lower.contains("science") {
        resources.push(SupplementaryResource {
            title: format!("Khan Academy: {}", truncate(title, 50)),
            url: format!("https://www.khanacademy.org/search?referer=%2F&page_search_query={search_term}"),
            source: ResourceSource::KhanAcademy,
            content_type: ResourceType::Video,
            relevance_score: 85,
            aligned_standard: Some(node_id.to_string()),
        });
    }

    // Wikipedia concept reference
    let wiki_term = title
        .split(|c: char| c == ':' || c == '(' || c == ',')
        .next()
        .unwrap_or(title)
        .trim()
        .replace(' ', "_");
    resources.push(SupplementaryResource {
        title: format!("Wikipedia: {}", truncate(title, 50)),
        url: format!("https://en.wikipedia.org/wiki/{wiki_term}"),
        source: ResourceSource::Wikipedia,
        content_type: ResourceType::Article,
        relevance_score: 60,
        aligned_standard: None,
    });

    // YouTube educational videos
    resources.push(SupplementaryResource {
        title: format!("YouTube: {}", truncate(title, 50)),
        url: format!("https://www.youtube.com/results?search_query={search_term}+education"),
        source: ResourceSource::YouTube,
        content_type: ResourceType::Video,
        relevance_score: 70,
        aligned_standard: None,
    });

    // Desmos for Math
    if subject_lower.contains("math") {
        resources.push(SupplementaryResource {
            title: "Desmos Graphing Calculator".to_string(),
            url: "https://www.desmos.com/calculator".to_string(),
            source: ResourceSource::Custom("Desmos".into()),
            content_type: ResourceType::Interactive,
            relevance_score: 75,
            aligned_standard: None,
        });
    }

    // PhET simulations for Physics and Chemistry
    if subject_lower.contains("science") || subject_lower.contains("physics") || subject_lower.contains("chemistry") {
        let phet_query = tags
            .first()
            .cloned()
            .unwrap_or_else(|| title.to_string())
            .replace(' ', "+");
        resources.push(SupplementaryResource {
            title: "PhET Interactive Simulations".to_string(),
            url: format!("https://phet.colorado.edu/en/simulations/filter?q={phet_query}"),
            source: ResourceSource::Custom("PhET".into()),
            content_type: ResourceType::Interactive,
            relevance_score: 80,
            aligned_standard: None,
        });
    }

    resources.sort_by(|a, b| b.relevance_score.cmp(&a.relevance_score));
    resources
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

fn caps_node(
    grade: u8,
    subject_area: &str,
    topic: &TopicDef,
) -> CurriculumNode {
    let node_id = format!("CAPS.{}.Gr{}.{}", subject_area.replace(' ', ""), grade, topic.code);
    let grade_level = format!("Grade{}", grade);
    let tags: Vec<String> = topic
        .tags
        .iter()
        .map(|t| t.to_string())
        .chain(["caps".to_string(), "south-africa".to_string(), "matric".to_string()])
        .collect();

    let mut resources = build_caps_resources(&node_id, topic.title, subject_area, &tags);

    // Add DBE past paper link for Gr12 nodes
    if grade == 12 {
        resources.push(SupplementaryResource {
            title: "DBE Past NSC Exam Papers".to_string(),
            url: "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx".to_string(),
            source: ResourceSource::Custom("DBE".into()),
            content_type: ResourceType::Exercise,
            relevance_score: 90,
            aligned_standard: Some(node_id.clone()),
        });
        resources.sort_by(|a, b| b.relevance_score.cmp(&a.relevance_score));
    }

    let exam_weight = topic.exam_marks.map(|(paper, marks, total)| ExamWeight {
        paper,
        marks,
        total_paper_marks: total,
        percentage: (marks as f32 / total as f32) * 100.0,
    });

    CurriculumNode {
        id: node_id.clone(),
        title: topic.title.to_string(),
        description: topic.description.to_string(),
        node_type: topic.node_type.to_string(),
        difficulty: match grade {
            9 => "Beginner",
            10 => "Intermediate",
            11 => "Advanced",
            _ => "Advanced",
        }
        .into(),
        domain: subject_area.to_string(),
        subdomain: topic.subdomain.to_string(),
        tags,
        estimated_hours: topic.hours,
        grade_levels: vec![grade_level.clone()],
        bloom_level: topic.bloom.to_string(),
        subject_area: subject_area.to_string(),
        academic_standards: vec![AcademicStandardRef {
            framework: "CAPS (South Africa DBE)".into(),
            code: node_id,
            description: topic.description.to_string(),
            grade_level,
        }],
        credit_hours: None,
        course_level: None,
        cip_code: None,
        program_id: None,
        corequisites: vec![],
        supplementary_resources: resources,
        exam_weight,
    }
}

fn caps_edge(from: &str, to: &str, rationale: &str, strength: u16) -> CurriculumEdge {
    CurriculumEdge {
        from: from.to_string(),
        to: to.to_string(),
        edge_type: "Requires".into(),
        strength_permille: strength,
        rationale: rationale.to_string(),
    }
}

fn caps_metadata(
    title: &str,
    grade: u8,
    subject: &str,
    domains: Vec<String>,
    node_count: usize,
    notes: &str,
) -> CurriculumMetadata {
    CurriculumMetadata {
        title: title.into(),
        framework: "CAPS (South Africa DBE)".into(),
        source: "https://www.education.gov.za/Curriculum/CurriculumAssessmentPolicyStatements.aspx"
            .into(),
        grade_level: format!("Grade{}", grade),
        subject_area: subject.into(),
        domain: subject.into(),
        version: "2011 (amended 2022)".into(),
        total_standards: node_count,
        domains,
        created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
        notes: notes.into(),
        academic_level: None,
        institution: None,
        cip_code: None,
        total_credits: None,
        duration_semesters: None,
    }
}

// ============================================================
// Grades 1-8 Mathematics (Foundation, Intermediate, Senior Phase)
// ============================================================

fn build_math_grade(grade: u8) -> CurriculumDocument {
    let topics: Vec<TopicDef> = match grade {
        1 => vec![
            TopicDef { code: "CNT", title: "Counting", description: "Count objects to 100. Count forwards and backwards in 1s, 2s, 5s, 10s. Compare and order numbers to 20", subdomain: "Numbers", node_type: "Skill", bloom: "Remember", hours: 10, tags: &["counting", "numbers", "ordering"], exam_marks: None },
            TopicDef { code: "ADD", title: "Addition and Subtraction", description: "Add and subtract within 20. Number bonds. Use concrete objects, drawings, and number lines", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 12, tags: &["addition", "subtraction", "number-bonds"], exam_marks: None },
            TopicDef { code: "PAT", title: "Patterns", description: "Copy, extend, and create simple patterns using objects, shapes, and numbers", subdomain: "Patterns", node_type: "Concept", bloom: "Understand", hours: 6, tags: &["patterns", "shapes", "repeating"], exam_marks: None },
            TopicDef { code: "SHP", title: "Space and Shape", description: "Recognise and name 2D shapes (circle, square, triangle, rectangle). Describe position (above, below, next to)", subdomain: "Geometry", node_type: "Concept", bloom: "Remember", hours: 6, tags: &["shapes", "2d-shapes", "position"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Compare and order objects by length, mass, and capacity using non-standard units. Time: days of the week, months", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 6, tags: &["measurement", "length", "mass", "time"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling", description: "Collect and sort objects. Draw picture graphs. Answer questions about data", subdomain: "Data", node_type: "Skill", bloom: "Understand", hours: 4, tags: &["data", "picture-graphs", "sorting"], exam_marks: None },
        ],
        2 => vec![
            TopicDef { code: "NUM", title: "Whole Numbers", description: "Place value to 99. Count in 2s, 3s, 5s, 10s to 200. Compare and order numbers. Even and odd numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Understand", hours: 10, tags: &["place-value", "numbers", "even-odd"], exam_marks: None },
            TopicDef { code: "ADD", title: "Addition and Subtraction", description: "Add and subtract within 99. Mental strategies. Word problems. Number sentences", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 14, tags: &["addition", "subtraction", "word-problems"], exam_marks: None },
            TopicDef { code: "MUL", title: "Multiplication", description: "Repeated addition. Multiply by 2, 5, 10. Array model. Sharing equally (division intro)", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["multiplication", "arrays", "repeated-addition"], exam_marks: None },
            TopicDef { code: "FRC", title: "Fractions", description: "Halves and quarters of shapes and groups. Equal sharing", subdomain: "Numbers", node_type: "Concept", bloom: "Understand", hours: 6, tags: &["fractions", "halves", "quarters"], exam_marks: None },
            TopicDef { code: "PAT", title: "Patterns and Sequences", description: "Number patterns with constant difference. Geometric patterns. Describe and extend patterns", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["patterns", "sequences", "constant-difference"], exam_marks: None },
            TopicDef { code: "SHP", title: "Space and Shape", description: "2D and 3D shapes. Symmetry. Views of objects. Directions and positions", subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 6, tags: &["shapes", "3d-shapes", "symmetry"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Length in cm and m. Mass in kg. Capacity in litres. Time: analogue clock to half-hour", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "centimetres", "metres", "clock"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling", description: "Collect data using tally marks. Draw and read pictographs and bar graphs", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 4, tags: &["data", "tally", "bar-graphs"], exam_marks: None },
        ],
        3 => vec![
            TopicDef { code: "NUM", title: "Whole Numbers", description: "Place value to 999. Round to nearest 10 and 100. Compare and order 3-digit numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Understand", hours: 10, tags: &["place-value", "rounding", "three-digit"], exam_marks: None },
            TopicDef { code: "ADD", title: "Addition and Subtraction", description: "Add and subtract within 999. Column method. Estimate. Solve word problems with two operations", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 14, tags: &["addition", "subtraction", "column-method", "estimation"], exam_marks: None },
            TopicDef { code: "MUL", title: "Multiplication and Division", description: "Times tables 2-10. Multiply 2-digit by 1-digit. Division as sharing and grouping. Remainders", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 14, tags: &["multiplication", "division", "times-tables", "remainders"], exam_marks: None },
            TopicDef { code: "FRC", title: "Common Fractions", description: "Fractions of whole: halves, thirds, quarters, fifths, sixths, eighths. Compare fractions. Equivalent fractions intro", subdomain: "Numbers", node_type: "Concept", bloom: "Understand", hours: 8, tags: &["fractions", "equivalent", "comparing-fractions"], exam_marks: None },
            TopicDef { code: "PAT", title: "Number Patterns", description: "Patterns in addition and multiplication tables. Describe rules. Input-output tables", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["patterns", "input-output", "rules"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Properties of 2D shapes. Angles (right, acute, obtuse). Lines of symmetry. Tessellations", subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 8, tags: &["geometry", "angles", "symmetry", "tessellation"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Perimeter of shapes. Area by counting squares. Time: analogue and digital to 5 minutes. Calendars", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["perimeter", "area", "time", "calendar"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling", description: "Collect, organise data. Bar graphs and pictographs. Mode. Interpret data and draw conclusions", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "bar-graphs", "mode", "interpret"], exam_marks: None },
        ],
        4 => vec![
            TopicDef { code: "NUM", title: "Whole Numbers", description: "Place value to 9999. Rounding. Properties of numbers (even, odd, prime, composite). Factors and multiples", subdomain: "Numbers", node_type: "Skill", bloom: "Understand", hours: 10, tags: &["place-value", "prime", "factors", "multiples"], exam_marks: None },
            TopicDef { code: "OPS", title: "Operations", description: "All four operations. Long multiplication. Long division. Order of operations. Word problems", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 14, tags: &["multiplication", "division", "order-of-operations", "long-division"], exam_marks: None },
            TopicDef { code: "FRC", title: "Common Fractions", description: "Equivalent fractions. Simplify fractions. Add and subtract fractions with same denominator. Mixed numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["fractions", "equivalent", "simplify", "mixed-numbers"], exam_marks: None },
            TopicDef { code: "DEC", title: "Decimal Fractions", description: "Tenths and hundredths. Convert between fractions and decimals. Place value with decimals. Add/subtract decimals", subdomain: "Numbers", node_type: "Concept", bloom: "Understand", hours: 8, tags: &["decimals", "tenths", "hundredths", "place-value"], exam_marks: None },
            TopicDef { code: "PAT", title: "Numeric Patterns", description: "Investigate patterns in number sequences. Describe rules for patterns. Complete flow diagrams", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["patterns", "sequences", "flow-diagrams", "rules"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Properties of quadrilaterals. Classify triangles by sides and angles. Parallel and perpendicular lines. Coordinate grid", subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 8, tags: &["geometry", "quadrilaterals", "triangles", "coordinates"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Convert between mm, cm, m, km. Area of rectangle. Volume by counting cubes. Temperature in °C", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "conversion", "area", "volume", "temperature"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling", description: "Collect and organise data. Bar graphs, pie charts intro. Mean of small data sets. Chance events (certain, likely, unlikely, impossible)", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "pie-charts", "mean", "chance"], exam_marks: None },
        ],
        5 => vec![
            TopicDef { code: "NUM", title: "Whole Numbers", description: "Place value to 999999. Properties of numbers. Common factors and common multiples. Divisibility rules", subdomain: "Numbers", node_type: "Skill", bloom: "Understand", hours: 8, tags: &["place-value", "divisibility", "common-factors"], exam_marks: None },
            TopicDef { code: "OPS", title: "Operations", description: "Multiply 3-digit by 2-digit. Division with remainders. Multistep problems. Estimation and checking", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 12, tags: &["multiplication", "division", "estimation", "multistep"], exam_marks: None },
            TopicDef { code: "FRC", title: "Common Fractions", description: "Add and subtract fractions with different denominators. Multiply fractions by whole numbers. Fractions of amounts", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["fractions", "different-denominators", "multiply-fractions"], exam_marks: None },
            TopicDef { code: "DEC", title: "Decimal Fractions", description: "Thousandths. Multiply and divide decimals by 10, 100, 1000. Rounding decimals. Percentages intro (50%, 25%, 10%)", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["decimals", "thousandths", "percentages", "rounding"], exam_marks: None },
            TopicDef { code: "PAT", title: "Patterns and Relationships", description: "Number patterns with two operations. Input-output tables. Flow diagrams. Sequences with constant difference", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["patterns", "input-output", "flow-diagrams"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Properties of 3D objects. Nets of cubes and prisms. Transformations: reflection, translation. Angles on straight line", subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 8, tags: &["geometry", "3d-objects", "nets", "transformations", "angles"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Perimeter and area of irregular shapes. Volume of rectangular prisms. Convert between units. 24-hour time", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "perimeter", "area", "volume", "24-hour-time"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling and Probability", description: "Double bar graphs. Mean, median, mode. Probability as fraction. Experimental vs theoretical probability", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "double-bar-graph", "median", "probability"], exam_marks: None },
        ],
        6 => vec![
            TopicDef { code: "NUM", title: "Whole Numbers and Integers", description: "Numbers to millions. Introduction to integers (positive and negative). Number line with integers. Ordering integers", subdomain: "Numbers", node_type: "Skill", bloom: "Understand", hours: 8, tags: &["integers", "millions", "number-line", "ordering"], exam_marks: None },
            TopicDef { code: "OPS", title: "Operations", description: "All operations with whole numbers. Order of operations (BODMAS). Multiply and divide by multiples of 10, 100", subdomain: "Operations", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["operations", "bodmas", "multiples-of-10"], exam_marks: None },
            TopicDef { code: "FRC", title: "Fractions, Decimals, and Percentages", description: "Convert between fractions, decimals, percentages. Calculate percentage of amount. Ratio and rate intro", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 12, tags: &["fractions", "decimals", "percentages", "ratio", "rate"], exam_marks: None },
            TopicDef { code: "EXP", title: "Exponents", description: "Squares and cubes. Square roots and cube roots. Exponent notation intro (2³ = 8). Calculate with exponents", subdomain: "Numbers", node_type: "Concept", bloom: "Understand", hours: 6, tags: &["exponents", "squares", "cubes", "square-roots"], exam_marks: None },
            TopicDef { code: "PAT", title: "Patterns and Relationships", description: "Describe rules for patterns algebraically. Input-output with two operations. Sequences with variable difference", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["patterns", "algebraic-rules", "input-output"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Angles: measuring with protractor, calculating angles on straight lines and at a point. Transformation geometry: reflection, rotation, translation", subdomain: "Geometry", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["geometry", "angles", "protractor", "rotation", "reflection"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Area of triangles. Volume of triangular prisms. Convert between metric units. Capacity. Temperature conversions", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "area-triangle", "volume-prism", "conversions"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling and Probability", description: "Histograms. Pie charts (reading and interpreting). Range. Probability of simple events as fractions and percentages", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "histograms", "pie-charts", "range", "probability"], exam_marks: None },
        ],
        7 => vec![
            TopicDef { code: "NUM", title: "Integers and Rational Numbers", description: "Operations with integers. Rules for multiplying/dividing positive and negative numbers. Introduction to rational numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["integers", "rational-numbers", "negative-numbers"], exam_marks: None },
            TopicDef { code: "EXP", title: "Exponents", description: "Laws of exponents: product rule, quotient rule, power rule. Calculate with exponents including negative bases. Scientific notation intro", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["exponents", "laws-of-exponents", "scientific-notation"], exam_marks: None },
            TopicDef { code: "ALG", title: "Algebraic Expressions", description: "Variables and constants. Algebraic language. Evaluate expressions by substitution. Simplify by collecting like terms", subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 10, tags: &["algebra", "variables", "substitution", "simplify"], exam_marks: None },
            TopicDef { code: "EQN", title: "Equations", description: "Solve simple linear equations. Write equations from word problems. Number sentences with unknowns", subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["equations", "linear", "word-problems", "unknowns"], exam_marks: None },
            TopicDef { code: "PAT", title: "Patterns and Functions", description: "Input-output relationships. Tables of values. Plot points on Cartesian plane. Linear relationships intro", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 8, tags: &["patterns", "cartesian-plane", "linear", "functions-intro"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Properties of triangles and quadrilaterals. Angle relationships (complementary, supplementary). Congruency intro. Constructions with compass and ruler", subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 10, tags: &["geometry", "constructions", "congruency", "angle-relationships"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Circumference and area of circle (π). Surface area of prisms. Volume of cylinders. Speed, distance, time", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "circle", "circumference", "pi", "speed"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling and Probability", description: "Mean, median, mode of grouped data. Scatter plots intro. Probability of combined events. Tree diagrams intro", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "grouped-data", "scatter-plots", "tree-diagrams"], exam_marks: None },
        ],
        8 => vec![
            TopicDef { code: "NUM", title: "Rational Numbers", description: "All operations with rational numbers (fractions, decimals, integers). Convert between forms. Properties of rational numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["rational-numbers", "operations", "fractions", "decimals"], exam_marks: None },
            TopicDef { code: "EXP", title: "Exponents", description: "Exponents with integer exponents including zero and negative. Simplify exponential expressions. Scientific notation with large and small numbers", subdomain: "Numbers", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["exponents", "negative-exponents", "scientific-notation"], exam_marks: None },
            TopicDef { code: "ALG", title: "Algebraic Expressions and Equations", description: "Expand brackets. Factorise common factor. Solve linear equations with variables on both sides. Inequalities on number line", subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 12, tags: &["algebra", "expand", "factorise", "linear-equations", "inequalities"], exam_marks: None },
            TopicDef { code: "FN", title: "Functions and Relationships", description: "Input-output tables. Represent functions as equations and graphs. Linear functions y = mx + c. Gradient and y-intercept", subdomain: "Patterns", node_type: "Concept", bloom: "Analyze", hours: 10, tags: &["functions", "linear", "gradient", "y-intercept", "graphs"], exam_marks: None },
            TopicDef { code: "GEO", title: "Geometry", description: "Angles in triangles and quadrilaterals. Parallel lines cut by transversal. Pythagoras theorem. Congruency and similarity intro", subdomain: "Geometry", node_type: "Concept", bloom: "Apply", hours: 10, tags: &["geometry", "pythagoras", "parallel-lines", "congruency", "similarity"], exam_marks: None },
            TopicDef { code: "MEA", title: "Measurement", description: "Area and perimeter of composite shapes. Surface area and volume of cubes, prisms, cylinders. Effect of scaling on measurements", subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8, tags: &["measurement", "composite-shapes", "surface-area", "scaling"], exam_marks: None },
            TopicDef { code: "DAT", title: "Data Handling", description: "Frequency tables. Histograms from grouped data. Mean, median, mode. Identify outliers. Scatter plots and trend lines", subdomain: "Data", node_type: "Skill", bloom: "Analyze", hours: 6, tags: &["data", "frequency-tables", "histograms", "outliers", "trend-lines"], exam_marks: None },
            TopicDef { code: "PRB", title: "Probability", description: "Probability of events. Relative frequency. Compare experimental and theoretical probability. Tree diagrams for compound events", subdomain: "Probability", node_type: "Concept", bloom: "Analyze", hours: 6, tags: &["probability", "relative-frequency", "compound-events", "tree-diagrams"], exam_marks: None },
        ],
        _ => vec![
            TopicDef { code: "GEN", title: "Mathematics", description: "General mathematics content", subdomain: "General", node_type: "Concept", bloom: "Understand", hours: 10, tags: &["mathematics"], exam_marks: None },
        ],
    };

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(grade, "Mathematics", t)).collect();

    // Cross-grade edges: connect to next grade
    let mut edges = Vec::new();
    if grade < 9 {
        let next = grade + 1;
        // Connect each topic to its counterpart in the next grade (same subdomain)
        for t in &topics {
            let from_id = format!("CAPS.Mathematics.Gr{}.{}", grade, t.code);
            // Find matching topic in next grade by subdomain overlap
            let to_codes: Vec<&str> = match t.subdomain {
                "Numbers" | "Operations" => {
                    if next <= 3 { vec!["NUM", "ADD", "MUL"] }
                    else if next <= 6 { vec!["NUM", "OPS"] }
                    else if next <= 8 { vec!["NUM", "ALG"] }
                    else { vec!["NUM.1", "ALG.1"] }
                },
                "Patterns" => {
                    if next <= 8 { vec!["PAT"] } else { vec!["PAT.1"] }
                },
                "Geometry" => {
                    if next <= 8 { vec!["GEO"] } else { vec!["GEOM.1"] }
                },
                "Measurement" => {
                    if next <= 8 { vec!["MEA"] } else { vec!["MEAS.1"] }
                },
                "Data" | "Probability" => {
                    if next <= 8 { vec!["DAT"] } else { vec!["STAT.1", "PROB.1"] }
                },
                "Algebra" => {
                    if next <= 8 { vec!["ALG", "EQN"] } else { vec!["ALG.1", "ALG.2"] }
                },
                _ => vec![],
            };
            for to_code in to_codes {
                let to_id = format!("CAPS.Mathematics.Gr{}.{}", next, to_code);
                edges.push(caps_edge(&from_id, &to_id,
                    &format!("Grade {} {} feeds into Grade {}", grade, t.subdomain, next), 800));
            }
        }
    }

    let node_count = nodes.len();
    let phase = match grade {
        1..=3 => "Foundation",
        4..=6 => "Intermediate",
        7..=8 => "Senior",
        _ => "General",
    };

    CurriculumDocument {
        metadata: caps_metadata(
            &format!("CAPS Mathematics Grade {}", grade),
            grade,
            "Mathematics",
            topics.iter().map(|t| t.subdomain.to_string()).collect::<std::collections::HashSet<_>>().into_iter().collect(),
            node_count,
            &format!("Curated from CAPS Mathematics {} Phase (DBE, 2011 amended 2022). Grade {}.", phase, grade),
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 9 Mathematics (Senior Phase — foundations for FET)
// ============================================================

fn build_math_grade_9() -> CurriculumDocument {
    let topics = vec![
        TopicDef {
            code: "NUM.1", title: "Whole Numbers and Integers",
            description: "Properties of integers. Order of operations (BODMAS). Calculations with integers including multiplication and division of negative numbers. Prime factorisation, HCF, LCM",
            subdomain: "Numbers & Operations", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["integers", "operations", "bodmas", "prime-factorisation", "hcf", "lcm"],
            exam_marks: None,
        },
        TopicDef {
            code: "NUM.2", title: "Common Fractions, Decimal Fractions, and Percentages",
            description: "Operations with fractions and decimals. Converting between fractions, decimals, and percentages. Percentage increase/decrease. Ratio and proportion",
            subdomain: "Numbers & Operations", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["fractions", "decimals", "percentages", "ratio", "proportion"],
            exam_marks: None,
        },
        TopicDef {
            code: "EXP.1", title: "Exponents",
            description: "Laws of exponents with integer exponents: aᵐ × aⁿ = aᵐ⁺ⁿ, aᵐ ÷ aⁿ = aᵐ⁻ⁿ, (aᵐ)ⁿ = aᵐⁿ, (ab)ⁿ = aⁿbⁿ, a⁰ = 1. Scientific notation",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["exponents", "laws-of-exponents", "scientific-notation", "indices"],
            exam_marks: None,
        },
        TopicDef {
            code: "ALG.1", title: "Algebraic Expressions",
            description: "Identify and classify like terms, coefficients, constants. Expand and simplify expressions. Multiply binomials. Factorise: common factor, difference of squares, trinomials",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["algebra", "expressions", "expand", "simplify", "factorise", "binomials"],
            exam_marks: None,
        },
        TopicDef {
            code: "ALG.2", title: "Algebraic Equations",
            description: "Solve linear equations including equations with fractions. Solve simple quadratic equations by factorisation (x² = c, ax² + bx = 0). Word problems leading to equations",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["equations", "linear-equations", "word-problems", "solving"],
            exam_marks: None,
        },
        TopicDef {
            code: "PAT.1", title: "Numeric and Geometric Patterns",
            description: "Investigate and describe numeric patterns (including geometric). Determine input and output values for given rules. Describe the general rule for a pattern in words and algebraically",
            subdomain: "Patterns & Functions", node_type: "Concept", bloom: "Analyze", hours: 6,
            tags: &["patterns", "sequences", "general-rule", "input-output"],
            exam_marks: None,
        },
        TopicDef {
            code: "FN.1", title: "Functions and Relationships",
            description: "Input-output tables. Represent functions as equations (y = mx + c), tables, and graphs. Interpret and plot points on the Cartesian plane. Linear functions",
            subdomain: "Patterns & Functions", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["functions", "cartesian-plane", "linear", "input-output", "graphs"],
            exam_marks: None,
        },
        TopicDef {
            code: "GEOM.1", title: "Geometry of Straight Lines",
            description: "Angles on a straight line (180°), vertically opposite angles, angles formed by parallel lines cut by a transversal (corresponding, alternate, co-interior)",
            subdomain: "Geometry", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["geometry", "angles", "parallel-lines", "transversal", "corresponding-angles"],
            exam_marks: None,
        },
        TopicDef {
            code: "GEOM.2", title: "Geometry of 2D Shapes",
            description: "Properties of triangles and quadrilaterals. Congruency (SSS, SAS, AAS, RHS). Similarity. The theorem of Pythagoras and its converse",
            subdomain: "Geometry", node_type: "Concept", bloom: "Apply", hours: 10,
            tags: &["geometry", "triangles", "quadrilaterals", "congruency", "pythagoras", "similarity"],
            exam_marks: None,
        },
        TopicDef {
            code: "MEAS.1", title: "Area, Perimeter, and Volume",
            description: "Area and perimeter of polygons. Surface area and volume of cubes, rectangular prisms, triangular prisms, and cylinders. Conversions between units",
            subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["measurement", "area", "perimeter", "volume", "surface-area", "prisms"],
            exam_marks: None,
        },
        TopicDef {
            code: "STAT.1", title: "Data Handling",
            description: "Collect, organise, summarise, and represent data. Mean, median, mode, range. Bar graphs, histograms, pie charts, line graphs. Interpret data and draw conclusions",
            subdomain: "Statistics", node_type: "Skill", bloom: "Analyze", hours: 6,
            tags: &["statistics", "data-handling", "mean", "median", "mode", "graphs"],
            exam_marks: None,
        },
        TopicDef {
            code: "PROB.1", title: "Probability",
            description: "Probability of single events. Probability scale (0 to 1). Relative frequency. Comparing experimental and theoretical probability. Tree diagrams for compound events",
            subdomain: "Probability", node_type: "Concept", bloom: "Understand", hours: 6,
            tags: &["probability", "events", "relative-frequency", "tree-diagrams"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(9, "Mathematics", t)).collect();
    let edges = vec![
        caps_edge("CAPS.Mathematics.Gr9.NUM.1", "CAPS.Mathematics.Gr9.NUM.2",
            "Integer operations before fractions and percentages", 800),
        caps_edge("CAPS.Mathematics.Gr9.NUM.1", "CAPS.Mathematics.Gr9.EXP.1",
            "Integer operations before exponent laws", 800),
        caps_edge("CAPS.Mathematics.Gr9.EXP.1", "CAPS.Mathematics.Gr9.ALG.1",
            "Exponent laws needed for algebraic expressions", 850),
        caps_edge("CAPS.Mathematics.Gr9.ALG.1", "CAPS.Mathematics.Gr9.ALG.2",
            "Expression manipulation before equation solving", 900),
        caps_edge("CAPS.Mathematics.Gr9.ALG.2", "CAPS.Mathematics.Gr9.PAT.1",
            "Equations needed for pattern generalisation", 700),
        caps_edge("CAPS.Mathematics.Gr9.PAT.1", "CAPS.Mathematics.Gr9.FN.1",
            "Patterns lead to function concepts", 800),
        caps_edge("CAPS.Mathematics.Gr9.GEOM.1", "CAPS.Mathematics.Gr9.GEOM.2",
            "Angle properties before shape properties", 850),
        // Gr9 → Gr10 cross-grade edges
        caps_edge("CAPS.Mathematics.Gr9.ALG.1", "CAPS.Mathematics.Gr10.ALG.1",
            "Grade 9 factorisation is prerequisite for Grade 10 algebra", 950),
        caps_edge("CAPS.Mathematics.Gr9.ALG.2", "CAPS.Mathematics.Gr10.ALG.2",
            "Grade 9 equations before Grade 10 equations", 900),
        caps_edge("CAPS.Mathematics.Gr9.EXP.1", "CAPS.Mathematics.Gr10.ALG.1",
            "Grade 9 exponents needed for Grade 10 expressions", 850),
        caps_edge("CAPS.Mathematics.Gr9.FN.1", "CAPS.Mathematics.Gr10.FN.1",
            "Grade 9 function concepts before Grade 10 functions", 900),
        caps_edge("CAPS.Mathematics.Gr9.GEOM.2", "CAPS.Mathematics.Gr10.GEOM.1",
            "Grade 9 shape properties before Grade 10 quadrilateral properties", 850),
        caps_edge("CAPS.Mathematics.Gr9.STAT.1", "CAPS.Mathematics.Gr10.STAT.1",
            "Grade 9 data handling before Grade 10 statistics", 800),
        caps_edge("CAPS.Mathematics.Gr9.PROB.1", "CAPS.Mathematics.Gr10.PROB.1",
            "Grade 9 basic probability before Grade 10 Venn diagrams", 850),
        caps_edge("CAPS.Mathematics.Gr9.NUM.2", "CAPS.Mathematics.Gr10.FIN.1",
            "Fractions and percentages needed for Grade 10 finance", 800),
        caps_edge("CAPS.Mathematics.Gr9.MEAS.1", "CAPS.Mathematics.Gr10.MEAS.1",
            "Grade 9 volume/area before Grade 10 prisms and cylinders", 800),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Mathematics Grade 9",
            9,
            "Mathematics",
            vec!["Numbers & Operations".into(), "Algebra".into(), "Patterns & Functions".into(),
                 "Geometry".into(), "Measurement".into(), "Statistics".into(), "Probability".into()],
            node_count,
            "Curated from CAPS Mathematics Senior Phase (DBE, 2011 amended 2022). Grade 9 — foundational skills for FET Phase.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 9 Natural Sciences (Senior Phase — physics/chemistry foundations)
// ============================================================

fn build_natsci_grade_9() -> CurriculumDocument {
    let topics = vec![
        TopicDef {
            code: "MM.1", title: "Properties of Materials",
            description: "Classify materials by properties (density, hardness, flexibility). Acids and bases in everyday life. pH indicators. Metals and non-metals. Physical and chemical changes",
            subdomain: "Matter & Materials", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["matter", "materials", "properties", "acids-bases", "metals", "physical-change", "chemical-change"],
            exam_marks: None,
        },
        TopicDef {
            code: "MM.2", title: "Particle Model of Matter",
            description: "The particle model to explain states of matter, changes of state, density, and gas pressure. Atoms as building blocks. Elements, compounds, and mixtures",
            subdomain: "Matter & Materials", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["particles", "atoms", "elements", "compounds", "mixtures", "states-of-matter"],
            exam_marks: None,
        },
        TopicDef {
            code: "MM.3", title: "Chemical Reactions",
            description: "Reactions of metals with oxygen, water, and acids. Reaction of non-metals with oxygen. Conservation of mass. Word equations for simple reactions",
            subdomain: "Matter & Materials", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["chemical-reactions", "metals", "oxides", "conservation-of-mass", "word-equations"],
            exam_marks: None,
        },
        TopicDef {
            code: "EC.1", title: "Energy and Electricity",
            description: "Energy transfer and transformation. Series and parallel circuits (basic). Potential difference and current. Resistance. Cost of electricity (kWh)",
            subdomain: "Energy & Change", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["energy", "electricity", "circuits", "current", "voltage", "resistance"],
            exam_marks: None,
        },
        TopicDef {
            code: "EC.2", title: "Forces and Motion",
            description: "Contact and non-contact forces. Gravitational force and weight (w = mg). Balanced and unbalanced forces. Speed, distance, time calculations. Speed-time graphs",
            subdomain: "Energy & Change", node_type: "Concept", bloom: "Apply", hours: 10,
            tags: &["forces", "motion", "gravity", "weight", "speed", "distance-time"],
            exam_marks: None,
        },
        TopicDef {
            code: "EC.3", title: "Electric Cells and Circuits",
            description: "How batteries work. Series and parallel circuits. Measuring voltage and current. Resistors in circuits. Safety with electricity",
            subdomain: "Energy & Change", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["circuits", "batteries", "series-parallel", "voltage", "current", "safety"],
            exam_marks: None,
        },
        TopicDef {
            code: "EC.4", title: "Waves, Sound, and Light",
            description: "Transverse and longitudinal waves. Properties: wavelength, frequency, amplitude. Sound: pitch, loudness, echoes. Light: reflection, refraction, colours",
            subdomain: "Energy & Change", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["waves", "sound", "light", "reflection", "refraction", "frequency"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(9, "Natural Sciences", t)).collect();
    let edges = vec![
        caps_edge("CAPS.NaturalSciences.Gr9.MM.1", "CAPS.NaturalSciences.Gr9.MM.2",
            "Material properties before particle model", 800),
        caps_edge("CAPS.NaturalSciences.Gr9.MM.2", "CAPS.NaturalSciences.Gr9.MM.3",
            "Particle model before chemical reactions", 850),
        caps_edge("CAPS.NaturalSciences.Gr9.EC.1", "CAPS.NaturalSciences.Gr9.EC.3",
            "Energy concepts before circuit details", 800),
        // Gr9 → Gr10 cross-grade edges (Natural Sciences → Physical Sciences)
        caps_edge("CAPS.NaturalSciences.Gr9.MM.2", "CAPS.PhysicalSciences.Gr10.CHM.1",
            "Grade 9 atoms/elements before Grade 10 atomic structure", 900),
        caps_edge("CAPS.NaturalSciences.Gr9.MM.3", "CAPS.PhysicalSciences.Gr10.CHM.5",
            "Grade 9 chemical reactions before Grade 10 reaction types", 850),
        caps_edge("CAPS.NaturalSciences.Gr9.MM.1", "CAPS.PhysicalSciences.Gr10.CHM.3",
            "Grade 9 acids-bases before Grade 10 intermolecular forces", 750),
        caps_edge("CAPS.NaturalSciences.Gr9.EC.2", "CAPS.PhysicalSciences.Gr10.PHY.1",
            "Grade 9 forces/motion before Grade 10 kinematics", 900),
        caps_edge("CAPS.NaturalSciences.Gr9.EC.3", "CAPS.PhysicalSciences.Gr10.PHY.7",
            "Grade 9 basic circuits before Grade 10 Ohm's law circuits", 900),
        caps_edge("CAPS.NaturalSciences.Gr9.EC.4", "CAPS.PhysicalSciences.Gr10.PHY.3",
            "Grade 9 waves basics before Grade 10 wave properties", 850),
        caps_edge("CAPS.NaturalSciences.Gr9.EC.1", "CAPS.PhysicalSciences.Gr10.PHY.6",
            "Grade 9 energy/electricity before Grade 10 electrostatics", 800),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Natural Sciences Grade 9",
            9,
            "Natural Sciences",
            vec!["Matter & Materials".into(), "Energy & Change".into()],
            node_count,
            "Curated from CAPS Natural Sciences Senior Phase (DBE, 2011 amended 2022). Grade 9 — physics and chemistry foundations for FET Physical Sciences.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 10 Mathematics
// ============================================================

fn build_math_grade_10() -> CurriculumDocument {
    let topics = vec![
        TopicDef {
            code: "ALG.1", title: "Algebraic Expressions",
            description: "Simplify, factorise, and manipulate algebraic expressions including products, factors (common factor, grouping, trinomials, difference of squares, sum/difference of cubes)",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["algebra", "factorisation", "expressions"],
            exam_marks: None,
        },
        TopicDef {
            code: "ALG.2", title: "Equations and Inequalities",
            description: "Solve linear, quadratic, literal, and simultaneous linear equations. Solve linear inequalities and represent on number line",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["equations", "inequalities", "linear", "quadratic"],
            exam_marks: None,
        },
        TopicDef {
            code: "PAT.1", title: "Number Patterns",
            description: "Investigate and extend numeric and geometric patterns. Determine the general term of linear sequences (Tn = an + b)",
            subdomain: "Patterns & Sequences", node_type: "Concept", bloom: "Analyze", hours: 6,
            tags: &["patterns", "sequences", "linear-sequences"],
            exam_marks: None,
        },
        TopicDef {
            code: "FIN.1", title: "Finance and Growth",
            description: "Simple and compound interest. Use formulae A = P(1+in) and A = P(1+i)^n. Hire purchase and inflation",
            subdomain: "Finance", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["finance", "simple-interest", "compound-interest", "growth"],
            exam_marks: None,
        },
        TopicDef {
            code: "FN.1", title: "Functions — Linear and Quadratic",
            description: "Define function, domain, range. Investigate and sketch y = ax + q, y = ax² + q. Effect of a and q on graphs. Points of intersection",
            subdomain: "Functions", node_type: "Concept", bloom: "Analyze", hours: 14,
            tags: &["functions", "linear-function", "quadratic-function", "parabola"],
            exam_marks: None,
        },
        TopicDef {
            code: "FN.2", title: "Functions — Hyperbola and Exponential",
            description: "Investigate and sketch y = a/x + q (hyperbola) and y = ab^x + q (exponential). Asymptotes, domain, range",
            subdomain: "Functions", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["functions", "hyperbola", "exponential-function", "asymptote"],
            exam_marks: None,
        },
        TopicDef {
            code: "TRIG.1", title: "Trigonometry — Definitions and Identities",
            description: "Define sin, cos, tan using right-angled triangles and the unit circle. Derive and use identities: tan θ = sin θ/cos θ, sin²θ + cos²θ = 1. Solve 2D triangle problems",
            subdomain: "Trigonometry", node_type: "Concept", bloom: "Understand", hours: 12,
            tags: &["trigonometry", "sine", "cosine", "tangent", "identities"],
            exam_marks: None,
        },
        TopicDef {
            code: "TRIG.2", title: "Trigonometric Functions",
            description: "Sketch and interpret graphs of y = sin θ, y = cos θ, y = tan θ for θ ∈ [0°, 360°]. Period, amplitude, asymptotes",
            subdomain: "Trigonometry", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["trigonometry", "trig-graphs", "sine-curve", "cosine-curve"],
            exam_marks: None,
        },
        TopicDef {
            code: "GEOM.1", title: "Euclidean Geometry — Properties",
            description: "Properties of special quadrilaterals (parallelogram, rectangle, rhombus, square, kite, trapezium). Midpoint theorem",
            subdomain: "Euclidean Geometry", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["geometry", "quadrilaterals", "midpoint-theorem", "euclidean"],
            exam_marks: None,
        },
        TopicDef {
            code: "ANAG.1", title: "Analytical Geometry",
            description: "Distance formula, midpoint formula, gradient. Inclination of a line. Equation of a straight line. Parallel and perpendicular lines",
            subdomain: "Analytical Geometry", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["analytical-geometry", "coordinate-geometry", "distance", "gradient"],
            exam_marks: None,
        },
        TopicDef {
            code: "STAT.1", title: "Statistics",
            description: "Collect, organise, and interpret univariate data. Measures of central tendency (mean, median, mode) and spread (range, percentiles, quartiles, IQR, five-number summary). Box-and-whisker plots",
            subdomain: "Statistics", node_type: "Skill", bloom: "Analyze", hours: 8,
            tags: &["statistics", "mean", "median", "box-plot", "data-handling"],
            exam_marks: None,
        },
        TopicDef {
            code: "PROB.1", title: "Probability",
            description: "Theoretical and experimental probability. Venn diagrams. Mutually exclusive and complementary events. The addition rule for mutually exclusive events",
            subdomain: "Probability", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["probability", "venn-diagrams", "events", "addition-rule"],
            exam_marks: None,
        },
        TopicDef {
            code: "MEAS.1", title: "Measurement",
            description: "Conversions, perimeter, area (2D shapes), surface area and volume of right prisms and cylinders",
            subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 6,
            tags: &["measurement", "area", "volume", "prisms", "cylinders"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(10, "Mathematics", t)).collect();
    let edges = vec![
        // Algebra prerequisites
        caps_edge("CAPS.Mathematics.Gr10.ALG.1", "CAPS.Mathematics.Gr10.ALG.2",
            "Algebraic manipulation is prerequisite for solving equations", 900),
        // Functions need algebra
        caps_edge("CAPS.Mathematics.Gr10.ALG.2", "CAPS.Mathematics.Gr10.FN.1",
            "Equations skills needed for function analysis", 800),
        caps_edge("CAPS.Mathematics.Gr10.FN.1", "CAPS.Mathematics.Gr10.FN.2",
            "Linear/quadratic foundation before hyperbola/exponential", 800),
        // Trig progression
        caps_edge("CAPS.Mathematics.Gr10.TRIG.1", "CAPS.Mathematics.Gr10.TRIG.2",
            "Trig definitions before trig graphs", 900),
        // Analytical geometry needs algebra
        caps_edge("CAPS.Mathematics.Gr10.ALG.2", "CAPS.Mathematics.Gr10.ANAG.1",
            "Linear equations needed for analytical geometry", 750),
        // Finance needs algebra
        caps_edge("CAPS.Mathematics.Gr10.ALG.1", "CAPS.Mathematics.Gr10.FIN.1",
            "Algebraic substitution needed for finance formulae", 700),
        // Patterns need algebra
        caps_edge("CAPS.Mathematics.Gr10.ALG.1", "CAPS.Mathematics.Gr10.PAT.1",
            "Algebraic expressions needed for general term", 700),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Mathematics Grade 10",
            10,
            "Mathematics",
            vec!["Algebra".into(), "Functions".into(), "Trigonometry".into(),
                 "Euclidean Geometry".into(), "Analytical Geometry".into(),
                 "Statistics".into(), "Probability".into(), "Measurement".into()],
            node_count,
            "Curated from CAPS Mathematics FET Phase (DBE, 2011 amended 2022). Grade 10 foundations.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 11 Mathematics
// ============================================================

fn build_math_grade_11() -> CurriculumDocument {
    let topics = vec![
        TopicDef {
            code: "ALG.1", title: "Algebraic Expressions — Surds and Simplification",
            description: "Simplify expressions involving surds. Apply laws of exponents to real-number exponents. Add, subtract, multiply, divide surds",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["algebra", "surds", "exponents", "simplification"],
            exam_marks: None,
        },
        TopicDef {
            code: "ALG.2", title: "Equations and Inequalities — Quadratic and Simultaneous",
            description: "Solve quadratic equations (factorisation, completing the square, quadratic formula). Nature of roots (discriminant). Simultaneous equations (one linear, one quadratic). Quadratic inequalities",
            subdomain: "Algebra", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["equations", "quadratic-formula", "discriminant", "simultaneous"],
            exam_marks: None,
        },
        TopicDef {
            code: "PAT.1", title: "Patterns, Sequences, and Series",
            description: "Arithmetic and geometric sequences. General term Tn. Sigma notation. Finite arithmetic and geometric series. Convergent geometric series (|r| < 1)",
            subdomain: "Patterns & Sequences", node_type: "Concept", bloom: "Analyze", hours: 12,
            tags: &["sequences", "series", "arithmetic", "geometric", "sigma-notation"],
            exam_marks: None,
        },
        TopicDef {
            code: "FIN.1", title: "Finance, Growth, and Decay",
            description: "Compound growth and decay: A = P(1+i)^n, A = P(1−i)^n. Effective and nominal interest rates. Simple and compound depreciation",
            subdomain: "Finance", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["finance", "compound-growth", "decay", "depreciation", "nominal-rate"],
            exam_marks: None,
        },
        TopicDef {
            code: "FN.1", title: "Functions — Parabola, Hyperbola, Exponential",
            description: "Investigate y = a(x−p)² + q, y = a/(x−p) + q, y = ab^(x−p) + q. Transformations: shifts, reflections, stretches. Axes of symmetry, turning points, asymptotes",
            subdomain: "Functions", node_type: "Concept", bloom: "Analyze", hours: 14,
            tags: &["functions", "transformations", "parabola", "hyperbola", "exponential"],
            exam_marks: None,
        },
        TopicDef {
            code: "FN.2", title: "Inverse Functions",
            description: "Concept of inverse function (reflection in y = x). Determine and sketch inverses of y = ax + q, y = ax², y = b^x. Logarithmic function as inverse of exponential",
            subdomain: "Functions", node_type: "Concept", bloom: "Analyze", hours: 8,
            tags: &["functions", "inverse", "logarithm", "reflection"],
            exam_marks: None,
        },
        TopicDef {
            code: "TRIG.1", title: "Trigonometry — Compound and Double Angles",
            description: "Derive and apply compound angle identities: sin(α±β), cos(α±β). Double angle formulae. Solve general trigonometric equations. Reduction formulae for (90°±θ), (180°±θ), (360°±θ)",
            subdomain: "Trigonometry", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["trigonometry", "compound-angles", "double-angles", "reduction-formulae"],
            exam_marks: None,
        },
        TopicDef {
            code: "TRIG.2", title: "Trigonometry — Sine, Cosine, and Area Rules",
            description: "Prove and apply sine rule, cosine rule, and area rule. Solve 2D and 3D problems involving these rules",
            subdomain: "Trigonometry", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["trigonometry", "sine-rule", "cosine-rule", "area-rule", "3d-problems"],
            exam_marks: None,
        },
        TopicDef {
            code: "GEOM.1", title: "Euclidean Geometry — Circle Geometry",
            description: "Theorems of the circle: line from centre perpendicular to chord, angle at centre, angles in same segment, cyclic quadrilaterals, tangent properties. Prove and apply",
            subdomain: "Euclidean Geometry", node_type: "Concept", bloom: "Evaluate", hours: 14,
            tags: &["geometry", "circle-theorems", "cyclic-quadrilateral", "tangent", "euclidean"],
            exam_marks: None,
        },
        TopicDef {
            code: "ANAG.1", title: "Analytical Geometry — Circles",
            description: "Equation of a circle with centre at origin and centre (a, b). Determine equation from given information. Tangent to circle at a point",
            subdomain: "Analytical Geometry", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["analytical-geometry", "circle-equation", "tangent-line"],
            exam_marks: None,
        },
        TopicDef {
            code: "STAT.1", title: "Statistics — Bivariate Data",
            description: "Scatter plots. Intuitive line of best fit. Correlation (strong/weak, positive/negative). Regression line (interpretation, not calculation). Ogives and cumulative frequency",
            subdomain: "Statistics", node_type: "Skill", bloom: "Analyze", hours: 8,
            tags: &["statistics", "scatter-plots", "correlation", "regression", "ogives"],
            exam_marks: None,
        },
        TopicDef {
            code: "PROB.1", title: "Probability — Dependent and Independent Events",
            description: "Dependent and independent events. Product rule. Tree diagrams and Venn diagrams. Contingency tables. Mutually exclusive vs independent",
            subdomain: "Probability", node_type: "Concept", bloom: "Analyze", hours: 8,
            tags: &["probability", "independent-events", "tree-diagrams", "contingency-tables"],
            exam_marks: None,
        },
        TopicDef {
            code: "MEAS.1", title: "Measurement — Pyramids, Cones, and Spheres",
            description: "Surface area and volume of pyramids (right), right cones, and spheres. Combinations of these solids",
            subdomain: "Measurement", node_type: "Skill", bloom: "Apply", hours: 6,
            tags: &["measurement", "pyramids", "cones", "spheres", "volume"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(11, "Mathematics", t)).collect();
    let edges = vec![
        // Algebra chain
        caps_edge("CAPS.Mathematics.Gr11.ALG.1", "CAPS.Mathematics.Gr11.ALG.2",
            "Surd manipulation needed for quadratic equations", 850),
        // Functions need algebra
        caps_edge("CAPS.Mathematics.Gr11.ALG.2", "CAPS.Mathematics.Gr11.FN.1",
            "Quadratic equations needed for function analysis", 800),
        caps_edge("CAPS.Mathematics.Gr11.FN.1", "CAPS.Mathematics.Gr11.FN.2",
            "Function concepts needed for inverses", 900),
        // Trig chain
        caps_edge("CAPS.Mathematics.Gr11.TRIG.1", "CAPS.Mathematics.Gr11.TRIG.2",
            "Compound angle identities needed for sine/cosine rules", 800),
        // Finance needs algebra
        caps_edge("CAPS.Mathematics.Gr11.ALG.1", "CAPS.Mathematics.Gr11.FIN.1",
            "Exponent laws needed for growth/decay formulae", 750),
        // Series needs algebra
        caps_edge("CAPS.Mathematics.Gr11.ALG.1", "CAPS.Mathematics.Gr11.PAT.1",
            "Algebraic manipulation for series formulae", 750),
        // Cross-grade: Gr10 → Gr11
        caps_edge("CAPS.Mathematics.Gr10.ALG.2", "CAPS.Mathematics.Gr11.ALG.1",
            "Grade 10 equations are prerequisite for Grade 11 algebra", 900),
        caps_edge("CAPS.Mathematics.Gr10.FN.2", "CAPS.Mathematics.Gr11.FN.1",
            "Grade 10 function types foundation for Grade 11 transformations", 900),
        caps_edge("CAPS.Mathematics.Gr10.TRIG.2", "CAPS.Mathematics.Gr11.TRIG.1",
            "Grade 10 trig graphs before compound angles", 900),
        caps_edge("CAPS.Mathematics.Gr10.GEOM.1", "CAPS.Mathematics.Gr11.GEOM.1",
            "Quadrilateral properties before circle geometry", 850),
        caps_edge("CAPS.Mathematics.Gr10.ANAG.1", "CAPS.Mathematics.Gr11.ANAG.1",
            "Straight-line analytical geometry before circle equations", 850),
        caps_edge("CAPS.Mathematics.Gr10.STAT.1", "CAPS.Mathematics.Gr11.STAT.1",
            "Univariate statistics before bivariate data", 800),
        caps_edge("CAPS.Mathematics.Gr10.PROB.1", "CAPS.Mathematics.Gr11.PROB.1",
            "Basic probability before dependent/independent events", 850),
        caps_edge("CAPS.Mathematics.Gr10.FIN.1", "CAPS.Mathematics.Gr11.FIN.1",
            "Simple/compound interest before nominal rates and depreciation", 850),
        caps_edge("CAPS.Mathematics.Gr10.PAT.1", "CAPS.Mathematics.Gr11.PAT.1",
            "Linear sequences before arithmetic/geometric series", 850),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Mathematics Grade 11",
            11,
            "Mathematics",
            vec!["Algebra".into(), "Functions".into(), "Trigonometry".into(),
                 "Euclidean Geometry".into(), "Analytical Geometry".into(),
                 "Statistics".into(), "Probability".into(), "Measurement".into()],
            node_count,
            "Curated from CAPS Mathematics FET Phase (DBE, 2011 amended 2022). Grade 11 intermediate.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 12 Mathematics (Matric)
// ============================================================

fn build_math_grade_12() -> CurriculumDocument {
    let topics = vec![
        // Paper 1 topics
        TopicDef {
            code: "P1.ALG", title: "Algebra, Equations, and Inequalities",
            description: "Solve equations: quadratic (all methods), surd, exponential, simultaneous (two quadratics). Quadratic inequalities. Nature of roots (discriminant Δ = b²−4ac). Apply to real-world problems",
            subdomain: "Paper 1 — Algebra", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["algebra", "quadratic", "equations", "discriminant", "inequalities"],
            exam_marks: Some((1, 25, 150)),
        },
        TopicDef {
            code: "P1.SEQ", title: "Patterns, Sequences, and Series",
            description: "Arithmetic sequences (Tn = a + (n−1)d, Sn = n/2[2a + (n−1)d]). Geometric sequences (Tn = ar^(n−1), Sn = a(r^n − 1)/(r − 1)). Sum to infinity S∞ = a/(1−r) for |r|<1. Sigma notation. Proof by induction (enrichment)",
            subdomain: "Paper 1 — Sequences", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["sequences", "series", "arithmetic", "geometric", "sigma-notation", "sum-to-infinity"],
            exam_marks: Some((1, 25, 150)),
        },
        TopicDef {
            code: "P1.FIN", title: "Finance, Growth, and Decay",
            description: "Future value and present value annuities: F = x[(1+i)^n − 1]/i, P = x[1 − (1+i)^−n]/i. Sinking funds, loan repayments, outstanding balance. Effective vs nominal rates. Deferred payments. Pyramidal schemes",
            subdomain: "Paper 1 — Finance", node_type: "Skill", bloom: "Analyze", hours: 12,
            tags: &["finance", "annuities", "future-value", "present-value", "loans", "sinking-fund"],
            exam_marks: Some((1, 25, 150)),
        },
        TopicDef {
            code: "P1.FN", title: "Functions and Graphs",
            description: "Revise linear, quadratic (y = a(x−p)²+q), hyperbolic (y = a/(x−p)+q), exponential (y = ab^(x−p)+q). Logarithmic function. Average gradient, increasing/decreasing intervals. Interpret real-world graphs. Transformations and reflections",
            subdomain: "Paper 1 — Functions", node_type: "Concept", bloom: "Analyze", hours: 14,
            tags: &["functions", "parabola", "hyperbola", "exponential", "logarithm", "transformations"],
            exam_marks: Some((1, 25, 150)),
        },
        TopicDef {
            code: "P1.CALC", title: "Differential Calculus",
            description: "Limits (intuitive). First principles: f'(x) = lim[h→0] (f(x+h)−f(x))/h. Differentiation rules: power rule, constant rule, sum/difference. Equations of tangent lines. Sketch cubic functions using first derivative (turning points, concavity). Optimisation problems",
            subdomain: "Paper 1 — Calculus", node_type: "Skill", bloom: "Evaluate", hours: 18,
            tags: &["calculus", "differentiation", "first-principles", "tangent", "cubic", "optimisation"],
            exam_marks: Some((1, 35, 150)),
        },
        TopicDef {
            code: "P1.COUNT", title: "Counting Principles and Probability",
            description: "Fundamental counting principle. Factorial notation (n!). Permutations and combinations. Probability using counting principles. Mutually exclusive and complementary events revisited. Independent events. Tree diagrams",
            subdomain: "Paper 1 — Probability", node_type: "Skill", bloom: "Analyze", hours: 12,
            tags: &["probability", "counting", "permutations", "combinations", "factorial"],
            exam_marks: Some((1, 15, 150)),
        },

        // Paper 2 topics
        TopicDef {
            code: "P2.GEOM", title: "Euclidean Geometry",
            description: "Circle geometry theorems (prove examinable theorems, apply all). Proportionality theorem. Similar and congruent triangles applied to circle geometry. Riders combining multiple theorems",
            subdomain: "Paper 2 — Geometry", node_type: "Concept", bloom: "Evaluate", hours: 16,
            tags: &["geometry", "circle-theorems", "similar-triangles", "proportionality", "riders"],
            exam_marks: Some((2, 40, 150)),
        },
        TopicDef {
            code: "P2.ANAG", title: "Analytical Geometry",
            description: "Revise distance, midpoint, gradient, collinear points. Equation of a circle (centre origin and general). Tangent to circle. Properties of quadrilaterals proven analytically. Inclination and angle between two lines",
            subdomain: "Paper 2 — Analytical Geometry", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["analytical-geometry", "circle", "tangent", "inclination", "coordinate-geometry"],
            exam_marks: Some((2, 40, 150)),
        },
        TopicDef {
            code: "P2.TRIG", title: "Trigonometry",
            description: "Compound angles: sin(α±β), cos(α±β), sin2α, cos2α. Solve general equations. Trigonometric identities (prove). Sine rule, cosine rule, area rule. 2D and 3D problems. Trigonometric graphs with amplitude, period, and phase shift",
            subdomain: "Paper 2 — Trigonometry", node_type: "Skill", bloom: "Evaluate", hours: 16,
            tags: &["trigonometry", "compound-angles", "identities", "sine-rule", "cosine-rule", "3d-problems"],
            exam_marks: Some((2, 40, 150)),
        },
        TopicDef {
            code: "P2.STAT", title: "Statistics",
            description: "Symmetric and skewed data. Variance and standard deviation (σ = √(Σ(x−x̄)²/n)). Identify outliers. Ogives, box-and-whisker plots, histograms, frequency polygons. Normal distribution (intuitive). Bivariate data: scatter plots, least-squares regression line ŷ = a + bx, correlation coefficient r",
            subdomain: "Paper 2 — Statistics", node_type: "Skill", bloom: "Analyze", hours: 12,
            tags: &["statistics", "standard-deviation", "variance", "regression", "correlation", "normal-distribution"],
            exam_marks: Some((2, 30, 150)),
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(12, "Mathematics", t)).collect();
    let edges = vec![
        // Within Paper 1
        caps_edge("CAPS.Mathematics.Gr12.P1.ALG", "CAPS.Mathematics.Gr12.P1.SEQ",
            "Algebraic skills needed for series formulae", 800),
        caps_edge("CAPS.Mathematics.Gr12.P1.ALG", "CAPS.Mathematics.Gr12.P1.FIN",
            "Equation solving needed for annuity calculations", 850),
        caps_edge("CAPS.Mathematics.Gr12.P1.ALG", "CAPS.Mathematics.Gr12.P1.FN",
            "Algebraic manipulation for function analysis", 800),
        caps_edge("CAPS.Mathematics.Gr12.P1.FN", "CAPS.Mathematics.Gr12.P1.CALC",
            "Function knowledge is prerequisite for calculus", 950),
        caps_edge("CAPS.Mathematics.Gr12.P1.ALG", "CAPS.Mathematics.Gr12.P1.COUNT",
            "Algebraic reasoning for counting problems", 700),

        // Within Paper 2
        caps_edge("CAPS.Mathematics.Gr12.P2.GEOM", "CAPS.Mathematics.Gr12.P2.ANAG",
            "Geometric properties applied analytically", 700),

        // Cross-grade: Gr11 → Gr12
        caps_edge("CAPS.Mathematics.Gr11.ALG.2", "CAPS.Mathematics.Gr12.P1.ALG",
            "Grade 11 quadratic equations foundation for Grade 12 algebra", 900),
        caps_edge("CAPS.Mathematics.Gr11.PAT.1", "CAPS.Mathematics.Gr12.P1.SEQ",
            "Grade 11 series foundation for Grade 12 sequences", 900),
        caps_edge("CAPS.Mathematics.Gr11.FIN.1", "CAPS.Mathematics.Gr12.P1.FIN",
            "Grade 11 growth/decay before Grade 12 annuities", 900),
        caps_edge("CAPS.Mathematics.Gr11.FN.2", "CAPS.Mathematics.Gr12.P1.FN",
            "Grade 11 inverse functions before Grade 12 function revision", 900),
        caps_edge("CAPS.Mathematics.Gr11.TRIG.1", "CAPS.Mathematics.Gr12.P2.TRIG",
            "Grade 11 compound angles before Grade 12 trig", 900),
        caps_edge("CAPS.Mathematics.Gr11.GEOM.1", "CAPS.Mathematics.Gr12.P2.GEOM",
            "Grade 11 circle geometry before Grade 12 proofs", 950),
        caps_edge("CAPS.Mathematics.Gr11.ANAG.1", "CAPS.Mathematics.Gr12.P2.ANAG",
            "Grade 11 circle equations before Grade 12 analytical geometry", 900),
        caps_edge("CAPS.Mathematics.Gr11.STAT.1", "CAPS.Mathematics.Gr12.P2.STAT",
            "Grade 11 bivariate data before Grade 12 regression", 850),
        caps_edge("CAPS.Mathematics.Gr11.PROB.1", "CAPS.Mathematics.Gr12.P1.COUNT",
            "Grade 11 probability before Grade 12 counting principles", 850),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Mathematics Grade 12 (Matric)",
            12,
            "Mathematics",
            vec![
                "Paper 1 — Algebra".into(), "Paper 1 — Sequences".into(),
                "Paper 1 — Finance".into(), "Paper 1 — Functions".into(),
                "Paper 1 — Calculus".into(), "Paper 1 — Probability".into(),
                "Paper 2 — Geometry".into(), "Paper 2 — Analytical Geometry".into(),
                "Paper 2 — Trigonometry".into(), "Paper 2 — Statistics".into(),
            ],
            node_count,
            "Curated from CAPS Mathematics FET Phase (DBE, 2011 amended 2022). NSC Matric examination syllabus — Paper 1 (150 marks, 3 hours) and Paper 2 (150 marks, 3 hours).",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 10 Physical Sciences
// ============================================================

fn build_physics_grade_10() -> CurriculumDocument {
    let topics = vec![
        // Physics topics
        TopicDef {
            code: "PHY.1", title: "Mechanics — Motion in One Dimension",
            description: "Vectors and scalars. Reference frames. Position, displacement, speed, velocity, acceleration. Equations of motion: v = u + at, s = ut + ½at², v² = u² + 2as. Graphs of motion (x-t, v-t, a-t)",
            subdomain: "Physics — Mechanics", node_type: "Concept", bloom: "Apply", hours: 14,
            tags: &["physics", "mechanics", "kinematics", "motion", "equations-of-motion"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.2", title: "Mechanics — Energy",
            description: "Gravitational potential energy (Ep = mgh). Kinetic energy (Ek = ½mv²). Mechanical energy. Conservation of mechanical energy (non-conservative forces absent). Work done by a force (W = FΔx cos θ)",
            subdomain: "Physics — Mechanics", node_type: "Concept", bloom: "Apply", hours: 10,
            tags: &["physics", "energy", "kinetic-energy", "potential-energy", "conservation"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.3", title: "Waves, Sound, and Light — Transverse and Longitudinal",
            description: "Pulse, wavelength, frequency, period, amplitude. Transverse and longitudinal waves. Wave equation v = fλ. Superposition. Standing waves (strings)",
            subdomain: "Physics — Waves", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["physics", "waves", "transverse", "longitudinal", "frequency", "wavelength"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.4", title: "Waves — Sound",
            description: "Speed of sound. Pitch and loudness. Ultrasound applications. Resonance and standing waves in pipes",
            subdomain: "Physics — Waves", node_type: "Concept", bloom: "Understand", hours: 6,
            tags: &["physics", "sound", "pitch", "loudness", "resonance"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.5", title: "Waves — Electromagnetic Radiation",
            description: "Electromagnetic spectrum. Nature of EM waves. Properties: wavelength, frequency, speed (c = 3×10⁸ m/s). Applications of different EM bands",
            subdomain: "Physics — Waves", node_type: "Concept", bloom: "Understand", hours: 6,
            tags: &["physics", "electromagnetic", "spectrum", "light", "radiation"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.6", title: "Electricity and Magnetism — Electrostatics",
            description: "Two kinds of charge. Coulomb's law: F = kQ₁Q₂/r². Electric field. Electric field between parallel plates. Charge conservation and quantisation",
            subdomain: "Physics — Electricity", node_type: "Concept", bloom: "Apply", hours: 10,
            tags: &["physics", "electrostatics", "coulombs-law", "electric-field", "charge"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.7", title: "Electricity — Electric Circuits",
            description: "EMF and internal resistance. Ohm's law (V = IR). Series and parallel circuits. Power (P = VI = I²R = V²/R). Energy (E = VIt). Kirchhoff's rules (simple circuits)",
            subdomain: "Physics — Electricity", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["physics", "circuits", "ohms-law", "resistance", "power", "kirchhoff"],
            exam_marks: None,
        },

        // Chemistry topics
        TopicDef {
            code: "CHM.1", title: "Matter and Materials — Atomic Structure",
            description: "Models of the atom (Bohr). Electron configuration. Periodic table structure. Periodic trends: atomic radius, ionisation energy, electronegativity, electron affinity",
            subdomain: "Chemistry — Atomic", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["chemistry", "atoms", "electron-configuration", "periodic-table", "trends"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.2", title: "Chemical Bonding",
            description: "Ionic bonding (electron transfer, lattice structures). Covalent bonding (Lewis dot structures, molecular shapes — VSEPR). Metallic bonding. Bond polarity and molecular polarity",
            subdomain: "Chemistry — Bonding", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["chemistry", "bonding", "ionic", "covalent", "metallic", "vsepr", "lewis-structures"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.3", title: "Intermolecular Forces",
            description: "Van der Waals forces (London, dipole-dipole). Hydrogen bonding. Relationship between IMFs and physical properties (boiling point, viscosity, surface tension). States of matter",
            subdomain: "Chemistry — Bonding", node_type: "Concept", bloom: "Analyze", hours: 8,
            tags: &["chemistry", "intermolecular-forces", "hydrogen-bonding", "van-der-waals"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.4", title: "Chemical Change — Stoichiometry",
            description: "Mole concept (n = m/M, n = cV, n = N/Nₐ). Balanced equations. Limiting reagents. Percentage yield. Concentration (mol/dm³). Volume relationships (molar gas volume at STP)",
            subdomain: "Chemistry — Quantitative", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["chemistry", "stoichiometry", "moles", "concentration", "balanced-equations"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.5", title: "Chemical Change — Types of Reactions",
            description: "Acid-base reactions. Redox reactions (oxidation numbers). Combustion reactions. Synthesis and decomposition. Neutralisation. Net ionic equations",
            subdomain: "Chemistry — Reactions", node_type: "Concept", bloom: "Understand", hours: 8,
            tags: &["chemistry", "reactions", "acid-base", "redox", "oxidation", "combustion"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.6", title: "Hydrosphere — Water Chemistry",
            description: "Water as a resource. The water cycle. Water quality (pH, dissolved oxygen, turbidity). Water purification and treatment. Pollution",
            subdomain: "Chemistry — Environmental", node_type: "Concept", bloom: "Understand", hours: 4,
            tags: &["chemistry", "water", "purification", "pollution", "environment"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(10, "Physical Sciences", t)).collect();
    let edges = vec![
        // Physics: mechanics → energy
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.1", "CAPS.PhysicalSciences.Gr10.PHY.2",
            "Kinematics concepts needed for energy calculations", 850),
        // Waves chain
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.3", "CAPS.PhysicalSciences.Gr10.PHY.4",
            "General wave properties before sound", 800),
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.3", "CAPS.PhysicalSciences.Gr10.PHY.5",
            "General wave properties before EM radiation", 800),
        // Electricity chain
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.6", "CAPS.PhysicalSciences.Gr10.PHY.7",
            "Electrostatics concepts before circuits", 850),
        // Chemistry: structure → bonding → IMF
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.1", "CAPS.PhysicalSciences.Gr10.CHM.2",
            "Atomic structure needed for bonding", 900),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.2", "CAPS.PhysicalSciences.Gr10.CHM.3",
            "Chemical bonding before intermolecular forces", 900),
        // Stoichiometry needs bonding knowledge
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.2", "CAPS.PhysicalSciences.Gr10.CHM.4",
            "Bonding and formulae needed for stoichiometry", 800),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.4", "CAPS.PhysicalSciences.Gr10.CHM.5",
            "Mole calculations needed for reaction types", 800),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Physical Sciences Grade 10",
            10,
            "Physical Sciences",
            vec!["Physics — Mechanics".into(), "Physics — Waves".into(),
                 "Physics — Electricity".into(), "Chemistry — Atomic".into(),
                 "Chemistry — Bonding".into(), "Chemistry — Quantitative".into(),
                 "Chemistry — Reactions".into()],
            node_count,
            "Curated from CAPS Physical Sciences FET Phase (DBE, 2011 amended 2022). Grade 10 foundations — Physics and Chemistry.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 11 Physical Sciences
// ============================================================

fn build_physics_grade_11() -> CurriculumDocument {
    let topics = vec![
        // Physics
        TopicDef {
            code: "PHY.1", title: "Mechanics — Vectors in Two Dimensions",
            description: "Resolution of vectors into components. Addition by components. Newton's laws in 2D with resolved forces. Inclined planes. Friction (static and kinetic)",
            subdomain: "Physics — Mechanics", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["physics", "vectors", "components", "inclined-planes", "friction"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.2", title: "Mechanics — Newton's Laws and Application",
            description: "Newton's first, second (Fnet = ma), and third laws. Free-body diagrams. Applications: lifts, connected objects, inclined planes with friction. Normal force and weight",
            subdomain: "Physics — Mechanics", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["physics", "newtons-laws", "force", "free-body-diagram", "acceleration"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.3", title: "Mechanics — Momentum and Impulse",
            description: "Momentum (p = mv). Newton's second law in terms of momentum (FnetΔt = Δp). Impulse-momentum theorem. Conservation of momentum in isolated systems. Elastic and inelastic collisions",
            subdomain: "Physics — Mechanics", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["physics", "momentum", "impulse", "collisions", "conservation"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.4", title: "Mechanics — Work, Energy, and Power",
            description: "Work (W = FΔx cos θ). Work-energy theorem. Conservation of energy with non-conservative forces. Power (P = W/Δt = Fv). Efficiency",
            subdomain: "Physics — Mechanics", node_type: "Skill", bloom: "Analyze", hours: 10,
            tags: &["physics", "work", "energy", "power", "efficiency", "conservation"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.5", title: "Waves — Geometrical Optics",
            description: "Refraction (Snell's law: n₁sin θ₁ = n₂sin θ₂). Total internal reflection. Critical angle. Lenses: converging and diverging. Thin lens equation (1/f = 1/do + 1/di). Magnification. Ray diagrams",
            subdomain: "Physics — Waves", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["physics", "optics", "refraction", "snells-law", "lenses", "total-internal-reflection"],
            exam_marks: None,
        },
        TopicDef {
            code: "PHY.6", title: "Electricity — Electrodynamics",
            description: "Electrical machines: generators (AC and DC) and motors. Faraday's law. Alternating current (peak, RMS values: Vrms = Vmax/√2). Transformers",
            subdomain: "Physics — Electricity", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["physics", "electrodynamics", "generators", "motors", "faraday", "alternating-current"],
            exam_marks: None,
        },

        // Chemistry
        TopicDef {
            code: "CHM.1", title: "Matter and Materials — Organic Molecules (Introduction)",
            description: "Molecular and structural formulae. Functional groups: alkanes, alkenes, alkynes, alcohols, carboxylic acids, esters, aldehydes, ketones. IUPAC nomenclature. Physical properties and intermolecular forces",
            subdomain: "Chemistry — Organic", node_type: "Concept", bloom: "Understand", hours: 12,
            tags: &["chemistry", "organic", "functional-groups", "iupac", "nomenclature", "hydrocarbons"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.2", title: "Chemical Change — Reaction Rates",
            description: "Rate of reaction. Factors affecting rate: concentration, temperature, surface area, catalysts. Collision theory. Energy profiles (activation energy, exothermic, endothermic). Maxwell-Boltzmann distribution",
            subdomain: "Chemistry — Kinetics", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["chemistry", "reaction-rates", "kinetics", "activation-energy", "catalysts", "collision-theory"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.3", title: "Chemical Change — Chemical Equilibrium",
            description: "Open and closed systems. Reversible reactions. Dynamic equilibrium. Le Chatelier's principle (concentration, temperature, pressure). Equilibrium constant Kc. Equilibrium calculations",
            subdomain: "Chemistry — Equilibrium", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["chemistry", "equilibrium", "le-chatelier", "reversible-reactions", "kc"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.4", title: "Acids and Bases",
            description: "Arrhenius, Brønsted-Lowry definitions. Strong/weak acids and bases. pH scale (pH = −log[H⁺]). Neutralisation. Titrations. Indicators. Acid-base reactions with oxides, carbonates, metals",
            subdomain: "Chemistry — Acids & Bases", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["chemistry", "acids", "bases", "ph", "titration", "neutralisation", "bronsted-lowry"],
            exam_marks: None,
        },
        TopicDef {
            code: "CHM.5", title: "Electrochemistry — Galvanic and Electrolytic Cells",
            description: "Oxidation and reduction (electron transfer). Galvanic (voltaic) cells. Cell notation. Standard electrode potentials (E°). EMF calculation. Electrolytic cells. Faraday's laws of electrolysis. Applications (electroplating, extraction)",
            subdomain: "Chemistry — Electrochemistry", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["chemistry", "electrochemistry", "galvanic-cell", "electrolysis", "electrode-potential", "redox"],
            exam_marks: None,
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(11, "Physical Sciences", t)).collect();
    let edges = vec![
        // Physics: vectors → Newton → momentum → work/energy
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.1", "CAPS.PhysicalSciences.Gr11.PHY.2",
            "Vector resolution needed for 2D Newton's law problems", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.2", "CAPS.PhysicalSciences.Gr11.PHY.3",
            "Newton's second law foundation for momentum", 850),
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.2", "CAPS.PhysicalSciences.Gr11.PHY.4",
            "Force concepts needed for work-energy theorem", 850),
        // Chemistry: organic → kinetics → equilibrium → acids → electrochem
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.2", "CAPS.PhysicalSciences.Gr11.CHM.3",
            "Reaction rates before equilibrium", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.3", "CAPS.PhysicalSciences.Gr11.CHM.4",
            "Equilibrium concepts before acid-base equilibria", 800),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.4", "CAPS.PhysicalSciences.Gr11.CHM.5",
            "Acid-base redox reactions before electrochemistry", 800),
        // Cross-grade: Gr10 → Gr11
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.1", "CAPS.PhysicalSciences.Gr11.PHY.1",
            "Grade 10 kinematics before Grade 11 2D vectors", 900),
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.2", "CAPS.PhysicalSciences.Gr11.PHY.4",
            "Grade 10 energy concepts before Grade 11 work-energy theorem", 850),
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.3", "CAPS.PhysicalSciences.Gr11.PHY.5",
            "Grade 10 waves before Grade 11 optics", 850),
        caps_edge("CAPS.PhysicalSciences.Gr10.PHY.7", "CAPS.PhysicalSciences.Gr11.PHY.6",
            "Grade 10 circuits before Grade 11 electrodynamics", 850),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.2", "CAPS.PhysicalSciences.Gr11.CHM.1",
            "Grade 10 bonding before Grade 11 organic chemistry", 850),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.4", "CAPS.PhysicalSciences.Gr11.CHM.2",
            "Grade 10 stoichiometry before Grade 11 reaction rates", 850),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.5", "CAPS.PhysicalSciences.Gr11.CHM.4",
            "Grade 10 acid-base basics before Grade 11 acids and bases", 800),
        caps_edge("CAPS.PhysicalSciences.Gr10.CHM.5", "CAPS.PhysicalSciences.Gr11.CHM.5",
            "Grade 10 redox basics before Grade 11 electrochemistry", 800),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Physical Sciences Grade 11",
            11,
            "Physical Sciences",
            vec!["Physics — Mechanics".into(), "Physics — Waves".into(),
                 "Physics — Electricity".into(), "Chemistry — Organic".into(),
                 "Chemistry — Kinetics".into(), "Chemistry — Equilibrium".into(),
                 "Chemistry — Acids & Bases".into(), "Chemistry — Electrochemistry".into()],
            node_count,
            "Curated from CAPS Physical Sciences FET Phase (DBE, 2011 amended 2022). Grade 11 intermediate — Physics and Chemistry.",
        ),
        nodes,
        edges,
    }
}

// ============================================================
// Grade 12 Physical Sciences (Matric)
// ============================================================

fn build_physics_grade_12() -> CurriculumDocument {
    let topics = vec![
        // Paper 1 — Physics
        TopicDef {
            code: "P1.MECH1", title: "Mechanics — Momentum and Impulse",
            description: "Conservation of linear momentum (Σpi = Σpf). Impulse (FnetΔt = mΔv = Δp). Elastic and inelastic collisions in one dimension. Newton's second law in terms of momentum. Application to safety: crumple zones, airbags, seatbelts",
            subdomain: "Paper 1 — Mechanics", node_type: "Skill", bloom: "Apply", hours: 10,
            tags: &["physics", "momentum", "impulse", "collisions", "safety"],
            exam_marks: Some((1, 12, 150)),
        },
        TopicDef {
            code: "P1.MECH2", title: "Mechanics — Vertical Projectile Motion",
            description: "Motion in one dimension under gravity. Equations of motion with g = 9.8 m/s². Free-fall, bouncing ball, projectile launched vertically. Graphs of motion (x-t, v-t). Reaction time experiments",
            subdomain: "Paper 1 — Mechanics", node_type: "Skill", bloom: "Apply", hours: 8,
            tags: &["physics", "projectile", "free-fall", "gravity", "kinematics"],
            exam_marks: Some((1, 15, 150)),
        },
        TopicDef {
            code: "P1.MECH3", title: "Mechanics — Work, Energy, and Power",
            description: "Work (W = FΔx cos θ, positive/negative/zero work). Work-energy theorem (Wnet = ΔEk). Conservation of energy including non-conservative forces (Wnc = ΔEk + ΔEp). Power (P = W/Δt = Fv). Efficiency",
            subdomain: "Paper 1 — Mechanics", node_type: "Skill", bloom: "Analyze", hours: 10,
            tags: &["physics", "work", "energy", "power", "efficiency", "conservation"],
            exam_marks: Some((1, 30, 150)),
        },
        TopicDef {
            code: "P1.DOP", title: "Waves — Doppler Effect",
            description: "The Doppler effect with sound. Observer frequency formula: fL = fs × (v ± vL)/(v ± vs). Applications: radar, medical ultrasound, astronomy (red shift). Supersonic motion and shock waves (Mach cone)",
            subdomain: "Paper 1 — Waves", node_type: "Concept", bloom: "Apply", hours: 8,
            tags: &["physics", "doppler-effect", "waves", "sound", "red-shift", "ultrasound"],
            exam_marks: Some((1, 10, 150)),
        },
        TopicDef {
            code: "P1.ELEC1", title: "Electricity — Electric Circuits",
            description: "Internal resistance and EMF (ε = I(R + r)). Series-parallel networks. Power dissipation. Kirchhoff's voltage and current laws. Wheatstone bridge (enrichment). Motor effect",
            subdomain: "Paper 1 — Electricity", node_type: "Skill", bloom: "Apply", hours: 12,
            tags: &["physics", "circuits", "internal-resistance", "emf", "kirchhoff", "power"],
            exam_marks: Some((1, 35, 150)),
        },
        TopicDef {
            code: "P1.ELEC2", title: "Electricity — Electrodynamics",
            description: "Generators (AC and DC principles). Faraday's law of electromagnetic induction (ε = −NΔΦ/Δt). Motors (DC motor operation). Alternating current: Vrms, Irms, Pavg. Transformers (Vs/Vp = Ns/Np). National electricity grid",
            subdomain: "Paper 1 — Electricity", node_type: "Concept", bloom: "Understand", hours: 10,
            tags: &["physics", "electrodynamics", "generators", "faraday", "transformers", "alternating-current"],
            exam_marks: Some((1, 20, 150)),
        },
        TopicDef {
            code: "P1.MOD1", title: "Modern Physics — Photoelectric Effect",
            description: "Dual nature of light. Photoelectric effect: threshold frequency (f₀), work function (W₀ = hf₀), Einstein's equation (E = W₀ + Ekmax = hf). Particle nature of light. Photon energy (E = hf = hc/λ). Photocell applications",
            subdomain: "Paper 1 — Modern Physics", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["physics", "photoelectric-effect", "photons", "work-function", "planck", "dual-nature"],
            exam_marks: Some((1, 18, 150)),
        },
        TopicDef {
            code: "P1.MOD2", title: "Modern Physics — Emission and Absorption Spectra",
            description: "Atomic energy levels. Emission spectra (continuous, line/bright). Absorption spectra (dark lines). Bohr model. E = hf relationship for photon emission/absorption. Applications: stellar composition, flame tests",
            subdomain: "Paper 1 — Modern Physics", node_type: "Concept", bloom: "Analyze", hours: 6,
            tags: &["physics", "spectra", "emission", "absorption", "bohr-model", "energy-levels"],
            exam_marks: Some((1, 10, 150)),
        },

        // Paper 2 — Chemistry
        TopicDef {
            code: "P2.ORG1", title: "Organic Chemistry — Nomenclature and Reactions",
            description: "IUPAC naming: alkanes, alkenes, alkynes, alcohols (primary/secondary/tertiary), carboxylic acids, esters, aldehydes, ketones, amines, amides, alkyl halides. Structural, condensed, and molecular formulae. Functional group identification. Isomers (structural, functional group, positional, chain)",
            subdomain: "Paper 2 — Organic Chemistry", node_type: "Concept", bloom: "Understand", hours: 12,
            tags: &["chemistry", "organic", "nomenclature", "iupac", "isomers", "functional-groups"],
            exam_marks: Some((2, 15, 150)),
        },
        TopicDef {
            code: "P2.ORG2", title: "Organic Chemistry — Reactions and Applications",
            description: "Substitution (haloalkanes + OH⁻). Elimination (dehydrohalogenation). Addition (alkenes: HX, X₂, H₂O, H₂). Esterification and hydrolysis. Combustion. Oxidation of alcohols. Polymerisation (addition and condensation). Plastics and polymers. Petrochemicals and the refining process (fractional distillation, cracking)",
            subdomain: "Paper 2 — Organic Chemistry", node_type: "Skill", bloom: "Apply", hours: 14,
            tags: &["chemistry", "organic-reactions", "substitution", "elimination", "addition", "polymers", "esterification"],
            exam_marks: Some((2, 25, 150)),
        },
        TopicDef {
            code: "P2.RATE", title: "Chemical Change — Reaction Rates",
            description: "Qualitative and quantitative rate. Rate = −Δ[reactant]/Δt = +Δ[product]/Δt. Factors: concentration, temperature, surface area, catalysts. Collision theory. Activation energy and energy profiles. Maxwell-Boltzmann distribution. Mechanism of catalysis",
            subdomain: "Paper 2 — Reaction Rates", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["chemistry", "reaction-rates", "kinetics", "activation-energy", "catalysts", "maxwell-boltzmann"],
            exam_marks: Some((2, 20, 150)),
        },
        TopicDef {
            code: "P2.EQUIL", title: "Chemical Equilibrium",
            description: "Reversible reactions and dynamic equilibrium. Le Chatelier's principle applied to concentration, temperature, pressure changes. Equilibrium constant Kc and its interpretation. Effect of catalyst on equilibrium (no shift, faster attainment). Industrial applications: Haber process, Contact process",
            subdomain: "Paper 2 — Equilibrium", node_type: "Concept", bloom: "Analyze", hours: 10,
            tags: &["chemistry", "equilibrium", "le-chatelier", "haber-process", "contact-process", "kc"],
            exam_marks: Some((2, 25, 150)),
        },
        TopicDef {
            code: "P2.ACID", title: "Acids and Bases",
            description: "Brønsted-Lowry model. Conjugate acid-base pairs. Strong vs weak acids/bases (degree of ionisation). Ka, Kb, Kw. pH calculations: pH = −log[H₃O⁺]. Titration curves (strong/strong, strong/weak). Indicators and equivalence point. Hydrolysis of salts. Buffer solutions (conceptual)",
            subdomain: "Paper 2 — Acids & Bases", node_type: "Skill", bloom: "Analyze", hours: 12,
            tags: &["chemistry", "acids", "bases", "ph", "titration-curves", "ka", "hydrolysis", "buffers"],
            exam_marks: Some((2, 30, 150)),
        },
        TopicDef {
            code: "P2.ECHEM", title: "Electrochemistry — Galvanic and Electrolytic Cells",
            description: "Galvanic cells: spontaneous redox reactions, cell notation, standard hydrogen electrode, EMF calculation (E°cell = E°cathode − E°anode). Electrolytic cells: non-spontaneous, external voltage, anode/cathode identification. Faraday's laws. Applications: electroplating, refining of copper, chlor-alkali process. Comparison of galvanic and electrolytic cells",
            subdomain: "Paper 2 — Electrochemistry", node_type: "Skill", bloom: "Analyze", hours: 10,
            tags: &["chemistry", "electrochemistry", "galvanic", "electrolytic", "emf", "faraday", "electrode-potential"],
            exam_marks: Some((2, 20, 150)),
        },
        TopicDef {
            code: "P2.FERT", title: "Chemical Industry — Fertilisers",
            description: "N, P, K in fertilisers. Haber process (NH₃). Ostwald process (HNO₃). Contact process (H₂SO₄). Superphosphate production. NPK ratios. Environmental impact: eutrophication, soil acidification",
            subdomain: "Paper 2 — Chemical Industry", node_type: "Concept", bloom: "Understand", hours: 6,
            tags: &["chemistry", "fertilisers", "haber-process", "ostwald-process", "eutrophication", "industry"],
            exam_marks: Some((2, 15, 150)),
        },
    ];

    let nodes: Vec<CurriculumNode> = topics.iter().map(|t| caps_node(12, "Physical Sciences", t)).collect();
    let edges = vec![
        // Paper 1 — Physics internal
        caps_edge("CAPS.PhysicalSciences.Gr12.P1.MECH1", "CAPS.PhysicalSciences.Gr12.P1.MECH2",
            "Momentum concepts before projectile motion applications", 750),
        caps_edge("CAPS.PhysicalSciences.Gr12.P1.MECH2", "CAPS.PhysicalSciences.Gr12.P1.MECH3",
            "Kinematics before work-energy theorem", 800),
        caps_edge("CAPS.PhysicalSciences.Gr12.P1.ELEC1", "CAPS.PhysicalSciences.Gr12.P1.ELEC2",
            "DC circuits before electrodynamics", 900),
        caps_edge("CAPS.PhysicalSciences.Gr12.P1.MOD1", "CAPS.PhysicalSciences.Gr12.P1.MOD2",
            "Photoelectric effect before emission spectra", 850),

        // Paper 2 — Chemistry internal
        caps_edge("CAPS.PhysicalSciences.Gr12.P2.ORG1", "CAPS.PhysicalSciences.Gr12.P2.ORG2",
            "Nomenclature before organic reactions", 950),
        caps_edge("CAPS.PhysicalSciences.Gr12.P2.RATE", "CAPS.PhysicalSciences.Gr12.P2.EQUIL",
            "Reaction rates before equilibrium", 900),
        caps_edge("CAPS.PhysicalSciences.Gr12.P2.EQUIL", "CAPS.PhysicalSciences.Gr12.P2.ACID",
            "Equilibrium concepts before acid-base equilibria", 850),
        caps_edge("CAPS.PhysicalSciences.Gr12.P2.ACID", "CAPS.PhysicalSciences.Gr12.P2.ECHEM",
            "Acid-base redox concepts before electrochemistry", 800),
        caps_edge("CAPS.PhysicalSciences.Gr12.P2.EQUIL", "CAPS.PhysicalSciences.Gr12.P2.FERT",
            "Equilibrium needed for industrial processes (Haber, Contact)", 750),

        // Cross-grade: Gr11 → Gr12
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.3", "CAPS.PhysicalSciences.Gr12.P1.MECH1",
            "Grade 11 momentum before Grade 12 momentum", 950),
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.4", "CAPS.PhysicalSciences.Gr12.P1.MECH3",
            "Grade 11 work-energy before Grade 12 work-energy-power", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.5", "CAPS.PhysicalSciences.Gr12.P1.DOP",
            "Grade 11 optics/waves before Grade 12 Doppler effect", 800),
        caps_edge("CAPS.PhysicalSciences.Gr11.PHY.6", "CAPS.PhysicalSciences.Gr12.P1.ELEC2",
            "Grade 11 electrodynamics before Grade 12 generators/motors", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.1", "CAPS.PhysicalSciences.Gr12.P2.ORG1",
            "Grade 11 organic introduction before Grade 12 organic chemistry", 950),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.2", "CAPS.PhysicalSciences.Gr12.P2.RATE",
            "Grade 11 reaction rates before Grade 12 rates", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.3", "CAPS.PhysicalSciences.Gr12.P2.EQUIL",
            "Grade 11 equilibrium before Grade 12 equilibrium", 900),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.4", "CAPS.PhysicalSciences.Gr12.P2.ACID",
            "Grade 11 acids and bases before Grade 12 acids and bases", 950),
        caps_edge("CAPS.PhysicalSciences.Gr11.CHM.5", "CAPS.PhysicalSciences.Gr12.P2.ECHEM",
            "Grade 11 electrochemistry before Grade 12 electrochemistry", 950),
    ];

    let node_count = nodes.len();

    CurriculumDocument {
        metadata: caps_metadata(
            "CAPS Physical Sciences Grade 12 (Matric)",
            12,
            "Physical Sciences",
            vec![
                "Paper 1 — Mechanics".into(), "Paper 1 — Waves".into(),
                "Paper 1 — Electricity".into(), "Paper 1 — Modern Physics".into(),
                "Paper 2 — Organic Chemistry".into(), "Paper 2 — Reaction Rates".into(),
                "Paper 2 — Equilibrium".into(), "Paper 2 — Acids & Bases".into(),
                "Paper 2 — Electrochemistry".into(), "Paper 2 — Chemical Industry".into(),
            ],
            node_count,
            "Curated from CAPS Physical Sciences FET Phase (DBE, 2011 amended 2022). NSC Matric examination syllabus — Paper 1 Physics (150 marks, 3 hours) and Paper 2 Chemistry (150 marks, 3 hours).",
        ),
        nodes,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caps_list_available() {
        let source = CapsSource::new();
        let entries = source.list_available().unwrap();
        assert_eq!(entries.len(), 16); // 8 Gr1-8 + 2 Gr9 + 6 Gr10-12
        assert!(entries.iter().any(|e| e.id == "caps-math-12"));
        assert!(entries.iter().any(|e| e.id == "caps-physics-12"));
    }

    #[test]
    fn test_caps_math_grade_10() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-math-10").unwrap();
        assert_eq!(doc.nodes.len(), 13);
        assert!(!doc.edges.is_empty());
        assert!(doc.nodes.iter().any(|n| n.title.contains("Algebraic Expressions")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Differential Calculus") == false));
        // All nodes should have CAPS framework
        for node in &doc.nodes {
            assert!(node.academic_standards[0].framework.contains("CAPS"));
            assert!(node.tags.contains(&"south-africa".to_string()));
        }
    }

    #[test]
    fn test_caps_math_grade_11() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-math-11").unwrap();
        assert_eq!(doc.nodes.len(), 13);
        assert!(doc.nodes.iter().any(|n| n.title.contains("Compound and Double Angles")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Inverse Functions")));
        // Check cross-grade edges exist
        let cross_grade = doc.edges.iter().filter(|e| e.from.contains("Gr10")).count();
        assert!(cross_grade >= 6, "Should have Gr10→Gr11 prerequisite edges");
    }

    #[test]
    fn test_caps_math_grade_12() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-math-12").unwrap();
        assert_eq!(doc.nodes.len(), 10);
        // Paper 1 topics
        assert!(doc.nodes.iter().any(|n| n.title.contains("Differential Calculus")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Counting Principles")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Annuities") || n.title.contains("Finance")));
        // Paper 2 topics
        assert!(doc.nodes.iter().any(|n| n.title.contains("Euclidean Geometry")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Trigonometry")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Statistics")));
        // Check cross-grade edges exist
        let cross_grade = doc.edges.iter().filter(|e| e.from.contains("Gr11")).count();
        assert!(cross_grade >= 6, "Should have Gr11→Gr12 prerequisite edges");
    }

    #[test]
    fn test_caps_physics_grade_10() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-physics-10").unwrap();
        assert_eq!(doc.nodes.len(), 13);
        // Physics topics
        assert!(doc.nodes.iter().any(|n| n.title.contains("Motion in One Dimension")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Electric Circuits")));
        // Chemistry topics
        assert!(doc.nodes.iter().any(|n| n.title.contains("Atomic Structure")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Stoichiometry")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Chemical Bonding")));
    }

    #[test]
    fn test_caps_physics_grade_11() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-physics-11").unwrap();
        assert_eq!(doc.nodes.len(), 11);
        assert!(doc.nodes.iter().any(|n| n.title.contains("Newton's Laws")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Geometrical Optics")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Organic Molecules")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Electrochemistry")));
    }

    #[test]
    fn test_caps_physics_grade_12() {
        let source = CapsSource::new();
        let doc = source.fetch("caps-physics-12").unwrap();
        assert_eq!(doc.nodes.len(), 15);
        // Paper 1 — Physics
        assert!(doc.nodes.iter().any(|n| n.title.contains("Photoelectric Effect")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Doppler Effect")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Electrodynamics")));
        // Paper 2 — Chemistry
        assert!(doc.nodes.iter().any(|n| n.title.contains("Organic Chemistry")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Acids and Bases")));
        assert!(doc.nodes.iter().any(|n| n.title.contains("Fertilisers")));
    }

    #[test]
    fn test_caps_all_nodes_have_resources() {
        let source = CapsSource::new();
        for entry in source.list_available().unwrap() {
            let doc = source.fetch(&entry.id).unwrap();
            for node in &doc.nodes {
                assert!(
                    !node.supplementary_resources.is_empty(),
                    "Node {} should have supplementary resources",
                    node.id
                );
                // Math nodes should have Khan Academy
                if node.subject_area == "Mathematics" {
                    assert!(
                        node.supplementary_resources.iter().any(|r| r.title.contains("Khan Academy")),
                        "Math node {} should have Khan Academy resource",
                        node.id
                    );
                }
            }
        }
    }

    #[test]
    fn test_caps_prerequisite_chain_integrity() {
        let source = CapsSource::new();
        // Load all 3 grades of math
        let gr10 = source.fetch("caps-math-10").unwrap();
        let gr11 = source.fetch("caps-math-11").unwrap();
        let gr12 = source.fetch("caps-math-12").unwrap();

        // Collect all node IDs
        let mut all_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for doc in [&gr10, &gr11, &gr12] {
            for node in &doc.nodes {
                all_ids.insert(node.id.clone());
            }
        }

        // All edge targets should reference real nodes
        for doc in [&gr10, &gr11, &gr12] {
            for edge in &doc.edges {
                assert!(
                    all_ids.contains(&edge.from),
                    "Edge source '{}' not found in any grade",
                    edge.from
                );
                assert!(
                    all_ids.contains(&edge.to),
                    "Edge target '{}' not found in any grade",
                    edge.to
                );
            }
        }
    }

    #[test]
    fn test_caps_total_node_count() {
        let source = CapsSource::new();
        let mut total = 0;
        for entry in source.list_available().unwrap() {
            let doc = source.fetch(&entry.id).unwrap();
            total += doc.nodes.len();
        }
        // Math: 13 + 13 + 10 = 36
        // Physics: 13 + 11 + 15 = 39
        // Gr9: 12 math + 7 natsci = 19
        // Gr10: 13 math + 13 physics = 26
        // Gr11: 13 math + 11 physics = 24
        // Gr12: 10 math + 15 physics = 25
        // Total: 94
        // Gr1-8: 62 nodes, Gr9-12: 94 nodes = 156 total
        assert!(total >= 150, "Total CAPS nodes should be at least 150, got {}", total);
    }

    #[test]
    fn test_caps_not_found_error() {
        let source = CapsSource::new();
        assert!(source.fetch("caps-math-13").is_err());
        assert!(source.fetch("nonexistent").is_err());
    }

    #[test]
    fn test_caps_node_ids_unique() {
        let source = CapsSource::new();
        let mut all_ids = std::collections::HashSet::new();
        for entry in source.list_available().unwrap() {
            let doc = source.fetch(&entry.id).unwrap();
            for node in &doc.nodes {
                assert!(
                    all_ids.insert(node.id.clone()),
                    "Duplicate node ID: {}",
                    node.id
                );
            }
        }
    }
}
