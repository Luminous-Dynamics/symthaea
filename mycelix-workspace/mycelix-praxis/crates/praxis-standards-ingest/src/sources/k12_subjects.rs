// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Additional K-12 subject sources that fill gaps not covered by CSP/NGSS.
//!
//! - **C3 Framework** (NCSS): Social Studies — Civics, Economics, Geography, History
//! - **ISTE Standards**: Technology — 7 student competency standards
//! - **National Core Arts Standards**: Arts — Dance, Media Arts, Music, Theatre, Visual Arts
//! - **SHAPE America**: Physical Education — 5 national PE standards
//! - **CEFR**: Foreign Language proficiency levels A1-C2
//!
//! All sources are embedded (curated from published frameworks) since these
//! don't have public APIs like the Common Standards Project.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
};

// ============================================================
// C3 Framework — Social Studies (NCSS)
// ============================================================

/// C3 Social Studies source — Civics, Economics, Geography, History.
pub struct C3Source;

impl C3Source {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for C3Source {
    fn name(&self) -> &str { "C3 Framework (Social Studies)" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "c3-k2".into(), title: "C3 Social Studies K-2".into(), subject: "Social Studies".into(), level: "K-12".into(), description: "4 disciplines, K-2 grade band".into() },
            SourceEntry { id: "c3-35".into(), title: "C3 Social Studies 3-5".into(), subject: "Social Studies".into(), level: "K-12".into(), description: "4 disciplines, 3-5 grade band".into() },
            SourceEntry { id: "c3-68".into(), title: "C3 Social Studies 6-8".into(), subject: "Social Studies".into(), level: "K-12".into(), description: "4 disciplines, 6-8 grade band".into() },
            SourceEntry { id: "c3-912".into(), title: "C3 Social Studies 9-12".into(), subject: "Social Studies".into(), level: "K-12".into(), description: "4 disciplines, 9-12 grade band".into() },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let (grade_level, grades_label) = match id {
            "c3-k2" => ("Kindergarten", "K-2"),
            "c3-35" => ("Grade3", "3-5"),
            "c3-68" => ("Grade6", "6-8"),
            "c3-912" => ("Grade9", "9-12"),
            _ => return Err(SourceError::NotFound(id.to_string())),
        };

        let disciplines = [
            ("Civics", vec![
                ("Civic ideals and practices", "Understand the foundations of government and civic participation"),
                ("Rights and responsibilities", "Analyze rights, responsibilities, and roles of citizens"),
                ("Government structures", "Examine how governments are organized and function"),
            ]),
            ("Economics", vec![
                ("Economic decision making", "Apply economic reasoning to personal and societal decisions"),
                ("Markets and exchange", "Understand how markets coordinate economic activity"),
                ("National and global economy", "Analyze economic systems and their interconnections"),
            ]),
            ("Geography", vec![
                ("Spatial thinking", "Use geographic tools to analyze spatial patterns"),
                ("Human-environment interaction", "Examine relationships between people and their environment"),
                ("Movement and regions", "Understand migration, trade, and cultural diffusion"),
            ]),
            ("History", vec![
                ("Historical thinking", "Analyze change and continuity over time"),
                ("Historical sources", "Evaluate primary and secondary sources for evidence"),
                ("Causation and context", "Explain historical events through multiple perspectives"),
            ]),
        ];

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (disc, topics) in &disciplines {
            for (i, (title, desc)) in topics.iter().enumerate() {
                let node_id = format!("C3.{}.{}.{}", disc, grades_label, i + 1);
                nodes.push(CurriculumNode {
                    id: node_id.clone(),
                    title: title.to_string(),
                    description: desc.to_string(),
                    node_type: if i == 0 { "Concept" } else { "Skill" }.into(),
                    difficulty: match id { "c3-k2" => "Beginner", "c3-35" => "Beginner", "c3-68" => "Intermediate", _ => "Advanced" }.into(),
                    domain: "Social Studies".into(),
                    subdomain: disc.to_string(),
                    tags: vec![disc.to_lowercase(), "social-studies".into(), "c3-framework".into()],
                    estimated_hours: match id { "c3-k2" => 6, "c3-35" => 8, "c3-68" => 10, _ => 12 },
                    grade_levels: vec![grade_level.to_string()],
                    bloom_level: match id { "c3-k2" => "Understand", "c3-35" => "Apply", "c3-68" => "Analyze", _ => "Evaluate" }.into(),
                    subject_area: "Social Studies".into(),
                    academic_standards: vec![AcademicStandardRef {
                        framework: "C3 Framework".into(),
                        code: node_id.clone(),
                        description: desc.to_string(),
                        grade_level: grade_level.to_string(),
                    }],
                    credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
                    exam_weight: None,
                });

                if i > 0 {
                    let prev_id = format!("C3.{}.{}.{}", disc, grades_label, i);
                    edges.push(CurriculumEdge {
                        from: prev_id, to: node_id, edge_type: "Requires".into(),
                        strength_permille: 750, rationale: format!("Sequential within {} discipline", disc),
                    });
                }
            }
        }

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: format!("C3 Social Studies {grades_label}"),
                framework: "C3 Framework".into(),
                source: "https://www.socialstudies.org/standards/c3".into(),
                grade_level: grade_level.into(), subject_area: "Social Studies".into(),
                domain: "Social Studies".into(), version: "2013".into(),
                total_standards: nodes.len(), domains: vec!["Civics".into(), "Economics".into(), "Geography".into(), "History".into()],
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "Curated from C3 Framework for Social Studies State Standards (NCSS, 2013).".into(),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes,
            edges,
        })
    }
}

// ============================================================
// ISTE Standards — Technology
// ============================================================

/// ISTE student technology standards.
pub struct IsteSource;

impl IsteSource {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for IsteSource {
    fn name(&self) -> &str { "ISTE Standards for Students" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![SourceEntry {
            id: "iste-students".into(), title: "ISTE Standards for Students".into(),
            subject: "Technology".into(), level: "K-12".into(),
            description: "7 student standards with indicators".into(),
        }])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        if id != "iste-students" { return Err(SourceError::NotFound(id.to_string())); }

        let standards = [
            ("Empowered Learner", "Students leverage technology to take an active role in choosing, achieving, and demonstrating competency in their learning goals"),
            ("Digital Citizen", "Students recognize the rights, responsibilities, and opportunities of living, learning, and working in an interconnected digital world"),
            ("Knowledge Constructor", "Students critically curate a variety of resources using digital tools to construct knowledge and produce creative artifacts"),
            ("Innovative Designer", "Students use a variety of technologies within a design process to identify and solve problems by creating new, useful, or imaginative solutions"),
            ("Computational Thinker", "Students develop and employ strategies for understanding and solving problems in ways that leverage the power of technological methods"),
            ("Creative Communicator", "Students communicate clearly and express themselves creatively using platforms, tools, styles, formats, and digital media"),
            ("Global Collaborator", "Students use digital tools to broaden their perspectives and enrich their learning by collaborating with others locally and globally"),
        ];

        let nodes: Vec<CurriculumNode> = standards.iter().enumerate().map(|(i, (title, desc))| {
            CurriculumNode {
                id: format!("ISTE.S.{}", i + 1),
                title: title.to_string(), description: desc.to_string(),
                node_type: "Skill".into(), difficulty: "Intermediate".into(),
                domain: "Technology".into(), subdomain: "Digital Literacy".into(),
                tags: vec!["technology".into(), "digital-literacy".into(), "iste".into(), title.to_lowercase().replace(' ', "-")],
                estimated_hours: 20,
                grade_levels: vec!["Grade3".into(), "Grade6".into(), "Grade9".into()],
                bloom_level: if i < 3 { "Understand" } else { "Create" }.into(),
                subject_area: "Technology".into(),
                academic_standards: vec![AcademicStandardRef {
                    framework: "ISTE".into(), code: format!("ISTE.S.{}", i + 1),
                    description: desc.to_string(), grade_level: "K-12".into(),
                }],
                credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
                exam_weight: None,
            }
        }).collect();

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: "ISTE Standards for Students".into(), framework: "ISTE".into(),
                source: "https://www.iste.org/standards/iste-standards-for-students".into(),
                grade_level: "Grade6".into(), subject_area: "Technology".into(), domain: "Technology".into(),
                version: "2016".into(), total_standards: nodes.len(),
                domains: vec!["Digital Literacy".into(), "Computational Thinking".into(), "Digital Citizenship".into()],
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "ISTE Standards for Students (2016 revision). K-12 technology competencies.".into(),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes,
            edges: vec![],
        })
    }
}

// ============================================================
// National Core Arts Standards
// ============================================================

/// National Core Arts Standards source — 5 art forms.
pub struct ArtsSource;

impl ArtsSource {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for ArtsSource {
    fn name(&self) -> &str { "National Core Arts Standards" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "arts-elementary".into(), title: "Arts Standards K-5".into(), subject: "Arts".into(), level: "K-12".into(), description: "5 art forms, elementary".into() },
            SourceEntry { id: "arts-secondary".into(), title: "Arts Standards 6-12".into(), subject: "Arts".into(), level: "K-12".into(), description: "5 art forms, secondary".into() },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let (grade_level, is_secondary) = match id {
            "arts-elementary" => ("Grade3", false),
            "arts-secondary" => ("Grade9", true),
            _ => return Err(SourceError::NotFound(id.to_string())),
        };

        let art_forms = ["Dance", "Media Arts", "Music", "Theatre", "Visual Arts"];
        let processes = [
            ("Creating", "Generate and conceptualize artistic ideas and work", "Create"),
            ("Performing", "Realize artistic ideas and work through interpretation and presentation", "Apply"),
            ("Responding", "Understand and evaluate how the arts convey meaning", "Analyze"),
            ("Connecting", "Relate artistic ideas and work with personal meaning and external context", "Evaluate"),
        ];

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for art_form in &art_forms {
            for (i, (process, desc, bloom)) in processes.iter().enumerate() {
                let node_id = format!("NCAS.{}.{}.{}", art_form.replace(' ', ""), process, if is_secondary { "HS" } else { "E" });
                nodes.push(CurriculumNode {
                    id: node_id.clone(),
                    title: format!("{}: {} ({})", art_form, process, if is_secondary { "HS" } else { "Elementary" }),
                    description: format!("{} in {}. {}", process, art_form, desc),
                    node_type: if i < 2 { "Skill" } else { "Concept" }.into(),
                    difficulty: if is_secondary { "Advanced" } else { "Beginner" }.into(),
                    domain: "Arts".into(), subdomain: art_form.to_string(),
                    tags: vec!["arts".into(), art_form.to_lowercase(), process.to_lowercase()],
                    estimated_hours: if is_secondary { 15 } else { 10 },
                    grade_levels: vec![grade_level.to_string()],
                    bloom_level: bloom.to_string(),
                    subject_area: "Arts".into(),
                    academic_standards: vec![AcademicStandardRef {
                        framework: "National Core Arts Standards".into(), code: node_id.clone(),
                        description: desc.to_string(), grade_level: grade_level.to_string(),
                    }],
                    credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
                    exam_weight: None,
                });

                if i > 0 {
                    let prev = format!("NCAS.{}.{}.{}", art_form.replace(' ', ""), processes[i-1].0, if is_secondary { "HS" } else { "E" });
                    edges.push(CurriculumEdge {
                        from: prev, to: node_id, edge_type: "Recommends".into(),
                        strength_permille: 600, rationale: format!("Artistic process progression in {}", art_form),
                    });
                }
            }
        }

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: format!("National Core Arts Standards {}", if is_secondary { "6-12" } else { "K-5" }),
                framework: "National Core Arts Standards".into(),
                source: "https://www.nationalartsstandards.org/".into(),
                grade_level: grade_level.into(), subject_area: "Arts".into(), domain: "Arts".into(),
                version: "2014".into(), total_standards: nodes.len(),
                domains: art_forms.iter().map(|s| s.to_string()).collect(),
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "Curated from National Core Arts Standards (NCCAS, 2014). 5 art forms × 4 processes.".into(),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        })
    }
}

// ============================================================
// SHAPE America — Physical Education
// ============================================================

/// SHAPE America PE standards source.
pub struct PeSource;

impl PeSource {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for PeSource {
    fn name(&self) -> &str { "SHAPE America PE Standards" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "pe-elementary".into(), title: "PE Standards K-5".into(), subject: "Physical Education".into(), level: "K-12".into(), description: "5 standards, elementary".into() },
            SourceEntry { id: "pe-secondary".into(), title: "PE Standards 6-12".into(), subject: "Physical Education".into(), level: "K-12".into(), description: "5 standards, secondary".into() },
        ])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let (grade_level, is_secondary) = match id {
            "pe-elementary" => ("Grade3", false),
            "pe-secondary" => ("Grade9", true),
            _ => return Err(SourceError::NotFound(id.to_string())),
        };

        let standards = [
            ("Motor competency", "Demonstrate competency in a variety of motor skills and movement patterns", "Apply"),
            ("Movement concepts", "Apply knowledge of concepts, principles, strategies, and tactics related to movement and performance", "Analyze"),
            ("Physical activity", "Demonstrate the knowledge and skills to achieve and maintain a health-enhancing level of physical activity and fitness", "Apply"),
            ("Responsible behavior", "Exhibit responsible personal and social behavior that respects self and others in physical activity settings", "Understand"),
            ("Value of activity", "Recognize the value of physical activity for health, enjoyment, challenge, self-expression, and social interaction", "Evaluate"),
        ];

        let nodes: Vec<CurriculumNode> = standards.iter().enumerate().map(|(i, (title, desc, bloom))| {
            let suffix = if is_secondary { "HS" } else { "E" };
            CurriculumNode {
                id: format!("SHAPE.PE.{}.{}", i + 1, suffix),
                title: format!("{} ({})", title, if is_secondary { "Secondary" } else { "Elementary" }),
                description: desc.to_string(),
                node_type: if i < 3 { "Skill" } else { "Concept" }.into(),
                difficulty: if is_secondary { "Intermediate" } else { "Beginner" }.into(),
                domain: "Physical Education".into(), subdomain: "Health & Fitness".into(),
                tags: vec!["physical-education".into(), "fitness".into(), "motor-skills".into(), "health".into()],
                estimated_hours: if is_secondary { 15 } else { 10 },
                grade_levels: vec![grade_level.to_string()],
                bloom_level: bloom.to_string(),
                subject_area: "Physical Education".into(),
                academic_standards: vec![AcademicStandardRef {
                    framework: "SHAPE America".into(), code: format!("SHAPE.PE.{}", i + 1),
                    description: desc.to_string(), grade_level: grade_level.to_string(),
                }],
                credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
                exam_weight: None,
            }
        }).collect();

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: format!("SHAPE America PE Standards {}", if is_secondary { "6-12" } else { "K-5" }),
                framework: "SHAPE America".into(),
                source: "https://www.shapeamerica.org/standards/pe/".into(),
                grade_level: grade_level.into(), subject_area: "Physical Education".into(),
                domain: "Physical Education".into(), version: "2024".into(),
                total_standards: nodes.len(), domains: vec!["Motor Skills".into(), "Fitness".into(), "Personal Behavior".into()],
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "Curated from SHAPE America National Standards for K-12 Physical Education (2024 revision).".into(),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes, edges: vec![],
        })
    }
}

// ============================================================
// CEFR — Foreign Language Proficiency
// ============================================================

/// CEFR language proficiency framework.
pub struct CefrSource;

impl CefrSource {
    pub fn new() -> Self { Self }
}

impl CurriculumSource for CefrSource {
    fn name(&self) -> &str { "CEFR Language Proficiency" }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![SourceEntry {
            id: "cefr".into(), title: "CEFR Language Proficiency Framework".into(),
            subject: "Foreign Language".into(), level: "K-12".into(),
            description: "A1-C2 levels × 5 skills".into(),
        }])
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        if id != "cefr" { return Err(SourceError::NotFound(id.to_string())); }

        let levels = [
            ("A1", "Breakthrough", "Grade3", "Beginner", "Remember",
             "Can understand and use familiar everyday expressions and very basic phrases"),
            ("A2", "Waystage", "Grade5", "Beginner", "Understand",
             "Can understand sentences and frequently used expressions related to areas of most immediate relevance"),
            ("B1", "Threshold", "Grade8", "Intermediate", "Apply",
             "Can deal with most situations likely to arise while travelling in an area where the language is spoken"),
            ("B2", "Vantage", "Grade11", "Advanced", "Analyze",
             "Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers possible"),
            ("C1", "Effective Proficiency", "Undergraduate", "Expert", "Evaluate",
             "Can express ideas fluently and spontaneously without much searching for expressions; use language flexibly for social, academic, and professional purposes"),
            ("C2", "Mastery", "Graduate", "Expert", "Create",
             "Can understand with ease virtually everything heard or read; summarize information from different sources, reconstructing arguments coherently"),
        ];

        let skills = ["Listening", "Reading", "Spoken Interaction", "Spoken Production", "Writing"];

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (i, (code, name, grade, diff, bloom, desc)) in levels.iter().enumerate() {
            for skill in &skills {
                let node_id = format!("CEFR.{}.{}", code, skill.replace(' ', ""));
                nodes.push(CurriculumNode {
                    id: node_id.clone(),
                    title: format!("{} {} ({})", code, skill, name),
                    description: format!("{} — {} at {} level.", skill, desc, code),
                    node_type: "Skill".into(), difficulty: diff.to_string(),
                    domain: "Foreign Language".into(), subdomain: skill.to_string(),
                    tags: vec!["language".into(), "cefr".into(), skill.to_lowercase(), code.to_lowercase()],
                    estimated_hours: match *code { "A1" => 100, "A2" => 200, "B1" => 400, "B2" => 600, "C1" => 800, _ => 1000 } / skills.len() as u32,
                    grade_levels: vec![grade.to_string()],
                    bloom_level: bloom.to_string(),
                    subject_area: "Foreign Language".into(),
                    academic_standards: vec![AcademicStandardRef {
                        framework: "CEFR".into(), code: format!("CEFR.{}", code),
                        description: desc.to_string(), grade_level: grade.to_string(),
                    }],
                    credit_hours: None, course_level: None, cip_code: None, program_id: None, corequisites: vec![], supplementary_resources: vec![],
                    exam_weight: None,
                });

                // Prerequisite: same skill at previous level
                if i > 0 {
                    let prev_code = levels[i - 1].0;
                    let prev_id = format!("CEFR.{}.{}", prev_code, skill.replace(' ', ""));
                    edges.push(CurriculumEdge {
                        from: prev_id, to: node_id, edge_type: "Requires".into(),
                        strength_permille: 900,
                        rationale: format!("{} {} requires {} proficiency", code, skill, prev_code),
                    });
                }
            }
        }

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: "CEFR Language Proficiency Framework".into(),
                framework: "CEFR".into(),
                source: "https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions".into(),
                grade_level: "Grade5".into(), subject_area: "Foreign Language".into(),
                domain: "Foreign Language".into(), version: "2020".into(),
                total_standards: nodes.len(),
                domains: skills.iter().map(|s| s.to_string()).collect(),
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: "CEFR (Council of Europe, 2001/2020 Companion Volume). 6 levels × 5 skills = 30 competency nodes.".into(),
                academic_level: None, institution: None, cip_code: None, total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c3_all_grade_bands() {
        let source = C3Source::new();
        for entry in source.list_available().unwrap() {
            let doc = source.fetch(&entry.id).unwrap();
            assert_eq!(doc.nodes.len(), 12); // 4 disciplines × 3 topics
            assert!(!doc.edges.is_empty());
        }
    }

    #[test]
    fn test_iste_standards() {
        let source = IsteSource::new();
        let doc = source.fetch("iste-students").unwrap();
        assert_eq!(doc.nodes.len(), 7);
        assert!(doc.nodes.iter().any(|n| n.title.contains("Computational Thinker")));
    }

    #[test]
    fn test_arts_standards() {
        let source = ArtsSource::new();
        let doc = source.fetch("arts-elementary").unwrap();
        assert_eq!(doc.nodes.len(), 20); // 5 art forms × 4 processes
        assert!(!doc.edges.is_empty());
        let doc_hs = source.fetch("arts-secondary").unwrap();
        assert_eq!(doc_hs.nodes.len(), 20);
    }

    #[test]
    fn test_pe_standards() {
        let source = PeSource::new();
        let doc = source.fetch("pe-elementary").unwrap();
        assert_eq!(doc.nodes.len(), 5);
        assert!(doc.nodes.iter().any(|n| n.title.contains("Motor")));
    }

    #[test]
    fn test_cefr_framework() {
        let source = CefrSource::new();
        let doc = source.fetch("cefr").unwrap();
        assert_eq!(doc.nodes.len(), 30); // 6 levels × 5 skills
        assert_eq!(doc.edges.len(), 25); // 5 levels of prereqs × 5 skills
        // Verify progression: A1 → A2 → B1 → B2 → C1 → C2 per skill
        let listening_edges: Vec<_> = doc.edges.iter().filter(|e| e.to.contains("Listening")).collect();
        assert_eq!(listening_edges.len(), 5);
    }

    #[test]
    fn test_all_sources_listable() {
        assert!(!C3Source::new().list_available().unwrap().is_empty());
        assert!(!IsteSource::new().list_available().unwrap().is_empty());
        assert!(!ArtsSource::new().list_available().unwrap().is_empty());
        assert!(!PeSource::new().list_available().unwrap().is_empty());
        assert!(!CefrSource::new().list_available().unwrap().is_empty());
    }
}
