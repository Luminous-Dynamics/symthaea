// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PhD progression template source.
//!
//! Provides canonical doctoral program milestones that are remarkably consistent
//! across institutions and disciplines. Templates can be customized per discipline
//! via CIP code association.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{self, CurriculumDocument};
use crate::higher_ed_types::*;

/// PhD template source.
pub struct PhDSource {
    templates: Vec<PhDTemplate>,
}

/// A discipline-specific PhD template.
#[derive(Debug, Clone)]
struct PhDTemplate {
    id: String,
    discipline: String,
    cip_code: Option<String>,
    milestones: Vec<PhDMilestone>,
}

impl PhDSource {
    pub fn new() -> Self {
        Self {
            templates: embedded_phd_templates(),
        }
    }
}

impl CurriculumSource for PhDSource {
    fn name(&self) -> &str {
        "PhD Progression Templates"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(self
            .templates
            .iter()
            .map(|t| SourceEntry {
                id: t.id.clone(),
                title: format!("PhD in {}", t.discipline),
                subject: t.discipline.clone(),
                level: "Doctoral".to_string(),
                description: format!("{} milestones", t.milestones.len()),
            })
            .collect())
    }

    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let template = self
            .templates
            .iter()
            .find(|t| t.id == id)
            .ok_or_else(|| SourceError::NotFound(format!("PhD template {id}")))?;

        Ok(converter::convert_phd_template(
            &template.discipline,
            template.cip_code.as_deref(),
            &template.milestones,
        ))
    }
}

/// Canonical PhD milestones shared across disciplines.
fn core_milestones(prefix: &str) -> Vec<PhDMilestone> {
    vec![
        PhDMilestone {
            id: format!("{prefix}/coursework"),
            title: "Core Coursework".to_string(),
            description: "Complete required doctoral-level courses in the field. Typically \
                          4-6 courses covering foundational theory, research methods, \
                          and specialized topics."
                .to_string(),
            milestone_type: PhDMilestoneType::Coursework,
            typical_year: 1,
            estimated_hours: 810, // ~18 credits * 45 hours
            prerequisites: vec![],
        },
        PhDMilestone {
            id: format!("{prefix}/seminar"),
            title: "Research Seminar Participation".to_string(),
            description: "Attend and present in departmental research seminars. \
                          Develop ability to critically analyze current research."
                .to_string(),
            milestone_type: PhDMilestoneType::Coursework,
            typical_year: 1,
            estimated_hours: 180,
            prerequisites: vec![],
        },
        PhDMilestone {
            id: format!("{prefix}/teaching"),
            title: "Teaching Requirement".to_string(),
            description: "Serve as teaching assistant for undergraduate courses. \
                          Develop pedagogy skills and demonstrate mastery of \
                          foundational material."
                .to_string(),
            milestone_type: PhDMilestoneType::Teaching,
            typical_year: 2,
            estimated_hours: 360,
            prerequisites: vec![format!("{prefix}/coursework")],
        },
        PhDMilestone {
            id: format!("{prefix}/qualifying-exam"),
            title: "Qualifying Examination".to_string(),
            description: "Pass written and/or oral comprehensive examinations \
                          demonstrating breadth and depth of knowledge in the field. \
                          Advancement to candidacy upon passing."
                .to_string(),
            milestone_type: PhDMilestoneType::QualifyingExam,
            typical_year: 2,
            estimated_hours: 450, // extensive preparation
            prerequisites: vec![format!("{prefix}/coursework")],
        },
        PhDMilestone {
            id: format!("{prefix}/literature-review"),
            title: "Comprehensive Literature Review".to_string(),
            description: "Survey and synthesize existing research to identify gaps \
                          and position the dissertation within the scholarly landscape."
                .to_string(),
            milestone_type: PhDMilestoneType::Research,
            typical_year: 2,
            estimated_hours: 360,
            prerequisites: vec![format!("{prefix}/qualifying-exam")],
        },
        PhDMilestone {
            id: format!("{prefix}/proposal"),
            title: "Dissertation Proposal Defense".to_string(),
            description: "Present and defend the dissertation proposal before the \
                          committee. Define research questions, methodology, and \
                          expected contributions."
                .to_string(),
            milestone_type: PhDMilestoneType::ProposalDefense,
            typical_year: 3,
            estimated_hours: 270,
            prerequisites: vec![
                format!("{prefix}/qualifying-exam"),
                format!("{prefix}/literature-review"),
            ],
        },
        PhDMilestone {
            id: format!("{prefix}/research"),
            title: "Independent Research".to_string(),
            description: "Conduct original research contributing new knowledge to \
                          the field. Design and execute experiments, collect and \
                          analyze data, iterate on methodology."
                .to_string(),
            milestone_type: PhDMilestoneType::Research,
            typical_year: 4,
            estimated_hours: 2700, // ~1.5 years full-time
            prerequisites: vec![format!("{prefix}/proposal")],
        },
        PhDMilestone {
            id: format!("{prefix}/publication"),
            title: "Peer-Reviewed Publication".to_string(),
            description: "Publish research findings in peer-reviewed journals or \
                          top-tier conference proceedings. Demonstrate ability to \
                          contribute to the scholarly record."
                .to_string(),
            milestone_type: PhDMilestoneType::Publication,
            typical_year: 4,
            estimated_hours: 360,
            prerequisites: vec![format!("{prefix}/research")],
        },
        PhDMilestone {
            id: format!("{prefix}/dissertation-writing"),
            title: "Dissertation Writing".to_string(),
            description: "Write the doctoral dissertation: introduction, literature \
                          review, methodology, results, discussion, and conclusions. \
                          Integrate all research into a coherent scholarly work."
                .to_string(),
            milestone_type: PhDMilestoneType::DissertationWriting,
            typical_year: 5,
            estimated_hours: 900,
            prerequisites: vec![
                format!("{prefix}/research"),
                format!("{prefix}/publication"),
            ],
        },
        PhDMilestone {
            id: format!("{prefix}/defense"),
            title: "Dissertation Defense".to_string(),
            description: "Present and defend the completed dissertation before the \
                          committee and academic community. Respond to questions \
                          and demonstrate mastery of the research."
                .to_string(),
            milestone_type: PhDMilestoneType::DissertationDefense,
            typical_year: 5,
            estimated_hours: 90,
            prerequisites: vec![format!("{prefix}/dissertation-writing")],
        },
    ]
}

/// Embedded PhD templates for major disciplines.
fn embedded_phd_templates() -> Vec<PhDTemplate> {
    vec![
        PhDTemplate {
            id: "phd-cs".into(),
            discipline: "Computer Science".into(),
            cip_code: Some("11.0701".into()),
            milestones: {
                let mut m = core_milestones("phd-cs");
                // CS-specific: systems project requirement
                m.push(PhDMilestone {
                    id: "phd-cs/systems-project".into(),
                    title: "Systems Implementation Project".to_string(),
                    description: "Design and implement a significant software system \
                                  demonstrating systems-building competency."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 3,
                    estimated_hours: 450,
                    prerequisites: vec!["phd-cs/qualifying-exam".into()],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-math".into(),
            discipline: "Mathematics".into(),
            cip_code: Some("27.0101".into()),
            milestones: core_milestones("phd-math"),
        },
        PhDTemplate {
            id: "phd-physics".into(),
            discipline: "Physics".into(),
            cip_code: Some("40.0801".into()),
            milestones: {
                let mut m = core_milestones("phd-physics");
                m.push(PhDMilestone {
                    id: "phd-physics/lab-rotation".into(),
                    title: "Laboratory Rotations".to_string(),
                    description: "Rotate through 2-3 research groups to gain exposure \
                                  to different experimental and theoretical approaches."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 1,
                    estimated_hours: 360,
                    prerequisites: vec![],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-biology".into(),
            discipline: "Biological Sciences".into(),
            cip_code: Some("26.0101".into()),
            milestones: {
                let mut m = core_milestones("phd-biology");
                m.push(PhDMilestone {
                    id: "phd-biology/lab-rotation".into(),
                    title: "Laboratory Rotations".to_string(),
                    description: "Complete 3 lab rotations across different research groups \
                                  to identify dissertation advisor and lab."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 1,
                    estimated_hours: 540,
                    prerequisites: vec![],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-engineering".into(),
            discipline: "Engineering".into(),
            cip_code: Some("14.0101".into()),
            milestones: core_milestones("phd-engineering"),
        },
        PhDTemplate {
            id: "phd-psychology".into(),
            discipline: "Psychology".into(),
            cip_code: Some("42.0101".into()),
            milestones: {
                let mut m = core_milestones("phd-psychology");
                m.push(PhDMilestone {
                    id: "phd-psychology/clinical-practicum".into(),
                    title: "Clinical Practicum".to_string(),
                    description: "Complete supervised clinical training hours. \
                                  Develop assessment and intervention competencies."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Teaching, // closest match
                    typical_year: 3,
                    estimated_hours: 1200,
                    prerequisites: vec!["phd-psychology/qualifying-exam".into()],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-education".into(),
            discipline: "Education".into(),
            cip_code: Some("13.0401".into()),
            milestones: core_milestones("phd-education"),
        },
        PhDTemplate {
            id: "phd-economics".into(),
            discipline: "Economics".into(),
            cip_code: Some("45.0601".into()),
            milestones: core_milestones("phd-economics"),
        },
        // ---- Humanities & Social Sciences ----
        PhDTemplate {
            id: "phd-literature".into(),
            discipline: "Literature".into(),
            cip_code: Some("23.0101".into()),
            milestones: {
                let mut m = core_milestones("phd-literature");
                m.push(PhDMilestone {
                    id: "phd-literature/language-exam".into(),
                    title: "Foreign Language Examination".to_string(),
                    description: "Demonstrate reading proficiency in one or two foreign \
                                  languages relevant to the research area."
                        .to_string(),
                    milestone_type: PhDMilestoneType::QualifyingExam,
                    typical_year: 2,
                    estimated_hours: 180,
                    prerequisites: vec!["phd-literature/coursework".into()],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-philosophy".into(),
            discipline: "Philosophy".into(),
            cip_code: Some("38.0101".into()),
            milestones: core_milestones("phd-philosophy"),
        },
        PhDTemplate {
            id: "phd-history".into(),
            discipline: "History".into(),
            cip_code: Some("54.0101".into()),
            milestones: {
                let mut m = core_milestones("phd-history");
                m.push(PhDMilestone {
                    id: "phd-history/archival-research".into(),
                    title: "Archival Research".to_string(),
                    description: "Conduct primary source research in archives. Travel to \
                                  relevant collections and develop expertise in paleography \
                                  and source analysis."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 3,
                    estimated_hours: 720,
                    prerequisites: vec!["phd-history/proposal".into()],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-linguistics".into(),
            discipline: "Linguistics".into(),
            cip_code: Some("16.0102".into()),
            milestones: {
                let mut m = core_milestones("phd-linguistics");
                m.push(PhDMilestone {
                    id: "phd-linguistics/fieldwork".into(),
                    title: "Linguistic Fieldwork".to_string(),
                    description: "Conduct fieldwork with language communities. Collect and \
                                  analyze primary linguistic data using appropriate ethical \
                                  frameworks."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 3,
                    estimated_hours: 540,
                    prerequisites: vec!["phd-linguistics/proposal".into()],
                });
                m
            },
        },
        PhDTemplate {
            id: "phd-political-science".into(),
            discipline: "Political Science".into(),
            cip_code: Some("45.1001".into()),
            milestones: core_milestones("phd-political-science"),
        },
        PhDTemplate {
            id: "phd-sociology".into(),
            discipline: "Sociology".into(),
            cip_code: Some("45.1101".into()),
            milestones: core_milestones("phd-sociology"),
        },
        PhDTemplate {
            id: "phd-chemistry".into(),
            discipline: "Chemistry".into(),
            cip_code: Some("40.0501".into()),
            milestones: {
                let mut m = core_milestones("phd-chemistry");
                m.push(PhDMilestone {
                    id: "phd-chemistry/lab-rotation".into(),
                    title: "Laboratory Rotations".to_string(),
                    description: "Complete 2-3 rotations in different research groups \
                                  to identify dissertation advisor and research area."
                        .to_string(),
                    milestone_type: PhDMilestoneType::Research,
                    typical_year: 1,
                    estimated_hours: 360,
                    prerequisites: vec![],
                });
                m
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phd_source_creation() {
        let source = PhDSource::new();
        let available = source.list_available().unwrap();
        assert!(available.len() >= 8);
    }

    #[test]
    fn test_fetch_cs_phd() {
        let source = PhDSource::new();
        let doc = source.fetch("phd-cs").unwrap();
        assert!(doc.metadata.title.contains("Computer Science"));
        assert_eq!(doc.metadata.grade_level, "Doctoral");
        assert!(doc.metadata.cip_code.as_deref() == Some("11.0701"));
        // Should have 10 core + 1 CS-specific milestone = 11
        assert!(doc.nodes.len() >= 11);
        // Should have LeadsTo edges from prerequisites
        assert!(!doc.edges.is_empty());
    }

    #[test]
    fn test_milestone_progression() {
        let source = PhDSource::new();
        let doc = source.fetch("phd-cs").unwrap();

        // Verify the defense depends on dissertation writing
        let defense_edges: Vec<_> = doc
            .edges
            .iter()
            .filter(|e| e.to.contains("defense") && !e.to.contains("proposal"))
            .collect();
        assert!(!defense_edges.is_empty());
        assert!(defense_edges[0].from.contains("dissertation-writing"));
    }

    #[test]
    fn test_all_templates_fetchable() {
        let source = PhDSource::new();
        for entry in source.list_available().unwrap() {
            let doc = source.fetch(&entry.id).unwrap();
            assert!(!doc.nodes.is_empty());
            assert_eq!(doc.metadata.grade_level, "Doctoral");
        }
    }

    #[test]
    fn test_phd_bloom_levels_high() {
        let source = PhDSource::new();
        let doc = source.fetch("phd-cs").unwrap();
        let high_bloom: Vec<_> = doc
            .nodes
            .iter()
            .filter(|n| n.bloom_level == "Create" || n.bloom_level == "Evaluate")
            .collect();
        // Most PhD milestones should be at Create or Evaluate
        assert!(high_bloom.len() >= doc.nodes.len() / 2);
    }

    #[test]
    fn test_unknown_template_returns_error() {
        let source = PhDSource::new();
        assert!(source.fetch("phd-underwater-basket-weaving").is_err());
    }
}
