// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bridge between `edunet-standards-ingest` curriculum JSON and the content
//! generation pipeline.
//!
//! This module reads the curriculum JSON files produced by the
//! `edunet-standards-ingest` CLI tool and converts them into [`StandardInput`]
//! values that the [`ContentPipeline`](crate::pipeline::ContentPipeline) can
//! consume directly.
//!
//! # Example
//!
//! ```rust,no_run
//! use edunet_content_gen::ingest::load_curriculum_file;
//! use edunet_content_gen::prelude::*;
//!
//! let standards = load_curriculum_file("examples/curriculum/math/ccss_math_grade-03.json")
//!     .expect("failed to load curriculum");
//! let pipeline = ContentPipeline::new(MockGenerator);
//! for standard in &standards {
//!     let lesson = pipeline.generate_lesson(standard).unwrap();
//!     println!("{}: {}", lesson.standard_code, lesson.title);
//! }
//! ```

use crate::pipeline::StandardInput;
use serde::Deserialize;
use std::path::Path;

/// A curriculum document as produced by `edunet-standards-ingest`.
#[derive(Debug, Deserialize)]
pub struct CurriculumDocument {
    pub metadata: CurriculumMetadata,
    pub nodes: Vec<CurriculumNode>,
    #[serde(default)]
    pub edges: Vec<CurriculumEdge>,
}

/// Metadata block from the curriculum JSON.
#[derive(Debug, Deserialize)]
pub struct CurriculumMetadata {
    pub title: String,
    pub framework: String,
    #[serde(default)]
    pub source: String,
    pub grade_level: String,
    pub subject_area: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub total_standards: usize,
    #[serde(default)]
    pub domains: Vec<String>,
}

/// A curriculum node from the JSON.
#[derive(Debug, Deserialize)]
pub struct CurriculumNode {
    pub id: String,
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub node_type: String,
    #[serde(default)]
    pub difficulty: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub subdomain: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub estimated_hours: u32,
    #[serde(default)]
    pub grade_levels: Vec<String>,
    #[serde(default)]
    pub bloom_level: String,
    #[serde(default)]
    pub subject_area: String,
    #[serde(default)]
    pub academic_standards: Vec<AcademicStandardRef>,
}

/// Academic standard reference within a node.
#[derive(Debug, Deserialize)]
pub struct AcademicStandardRef {
    pub framework: String,
    pub code: String,
    pub description: String,
    pub grade_level: String,
}

/// A prerequisite edge from the JSON.
#[derive(Debug, Deserialize)]
pub struct CurriculumEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    #[serde(default)]
    pub strength_permille: u16,
    #[serde(default)]
    pub rationale: String,
}

/// Load a curriculum JSON file and return [`StandardInput`] values ready for
/// the content generation pipeline.
pub fn load_curriculum_file<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<StandardInput>, CurriculumLoadError> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(CurriculumLoadError::Io)?;
    load_curriculum_json(&content)
}

/// Parse curriculum JSON string and return [`StandardInput`] values.
pub fn load_curriculum_json(json: &str) -> Result<Vec<StandardInput>, CurriculumLoadError> {
    let doc: CurriculumDocument =
        serde_json::from_str(json).map_err(CurriculumLoadError::Parse)?;
    Ok(convert_document(&doc))
}

/// Convert a parsed curriculum document into pipeline-ready standard inputs.
pub fn convert_document(doc: &CurriculumDocument) -> Vec<StandardInput> {
    // Build a prerequisite map: node_id -> list of prerequisite node_ids
    let mut prereq_map: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for edge in &doc.edges {
        if edge.edge_type == "Requires" {
            prereq_map
                .entry(edge.to.as_str())
                .or_default()
                .push(edge.from.as_str());
        }
    }

    doc.nodes
        .iter()
        .map(|node| {
            let prerequisites = prereq_map
                .get(node.id.as_str())
                .map(|ids| ids.iter().map(|id| id.to_string()).collect())
                .unwrap_or_default();

            let grade_level = node
                .grade_levels
                .first()
                .cloned()
                .unwrap_or_else(|| doc.metadata.grade_level.clone());

            StandardInput {
                code: node.id.clone(),
                description: node.description.clone(),
                grade_level,
                domain: if node.subdomain.is_empty() {
                    node.domain.clone()
                } else {
                    node.subdomain.clone()
                },
                prerequisites,
            }
        })
        .collect()
}

/// Errors when loading curriculum files.
#[derive(Debug, thiserror::Error)]
pub enum CurriculumLoadError {
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSON: &str = r#"{
        "metadata": {
            "title": "Test Curriculum",
            "framework": "CommonCore",
            "grade_level": "Grade3",
            "subject_area": "Mathematics"
        },
        "nodes": [
            {
                "id": "3.OA.A.1",
                "title": "Interpret products",
                "description": "Interpret products of whole numbers",
                "subdomain": "Operations & Algebraic Thinking",
                "grade_levels": ["Grade3"]
            },
            {
                "id": "3.OA.A.3",
                "title": "Solve word problems",
                "description": "Use multiplication and division to solve word problems",
                "subdomain": "Operations & Algebraic Thinking",
                "grade_levels": ["Grade3"]
            }
        ],
        "edges": [
            {
                "from": "3.OA.A.1",
                "to": "3.OA.A.3",
                "edge_type": "Requires",
                "strength_permille": 900,
                "rationale": "Must understand products before word problems"
            }
        ]
    }"#;

    #[test]
    fn test_load_curriculum_json() {
        let standards = load_curriculum_json(SAMPLE_JSON).unwrap();
        assert_eq!(standards.len(), 2);
        assert_eq!(standards[0].code, "3.OA.A.1");
        assert_eq!(standards[0].grade_level, "Grade3");
        assert_eq!(standards[0].domain, "Operations & Algebraic Thinking");
        assert!(standards[0].prerequisites.is_empty());
    }

    #[test]
    fn test_prerequisites_wired() {
        let standards = load_curriculum_json(SAMPLE_JSON).unwrap();
        // 3.OA.A.3 requires 3.OA.A.1
        let word_problems = &standards[1];
        assert_eq!(word_problems.code, "3.OA.A.3");
        assert_eq!(word_problems.prerequisites, vec!["3.OA.A.1"]);
    }

    #[test]
    fn test_grade_level_fallback() {
        let json = r#"{
            "metadata": {
                "title": "Test",
                "framework": "NGSS",
                "grade_level": "Grade5",
                "subject_area": "Science"
            },
            "nodes": [
                {
                    "id": "5-ESS1-1",
                    "title": "Star brightness",
                    "description": "Support an argument about star brightness",
                    "grade_levels": []
                }
            ],
            "edges": []
        }"#;
        let standards = load_curriculum_json(json).unwrap();
        // Falls back to metadata.grade_level when node has no grade_levels
        assert_eq!(standards[0].grade_level, "Grade5");
    }

    #[test]
    fn test_domain_falls_back_to_domain_field() {
        let json = r#"{
            "metadata": {
                "title": "Test",
                "framework": "CommonCore",
                "grade_level": "Grade3",
                "subject_area": "Mathematics"
            },
            "nodes": [
                {
                    "id": "3.OA.A.1",
                    "title": "Test",
                    "description": "Test",
                    "domain": "Mathematics",
                    "subdomain": ""
                }
            ],
            "edges": []
        }"#;
        let standards = load_curriculum_json(json).unwrap();
        assert_eq!(standards[0].domain, "Mathematics");
    }
}
