// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ESCO (European Skills, Competences, Qualifications and Occupations) source.
//!
//! Fetches occupations and their required skills from the EU ESCO API,
//! creating career pathway nodes that connect to educational curricula.
//!
//! API: `https://ec.europa.eu/esco/api/`

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

const ESCO_API: &str = "https://ec.europa.eu/esco/api";

/// ESCO search response.
#[derive(Debug, Deserialize)]
struct EscoSearchResponse {
    total: usize,
    #[serde(rename = "_embedded")]
    embedded: EscoEmbedded,
}

#[derive(Debug, Deserialize)]
struct EscoEmbedded {
    results: Vec<EscoSearchResult>,
}

#[derive(Debug, Deserialize)]
struct EscoSearchResult {
    uri: String,
    title: String,
    #[serde(rename = "className")]
    class_name: String,
}

/// ESCO occupation detail.
#[derive(Debug, Deserialize)]
struct EscoOccupation {
    uri: String,
    title: String,
    #[serde(default)]
    description: Option<EscoDescription>,
    #[serde(rename = "_links", default)]
    links: Option<EscoLinks>,
    #[serde(rename = "hasEssentialSkill", default)]
    has_essential_skill: Vec<EscoSkillRef>,
    #[serde(rename = "hasOptionalSkill", default)]
    has_optional_skill: Vec<EscoSkillRef>,
}

#[derive(Debug, Deserialize)]
struct EscoDescription {
    #[serde(default)]
    en: Option<EscoLiteral>,
}

#[derive(Debug, Deserialize)]
struct EscoLiteral {
    #[serde(default)]
    literal: String,
}

#[derive(Debug, Deserialize)]
struct EscoLinks {
    #[serde(rename = "hasEssentialSkill", default)]
    essential_skills: Option<Vec<EscoLinkEntry>>,
    #[serde(rename = "hasOptionalSkill", default)]
    optional_skills: Option<Vec<EscoLinkEntry>>,
}

#[derive(Debug, Deserialize)]
struct EscoLinkEntry {
    uri: String,
    title: String,
}

#[derive(Debug, Clone, Deserialize)]
struct EscoSkillRef {
    uri: String,
    title: String,
}

/// ESCO career pathway source.
pub struct EscoSource {
    http: Client,
}

impl EscoSource {
    pub fn new() -> Result<Self, SourceError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("edunet-standards-ingest/0.1.0")
            .build()?;
        Ok(Self { http })
    }

    /// Search for occupations by keyword.
    pub async fn search_occupations(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String)>, SourceError> {
        let url = format!(
            "{ESCO_API}/search?text={}&type=occupation&language=en&limit={limit}",
            urlencoding::encode(query)
        );
        let resp: EscoSearchResponse = self.http.get(&url).send().await?.json().await?;
        Ok(resp
            .embedded
            .results
            .into_iter()
            .map(|r| (r.uri, r.title))
            .collect())
    }

    /// Fetch an occupation with its skills.
    pub async fn get_occupation(&self, uri: &str) -> Result<EscoOccupation, SourceError> {
        let url = format!(
            "{ESCO_API}/resource/occupation?uri={}&language=en",
            urlencoding::encode(uri)
        );
        let resp: EscoOccupation = self.http.get(&url).send().await?.json().await?;
        Ok(resp)
    }

    /// Fetch occupations for a career field and convert to curriculum.
    pub async fn fetch_career_pathway(
        &self,
        field: &str,
        limit: usize,
    ) -> Result<CurriculumDocument, SourceError> {
        let results = self.search_occupations(field, limit).await?;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (uri, title) in &results {
            match self.get_occupation(uri).await {
                Ok(occ) => {
                    let occ_id = sanitize_uri(uri);
                    let desc = occ
                        .description
                        .and_then(|d| d.en.map(|l| l.literal))
                        .unwrap_or_default();

                    // Create occupation node
                    nodes.push(CurriculumNode {
                        id: occ_id.clone(),
                        title: title.clone(),
                        description: truncate(&desc, 500),
                        node_type: "Skill".into(),
                        difficulty: "Advanced".into(),
                        domain: field.to_string(),
                        subdomain: "Career".into(),
                        tags: vec![
                            "esco".into(),
                            "occupation".into(),
                            "career".into(),
                            field.to_lowercase(),
                        ],
                        estimated_hours: 2000, // typical career preparation
                        grade_levels: vec!["Undergraduate".into()],
                        bloom_level: "Apply".into(),
                        subject_area: field.to_string(),
                        academic_standards: vec![AcademicStandardRef {
                            framework: "ESCO".into(),
                            code: uri.clone(),
                            description: title.clone(),
                            grade_level: "Professional".into(),
                        }],
                        credit_hours: None,
                        course_level: None,
                        cip_code: None,
                        program_id: None,
                        corequisites: vec![], supplementary_resources: vec![],
                        exam_weight: None,
                    });

                    // Create skill nodes from essential skills
                    let skills: Vec<EscoSkillRef> = occ.has_essential_skill.clone();
                    for (i, skill) in skills.iter().take(5).enumerate() {
                        let skill_id = sanitize_uri(&skill.uri);
                        if !nodes.iter().any(|n| n.id == skill_id) {
                            nodes.push(CurriculumNode {
                                id: skill_id.clone(),
                                title: skill.title.clone(),
                                description: format!("Essential skill for {}: {}", title, skill.title),
                                node_type: "Skill".into(),
                                difficulty: "Intermediate".into(),
                                domain: field.to_string(),
                                subdomain: "Skills".into(),
                                tags: vec!["esco".into(), "skill".into(), field.to_lowercase()],
                                estimated_hours: 200,
                                grade_levels: vec!["Undergraduate".into()],
                                bloom_level: "Apply".into(),
                                subject_area: field.to_string(),
                                academic_standards: vec![AcademicStandardRef {
                                    framework: "ESCO".into(),
                                    code: skill.uri.clone(),
                                    description: skill.title.clone(),
                                    grade_level: "Professional".into(),
                                }],
                                credit_hours: None,
                                course_level: None,
                                cip_code: None,
                                program_id: None,
                                corequisites: vec![], supplementary_resources: vec![],
                                exam_weight: None,
                            });
                        }

                        // Skill → Occupation edge
                        edges.push(CurriculumEdge {
                            from: skill_id,
                            to: occ_id.clone(),
                            edge_type: "Requires".into(),
                            strength_permille: 800,
                            rationale: format!(
                                "Essential skill '{}' required for '{}'",
                                skill.title, title
                            ),
                        });
                    }
                }
                Err(e) => {
                    eprintln!("  Warning: failed to fetch {}: {}", title, e);
                }
            }
        }

        let domains: Vec<String> = nodes
            .iter()
            .map(|n| n.subdomain.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(CurriculumDocument {
            metadata: CurriculumMetadata {
                title: format!("ESCO Career Pathway: {field}"),
                framework: "ESCO".into(),
                source: "https://ec.europa.eu/esco/".into(),
                grade_level: "Undergraduate".into(),
                subject_area: field.to_string(),
                domain: field.to_string(),
                version: "1.2".into(),
                total_standards: nodes.len(),
                domains,
                created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
                notes: format!(
                    "Career pathway from ESCO API. {} occupations with essential skills.",
                    results.len()
                ),
                academic_level: Some("Professional".into()),
                institution: None,
                cip_code: None,
                total_credits: None,
                duration_semesters: None,
            },
            nodes,
            edges,
        })
    }
}

impl CurriculumSource for EscoSource {
    fn name(&self) -> &str {
        "ESCO (European Skills, Competences, Qualifications and Occupations)"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(CAREER_FIELDS
            .iter()
            .map(|(id, name)| SourceEntry {
                id: id.to_string(),
                title: format!("ESCO: {name}"),
                subject: name.to_string(),
                level: "Professional".to_string(),
                description: "Career pathway with occupations and skills".to_string(),
            })
            .collect())
    }

    fn fetch(&self, _id: &str) -> Result<CurriculumDocument, SourceError> {
        Err(SourceError::Other(
            "Use fetch_career_pathway() for async ESCO queries".to_string(),
        ))
    }
}

/// Predefined career fields for browsing.
const CAREER_FIELDS: &[(&str, &str)] = &[
    ("software", "Software Development"),
    ("data-science", "Data Science & AI"),
    ("cybersecurity", "Cybersecurity"),
    ("engineering", "Engineering"),
    ("healthcare", "Healthcare"),
    ("education", "Education & Training"),
    ("finance", "Finance & Economics"),
    ("science", "Scientific Research"),
    ("law", "Legal Professions"),
    ("arts", "Creative Arts & Design"),
];

fn sanitize_uri(uri: &str) -> String {
    uri.rsplit('/')
        .next()
        .unwrap_or(uri)
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let at = s.char_indices().take_while(|(i, _)| *i <= max - 3).last().map(|(i, _)| i).unwrap_or(max - 3);
        format!("{}...", &s[..at])
    }
}

// Inline minimal URL encoding (avoid adding urlencoding crate dependency)
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u32),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esco_source_list_available() {
        let source = EscoSource::new().unwrap();
        let entries = source.list_available().unwrap();
        assert!(entries.len() >= 10);
        assert!(entries.iter().any(|e| e.id == "software"));
    }

    #[test]
    fn test_sanitize_uri() {
        assert_eq!(
            sanitize_uri("http://data.europa.eu/esco/occupation/abc123"),
            "abc123"
        );
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoding::encode("software developer"), "software+developer");
        assert_eq!(urlencoding::encode("C++"), "C%2B%2B");
    }
}
