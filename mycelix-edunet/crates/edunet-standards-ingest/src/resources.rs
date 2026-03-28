// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Supplementary resource links and usage analytics.
//!
//! When generated content is insufficient, students can access external
//! resources (Khan Academy, OpenStax, Wikipedia, YouTube, etc.). Usage
//! tracking creates a feedback loop that identifies content gaps.
//!
//! # The Feedback Loop
//!
//! 1. Student views a curriculum node's generated content
//! 2. If insufficient, student clicks a supplementary resource link
//! 3. `ResourceAccess` event is recorded (node_id, resource_url, timestamp)
//! 4. Analytics aggregation reveals:
//!    - **High click-rate nodes**: our content is weak here → improve
//!    - **Frequently used sources**: this platform adds value → consider integrating
//!    - **Source preference by level**: younger students prefer videos, grad students prefer papers

use serde::{Deserialize, Serialize};

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
    pub content_type: ResourceContentType,
    /// How relevant this resource is to the node (0-100).
    pub relevance_score: u8,
    /// Standard code alignment (if resource explicitly covers this standard).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aligned_standard: Option<String>,
}

/// Platform/source of a supplementary resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceSource {
    KhanAcademy,
    OpenStax,
    MitOcw,
    Wikipedia,
    YouTube,
    Brilliant,
    Coursera,
    EdX,
    PhET,
    Desmos,
    GeoGebra,
    PubMed,
    ArXiv,
    Custom(String),
}

impl ResourceSource {
    pub fn label(&self) -> &str {
        match self {
            ResourceSource::KhanAcademy => "Khan Academy",
            ResourceSource::OpenStax => "OpenStax",
            ResourceSource::MitOcw => "MIT OpenCourseWare",
            ResourceSource::Wikipedia => "Wikipedia",
            ResourceSource::YouTube => "YouTube",
            ResourceSource::Brilliant => "Brilliant",
            ResourceSource::Coursera => "Coursera",
            ResourceSource::EdX => "edX",
            ResourceSource::PhET => "PhET Simulations",
            ResourceSource::Desmos => "Desmos",
            ResourceSource::GeoGebra => "GeoGebra",
            ResourceSource::PubMed => "PubMed",
            ResourceSource::ArXiv => "arXiv",
            ResourceSource::Custom(name) => name,
        }
    }

    /// Base URL for constructing resource links.
    pub fn base_url(&self) -> &str {
        match self {
            ResourceSource::KhanAcademy => "https://www.khanacademy.org",
            ResourceSource::OpenStax => "https://openstax.org",
            ResourceSource::MitOcw => "https://ocw.mit.edu",
            ResourceSource::Wikipedia => "https://en.wikipedia.org/wiki",
            ResourceSource::YouTube => "https://www.youtube.com",
            ResourceSource::Brilliant => "https://brilliant.org",
            ResourceSource::Coursera => "https://www.coursera.org",
            ResourceSource::EdX => "https://www.edx.org",
            ResourceSource::PhET => "https://phet.colorado.edu",
            ResourceSource::Desmos => "https://www.desmos.com",
            ResourceSource::GeoGebra => "https://www.geogebra.org",
            ResourceSource::PubMed => "https://pubmed.ncbi.nlm.nih.gov",
            ResourceSource::ArXiv => "https://arxiv.org",
            ResourceSource::Custom(_) => "",
        }
    }
}

/// Content type of a supplementary resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceContentType {
    Video,
    Textbook,
    Interactive,
    Article,
    Exercise,
    Course,
    Simulation,
    Paper,
    Tutorial,
}

/// A record of a student accessing a supplementary resource.
/// Used for analytics to identify content gaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAccess {
    /// Which curriculum node the student was on.
    pub node_id: String,
    /// Which resource they accessed.
    pub resource_url: String,
    /// Source platform.
    pub source: ResourceSource,
    /// Timestamp (ISO 8601).
    pub accessed_at: String,
    /// How long they spent (seconds, estimated from return).
    pub duration_secs: Option<u32>,
    /// Did the student find it helpful? (optional self-report)
    pub helpful: Option<bool>,
    /// Student's academic level at time of access.
    pub student_level: Option<String>,
}

/// Analytics summary for a curriculum node's supplementary resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalytics {
    /// Node ID.
    pub node_id: String,
    /// Total accesses to external resources for this node.
    pub total_accesses: u32,
    /// Unique students who accessed resources for this node.
    pub unique_students: u32,
    /// Most accessed resource URL.
    pub most_popular_resource: Option<String>,
    /// Most popular source platform.
    pub most_popular_source: Option<ResourceSource>,
    /// Access rate: what % of students viewing this node click external resources.
    /// High rate (>30%) = our content is insufficient.
    pub access_rate_pct: f32,
    /// Average helpfulness rating (0.0-1.0).
    pub avg_helpfulness: Option<f32>,
    /// Content gap severity: computed from access_rate + helpfulness.
    /// 0-100 where 100 = critical content gap.
    pub content_gap_score: u8,
}

impl ResourceAnalytics {
    /// Compute content gap severity.
    ///
    /// High access rate + low helpfulness = students need help but can't find it.
    /// High access rate + high helpfulness = external source provides something we don't.
    pub fn compute_gap_score(access_rate: f32, avg_helpfulness: Option<f32>) -> u8 {
        let base = (access_rate * 100.0).min(100.0);
        match avg_helpfulness {
            Some(h) if h < 0.5 => (base * 1.5).min(100.0) as u8, // students struggling AND external doesn't help = critical
            Some(h) if h > 0.8 => (base * 0.8) as u8, // external helps well = we should integrate it
            _ => base as u8,
        }
    }
}

/// Generate suggested supplementary resources for a curriculum node based on its metadata.
pub fn suggest_resources(
    node_id: &str,
    title: &str,
    subject_area: &str,
    grade_level: &str,
    tags: &[String],
) -> Vec<SupplementaryResource> {
    let mut resources = Vec::new();
    let subject_lower = subject_area.to_lowercase();

    // Khan Academy — great for K-12 Math and Science
    if subject_lower.contains("math") || subject_lower.contains("science") {
        let search_term = title.replace(' ', "+");
        resources.push(SupplementaryResource {
            title: format!("Khan Academy: {}", truncate(title, 50)),
            url: format!("https://www.khanacademy.org/search?referer=%2F&page_search_query={search_term}"),
            source: ResourceSource::KhanAcademy,
            content_type: ResourceContentType::Video,
            relevance_score: 85,
            aligned_standard: Some(node_id.to_string()),
        });
    }

    // Wikipedia — universal concept reference
    let wiki_term = title.split(|c: char| c == ':' || c == '(' || c == ',')
        .next().unwrap_or(title).trim().replace(' ', "_");
    resources.push(SupplementaryResource {
        title: format!("Wikipedia: {}", truncate(title, 50)),
        url: format!("https://en.wikipedia.org/wiki/{wiki_term}"),
        source: ResourceSource::Wikipedia,
        content_type: ResourceContentType::Article,
        relevance_score: 60,
        aligned_standard: None,
    });

    // YouTube — educational videos
    let yt_query = title.replace(' ', "+");
    resources.push(SupplementaryResource {
        title: format!("YouTube: {}", truncate(title, 50)),
        url: format!("https://www.youtube.com/results?search_query={yt_query}+education"),
        source: ResourceSource::YouTube,
        content_type: ResourceContentType::Video,
        relevance_score: 70,
        aligned_standard: None,
    });

    // Subject-specific resources
    if subject_lower.contains("math") {
        resources.push(SupplementaryResource {
            title: "Desmos Graphing Calculator".to_string(),
            url: "https://www.desmos.com/calculator".to_string(),
            source: ResourceSource::Desmos,
            content_type: ResourceContentType::Interactive,
            relevance_score: 75,
            aligned_standard: None,
        });
    }

    if subject_lower.contains("physics") || subject_lower.contains("chemistry") || subject_lower.contains("science") {
        let phet_query = tags.first().cloned().unwrap_or_else(|| title.to_string()).replace(' ', "+");
        resources.push(SupplementaryResource {
            title: "PhET Interactive Simulations".to_string(),
            url: format!("https://phet.colorado.edu/en/simulations/filter?q={phet_query}"),
            source: ResourceSource::PhET,
            content_type: ResourceContentType::Simulation,
            relevance_score: 80,
            aligned_standard: None,
        });
    }

    if subject_lower.contains("computer") || subject_lower.contains("software") || tags.iter().any(|t| t.contains("programming")) {
        resources.push(SupplementaryResource {
            title: format!("Brilliant: {}", truncate(title, 40)),
            url: format!("https://brilliant.org/courses/"),
            source: ResourceSource::Brilliant,
            content_type: ResourceContentType::Interactive,
            relevance_score: 75,
            aligned_standard: None,
        });
    }

    // Graduate/Doctoral level — research papers
    if grade_level == "Graduate" || grade_level == "Doctoral" {
        let arxiv_query = tags.first().cloned().unwrap_or_else(|| title.to_string()).replace(' ', "+");
        resources.push(SupplementaryResource {
            title: format!("arXiv papers on {}", truncate(title, 40)),
            url: format!("https://arxiv.org/search/?query={arxiv_query}&searchtype=all"),
            source: ResourceSource::ArXiv,
            content_type: ResourceContentType::Paper,
            relevance_score: 80,
            aligned_standard: None,
        });
    }

    // Sort by relevance
    resources.sort_by(|a, b| b.relevance_score.cmp(&a.relevance_score));
    resources
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let at = s.char_indices().take_while(|(i, _)| *i <= max - 3).last().map(|(i, _)| i).unwrap_or(max - 3);
        format!("{}...", &s[..at])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_resources_math() {
        let resources = suggest_resources(
            "3.OA.A.1", "Interpret products of whole numbers",
            "Mathematics", "Grade3", &["multiplication".into()],
        );
        assert!(resources.len() >= 3); // Khan Academy, Wikipedia, YouTube, Desmos
        assert!(resources.iter().any(|r| r.source == ResourceSource::KhanAcademy));
        assert!(resources.iter().any(|r| r.source == ResourceSource::Desmos));
    }

    #[test]
    fn test_suggest_resources_science() {
        let resources = suggest_resources(
            "NGSS-5-PS1-1", "Matter and its interactions",
            "Science", "Grade5", &["matter".into(), "energy".into()],
        );
        assert!(resources.iter().any(|r| r.source == ResourceSource::PhET));
    }

    #[test]
    fn test_suggest_resources_doctoral() {
        let resources = suggest_resources(
            "phd-cs/research", "Independent Research",
            "Computer Science", "Doctoral", &["research".into()],
        );
        assert!(resources.iter().any(|r| r.source == ResourceSource::ArXiv));
    }

    #[test]
    fn test_suggest_resources_cs() {
        let resources = suggest_resources(
            "6.006", "Introduction to Algorithms",
            "Computer Science", "Undergraduate", &["algorithms".into(), "programming".into()],
        );
        assert!(resources.iter().any(|r| r.source == ResourceSource::Brilliant));
    }

    #[test]
    fn test_gap_score_computation() {
        // High access rate + low helpfulness = critical gap
        assert!(ResourceAnalytics::compute_gap_score(0.5, Some(0.3)) > 50);
        // Low access rate = low gap
        assert!(ResourceAnalytics::compute_gap_score(0.05, None) < 10);
        // High access + high helpfulness = moderate (external adds value)
        let score = ResourceAnalytics::compute_gap_score(0.4, Some(0.9));
        assert!(score > 20 && score < 50);
    }

    #[test]
    fn test_resource_source_labels() {
        assert_eq!(ResourceSource::KhanAcademy.label(), "Khan Academy");
        assert_eq!(ResourceSource::PhET.label(), "PhET Simulations");
        assert_eq!(ResourceSource::Custom("MySource".into()).label(), "MySource");
    }

    #[test]
    fn test_resources_sorted_by_relevance() {
        let resources = suggest_resources(
            "3.OA.A.1", "Multiplication", "Mathematics", "Grade3", &[],
        );
        for window in resources.windows(2) {
            assert!(window[0].relevance_score >= window[1].relevance_score);
        }
    }
}
