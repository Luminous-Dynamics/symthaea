// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Types matching the Common Standards Project API v1 response format.

use serde::Deserialize;

/// Wrapper for all CSP API responses: `{"data": ...}`
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    pub data: T,
}

// ---- Jurisdictions ----

/// A jurisdiction (state, organization, or school).
#[derive(Debug, Clone, Deserialize)]
pub struct Jurisdiction {
    pub id: String,
    pub title: String,
    #[serde(rename = "type")]
    pub jurisdiction_type: String,
}

/// A jurisdiction with its standard sets.
#[derive(Debug, Deserialize)]
pub struct JurisdictionDetail {
    pub id: String,
    pub title: String,
    #[serde(rename = "type")]
    pub jurisdiction_type: String,
    #[serde(rename = "standardSets", default)]
    pub standard_sets: Vec<StandardSetSummary>,
}

/// Summary of a standard set within a jurisdiction.
#[derive(Debug, Clone, Deserialize)]
pub struct StandardSetSummary {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub subject: String,
    #[serde(rename = "educationLevels", default)]
    pub education_levels: Vec<String>,
    #[serde(default)]
    pub document: Option<DocumentInfo>,
}

/// Document metadata for a standard set.
#[derive(Debug, Clone, Deserialize)]
pub struct DocumentInfo {
    pub id: Option<String>,
    pub valid: Option<String>,
    pub title: Option<String>,
    #[serde(rename = "sourceURL")]
    pub source_url: Option<String>,
    #[serde(rename = "publicationStatus")]
    pub publication_status: Option<String>,
}

// ---- Standard Sets ----

/// A full standard set with all standards.
#[derive(Debug, Deserialize)]
pub struct StandardSet {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub subject: String,
    #[serde(rename = "normalizedSubject", default)]
    pub normalized_subject: Option<String>,
    #[serde(rename = "educationLevels", default)]
    pub education_levels: Vec<String>,
    #[serde(default)]
    pub document: Option<DocumentInfo>,
    #[serde(default)]
    pub jurisdiction: Option<JurisdictionRef>,
    #[serde(default)]
    pub license: Option<LicenseInfo>,
    /// Standards as an array (requires `?standardsAsArray=true`).
    #[serde(default)]
    pub standards: Vec<Standard>,
}

/// Reference to the jurisdiction owning a standard set.
#[derive(Debug, Clone, Deserialize)]
pub struct JurisdictionRef {
    pub id: String,
    pub title: String,
}

/// License information for a standard set.
#[derive(Debug, Clone, Deserialize)]
pub struct LicenseInfo {
    pub title: Option<String>,
    #[serde(rename = "URL")]
    pub url: Option<String>,
    #[serde(rename = "rightsHolder")]
    pub rights_holder: Option<String>,
}

// ---- Individual Standards ----

/// A single academic standard from the CSP API.
#[derive(Debug, Clone, Deserialize)]
pub struct Standard {
    pub id: String,
    /// ASN identifier (e.g., "S2605001").
    #[serde(rename = "asnIdentifier", default)]
    pub asn_identifier: Option<String>,
    /// Sort position within the set.
    #[serde(default)]
    pub position: i64,
    /// Hierarchy depth: 0 = domain, 1 = cluster, 2 = standard.
    #[serde(default)]
    pub depth: u32,
    /// Human-readable code (e.g., "1.G.A.3").
    #[serde(rename = "statementNotation", default)]
    pub statement_notation: Option<String>,
    /// Label: "Domain", "Cluster", or "Standard".
    #[serde(rename = "statementLabel", default)]
    pub statement_label: Option<String>,
    /// The standard description text.
    #[serde(default)]
    pub description: String,
    /// Parent standard ID.
    #[serde(rename = "parentId", default)]
    pub parent_id: Option<String>,
    /// Ancestor IDs (from bottom up).
    #[serde(rename = "ancestorIds", default)]
    pub ancestor_ids: Vec<String>,
}

impl Standard {
    /// True if this is a leaf standard (not a domain/cluster heading).
    pub fn is_leaf_standard(&self) -> bool {
        matches!(
            self.statement_label.as_deref(),
            Some("Standard") | None
        ) && self.depth >= 2
    }

    /// Get the notation code, falling back to a truncated description.
    pub fn code(&self) -> String {
        self.statement_notation
            .clone()
            .unwrap_or_else(|| {
                let desc = &self.description;
                if desc.len() > 30 {
                    format!("{}...", &desc[..27])
                } else {
                    desc.clone()
                }
            })
    }
}
