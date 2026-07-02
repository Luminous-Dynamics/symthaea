// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curriculum source providers.
//!
//! Each source fetches or loads academic standards/curricula from a different
//! data source and produces [`CurriculumDocument`](crate::converter::CurriculumDocument)
//! output compatible with the knowledge_zome data model.

pub mod acm;
pub mod caps;
pub mod cip;
pub mod csp;
pub mod cybersecurity;
pub mod esco;
pub mod k12_subjects;
pub mod luminous;
pub mod ocw;
pub mod phd;
pub mod philosophy;
pub mod universal;

use crate::converter::CurriculumDocument;

/// Errors from curriculum source providers.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Other(String),
}

/// Trait for curriculum data source providers.
///
/// Each implementation fetches or generates curriculum data from a specific
/// source (API, bundled dataset, template) and returns it as a
/// [`CurriculumDocument`] ready for JSON serialization.
pub trait CurriculumSource {
    /// Human-readable name of this source.
    fn name(&self) -> &str;

    /// List available programs/sets this source can provide.
    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError>;

    /// Fetch a specific curriculum by ID and return as a CurriculumDocument.
    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError>;
}

/// An entry in a source's catalog of available curricula.
#[derive(Debug, Clone)]
pub struct SourceEntry {
    /// Unique identifier within this source.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Subject area.
    pub subject: String,
    /// Academic level (e.g., "K-12", "Undergraduate", "Graduate", "Doctoral").
    pub level: String,
    /// Additional description.
    pub description: String,
}
