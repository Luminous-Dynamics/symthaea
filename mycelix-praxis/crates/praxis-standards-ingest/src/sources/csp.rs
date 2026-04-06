// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Common Standards Project source provider (K-12 standards).
//!
//! Wraps the existing [`CspClient`](crate::client::CspClient) as a
//! [`CurriculumSource`] implementation.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::client::CspClient;
use crate::converter::{self, CurriculumDocument};

/// K-12 standards from the Common Standards Project API.
pub struct CspSource {
    client: CspClient,
    jurisdiction_id: String,
}

impl CspSource {
    pub fn new(jurisdiction_id: String) -> Result<Self, SourceError> {
        let client = CspClient::new().map_err(|e| SourceError::Other(e.to_string()))?;
        Ok(Self {
            client,
            jurisdiction_id,
        })
    }

    /// List standard sets, optionally filtered by subject and grade.
    pub async fn list_sets(
        &self,
        subject: Option<&str>,
        grade: Option<&str>,
    ) -> Result<Vec<SourceEntry>, SourceError> {
        let detail = self
            .client
            .get_jurisdiction(&self.jurisdiction_id)
            .await
            .map_err(|e| SourceError::Other(e.to_string()))?;

        let entries = detail
            .standard_sets
            .iter()
            .filter(|s| {
                if let Some(subj) = subject {
                    if !s.subject.to_lowercase().contains(&subj.to_lowercase()) {
                        return false;
                    }
                }
                if let Some(g) = grade {
                    if !s.education_levels.iter().any(|l| l == g) {
                        return false;
                    }
                }
                true
            })
            .map(|s| SourceEntry {
                id: s.id.clone(),
                title: s.title.clone(),
                subject: s.subject.clone(),
                level: "K-12".to_string(),
                description: format!("Grades: {}", s.education_levels.join(", ")),
            })
            .collect();

        Ok(entries)
    }

    /// Fetch a standard set and convert to curriculum document.
    pub async fn fetch_set(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        let set = self
            .client
            .get_standard_set(id)
            .await
            .map_err(|e| SourceError::Other(e.to_string()))?;
        Ok(converter::convert_standard_set(&set))
    }
}

impl CurriculumSource for CspSource {
    fn name(&self) -> &str {
        "Common Standards Project (K-12)"
    }

    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        // Synchronous wrapper — callers should use list_sets() for async
        Err(SourceError::Other(
            "Use list_sets() for async CSP queries".to_string(),
        ))
    }

    fn fetch(&self, _id: &str) -> Result<CurriculumDocument, SourceError> {
        // Synchronous wrapper — callers should use fetch_set() for async
        Err(SourceError::Other(
            "Use fetch_set() for async CSP queries".to_string(),
        ))
    }
}
