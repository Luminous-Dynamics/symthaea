// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HTTP client for the Common Standards Project API v1.

use crate::api_types::*;
use reqwest::Client;
use std::time::Duration;

const BASE_URL: &str = "https://commonstandardsproject.com/api/v1";

/// Errors from the CSP API client.
#[derive(Debug, thiserror::Error)]
pub enum CspError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API returned no data for {0}")]
    NotFound(String),
}

/// Client for the Common Standards Project API.
pub struct CspClient {
    http: Client,
}

impl CspClient {
    /// Create a new client with sensible defaults.
    pub fn new() -> Result<Self, CspError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("edunet-standards-ingest/0.1.0")
            .build()?;
        Ok(Self { http })
    }

    /// List all jurisdictions (states, organizations, schools).
    pub async fn list_jurisdictions(&self) -> Result<Vec<Jurisdiction>, CspError> {
        let url = format!("{BASE_URL}/jurisdictions");
        let resp: ApiResponse<Vec<Jurisdiction>> = self.http.get(&url).send().await?.json().await?;
        Ok(resp.data)
    }

    /// Get a single jurisdiction with its standard sets.
    pub async fn get_jurisdiction(&self, id: &str) -> Result<JurisdictionDetail, CspError> {
        let url = format!("{BASE_URL}/jurisdictions/{id}");
        let resp: ApiResponse<JurisdictionDetail> = self
            .http
            .get(&url)
            .query(&[("hideHiddenSets", "true")])
            .send()
            .await?
            .json()
            .await?;
        Ok(resp.data)
    }

    /// Fetch a standard set with all its standards as an array.
    pub async fn get_standard_set(&self, id: &str) -> Result<StandardSet, CspError> {
        let url = format!("{BASE_URL}/standard_sets/{id}");
        let resp: ApiResponse<StandardSet> = self
            .http
            .get(&url)
            .query(&[("standardsAsArray", "true")])
            .send()
            .await?
            .json()
            .await?;
        Ok(resp.data)
    }
}

impl Default for CspClient {
    fn default() -> Self {
        Self::new().expect("failed to create HTTP client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = CspClient::new();
        assert!(client.is_ok());
    }
}
