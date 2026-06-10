// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! REST API types for building tax proof services.
//!
//! This module provides request/response types for REST API integration,
//! making it easy to expose tax proof functionality over HTTP.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_zk_tax::api::{ProofRequest, ProofResponse, ApiEndpoint};
//!
//! // Parse incoming request
//! let request: ProofRequest = serde_json::from_str(body)?;
//!
//! // Generate response
//! let response = ProofResponse::from_bracket_proof(proof);
//!
//! // Return JSON
//! let json = serde_json::to_string(&response)?;
//! ```

use crate::{
    error::Result, Error, FilingStatus, Jurisdiction,
    TaxBracketProof, TaxYear,
};
use serde::{Deserialize, Serialize};

// =============================================================================
// API Endpoint Definitions
// =============================================================================

/// Available API endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiEndpoint {
    /// Generate a basic tax bracket proof
    ProveBasket,
    /// Generate a range proof
    ProveRange,
    /// Generate a batch (multi-year) proof
    ProveBatch,
    /// Generate an effective rate proof
    ProveEffectiveRate,
    /// Generate a cross-jurisdiction proof
    ProveCrossJurisdiction,
    /// Verify any proof
    Verify,
    /// Get supported jurisdictions
    ListJurisdictions,
    /// Get brackets for a jurisdiction
    GetBrackets,
    /// Health check
    Health,
}

impl ApiEndpoint {
    /// Get the HTTP method for this endpoint.
    pub fn method(&self) -> &'static str {
        match self {
            ApiEndpoint::ProveBasket
            | ApiEndpoint::ProveRange
            | ApiEndpoint::ProveBatch
            | ApiEndpoint::ProveEffectiveRate
            | ApiEndpoint::ProveCrossJurisdiction
            | ApiEndpoint::Verify => "POST",
            ApiEndpoint::ListJurisdictions
            | ApiEndpoint::GetBrackets
            | ApiEndpoint::Health => "GET",
        }
    }

    /// Get the path for this endpoint.
    pub fn path(&self) -> &'static str {
        match self {
            ApiEndpoint::ProveBasket => "/api/v1/prove/bracket",
            ApiEndpoint::ProveRange => "/api/v1/prove/range",
            ApiEndpoint::ProveBatch => "/api/v1/prove/batch",
            ApiEndpoint::ProveEffectiveRate => "/api/v1/prove/effective-rate",
            ApiEndpoint::ProveCrossJurisdiction => "/api/v1/prove/cross-jurisdiction",
            ApiEndpoint::Verify => "/api/v1/verify",
            ApiEndpoint::ListJurisdictions => "/api/v1/jurisdictions",
            ApiEndpoint::GetBrackets => "/api/v1/brackets",
            ApiEndpoint::Health => "/api/v1/health",
        }
    }
}

// =============================================================================
// Request Types
// =============================================================================

/// Request to generate a basic tax bracket proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BracketProofRequest {
    /// Income in cents (e.g., 8500000 for $85,000)
    pub income_cents: u64,
    /// Jurisdiction code (e.g., "US", "UK")
    pub jurisdiction: String,
    /// Filing status (e.g., "single", "mfj")
    pub filing_status: String,
    /// Tax year
    pub tax_year: TaxYear,
    /// Whether to use dev mode (for testing)
    #[serde(default)]
    pub dev_mode: bool,
}

impl BracketProofRequest {
    /// Parse jurisdiction from string.
    pub fn parse_jurisdiction(&self) -> Result<Jurisdiction> {
        Jurisdiction::from_code(&self.jurisdiction)
            .ok_or_else(|| Error::unsupported_jurisdiction(&self.jurisdiction))
    }

    /// Parse filing status from string.
    pub fn parse_filing_status(&self) -> Result<FilingStatus> {
        match self.filing_status.to_lowercase().as_str() {
            "single" => Ok(FilingStatus::Single),
            "mfj" | "married_filing_jointly" => Ok(FilingStatus::MarriedFilingJointly),
            "mfs" | "married_filing_separately" => Ok(FilingStatus::MarriedFilingSeparately),
            "hoh" | "head_of_household" => Ok(FilingStatus::HeadOfHousehold),
            _ => Err(Error::invalid_filing_status(
                &self.filing_status,
                "unknown",
                &["single", "mfj", "mfs", "hoh"],
            )),
        }
    }

    /// Get income in dollars.
    pub fn income(&self) -> u64 {
        self.income_cents / 100
    }
}

/// Request to generate a range proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeProofRequest {
    /// Income in cents
    pub income_cents: u64,
    /// Minimum income to prove (optional)
    pub min_income_cents: Option<u64>,
    /// Maximum income to prove (optional)
    pub max_income_cents: Option<u64>,
    /// Tax year
    pub tax_year: TaxYear,
    /// Dev mode
    #[serde(default)]
    pub dev_mode: bool,
}

/// Request to generate a batch (multi-year) proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchProofRequest {
    /// List of (year, income_cents) pairs
    pub years: Vec<YearIncome>,
    /// Jurisdiction code
    pub jurisdiction: String,
    /// Filing status
    pub filing_status: String,
    /// Dev mode
    #[serde(default)]
    pub dev_mode: bool,
}

/// Year and income pair for batch requests.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YearIncome {
    pub year: TaxYear,
    pub income_cents: u64,
}

/// Request to verify a proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyRequest {
    /// The proof to verify (as JSON)
    pub proof: serde_json::Value,
    /// Expected commitment (optional, for additional validation)
    pub expected_commitment: Option<String>,
}

/// Request to get brackets for a jurisdiction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetBracketsRequest {
    /// Jurisdiction code
    pub jurisdiction: String,
    /// Tax year
    pub tax_year: TaxYear,
    /// Filing status
    pub filing_status: String,
}

// =============================================================================
// Response Types
// =============================================================================

/// Standard API response wrapper.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Whether the request succeeded
    pub success: bool,
    /// Response data (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    /// Error information (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ApiError>,
    /// Request metadata
    pub meta: ResponseMeta,
}

/// Error details for API responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiError {
    /// Error code (e.g., "E001_INCOME_OUT_OF_BOUNDS")
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ApiError {
    /// Create from SDK error.
    pub fn from_error(e: &Error) -> Self {
        Self {
            code: e.code().to_string(),
            message: e.to_string().lines().next().unwrap_or("Unknown error").to_string(),
            details: Some(e.to_string()),
        }
    }
}

/// Response metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseMeta {
    /// API version
    pub api_version: String,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Request ID (for tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl Default for ResponseMeta {
    fn default() -> Self {
        Self {
            api_version: "1.0.0".to_string(),
            processing_time_ms: 0,
            request_id: None,
        }
    }
}

impl<T> ApiResponse<T> {
    /// Create a success response.
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            meta: ResponseMeta::default(),
        }
    }

    /// Create an error response.
    pub fn error(err: ApiError) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(err),
            meta: ResponseMeta::default(),
        }
    }

    /// Set processing time.
    pub fn with_processing_time(mut self, ms: u64) -> Self {
        self.meta.processing_time_ms = ms;
        self
    }

    /// Set request ID.
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.meta.request_id = Some(id.into());
        self
    }
}

/// Bracket proof response data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BracketProofResponse {
    /// The generated proof
    pub proof: TaxBracketProof,
    /// Commitment hash (hex)
    pub commitment: String,
    /// Bracket information
    pub bracket_info: BracketInfo,
}

/// Bracket information for responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BracketInfo {
    /// Bracket index (0-based)
    pub index: u8,
    /// Marginal rate as percentage (e.g., 22.0)
    pub rate_percent: f64,
    /// Lower bound in dollars
    pub lower_bound: u64,
    /// Upper bound in dollars
    pub upper_bound: u64,
}

impl BracketProofResponse {
    /// Create from a bracket proof.
    pub fn from_proof(proof: TaxBracketProof) -> Self {
        let commitment = proof.commitment.to_hex();
        let bracket_info = BracketInfo {
            index: proof.bracket_index,
            rate_percent: proof.rate_bps as f64 / 100.0,
            lower_bound: proof.bracket_lower,
            upper_bound: proof.bracket_upper,
        };
        Self {
            proof,
            commitment,
            bracket_info,
        }
    }
}

/// Verify response data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyResponse {
    /// Whether the proof is valid
    pub valid: bool,
    /// Verification details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Jurisdiction list response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionListResponse {
    /// List of supported jurisdictions
    pub jurisdictions: Vec<JurisdictionInfo>,
}

/// Information about a jurisdiction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionInfo {
    /// Jurisdiction code (e.g., "US")
    pub code: String,
    /// Display name (e.g., "United States")
    pub name: String,
    /// Supported filing statuses
    pub filing_statuses: Vec<String>,
    /// Supported tax years
    pub tax_years: Vec<TaxYear>,
}

/// Brackets list response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BracketsResponse {
    /// Jurisdiction code
    pub jurisdiction: String,
    /// Tax year
    pub tax_year: TaxYear,
    /// Filing status
    pub filing_status: String,
    /// List of brackets
    pub brackets: Vec<BracketInfo>,
}

/// Health check response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// SDK version
    pub sdk_version: String,
    /// Supported features
    pub features: Vec<String>,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
            sdk_version: crate::VERSION.to_string(),
            features: vec![
                "bracket_proofs".to_string(),
                "range_proofs".to_string(),
                "batch_proofs".to_string(),
                "effective_rate_proofs".to_string(),
                "cross_jurisdiction_proofs".to_string(),
                "deduction_proofs".to_string(),
                "composite_proofs".to_string(),
                "proof_chains".to_string(),
                "combined_federal_state".to_string(),
                "stability_proofs".to_string(),
                "time_bound_proofs".to_string(),
                "merkle_aggregation".to_string(),
                "selective_disclosure".to_string(),
                "compression".to_string(),
            ],
        }
    }
}

// =============================================================================
// OpenAPI Spec Generation
// =============================================================================

/// Generate OpenAPI 3.0 specification for the API.
pub fn generate_openapi_spec() -> String {
    r##"{
  "openapi": "3.0.0",
  "info": {
    "title": "ZK Tax Proof API",
    "version": "1.0.0",
    "description": "Zero-knowledge tax bracket proofs for privacy-preserving compliance",
    "license": {
      "name": "MIT OR Apache-2.0"
    }
  },
  "servers": [
    {
      "url": "https://api.example.com",
      "description": "Production server"
    }
  ],
  "paths": {
    "/api/v1/prove/bracket": {
      "post": {
        "summary": "Generate a tax bracket proof",
        "tags": ["Proofs"],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/BracketProofRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Proof generated successfully",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/BracketProofResponse" }
              }
            }
          }
        }
      }
    },
    "/api/v1/verify": {
      "post": {
        "summary": "Verify a proof",
        "tags": ["Verification"],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/VerifyRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Verification result",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/VerifyResponse" }
              }
            }
          }
        }
      }
    },
    "/api/v1/health": {
      "get": {
        "summary": "Health check",
        "tags": ["System"],
        "responses": {
          "200": {
            "description": "Service health status",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HealthResponse" }
              }
            }
          }
        }
      }
    }
  }
}"##
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_proof_request_parsing() {
        let request = BracketProofRequest {
            income_cents: 8500000,
            jurisdiction: "US".to_string(),
            filing_status: "single".to_string(),
            tax_year: 2024,
            dev_mode: true,
        };

        assert_eq!(request.income(), 85000);
        assert!(request.parse_jurisdiction().is_ok());
        assert!(request.parse_filing_status().is_ok());
    }

    #[test]
    fn test_api_response_success() {
        let response: ApiResponse<String> = ApiResponse::success("test".to_string())
            .with_processing_time(100)
            .with_request_id("req-123");

        assert!(response.success);
        assert_eq!(response.data, Some("test".to_string()));
        assert_eq!(response.meta.processing_time_ms, 100);
        assert_eq!(response.meta.request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_health_response() {
        let health = HealthResponse::default();
        assert_eq!(health.status, "healthy");
        assert!(!health.features.is_empty());
    }
}
