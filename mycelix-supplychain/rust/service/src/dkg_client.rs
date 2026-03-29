// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG client for publishing claims to the distributed knowledge graph
//!
//! This module provides HTTP client functionality for interacting with the
//! mycelix-dkg service for supply chain provenance claims.

use anyhow::{Context, Result};
use claim_model::DkgClaim;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Configuration for the DKG client
#[derive(Debug, Clone)]
pub struct DkgClientConfig {
    /// Base URL for the mycelix-dkg service (e.g., "http://localhost:9090")
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retry attempts for transient failures
    pub max_retries: u32,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
}

impl Default for DkgClientConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("DKG_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:9090".to_string()),
            timeout_secs: std::env::var("DKG_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            max_retries: std::env::var("DKG_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            api_key: std::env::var("DKG_API_KEY").ok(),
        }
    }
}

/// DKG client for interacting with the distributed knowledge graph service
#[derive(Clone)]
pub struct DkgClient {
    client: Client,
    config: DkgClientConfig,
}

/// Response from DKG publish operation
#[derive(Debug, Serialize, Deserialize)]
pub struct PublishResponse {
    /// Transaction ID for the published claim
    pub tx_id: String,
    /// Hash of the claim in the DKG
    pub claim_hash: String,
    /// Block number (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_number: Option<u64>,
    /// Timestamp of publication
    pub timestamp: String,
}

/// Request body for publishing a claim
#[derive(Debug, Serialize)]
struct PublishRequest<'a> {
    claim: &'a DkgClaim,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<String>,
}

/// Response from DKG resolve operation
#[derive(Debug, Deserialize)]
struct ResolveResponse {
    claim: DkgClaim,
    #[serde(default)]
    verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    verification_proof: Option<String>,
}

/// Error types specific to DKG operations
#[derive(Debug, thiserror::Error)]
pub enum DkgError {
    #[error("DKG service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Claim not found: {0}")]
    ClaimNotFound(String),

    #[error("Invalid claim format: {0}")]
    InvalidClaim(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Rate limited, retry after {0} seconds")]
    RateLimited(u64),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl DkgClient {
    /// Create a new DKG client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(DkgClientConfig::default())
    }

    /// Create a new DKG client with custom configuration
    pub fn with_config(config: DkgClientConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(5)
            .build()
            .context("Failed to create HTTP client for DKG")?;

        Ok(Self { client, config })
    }

    /// Publish a claim to the DKG network
    ///
    /// This sends the claim to the mycelix-dkg service which will:
    /// 1. Validate the claim structure and signatures
    /// 2. Store it in the distributed knowledge graph
    /// 3. Return a transaction ID for tracking
    pub async fn publish(&self, claim: &DkgClaim) -> Result<PublishResponse, DkgError> {
        let url = format!("{}/api/v1/claims", self.config.base_url);

        info!(
            claim_id = %claim.id,
            batch_id = %claim.subject.batch_id,
            "Publishing claim to DKG"
        );

        let request_body = PublishRequest {
            claim,
            priority: None,
        };

        let mut attempt = 0;
        loop {
            attempt += 1;

            let mut request = self.client.post(&url).json(&request_body);

            if let Some(ref api_key) = self.config.api_key {
                request = request.header("X-API-Key", api_key);
            }

            match request.send().await {
                Ok(response) => {
                    let status = response.status();

                    match status {
                        StatusCode::OK | StatusCode::CREATED | StatusCode::ACCEPTED => {
                            let publish_response: PublishResponse = response
                                .json()
                                .await
                                .map_err(|e| DkgError::SerializationError(e.to_string()))?;

                            info!(
                                claim_id = %claim.id,
                                tx_id = %publish_response.tx_id,
                                "Claim published to DKG successfully"
                            );

                            return Ok(publish_response);
                        }
                        StatusCode::TOO_MANY_REQUESTS => {
                            let retry_after = response
                                .headers()
                                .get("Retry-After")
                                .and_then(|v| v.to_str().ok())
                                .and_then(|v| v.parse::<u64>().ok())
                                .unwrap_or(60);

                            return Err(DkgError::RateLimited(retry_after));
                        }
                        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                            let body = response.text().await.unwrap_or_default();
                            return Err(DkgError::AuthenticationFailed(body));
                        }
                        StatusCode::BAD_REQUEST => {
                            let body = response.text().await.unwrap_or_default();
                            return Err(DkgError::InvalidClaim(body));
                        }
                        StatusCode::SERVICE_UNAVAILABLE
                        | StatusCode::BAD_GATEWAY
                        | StatusCode::GATEWAY_TIMEOUT => {
                            if attempt < self.config.max_retries {
                                warn!(
                                    attempt = attempt,
                                    status = %status,
                                    "DKG service temporarily unavailable, retrying..."
                                );
                                tokio::time::sleep(self.backoff_delay(attempt)).await;
                                continue;
                            }
                            let body = response.text().await.unwrap_or_default();
                            return Err(DkgError::ServiceUnavailable(body));
                        }
                        _ => {
                            let body = response.text().await.unwrap_or_default();
                            error!(status = %status, body = %body, "DKG publish failed");
                            return Err(DkgError::NetworkError(format!(
                                "Unexpected status {}: {}",
                                status, body
                            )));
                        }
                    }
                }
                Err(e)
                    if attempt < self.config.max_retries && (e.is_timeout() || e.is_connect()) =>
                {
                    warn!(
                        attempt = attempt,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "DKG request failed, retrying..."
                    );
                    tokio::time::sleep(self.backoff_delay(attempt)).await;
                    continue;
                }
                Err(e) => {
                    error!(error = %e, "DKG publish request failed");
                    return Err(DkgError::NetworkError(e.to_string()));
                }
            }
        }
    }

    /// Resolve a claim from the DKG network by its ID
    ///
    /// This retrieves the claim from the distributed knowledge graph,
    /// including verification status if available.
    pub async fn resolve(&self, claim_id: &str) -> Result<DkgClaim, DkgError> {
        let url = format!("{}/api/v1/claims/{}", self.config.base_url, claim_id);

        debug!(claim_id = %claim_id, "Resolving claim from DKG");

        let mut request = self.client.get(&url);

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("X-API-Key", api_key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| DkgError::NetworkError(e.to_string()))?;

        let status = response.status();

        match status {
            StatusCode::OK => {
                let resolve_response: ResolveResponse = response
                    .json()
                    .await
                    .map_err(|e| DkgError::SerializationError(e.to_string()))?;

                debug!(
                    claim_id = %claim_id,
                    verified = resolve_response.verified,
                    "Claim resolved from DKG"
                );

                Ok(resolve_response.claim)
            }
            StatusCode::NOT_FOUND => Err(DkgError::ClaimNotFound(claim_id.to_string())),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                let body = response.text().await.unwrap_or_default();
                Err(DkgError::AuthenticationFailed(body))
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                let body = response.text().await.unwrap_or_default();
                Err(DkgError::ServiceUnavailable(body))
            }
            _ => {
                let body = response.text().await.unwrap_or_default();
                Err(DkgError::NetworkError(format!(
                    "Unexpected status {}: {}",
                    status, body
                )))
            }
        }
    }

    /// Calculate exponential backoff delay for retries
    fn backoff_delay(&self, attempt: u32) -> Duration {
        let base_ms = 1000u64;
        let max_delay_ms = 30000u64;
        let delay_ms = (base_ms * 2u64.pow(attempt - 1)).min(max_delay_ms);
        Duration::from_millis(delay_ms)
    }
}

// Global lazy-initialized client for convenience functions
static DKG_CLIENT: std::sync::OnceLock<DkgClient> = std::sync::OnceLock::new();

fn get_global_client() -> Result<&'static DkgClient> {
    DKG_CLIENT
        .get_or_try_init(DkgClient::new)
        .context("Failed to initialize DKG client")
}

/// Publish a claim to the DKG network (convenience function)
///
/// Uses a global client instance with default configuration.
/// For custom configuration, create a DkgClient instance directly.
pub async fn publish_claim(claim: &DkgClaim) -> Result<String> {
    let client = get_global_client()?;

    match client.publish(claim).await {
        Ok(response) => Ok(response.tx_id),
        Err(DkgError::ServiceUnavailable(msg)) => {
            warn!(
                claim_id = %claim.id,
                error = %msg,
                "DKG service unavailable - claim stored locally only"
            );
            // Return a local-only marker transaction ID
            Ok(format!("local-only:{}", claim.id))
        }
        Err(e) => {
            error!(claim_id = %claim.id, error = %e, "Failed to publish claim to DKG");
            Err(e.into())
        }
    }
}

/// Resolve a claim from the DKG network (convenience function)
///
/// Uses a global client instance with default configuration.
/// For custom configuration, create a DkgClient instance directly.
pub async fn resolve_claim(claim_id: &str) -> Result<DkgClaim> {
    let client = get_global_client()?;

    client.resolve(claim_id).await.map_err(|e| {
        error!(claim_id = %claim_id, error = %e, "Failed to resolve claim from DKG");
        e.into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        std::env::remove_var("DKG_SERVICE_URL");
        std::env::remove_var("DKG_TIMEOUT_SECS");
        std::env::remove_var("DKG_MAX_RETRIES");

        let config = DkgClientConfig::default();
        assert_eq!(config.base_url, "http://localhost:9090");
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("DKG_SERVICE_URL", "http://dkg.example.com:8080");
        std::env::set_var("DKG_TIMEOUT_SECS", "60");
        std::env::set_var("DKG_MAX_RETRIES", "5");
        std::env::set_var("DKG_API_KEY", "test-api-key");

        let config = DkgClientConfig::default();
        assert_eq!(config.base_url, "http://dkg.example.com:8080");
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.api_key, Some("test-api-key".to_string()));

        std::env::remove_var("DKG_SERVICE_URL");
        std::env::remove_var("DKG_TIMEOUT_SECS");
        std::env::remove_var("DKG_MAX_RETRIES");
        std::env::remove_var("DKG_API_KEY");
    }

    #[test]
    fn test_client_creation() {
        let config = DkgClientConfig {
            base_url: "http://localhost:9090".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            api_key: None,
        };

        let client = DkgClient::with_config(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_backoff_delay() {
        let config = DkgClientConfig::default();
        let client = DkgClient::with_config(config).unwrap();

        assert_eq!(client.backoff_delay(1), Duration::from_millis(1000));
        assert_eq!(client.backoff_delay(2), Duration::from_millis(2000));
        assert_eq!(client.backoff_delay(3), Duration::from_millis(4000));
        assert_eq!(client.backoff_delay(10), Duration::from_millis(30000)); // capped
    }
}
