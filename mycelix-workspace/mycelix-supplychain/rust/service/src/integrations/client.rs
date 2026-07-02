// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP Client for External Integrations
//!
//! Provides a reusable HTTP client with:
//! - Automatic retry with exponential backoff
//! - Rate limiting awareness
//! - Request/response logging
//! - Timeout handling

use super::IntegrationError;
use reqwest::{Client, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;

/// HTTP client with retry logic
#[derive(Clone)]
pub struct IntegrationClient {
    client: Client,
    max_retries: u32,
    #[allow(dead_code)] // Reserved for future per-request timeout configuration
    base_timeout: Duration,
}

impl IntegrationClient {
    /// Create a new integration client
    pub fn new() -> Result<Self, IntegrationError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| IntegrationError::NetworkError(e.to_string()))?;

        Ok(Self {
            client,
            max_retries: 3,
            base_timeout: Duration::from_secs(30),
        })
    }

    /// POST request with JSON body
    pub async fn post<T: Serialize, R: DeserializeOwned>(
        &self,
        url: &str,
        body: &T,
        auth_header: Option<(&str, &str)>,
    ) -> Result<R, IntegrationError> {
        let mut attempt = 0;
        loop {
            attempt += 1;

            let mut request = self.client
                .post(url)
                .json(body);

            if let Some((key, value)) = auth_header {
                request = request.header(key, value);
            }

            match request.send().await {
                Ok(response) => {
                    return self.handle_response(response).await;
                }
                Err(e) if attempt < self.max_retries && e.is_timeout() => {
                    tracing::warn!(
                        attempt = attempt,
                        max_retries = self.max_retries,
                        error = %e,
                        "Request timeout, retrying..."
                    );
                    tokio::time::sleep(self.backoff_delay(attempt)).await;
                    continue;
                }
                Err(e) => {
                    return Err(IntegrationError::NetworkError(e.to_string()));
                }
            }
        }
    }

    /// GET request
    pub async fn get<R: DeserializeOwned>(
        &self,
        url: &str,
        auth_header: Option<(&str, &str)>,
    ) -> Result<R, IntegrationError> {
        let mut request = self.client.get(url);

        if let Some((key, value)) = auth_header {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| IntegrationError::NetworkError(e.to_string()))?;

        self.handle_response(response).await
    }

    /// DELETE request
    pub async fn delete<R: DeserializeOwned>(
        &self,
        url: &str,
        auth_header: Option<(&str, &str)>,
    ) -> Result<R, IntegrationError> {
        let mut request = self.client.delete(url);

        if let Some((key, value)) = auth_header {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| IntegrationError::NetworkError(e.to_string()))?;

        self.handle_response(response).await
    }

    /// Handle response with error checking
    async fn handle_response<R: DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<R, IntegrationError> {
        let status = response.status();

        match status {
            StatusCode::OK | StatusCode::CREATED | StatusCode::ACCEPTED => {
                response
                    .json::<R>()
                    .await
                    .map_err(|e| IntegrationError::SerializationError(e.to_string()))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(60);
                Err(IntegrationError::RateLimited(retry_after))
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                let body = response.text().await.unwrap_or_default();
                Err(IntegrationError::ApiError(format!(
                    "Authentication failed: {}",
                    body
                )))
            }
            StatusCode::BAD_REQUEST => {
                let body = response.text().await.unwrap_or_default();
                Err(IntegrationError::ApiError(format!(
                    "Bad request: {}",
                    body
                )))
            }
            _ => {
                let body = response.text().await.unwrap_or_default();
                Err(IntegrationError::ApiError(format!(
                    "API error ({}): {}",
                    status, body
                )))
            }
        }
    }

    /// Calculate exponential backoff delay
    fn backoff_delay(&self, attempt: u32) -> Duration {
        let base_ms = 1000u64;
        let max_delay_ms = 30000u64;
        let delay_ms = (base_ms * 2u64.pow(attempt - 1)).min(max_delay_ms);
        Duration::from_millis(delay_ms)
    }
}

impl Default for IntegrationClient {
    fn default() -> Self {
        Self::new().expect("Failed to create HTTP client")
    }
}
