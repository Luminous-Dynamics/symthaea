// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Client for Symthaea inference server
//!
//! Connects to the symthaea-inference HTTP API for LLM generation.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Request to the inference server
#[derive(Debug, Serialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
}

/// Response from the inference server
#[derive(Debug, Deserialize)]
struct GenerateResponse {
    text: String,
    tokens_generated: u32,
    #[allow(dead_code)]
    model: String,
}

/// Health check response
#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    #[allow(dead_code)]
    model_loaded: bool,
}

/// Client for the Symthaea inference server
pub struct InferenceClient {
    base_url: String,
    http_client: reqwest::Client,
}

impl InferenceClient {
    /// Create a new inference client
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Check if the inference server is healthy
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);

        match self.http_client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let health: HealthResponse = response.json().await?;
                    Ok(health.status == "ok")
                } else {
                    Ok(false)
                }
            }
            Err(e) => {
                warn!("Inference health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Generate text from a prompt
    pub async fn generate(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String> {
        let url = format!("{}/generate", self.base_url);

        let request = GenerateRequest {
            prompt: prompt.to_string(),
            max_tokens: max_tokens.or(Some(512)),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: Some(vec![
                "\n\nHuman:".to_string(),
                "\n\nUser:".to_string(),
                "---".to_string(),
            ]),
        };

        debug!("Sending inference request to {}", url);

        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send inference request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Inference request failed ({}): {}", status, body));
        }

        let result: GenerateResponse = response.json().await
            .map_err(|e| anyhow!("Failed to parse inference response: {}", e))?;

        debug!("Generated {} tokens", result.tokens_generated);

        Ok(result.text.trim().to_string())
    }

    /// Generate with a civic assistance prompt
    pub async fn generate_civic_response(
        &self,
        question: &str,
        context: &str,
        location: Option<&str>,
    ) -> Result<String> {
        let prompt = format!(
            r#"You are Symthaea, a helpful civic AI assistant. Answer the citizen's question using the provided context. Be concise, accurate, and helpful. If you're not sure, say so and suggest where they might find more information.

Context from civic knowledge database:
{context}

{location_note}

Citizen's question: {question}

Symthaea:"#,
            context = if context.is_empty() { "No specific information found in the database." } else { context },
            location_note = location.map(|l| format!("Location: {}", l)).unwrap_or_default(),
            question = question,
        );

        self.generate(&prompt, Some(300)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = InferenceClient::new("http://localhost:3000");
        assert_eq!(client.base_url, "http://localhost:3000");
    }

    #[test]
    fn test_url_normalization() {
        let client = InferenceClient::new("http://localhost:3000/");
        assert_eq!(client.base_url, "http://localhost:3000");
    }
}
