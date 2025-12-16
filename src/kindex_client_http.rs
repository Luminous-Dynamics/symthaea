// Real K-Index Client - HTTP-based (v0.3)
//
// Connects to actual K-Index backend service to query pre-computed deltas.
// Uses blocking HTTP for now - can be made async if needed.

use crate::kindex_client::{KDelta, KIndexClient};
use serde::Deserialize;

/// Real K-Index client that talks to HTTP backend
pub struct RealKIndexClient {
    base_url: String,
    http: reqwest::blocking::Client,
}

/// Response from K-Index backend
#[derive(Debug, Deserialize)]
struct KDeltaResponse {
    dimension: String,
    delta: f32,
    timeframe: String,
    drivers: Vec<String>,
    confidence: Option<f32>,
}

impl RealKIndexClient {
    /// Create new client pointing to K-Index service
    ///
    /// # Example
    /// ```no_run
    /// use symthaea::kindex_client_http::RealKIndexClient;
    ///
    /// let client = RealKIndexClient::new("http://localhost:8000".to_string());
    /// ```
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            http: reqwest::blocking::Client::new(),
        }
    }

    /// Build URL for endpoint
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

impl KIndexClient for RealKIndexClient {
    fn get_delta(&self, dimension: &str, timeframe: &str) -> Option<KDelta> {
        let url = self.url(&format!(
            "/kindex/delta?dimension={}&timeframe={}",
            dimension, timeframe
        ));

        // Make HTTP request
        let res = self.http.get(&url).send().ok()?;

        // Check status
        if !res.status().is_success() {
            tracing::warn!(
                "K-Index request failed: {} for dimension={}, timeframe={}",
                res.status(),
                dimension,
                timeframe
            );
            return None;
        }

        // Parse response
        let payload: KDeltaResponse = res.json().ok()?;

        Some(KDelta {
            dimension: payload.dimension,
            delta: payload.delta,
            timeframe: payload.timeframe,
            drivers: payload.drivers,
            confidence: payload.confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_construction() {
        let client = RealKIndexClient::new("http://localhost:8000".to_string());
        let url = client.url("/kindex/delta?dimension=Knowledge&timeframe=Past7Days");

        assert_eq!(
            url,
            "http://localhost:8000/kindex/delta?dimension=Knowledge&timeframe=Past7Days"
        );
    }

    // Note: Real integration tests would require a running K-Index service
    // For now, these are just structural tests
}
