// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Experimental module: fields will be read when Mycelix SDK integration is wired
#![allow(dead_code)]
//! Mycelix Network Integration
//!
//! Provides async client for fetching K-Vectors from the Mycelix network
//! and computing Spectral K metrics from the live network graph.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    SPECTRAL K COMPUTATION                           │
//! │                                                                     │
//! │  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
//! │  │  Gateway API    │───▶│ K-Vector Fetch   │───▶│ Laplacian     │ │
//! │  │  /api/k-vectors │    │ NetworkClient    │    │ Eigenvalues   │ │
//! │  └─────────────────┘    └──────────────────┘    └───────────────┘ │
//! │                                  │                       │        │
//! │                                  ▼                       ▼        │
//! │                         ┌───────────────────────────────────────┐ │
//! │                         │         Spectral K Result             │ │
//! │                         │  • Network coherence (λ₂)             │ │
//! │                         │  • Algebraic connectivity             │ │
//! │                         │  • Graph energy                       │ │
//! │                         └───────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// K-Vector from network (8-dimensional consciousness signature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkKVector {
    /// Agent/node identifier
    pub agent: String,
    /// 8-dimensional K-Vector signature
    pub k_vector: Vec<f32>,
}

/// Spectral K computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralKResult {
    /// Number of nodes in the network
    pub node_count: usize,
    /// Algebraic connectivity (second smallest eigenvalue of Laplacian)
    /// Higher = more connected network
    pub algebraic_connectivity: f64,
    /// Spectral gap (λ₂ - λ₁, where λ₁ = 0)
    pub spectral_gap: f64,
    /// Graph energy (sum of absolute eigenvalues)
    pub graph_energy: f64,
    /// Network coherence (normalized algebraic connectivity)
    pub coherence: f64,
    /// Computation timestamp
    pub computed_at: u64,
}

impl SpectralKResult {
    /// Create empty result for disconnected network
    pub fn empty() -> Self {
        Self {
            node_count: 0,
            algebraic_connectivity: 0.0,
            spectral_gap: 0.0,
            graph_energy: 0.0,
            coherence: 0.0,
            computed_at: now_secs(),
        }
    }
}

/// Network client for fetching K-Vectors from Mycelix gateway
pub struct NetworkClient {
    /// Gateway base URL
    gateway_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Cache for K-Vectors
    cache: Option<(Vec<NetworkKVector>, Instant)>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl NetworkClient {
    /// Create new network client
    pub fn new(gateway_url: impl Into<String>) -> Self {
        Self {
            gateway_url: gateway_url.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            cache: None,
            cache_ttl: Duration::from_secs(30),
        }
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Fetch K-Vectors from the gateway
    pub async fn fetch_k_vectors(&mut self) -> Result<Vec<NetworkKVector>, NetworkError> {
        // Check cache
        if let Some((ref cached, timestamp)) = self.cache {
            if timestamp.elapsed() < self.cache_ttl {
                return Ok(cached.clone());
            }
        }

        // Fetch from gateway
        let url = format!("{}/api/k-vectors", self.gateway_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NetworkError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(NetworkError::RequestFailed(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let api_response: ApiResponse<Vec<KVectorEntry>> = response
            .json()
            .await
            .map_err(|e| NetworkError::ParseError(e.to_string()))?;

        if !api_response.success {
            return Err(NetworkError::ApiError(
                api_response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        let k_vectors: Vec<NetworkKVector> = api_response
            .data
            .unwrap_or_default()
            .into_iter()
            .map(|entry| NetworkKVector {
                agent: entry.agent,
                k_vector: entry.k_vector,
            })
            .collect();

        // Update cache
        self.cache = Some((k_vectors.clone(), Instant::now()));

        Ok(k_vectors)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

/// API response wrapper (matches gateway format)
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

/// K-Vector entry from API
#[derive(Debug, Deserialize)]
struct KVectorEntry {
    agent: String,
    k_vector: Vec<f32>,
}

/// Network errors
#[derive(Debug, Clone)]
pub enum NetworkError {
    /// HTTP request failed
    RequestFailed(String),
    /// Failed to parse response
    ParseError(String),
    /// API returned error
    ApiError(String),
    /// Network unavailable
    Unavailable,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestFailed(msg) => write!(f, "Request failed: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::ApiError(msg) => write!(f, "API error: {msg}"),
            Self::Unavailable => write!(f, "Network unavailable"),
        }
    }
}

impl std::error::Error for NetworkError {}

// ============================================================================
// SPECTRAL K COMPUTATION
// ============================================================================

/// Compute Spectral K metrics from K-Vectors
///
/// This computes network graph properties using the Laplacian matrix:
/// - Build similarity matrix from K-Vector cosine similarities
/// - Compute Laplacian: L = D - A (degree matrix minus adjacency)
/// - Find eigenvalues for spectral analysis
pub struct SpectralKComputer {
    /// Similarity threshold for edge creation
    pub similarity_threshold: f32,
}

impl Default for SpectralKComputer {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
        }
    }
}

impl SpectralKComputer {
    /// Create with custom similarity threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            similarity_threshold: threshold,
        }
    }

    /// Compute Spectral K from K-Vectors
    pub fn compute(&self, k_vectors: &[NetworkKVector]) -> SpectralKResult {
        let n = k_vectors.len();
        if n < 2 {
            return SpectralKResult::empty();
        }

        // Build adjacency matrix from K-Vector similarities
        let mut adjacency = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&k_vectors[i].k_vector, &k_vectors[j].k_vector);
                if sim >= self.similarity_threshold as f64 {
                    adjacency[i][j] = sim;
                    adjacency[j][i] = sim;
                }
            }
        }

        // Compute degree matrix
        let degrees: Vec<f64> = adjacency.iter().map(|row| row.iter().sum()).collect();

        // Build Laplacian: L = D - A
        let mut laplacian = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    laplacian[i][j] = degrees[i];
                } else {
                    laplacian[i][j] = -adjacency[i][j];
                }
            }
        }

        // Compute eigenvalues using power iteration approximation
        // (Full eigendecomposition would use nalgebra, but this is simpler)
        let eigenvalues = approximate_eigenvalues(&laplacian, 3);

        // Sort eigenvalues
        let mut sorted_eigenvalues = eigenvalues.clone();
        sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // λ₁ should be ~0 for connected graph, λ₂ is algebraic connectivity
        let lambda_2 = if sorted_eigenvalues.len() > 1 {
            sorted_eigenvalues[1].max(0.0)
        } else {
            0.0
        };

        // Graph energy = sum of |eigenvalues|
        let graph_energy: f64 = eigenvalues.iter().map(|e| e.abs()).sum();

        // Normalize coherence to [0, 1]
        let max_connectivity = (n - 1) as f64; // Complete graph
        let coherence = (lambda_2 / max_connectivity).clamp(0.0, 1.0);

        SpectralKResult {
            node_count: n,
            algebraic_connectivity: lambda_2,
            spectral_gap: lambda_2, // For unweighted, spectral gap = λ₂
            graph_energy,
            coherence,
            computed_at: now_secs(),
        }
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Approximate eigenvalues using power iteration
/// Returns the k largest eigenvalues
fn approximate_eigenvalues(matrix: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }

    let mut eigenvalues = Vec::with_capacity(k.min(n));
    let mut deflated = matrix.to_vec();

    for _ in 0..k.min(n) {
        // Power iteration
        let mut v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        for _ in 0..50 {
            // Multiply: w = Av
            let w: Vec<f64> = (0..n)
                .map(|i| deflated[i].iter().zip(&v).map(|(a, b)| a * b).sum())
                .collect();

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        // Compute Rayleigh quotient for eigenvalue
        let av: Vec<f64> = (0..n)
            .map(|i| deflated[i].iter().zip(&v).map(|(a, b)| a * b).sum())
            .collect();
        let eigenvalue: f64 = v.iter().zip(&av).map(|(a, b)| a * b).sum();

        eigenvalues.push(eigenvalue);

        // Deflate matrix: A' = A - λvv^T
        for i in 0..n {
            for j in 0..n {
                deflated[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    eigenvalues
}

/// Get current Unix timestamp in seconds
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// HIGH-LEVEL INTEGRATION
// ============================================================================

/// Mycelix network integration for Symthaea
///
/// Provides easy-to-use functions for fetching network state
/// and computing consciousness metrics from the distributed network.
pub struct MycelixNetworkBridge {
    client: NetworkClient,
    spectral_computer: SpectralKComputer,
    /// Last computation result
    last_result: Option<SpectralKResult>,
}

impl MycelixNetworkBridge {
    /// Create new bridge with gateway URL
    pub fn new(gateway_url: impl Into<String>) -> Self {
        Self {
            client: NetworkClient::new(gateway_url),
            spectral_computer: SpectralKComputer::default(),
            last_result: None,
        }
    }

    /// Fetch K-Vectors and compute Spectral K
    pub async fn compute_network_consciousness(&mut self) -> Result<SpectralKResult, NetworkError> {
        let k_vectors = self.client.fetch_k_vectors().await?;
        let result = self.spectral_computer.compute(&k_vectors);
        self.last_result = Some(result.clone());
        Ok(result)
    }

    /// Get last computed result without network fetch
    pub fn last_result(&self) -> Option<&SpectralKResult> {
        self.last_result.as_ref()
    }

    /// Get network coherence (0-1 normalized)
    pub fn network_coherence(&self) -> f64 {
        self.last_result
            .as_ref()
            .map(|r| r.coherence)
            .unwrap_or(0.0)
    }

    /// Check if network is well-connected (coherence > threshold)
    pub fn is_network_coherent(&self, threshold: f64) -> bool {
        self.network_coherence() >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_spectral_k_computation() {
        let k_vectors = vec![
            NetworkKVector {
                agent: "a".to_string(),
                k_vector: vec![0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 0.8, 0.3],
            },
            NetworkKVector {
                agent: "b".to_string(),
                k_vector: vec![0.7, 0.7, 0.6, 0.6, 0.8, 0.5, 0.7, 0.4],
            },
            NetworkKVector {
                agent: "c".to_string(),
                k_vector: vec![0.9, 0.5, 0.8, 0.4, 0.7, 0.6, 0.9, 0.2],
            },
        ];

        let computer = SpectralKComputer::default();
        let result = computer.compute(&k_vectors);

        assert_eq!(result.node_count, 3);
        assert!(result.coherence >= 0.0 && result.coherence <= 1.0);
    }

    #[test]
    fn test_empty_network() {
        let computer = SpectralKComputer::default();
        let result = computer.compute(&[]);
        assert_eq!(result.node_count, 0);
        assert_eq!(result.coherence, 0.0);
    }
}
