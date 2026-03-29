// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Production service layer with rate limiting, caching, and observability.
//!
//! This module provides a production-ready service wrapper around the core
//! proof generation functionality, including:
//!
//! - Rate limiting per client
//! - Proof caching
//! - Metrics and tracing
//! - Request validation
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_zk_tax::service::{ProofService, ServiceConfig};
//!
//! let config = ServiceConfig::default()
//!     .with_rate_limit(100, Duration::from_secs(60))
//!     .with_cache_size(10_000);
//!
//! let service = ProofService::new(config);
//!
//! let proof = service.prove_bracket(ProveRequest {
//!     client_id: "user-123".to_string(),
//!     income: 85_000,
//!     jurisdiction: Jurisdiction::US,
//!     filing_status: FilingStatus::Single,
//!     tax_year: 2024,
//! }).await?;
//! ```

use crate::{Error, Result, Jurisdiction, FilingStatus, TaxYear, TaxBracketProof};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

#[cfg(feature = "prover")]
use crate::prover::TaxBracketProver;

// =============================================================================
// Configuration
// =============================================================================

/// Service configuration.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Maximum requests per window per client
    pub rate_limit_requests: u32,
    /// Rate limit window duration
    pub rate_limit_window: Duration,
    /// Maximum cache size (number of proofs)
    pub cache_size: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Development mode (faster but insecure proofs)
    pub dev_mode: bool,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            rate_limit_requests: 100,
            rate_limit_window: Duration::from_secs(60),
            cache_size: 10_000,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            metrics_enabled: true,
            dev_mode: false,
        }
    }
}

impl ServiceConfig {
    /// Create a new config with rate limiting.
    pub fn with_rate_limit(mut self, requests: u32, window: Duration) -> Self {
        self.rate_limit_requests = requests;
        self.rate_limit_window = window;
        self
    }

    /// Set the cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable development mode.
    pub fn with_dev_mode(mut self) -> Self {
        self.dev_mode = true;
        self
    }
}

// =============================================================================
// Rate Limiter
// =============================================================================

/// Token bucket rate limiter.
#[derive(Clone)]
pub struct RateLimiter {
    /// Bucket state per client
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    /// Maximum tokens (requests) per bucket
    max_tokens: u32,
    /// Refill rate (tokens per second)
    refill_rate: f64,
}

#[derive(Clone)]
struct TokenBucket {
    tokens: f64,
    last_update: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_requests: u32, window: Duration) -> Self {
        let refill_rate = max_requests as f64 / window.as_secs_f64();

        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            max_tokens: max_requests,
            refill_rate,
        }
    }

    /// Check if a request is allowed for a client.
    pub fn check(&self, client_id: &str) -> bool {
        let mut buckets = self.buckets.write().unwrap();

        let now = Instant::now();
        let bucket = buckets.entry(client_id.to_string()).or_insert_with(|| {
            TokenBucket {
                tokens: self.max_tokens as f64,
                last_update: now,
            }
        });

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(bucket.last_update).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_rate).min(self.max_tokens as f64);
        bucket.last_update = now;

        // Check if we have a token
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Get remaining tokens for a client.
    pub fn remaining(&self, client_id: &str) -> u32 {
        let buckets = self.buckets.read().unwrap();

        buckets
            .get(client_id)
            .map(|b| b.tokens as u32)
            .unwrap_or(self.max_tokens)
    }

    /// Clear old buckets (for memory management).
    pub fn cleanup(&self, max_age: Duration) {
        let mut buckets = self.buckets.write().unwrap();
        let now = Instant::now();

        buckets.retain(|_, bucket| {
            now.duration_since(bucket.last_update) < max_age
        });
    }
}

// =============================================================================
// Proof Cache
// =============================================================================

/// LRU cache for proofs.
#[derive(Clone)]
pub struct ProofCacheService {
    /// Cached proofs
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache TTL
    ttl: Duration,
}

#[derive(Clone)]
struct CacheEntry {
    proof: TaxBracketProof,
    created_at: Instant,
    access_count: u64,
}

impl ProofCacheService {
    /// Create a new cache.
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            ttl,
        }
    }

    /// Generate a cache key for a proof request.
    pub fn cache_key(
        income: u64,
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> String {
        format!(
            "{}:{}:{}:{}",
            income,
            jurisdiction.code(),
            filing_status.code(),
            tax_year
        )
    }

    /// Get a cached proof.
    pub fn get(&self, key: &str) -> Option<TaxBracketProof> {
        let mut entries = self.entries.write().unwrap();

        if let Some(entry) = entries.get_mut(key) {
            // Check TTL
            if entry.created_at.elapsed() > self.ttl {
                entries.remove(key);
                return None;
            }

            entry.access_count += 1;
            return Some(entry.proof.clone());
        }

        None
    }

    /// Store a proof in the cache.
    pub fn put(&self, key: String, proof: TaxBracketProof) {
        let mut entries = self.entries.write().unwrap();

        // Evict if at capacity
        if entries.len() >= self.max_size {
            // Simple eviction: remove oldest entry
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone())
            {
                entries.remove(&oldest_key);
            }
        }

        entries.insert(
            key,
            CacheEntry {
                proof,
                created_at: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read().unwrap();

        let total_accesses: u64 = entries.values().map(|e| e.access_count).sum();
        let avg_age = if entries.is_empty() {
            Duration::ZERO
        } else {
            let total_age: Duration = entries.values().map(|e| e.created_at.elapsed()).sum();
            total_age / entries.len() as u32
        };

        CacheStats {
            size: entries.len(),
            max_size: self.max_size,
            total_accesses,
            avg_age,
        }
    }

    /// Clear expired entries.
    pub fn cleanup(&self) {
        let mut entries = self.entries.write().unwrap();

        entries.retain(|_, entry| entry.created_at.elapsed() <= self.ttl);
    }
}

/// Cache statistics.
#[derive(Clone, Debug, Serialize)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub total_accesses: u64,
    pub avg_age: Duration,
}

// =============================================================================
// Metrics
// =============================================================================

/// Service metrics collector.
#[derive(Clone)]
pub struct MetricsCollector {
    /// Request counts by endpoint
    request_counts: Arc<RwLock<HashMap<String, u64>>>,
    /// Error counts by type
    error_counts: Arc<RwLock<HashMap<String, u64>>>,
    /// Latency samples (in microseconds)
    latencies: Arc<RwLock<Vec<u64>>>,
    /// Start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            request_counts: Arc::new(RwLock::new(HashMap::new())),
            error_counts: Arc::new(RwLock::new(HashMap::new())),
            latencies: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    /// Record a request.
    pub fn record_request(&self, endpoint: &str) {
        let mut counts = self.request_counts.write().unwrap();
        *counts.entry(endpoint.to_string()).or_insert(0) += 1;
    }

    /// Record an error.
    pub fn record_error(&self, error_type: &str) {
        let mut counts = self.error_counts.write().unwrap();
        *counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Record request latency.
    pub fn record_latency(&self, duration: Duration) {
        let mut latencies = self.latencies.write().unwrap();

        // Keep only last 10,000 samples
        if latencies.len() >= 10_000 {
            latencies.remove(0);
        }

        latencies.push(duration.as_micros() as u64);
    }

    /// Get all metrics.
    pub fn get_metrics(&self) -> ServiceMetrics {
        let request_counts = self.request_counts.read().unwrap().clone();
        let error_counts = self.error_counts.read().unwrap().clone();
        let latencies = self.latencies.read().unwrap().clone();

        let avg_latency_us = if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        };

        let p99_latency_us = if latencies.is_empty() {
            0
        } else {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[sorted.len() * 99 / 100]
        };

        ServiceMetrics {
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_requests: request_counts.values().sum(),
            total_errors: error_counts.values().sum(),
            request_counts,
            error_counts,
            avg_latency_us,
            p99_latency_us,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Service metrics.
#[derive(Clone, Debug, Serialize)]
pub struct ServiceMetrics {
    pub uptime_secs: u64,
    pub total_requests: u64,
    pub total_errors: u64,
    pub request_counts: HashMap<String, u64>,
    pub error_counts: HashMap<String, u64>,
    pub avg_latency_us: u64,
    pub p99_latency_us: u64,
}

// =============================================================================
// Proof Service
// =============================================================================

/// Production-ready proof service.
#[cfg(feature = "prover")]
#[derive(Clone)]
pub struct ProofService {
    /// Configuration
    config: ServiceConfig,
    /// Prover instance
    prover: Arc<TaxBracketProver>,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Proof cache
    cache: ProofCacheService,
    /// Metrics collector
    metrics: MetricsCollector,
}

/// Request to generate a proof.
#[derive(Clone, Debug, Deserialize)]
pub struct ProveRequest {
    /// Client identifier (for rate limiting)
    pub client_id: String,
    /// Income in dollars
    pub income: u64,
    /// Jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,
    /// Use cached result if available
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
}

fn default_use_cache() -> bool {
    true
}

/// Response from proof generation.
#[derive(Clone, Debug, Serialize)]
pub struct ProveResponse {
    /// The generated proof
    pub proof: TaxBracketProof,
    /// Whether this was served from cache
    pub from_cache: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Remaining rate limit quota
    pub rate_limit_remaining: u32,
}

#[cfg(feature = "prover")]
impl ProofService {
    /// Create a new proof service.
    pub fn new(config: ServiceConfig) -> Self {
        let prover = if config.dev_mode {
            TaxBracketProver::dev_mode()
        } else {
            TaxBracketProver::new()
        };

        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests,
            config.rate_limit_window,
        );

        let cache = ProofCacheService::new(config.cache_size, config.cache_ttl);

        Self {
            config,
            prover: Arc::new(prover),
            rate_limiter,
            cache,
            metrics: MetricsCollector::new(),
        }
    }

    /// Generate a tax bracket proof.
    pub fn prove_bracket(&self, request: ProveRequest) -> Result<ProveResponse> {
        let start = Instant::now();

        // Record request
        self.metrics.record_request("prove_bracket");

        // Check rate limit
        if !self.rate_limiter.check(&request.client_id) {
            self.metrics.record_error("rate_limited");
            return Err(Error::proof_generation_with_context(
                "Rate limit exceeded",
                format!("Client {} exceeded rate limit", request.client_id),
                format!("Wait and retry. Limit: {} requests per {:?}",
                    self.config.rate_limit_requests,
                    self.config.rate_limit_window),
            ));
        }

        // Check cache
        let cache_key = ProofCacheService::cache_key(
            request.income,
            request.jurisdiction,
            request.filing_status,
            request.tax_year,
        );

        if request.use_cache {
            if let Some(proof) = self.cache.get(&cache_key) {
                let elapsed = start.elapsed();
                self.metrics.record_latency(elapsed);

                return Ok(ProveResponse {
                    proof,
                    from_cache: true,
                    processing_time_ms: elapsed.as_millis() as u64,
                    rate_limit_remaining: self.rate_limiter.remaining(&request.client_id),
                });
            }
        }

        // Generate proof
        let proof = self.prover.prove(
            request.income,
            request.jurisdiction,
            request.filing_status,
            request.tax_year,
        )?;

        // Cache the result
        self.cache.put(cache_key, proof.clone());

        let elapsed = start.elapsed();
        self.metrics.record_latency(elapsed);

        Ok(ProveResponse {
            proof,
            from_cache: false,
            processing_time_ms: elapsed.as_millis() as u64,
            rate_limit_remaining: self.rate_limiter.remaining(&request.client_id),
        })
    }

    /// Get service metrics.
    pub fn get_metrics(&self) -> ServiceMetrics {
        self.metrics.get_metrics()
    }

    /// Get cache statistics.
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Run background cleanup tasks.
    pub fn cleanup(&self) {
        self.cache.cleanup();
        self.rate_limiter.cleanup(Duration::from_secs(3600));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(3, Duration::from_secs(1));

        // First 3 requests should pass
        assert!(limiter.check("client1"));
        assert!(limiter.check("client1"));
        assert!(limiter.check("client1"));

        // 4th should fail
        assert!(!limiter.check("client1"));

        // Different client should still work
        assert!(limiter.check("client2"));
    }

    #[cfg(feature = "prover")]
    #[test]
    fn test_cache() {
        let cache = ProofCacheService::new(100, Duration::from_secs(60));

        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let key = ProofCacheService::cache_key(85_000, Jurisdiction::US, FilingStatus::Single, 2024);

        // Should not exist initially
        assert!(cache.get(&key).is_none());

        // Put and get
        cache.put(key.clone(), proof.clone());
        let cached = cache.get(&key).unwrap();
        assert_eq!(cached.bracket_index, proof.bracket_index);
    }

    #[cfg(feature = "prover")]
    #[test]
    fn test_service() {
        let config = ServiceConfig::default().with_dev_mode();
        let service = ProofService::new(config);

        let request = ProveRequest {
            client_id: "test-client".to_string(),
            income: 85_000,
            jurisdiction: Jurisdiction::US,
            filing_status: FilingStatus::Single,
            tax_year: 2024,
            use_cache: true,
        };

        // First request
        let response = service.prove_bracket(request.clone()).unwrap();
        assert!(!response.from_cache);

        // Second request should be cached
        let response2 = service.prove_bracket(request).unwrap();
        assert!(response2.from_cache);
    }
}
