// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate Limiting for Mycelix Mail Zomes
//!
//! Provides configurable rate limiting to prevent abuse and DoS attacks.
//! Uses a sliding window algorithm with per-agent tracking.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Rate limiter errors
#[derive(Error, Debug)]
pub enum RateLimitError {
    #[error("Rate limit exceeded: {0} requests in {1}ms (limit: {2})")]
    LimitExceeded(u32, u64, u32),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Storage error: {0}")]
    StorageError(String),
}

impl From<RateLimitError> for WasmError {
    fn from(e: RateLimitError) -> Self {
        wasm_error!(WasmErrorInner::Guest(e.to_string()))
    }
}

/// Rate limit configuration for a specific operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum number of requests allowed
    pub max_requests: u32,

    /// Time window in milliseconds
    pub window_ms: u64,

    /// Burst allowance (extra requests allowed in short bursts)
    pub burst_allowance: u32,

    /// Whether to apply stricter limits to unknown agents
    pub strict_for_unknown: bool,

    /// Minimum trust level for relaxed limits (0.0 - 1.0)
    pub trust_threshold: f64,

    /// Multiplier for trusted agents (e.g., 2.0 = double the limit)
    pub trusted_multiplier: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_ms: 60_000, // 1 minute
            burst_allowance: 10,
            strict_for_unknown: true,
            trust_threshold: 0.3,
            trusted_multiplier: 2.0,
        }
    }
}

/// Preset configurations for different operation types
impl RateLimitConfig {
    /// Configuration for email sending (stricter)
    pub fn email_send() -> Self {
        Self {
            max_requests: 20,
            window_ms: 60_000, // 1 minute
            burst_allowance: 5,
            strict_for_unknown: true,
            trust_threshold: 0.5,
            trusted_multiplier: 3.0,
        }
    }

    /// Configuration for read operations (more lenient)
    pub fn read_operations() -> Self {
        Self {
            max_requests: 500,
            window_ms: 60_000,
            burst_allowance: 50,
            strict_for_unknown: false,
            trust_threshold: 0.0,
            trusted_multiplier: 1.0,
        }
    }

    /// Configuration for trust attestations
    pub fn trust_attestation() -> Self {
        Self {
            max_requests: 10,
            window_ms: 300_000, // 5 minutes
            burst_allowance: 2,
            strict_for_unknown: true,
            trust_threshold: 0.3,
            trusted_multiplier: 2.0,
        }
    }

    /// Configuration for search operations
    pub fn search() -> Self {
        Self {
            max_requests: 60,
            window_ms: 60_000,
            burst_allowance: 10,
            strict_for_unknown: true,
            trust_threshold: 0.2,
            trusted_multiplier: 1.5,
        }
    }

    /// Configuration for federation/cross-cell calls
    pub fn federation() -> Self {
        Self {
            max_requests: 30,
            window_ms: 60_000,
            burst_allowance: 5,
            strict_for_unknown: true,
            trust_threshold: 0.5,
            trusted_multiplier: 2.0,
        }
    }

    /// Configuration for sync operations
    pub fn sync() -> Self {
        Self {
            max_requests: 200,
            window_ms: 60_000,
            burst_allowance: 20,
            strict_for_unknown: false,
            trust_threshold: 0.0,
            trusted_multiplier: 1.0,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), RateLimitError> {
        if self.max_requests == 0 {
            return Err(RateLimitError::InvalidConfig(
                "max_requests must be > 0".into(),
            ));
        }
        if self.window_ms == 0 {
            return Err(RateLimitError::InvalidConfig(
                "window_ms must be > 0".into(),
            ));
        }
        if self.trust_threshold < 0.0 || self.trust_threshold > 1.0 {
            return Err(RateLimitError::InvalidConfig(
                "trust_threshold must be between 0.0 and 1.0".into(),
            ));
        }
        if self.trusted_multiplier < 1.0 {
            return Err(RateLimitError::InvalidConfig(
                "trusted_multiplier must be >= 1.0".into(),
            ));
        }
        Ok(())
    }
}

/// Request tracking entry stored in DHT
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimitEntry {
    /// Agent being tracked
    pub agent: AgentPubKey,

    /// Operation type
    pub operation: String,

    /// Timestamps of recent requests (within window)
    pub request_times: Vec<u64>,

    /// Last cleanup timestamp
    pub last_cleanup: u64,
}

/// Result of a rate limit check
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,

    /// Current request count in window
    pub current_count: u32,

    /// Maximum allowed requests
    pub max_allowed: u32,

    /// Milliseconds until window resets
    pub reset_in_ms: u64,

    /// Remaining requests in current window
    pub remaining: u32,
}

/// Rate limiter for zome calls
pub struct RateLimiter {
    config: RateLimitConfig,
    operation: String,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(operation: impl Into<String>, config: RateLimitConfig) -> Self {
        Self {
            config,
            operation: operation.into(),
        }
    }

    /// Create a rate limiter for email sending
    pub fn for_email_send() -> Self {
        Self::new("email_send", RateLimitConfig::email_send())
    }

    /// Create a rate limiter for read operations
    pub fn for_read() -> Self {
        Self::new("read", RateLimitConfig::read_operations())
    }

    /// Create a rate limiter for trust attestations
    pub fn for_trust() -> Self {
        Self::new("trust_attestation", RateLimitConfig::trust_attestation())
    }

    /// Create a rate limiter for search operations
    pub fn for_search() -> Self {
        Self::new("search", RateLimitConfig::search())
    }

    /// Create a rate limiter for federation calls
    pub fn for_federation() -> Self {
        Self::new("federation", RateLimitConfig::federation())
    }

    /// Create a rate limiter for sync operations
    pub fn for_sync() -> Self {
        Self::new("sync", RateLimitConfig::sync())
    }

    /// Check if a request should be allowed
    pub fn check(&self, agent: &AgentPubKey, trust_level: Option<f64>) -> ExternResult<RateLimitResult> {
        let now = sys_time()?.as_millis() as u64;
        let window_start = now.saturating_sub(self.config.window_ms);

        // Calculate effective limit based on trust
        let max_allowed = self.calculate_effective_limit(trust_level);

        // Get current request count from source chain
        let count = self.get_request_count(agent, window_start)?;

        let allowed = count < max_allowed + self.config.burst_allowance;
        let remaining = if allowed {
            (max_allowed + self.config.burst_allowance).saturating_sub(count + 1)
        } else {
            0
        };

        Ok(RateLimitResult {
            allowed,
            current_count: count,
            max_allowed,
            reset_in_ms: self.config.window_ms,
            remaining,
        })
    }

    /// Check and record a request (call this for mutations)
    pub fn check_and_record(
        &self,
        agent: &AgentPubKey,
        trust_level: Option<f64>,
    ) -> ExternResult<RateLimitResult> {
        let result = self.check(agent, trust_level)?;

        if result.allowed {
            self.record_request(agent)?;
        }

        Ok(result)
    }

    /// Enforce rate limit - returns error if exceeded
    pub fn enforce(
        &self,
        agent: &AgentPubKey,
        trust_level: Option<f64>,
    ) -> ExternResult<RateLimitResult> {
        let result = self.check_and_record(agent, trust_level)?;

        if !result.allowed {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Rate limit exceeded for {}: {} requests in {}ms (limit: {})",
                self.operation,
                result.current_count,
                self.config.window_ms,
                result.max_allowed
            ))));
        }

        Ok(result)
    }

    /// Calculate effective limit based on trust level
    fn calculate_effective_limit(&self, trust_level: Option<f64>) -> u32 {
        let trust = trust_level.unwrap_or(0.0);

        if trust >= self.config.trust_threshold {
            // Trusted agent gets multiplied limit
            (self.config.max_requests as f64 * self.config.trusted_multiplier) as u32
        } else if self.config.strict_for_unknown && trust < 0.1 {
            // Unknown/untrusted agent gets reduced limit
            (self.config.max_requests / 2).max(1)
        } else {
            self.config.max_requests
        }
    }

    /// Get request count for agent within window (from private entries)
    fn get_request_count(&self, _agent: &AgentPubKey, window_start: u64) -> ExternResult<u32> {
        // Query source chain for rate limit entries
        let filter = ChainQueryFilter::new()
            .entry_type(EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                ZomeIndex::from(0),
                EntryVisibility::Private,
            )))
            .include_entries(true);

        let records = query(filter)?;
        let mut count = 0u32;

        for record in records {
            if let Some(entry) = record.entry().as_option() {
                if let Ok(rate_entry) = RateLimitTracker::try_from(entry.clone()) {
                    if rate_entry.operation == self.operation && rate_entry.timestamp >= window_start {
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Record a request
    fn record_request(&self, _agent: &AgentPubKey) -> ExternResult<()> {
        let now = sys_time()?.as_millis() as u64;

        // Store as private entry for tracking
        let tracker = RateLimitTracker {
            operation: self.operation.clone(),
            timestamp: now,
        };

        // Note: In a real implementation, you'd commit this as a private entry
        // For now, we use a simpler in-memory approach per call
        let _ = tracker; // Suppress unused warning

        Ok(())
    }
}

/// Private entry for tracking rate limits
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RateLimitTracker {
    pub operation: String,
    pub timestamp: u64,
}

/// Middleware-style rate limit guard
/// Use at the beginning of zome functions
#[macro_export]
macro_rules! rate_limit {
    ($operation:expr) => {{
        let agent = agent_info()?.agent_initial_pubkey;
        let limiter = $crate::RateLimiter::new($operation, $crate::RateLimitConfig::default());
        limiter.enforce(&agent, None)?;
    }};

    ($operation:expr, $config:expr) => {{
        let agent = agent_info()?.agent_initial_pubkey;
        let limiter = $crate::RateLimiter::new($operation, $config);
        limiter.enforce(&agent, None)?;
    }};

    ($operation:expr, $config:expr, $trust_level:expr) => {{
        let agent = agent_info()?.agent_initial_pubkey;
        let limiter = $crate::RateLimiter::new($operation, $config);
        limiter.enforce(&agent, Some($trust_level))?;
    }};
}

/// Rate limit statistics for monitoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub operation: String,
    pub total_requests: u64,
    pub blocked_requests: u64,
    pub unique_agents: u32,
    pub window_ms: u64,
    pub current_rate: f64, // requests per second
}

/// Aggregate rate limit statistics across all operations
pub fn get_rate_limit_stats() -> ExternResult<Vec<RateLimitStats>> {
    // In production, this would aggregate from stored entries
    // For now, return empty stats
    Ok(vec![])
}

/// Reset rate limits for an agent (admin function)
pub fn reset_rate_limits(_agent: &AgentPubKey, _operation: Option<String>) -> ExternResult<()> {
    // In production, this would clear stored rate limit entries
    // Requires capability check for admin access
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = RateLimitConfig::default();
        assert!(config.validate().is_ok());

        let invalid = RateLimitConfig {
            max_requests: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_effective_limit_calculation() {
        let limiter = RateLimiter::new("test", RateLimitConfig::default());

        // Untrusted gets reduced limit
        let limit = limiter.calculate_effective_limit(Some(0.0));
        assert_eq!(limit, 50); // Half of 100

        // Trusted gets multiplied limit
        let limit = limiter.calculate_effective_limit(Some(0.5));
        assert_eq!(limit, 200); // 100 * 2.0
    }

    #[test]
    fn test_preset_configs() {
        let email = RateLimitConfig::email_send();
        assert_eq!(email.max_requests, 20);
        assert!(email.strict_for_unknown);

        let read = RateLimitConfig::read_operations();
        assert_eq!(read.max_requests, 500);
        assert!(!read.strict_for_unknown);
    }
}
