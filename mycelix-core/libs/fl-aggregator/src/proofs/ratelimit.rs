//! Rate Limiting and Quotas for Proof Operations
//!
//! Provides protection against abuse with:
//! - Token bucket rate limiting
//! - Sliding window counters
//! - Per-user quotas
//! - Burst allowance
//! - Cost-based limiting

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::proofs::ProofType;

// ============================================================================
// Rate Limiter Configuration
// ============================================================================

/// Configuration for rate limiting
#[derive(Clone, Debug)]
pub struct RateLimitConfig {
    /// Maximum requests per second
    pub requests_per_second: f64,
    /// Burst allowance (tokens in bucket)
    pub burst_size: usize,
    /// Window size for sliding window
    pub window_size: Duration,
    /// Enable per-user limits
    pub per_user_limits: bool,
    /// Default quota per user
    pub default_user_quota: Quota,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100.0,
            burst_size: 50,
            window_size: Duration::from_secs(60),
            per_user_limits: true,
            default_user_quota: Quota::default(),
        }
    }
}

/// User quota configuration
#[derive(Clone, Debug)]
pub struct Quota {
    /// Maximum proofs per hour
    pub proofs_per_hour: u64,
    /// Maximum proofs per day
    pub proofs_per_day: u64,
    /// Maximum total proof bytes per day
    pub bytes_per_day: u64,
    /// Allowed proof types (empty = all)
    pub allowed_proof_types: Vec<ProofType>,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
}

impl Default for Quota {
    fn default() -> Self {
        Self {
            proofs_per_hour: 1000,
            proofs_per_day: 10000,
            bytes_per_day: 100 * 1024 * 1024, // 100MB
            allowed_proof_types: Vec::new(),  // All types allowed
            max_concurrent: 10,
        }
    }
}

impl Quota {
    /// Unlimited quota
    pub fn unlimited() -> Self {
        Self {
            proofs_per_hour: u64::MAX,
            proofs_per_day: u64::MAX,
            bytes_per_day: u64::MAX,
            allowed_proof_types: Vec::new(),
            max_concurrent: usize::MAX,
        }
    }

    /// Restricted quota for free tier
    pub fn free_tier() -> Self {
        Self {
            proofs_per_hour: 100,
            proofs_per_day: 500,
            bytes_per_day: 10 * 1024 * 1024, // 10MB
            allowed_proof_types: vec![ProofType::Range], // Only range proofs
            max_concurrent: 2,
        }
    }

    /// Standard quota
    pub fn standard() -> Self {
        Self::default()
    }

    /// Premium quota
    pub fn premium() -> Self {
        Self {
            proofs_per_hour: 10000,
            proofs_per_day: 100000,
            bytes_per_day: 1024 * 1024 * 1024, // 1GB
            allowed_proof_types: Vec::new(),
            max_concurrent: 50,
        }
    }
}

// ============================================================================
// Token Bucket
// ============================================================================

/// Token bucket for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    /// Current number of tokens
    tokens: f64,
    /// Maximum tokens (bucket size)
    max_tokens: f64,
    /// Token refill rate per second
    refill_rate: f64,
    /// Last refill time
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume tokens
    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Check available tokens without consuming
    pub fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Get time until tokens will be available
    pub fn time_until_available(&mut self, tokens: f64) -> Duration {
        self.refill();

        if self.tokens >= tokens {
            Duration::ZERO
        } else {
            let needed = tokens - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = elapsed.as_secs_f64() * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_refill = now;
    }
}

// ============================================================================
// Sliding Window Counter
// ============================================================================

/// Sliding window counter for request tracking
#[derive(Debug)]
pub struct SlidingWindow {
    /// Window size
    window_size: Duration,
    /// Request counts per time slot
    slots: Vec<WindowSlot>,
    /// Number of slots
    num_slots: usize,
    /// Slot duration
    slot_duration: Duration,
}

#[derive(Debug, Clone)]
struct WindowSlot {
    count: u64,
    start_time: Instant,
}

impl SlidingWindow {
    /// Create a new sliding window
    pub fn new(window_size: Duration, num_slots: usize) -> Self {
        let slot_duration = window_size / num_slots as u32;
        let now = Instant::now();

        let slots: Vec<WindowSlot> = (0..num_slots)
            .map(|i| WindowSlot {
                count: 0,
                start_time: now - slot_duration * i as u32,
            })
            .collect();

        Self {
            window_size,
            slots,
            num_slots,
            slot_duration,
        }
    }

    /// Record a request
    pub fn record(&mut self) {
        self.cleanup();

        let now = Instant::now();
        if let Some(slot) = self.slots.first_mut() {
            if now.duration_since(slot.start_time) < self.slot_duration {
                slot.count += 1;
            } else {
                // Need new slot
                self.slots.insert(
                    0,
                    WindowSlot {
                        count: 1,
                        start_time: now,
                    },
                );
            }
        } else {
            self.slots.push(WindowSlot {
                count: 1,
                start_time: now,
            });
        }
    }

    /// Get count in current window
    pub fn count(&mut self) -> u64 {
        self.cleanup();
        self.slots.iter().map(|s| s.count).sum()
    }

    /// Clean up old slots
    fn cleanup(&mut self) {
        let now = Instant::now();
        let cutoff = now - self.window_size;

        self.slots.retain(|s| s.start_time > cutoff);

        // Ensure we don't have too many slots
        while self.slots.len() > self.num_slots {
            self.slots.pop();
        }
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

/// Rate limiter result
#[derive(Clone, Debug)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Remaining requests
    pub remaining: u64,
    /// Time until reset
    pub reset_after: Duration,
    /// Retry after (if not allowed)
    pub retry_after: Option<Duration>,
    /// Limit that was hit (if any)
    pub limit_hit: Option<String>,
}

impl RateLimitResult {
    /// Create an allowed result
    pub fn allowed(remaining: u64, reset_after: Duration) -> Self {
        Self {
            allowed: true,
            remaining,
            reset_after,
            retry_after: None,
            limit_hit: None,
        }
    }

    /// Create a denied result
    pub fn denied(retry_after: Duration, limit: &str) -> Self {
        Self {
            allowed: false,
            remaining: 0,
            reset_after: retry_after,
            retry_after: Some(retry_after),
            limit_hit: Some(limit.to_string()),
        }
    }
}

/// Rate limiter for proof operations
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Global token bucket
    global_bucket: Arc<RwLock<TokenBucket>>,
    /// Per-user state
    user_state: Arc<RwLock<HashMap<String, UserState>>>,
    /// Statistics
    stats: Arc<RwLock<RateLimitStats>>,
}

/// Per-user rate limiting state
struct UserState {
    /// Token bucket for this user
    bucket: TokenBucket,
    /// Request counts
    hourly_window: SlidingWindow,
    daily_window: SlidingWindow,
    /// Bytes used today
    bytes_today: u64,
    /// Last byte reset
    byte_reset_time: Instant,
    /// Current concurrent requests
    concurrent: usize,
    /// User quota
    quota: Quota,
}

impl UserState {
    fn new(quota: Quota) -> Self {
        Self {
            bucket: TokenBucket::new(quota.max_concurrent as f64, 10.0),
            hourly_window: SlidingWindow::new(Duration::from_secs(3600), 60),
            daily_window: SlidingWindow::new(Duration::from_secs(86400), 1440),
            bytes_today: 0,
            byte_reset_time: Instant::now(),
            concurrent: 0,
            quota,
        }
    }

    fn maybe_reset_bytes(&mut self) {
        if self.byte_reset_time.elapsed() > Duration::from_secs(86400) {
            self.bytes_today = 0;
            self.byte_reset_time = Instant::now();
        }
    }
}

/// Rate limiter statistics
#[derive(Clone, Debug, Default)]
pub struct RateLimitStats {
    /// Total requests checked
    pub total_requests: u64,
    /// Requests allowed
    pub allowed_requests: u64,
    /// Requests denied
    pub denied_requests: u64,
    /// Denials by limit type
    pub denials_by_type: HashMap<String, u64>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            global_bucket: Arc::new(RwLock::new(TokenBucket::new(
                config.burst_size as f64,
                config.requests_per_second,
            ))),
            user_state: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RateLimitStats::default())),
            config,
        }
    }

    /// Check if a request should be allowed
    pub async fn check(&self, user_id: Option<&str>, proof_type: ProofType) -> RateLimitResult {
        self.check_with_cost(user_id, proof_type, 1.0).await
    }

    /// Check with a cost multiplier (for expensive operations)
    pub async fn check_with_cost(
        &self,
        user_id: Option<&str>,
        proof_type: ProofType,
        cost: f64,
    ) -> RateLimitResult {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;

        // Check global rate limit
        let mut global = self.global_bucket.write().await;
        if !global.try_consume(cost) {
            let retry = global.time_until_available(cost);
            stats.denied_requests += 1;
            *stats.denials_by_type.entry("global".to_string()).or_insert(0) += 1;
            return RateLimitResult::denied(retry, "global_rate_limit");
        }

        // Check per-user limits if enabled and user provided
        if self.config.per_user_limits {
            if let Some(uid) = user_id {
                let mut users = self.user_state.write().await;
                let state = users
                    .entry(uid.to_string())
                    .or_insert_with(|| UserState::new(self.config.default_user_quota.clone()));

                // Check proof type permission
                if !state.quota.allowed_proof_types.is_empty()
                    && !state.quota.allowed_proof_types.contains(&proof_type)
                {
                    stats.denied_requests += 1;
                    *stats.denials_by_type.entry("proof_type".to_string()).or_insert(0) += 1;
                    return RateLimitResult::denied(Duration::ZERO, "proof_type_not_allowed");
                }

                // Check concurrent limit
                if state.concurrent >= state.quota.max_concurrent {
                    stats.denied_requests += 1;
                    *stats.denials_by_type.entry("concurrent".to_string()).or_insert(0) += 1;
                    return RateLimitResult::denied(Duration::from_secs(1), "concurrent_limit");
                }

                // Check hourly limit
                if state.hourly_window.count() >= state.quota.proofs_per_hour {
                    stats.denied_requests += 1;
                    *stats.denials_by_type.entry("hourly".to_string()).or_insert(0) += 1;
                    return RateLimitResult::denied(Duration::from_secs(60), "hourly_limit");
                }

                // Check daily limit
                if state.daily_window.count() >= state.quota.proofs_per_day {
                    stats.denied_requests += 1;
                    *stats.denials_by_type.entry("daily".to_string()).or_insert(0) += 1;
                    return RateLimitResult::denied(Duration::from_secs(3600), "daily_limit");
                }

                // Record the request
                state.hourly_window.record();
                state.daily_window.record();
                state.concurrent += 1;

                let remaining = state.quota.proofs_per_hour - state.hourly_window.count();
                stats.allowed_requests += 1;

                return RateLimitResult::allowed(remaining, Duration::from_secs(3600));
            }
        }

        stats.allowed_requests += 1;
        RateLimitResult::allowed(
            global.available() as u64,
            Duration::from_secs_f64(1.0 / self.config.requests_per_second),
        )
    }

    /// Release a concurrent slot (call when request completes)
    pub async fn release(&self, user_id: &str) {
        let mut users = self.user_state.write().await;
        if let Some(state) = users.get_mut(user_id) {
            state.concurrent = state.concurrent.saturating_sub(1);
        }
    }

    /// Record bytes used by a user
    pub async fn record_bytes(&self, user_id: &str, bytes: u64) -> bool {
        let mut users = self.user_state.write().await;
        if let Some(state) = users.get_mut(user_id) {
            state.maybe_reset_bytes();

            if state.bytes_today + bytes > state.quota.bytes_per_day {
                return false;
            }

            state.bytes_today += bytes;
        }
        true
    }

    /// Set quota for a user
    pub async fn set_quota(&self, user_id: &str, quota: Quota) {
        let mut users = self.user_state.write().await;
        if let Some(state) = users.get_mut(user_id) {
            state.quota = quota;
        } else {
            users.insert(user_id.to_string(), UserState::new(quota));
        }
    }

    /// Get user's remaining quota
    pub async fn get_remaining(&self, user_id: &str) -> Option<RemainingQuota> {
        let mut users = self.user_state.write().await;
        users.get_mut(user_id).map(|state| {
            state.maybe_reset_bytes();
            RemainingQuota {
                hourly: state.quota.proofs_per_hour.saturating_sub(state.hourly_window.count()),
                daily: state.quota.proofs_per_day.saturating_sub(state.daily_window.count()),
                bytes: state.quota.bytes_per_day.saturating_sub(state.bytes_today),
                concurrent: state.quota.max_concurrent.saturating_sub(state.concurrent),
            }
        })
    }

    /// Get statistics
    pub async fn stats(&self) -> RateLimitStats {
        self.stats.read().await.clone()
    }

    /// Reset all rate limits (for testing)
    pub async fn reset(&self) {
        let mut global = self.global_bucket.write().await;
        *global = TokenBucket::new(
            self.config.burst_size as f64,
            self.config.requests_per_second,
        );

        self.user_state.write().await.clear();

        let mut stats = self.stats.write().await;
        *stats = RateLimitStats::default();
    }
}

/// Remaining quota for a user
#[derive(Clone, Debug)]
pub struct RemainingQuota {
    /// Remaining hourly proofs
    pub hourly: u64,
    /// Remaining daily proofs
    pub daily: u64,
    /// Remaining bytes
    pub bytes: u64,
    /// Remaining concurrent slots
    pub concurrent: usize,
}

// ============================================================================
// Cost Estimator
// ============================================================================

/// Estimate the cost of a proof operation
pub fn estimate_cost(proof_type: ProofType, params: &CostParams) -> f64 {
    let base_cost = match proof_type {
        ProofType::Range => 1.0,
        ProofType::Membership => 1.5,
        ProofType::GradientIntegrity => 2.0 + (params.data_size as f64 / 10000.0),
        ProofType::IdentityAssurance => 1.5,
        ProofType::VoteEligibility => 1.2,
    };

    let security_multiplier = match params.security_level {
        0 => 0.7,  // Standard96
        1 => 1.0,  // Standard128
        _ => 2.0,  // High256
    };

    base_cost * security_multiplier
}

/// Parameters for cost estimation
#[derive(Clone, Debug, Default)]
pub struct CostParams {
    /// Size of input data
    pub data_size: usize,
    /// Security level (0=96, 1=128, 2=256)
    pub security_level: u8,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_consume() {
        let mut bucket = TokenBucket::new(10.0, 1.0);

        assert!(bucket.try_consume(5.0));
        assert_eq!(bucket.available() as u64, 5);

        assert!(bucket.try_consume(5.0));
        assert!(!bucket.try_consume(1.0)); // Should be empty
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(10.0, 100.0); // Fast refill for testing

        bucket.try_consume(10.0);
        assert_eq!(bucket.available() as u64, 0);

        // Simulate time passing by manually setting last_refill
        bucket.last_refill = Instant::now() - Duration::from_millis(100);
        assert!(bucket.available() >= 9.0); // Should have refilled
    }

    #[test]
    fn test_sliding_window() {
        let mut window = SlidingWindow::new(Duration::from_secs(60), 60);

        for _ in 0..10 {
            window.record();
        }

        assert_eq!(window.count(), 10);
    }

    #[test]
    fn test_quota_tiers() {
        let free = Quota::free_tier();
        let standard = Quota::standard();
        let premium = Quota::premium();

        assert!(free.proofs_per_day < standard.proofs_per_day);
        assert!(standard.proofs_per_day < premium.proofs_per_day);
    }

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        let result = limiter.check(Some("user1"), ProofType::Range).await;
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_rate_limiter_concurrent() {
        let config = RateLimitConfig {
            default_user_quota: Quota {
                max_concurrent: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // First two should be allowed
        let r1 = limiter.check(Some("user1"), ProofType::Range).await;
        let r2 = limiter.check(Some("user1"), ProofType::Range).await;
        assert!(r1.allowed);
        assert!(r2.allowed);

        // Third should be denied (concurrent limit)
        let r3 = limiter.check(Some("user1"), ProofType::Range).await;
        assert!(!r3.allowed);
        assert_eq!(r3.limit_hit, Some("concurrent_limit".to_string()));

        // Release one slot
        limiter.release("user1").await;

        // Now should be allowed again
        let r4 = limiter.check(Some("user1"), ProofType::Range).await;
        assert!(r4.allowed);
    }

    #[tokio::test]
    async fn test_rate_limiter_proof_type_restriction() {
        let config = RateLimitConfig {
            default_user_quota: Quota {
                allowed_proof_types: vec![ProofType::Range],
                ..Default::default()
            },
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        let r1 = limiter.check(Some("user1"), ProofType::Range).await;
        assert!(r1.allowed);

        let r2 = limiter.check(Some("user1"), ProofType::GradientIntegrity).await;
        assert!(!r2.allowed);
        assert_eq!(r2.limit_hit, Some("proof_type_not_allowed".to_string()));
    }

    #[tokio::test]
    async fn test_rate_limiter_bytes() {
        let config = RateLimitConfig {
            default_user_quota: Quota {
                bytes_per_day: 1000,
                ..Default::default()
            },
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Initialize user
        let _ = limiter.check(Some("user1"), ProofType::Range).await;

        assert!(limiter.record_bytes("user1", 500).await);
        assert!(limiter.record_bytes("user1", 400).await);
        assert!(!limiter.record_bytes("user1", 200).await); // Would exceed limit
    }

    #[tokio::test]
    async fn test_rate_limiter_remaining() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        // Initialize user
        let _ = limiter.check(Some("user1"), ProofType::Range).await;

        let remaining = limiter.get_remaining("user1").await.unwrap();
        assert!(remaining.hourly > 0);
        assert!(remaining.daily > 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_stats() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        for _ in 0..5 {
            let _ = limiter.check(Some("user1"), ProofType::Range).await;
        }

        let stats = limiter.stats().await;
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.allowed_requests, 5);
    }

    #[test]
    fn test_cost_estimation() {
        let params = CostParams {
            data_size: 1000,
            security_level: 1,
        };

        let range_cost = estimate_cost(ProofType::Range, &params);
        let gradient_cost = estimate_cost(ProofType::GradientIntegrity, &params);

        assert!(gradient_cost > range_cost);
    }

    #[test]
    fn test_rate_limit_result() {
        let allowed = RateLimitResult::allowed(100, Duration::from_secs(60));
        assert!(allowed.allowed);
        assert_eq!(allowed.remaining, 100);

        let denied = RateLimitResult::denied(Duration::from_secs(30), "test_limit");
        assert!(!denied.allowed);
        assert_eq!(denied.retry_after, Some(Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn test_set_quota() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        limiter.set_quota("user1", Quota::premium()).await;

        // Initialize user
        let _ = limiter.check(Some("user1"), ProofType::Range).await;

        let remaining = limiter.get_remaining("user1").await.unwrap();
        // Premium has 10000 per hour
        assert!(remaining.hourly > 9000);
    }

    #[tokio::test]
    async fn test_reset() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        for _ in 0..5 {
            let _ = limiter.check(Some("user1"), ProofType::Range).await;
        }

        limiter.reset().await;

        let stats = limiter.stats().await;
        assert_eq!(stats.total_requests, 0);
    }
}
