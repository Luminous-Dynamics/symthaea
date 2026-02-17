//! Mock Ethereum Client for Testing
//!
//! Provides a mock implementation of Ethereum client operations
//! for integration testing without requiring actual blockchain connectivity.
//!
//! ## P3-04 Remediation: Rate Limiting & Realistic Delays
//!
//! The mock client now supports:
//! - Rate limiting (configurable tx/sec)
//! - Simulated block time delays
//! - Realistic network latency simulation

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

use super::types::*;

/// Result of a mock operation.
pub type MockResult<T> = Result<T, MockError>;

/// Mock error types.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MockError {
    #[error("Simulated transaction failure: {0}")]
    SimulatedFailure(String),

    #[error("Insufficient funds: {0}")]
    InsufficientFunds(String),

    #[error("Invalid address: {0}")]
    InvalidAddress(String),
}

/// Recorded payment distribution request for verification.
#[derive(Debug, Clone)]
pub struct RecordedPayment {
    pub model_id: String,
    pub round: u64,
    pub total_amount_wei: String,
    pub splits: Vec<PaymentSplit>,
    pub tx_hash: String,
    pub block_number: u64,
}

/// Recorded anchor request for verification.
#[derive(Debug, Clone)]
pub struct RecordedAnchor {
    pub agent_id: String,
    pub eth_address: String,
    pub score_bps: u64,
    pub round: u64,
    pub evidence_hash: Option<String>,
    pub tx_hash: String,
    pub block_number: u64,
}

/// Mock Ethereum client for testing.
///
/// Records all operations for later verification and allows
/// configuring success/failure behavior.
///
/// ## P3-04: Realistic Testing Features
///
/// Use `with_realistic_delays()` to enable rate limiting and latency simulation:
/// ```rust,ignore
/// let client = MockEthereumClient::with_realistic_delays(RateLimiterConfig::polygon_realistic());
/// ```
pub struct MockEthereumClient {
    /// Counter for generating unique tx hashes.
    tx_counter: AtomicU64,

    /// Current simulated block number.
    block_number: AtomicU64,

    /// Recorded payment distributions.
    payments: Arc<RwLock<Vec<RecordedPayment>>>,

    /// Recorded anchor operations.
    anchors: Arc<RwLock<Vec<RecordedAnchor>>>,

    /// Recorded contributions.
    contributions: Arc<RwLock<Vec<(String, u64, Vec<(String, u64)>)>>>,

    /// Failure configuration.
    failure_config: Arc<RwLock<FailureConfig>>,

    /// Simulated gas price.
    gas_price: u64,

    /// P3-04: Rate limiter for realistic testing.
    rate_limiter: RateLimiter,
}

/// Configuration for simulating failures.
#[derive(Debug, Clone, Default)]
pub struct FailureConfig {
    /// Fail the next N payment requests.
    pub fail_next_payments: u32,

    /// Fail the next N anchor requests.
    pub fail_next_anchors: u32,

    /// Specific model IDs that should fail.
    pub failing_model_ids: Vec<String>,

    /// Specific agent IDs that should fail.
    pub failing_agent_ids: Vec<String>,
}

/// P3-04: Rate limiting configuration for realistic testing.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Maximum transactions per second (0 = unlimited).
    pub max_tx_per_second: u32,

    /// Simulated block time in milliseconds.
    pub block_time_ms: u64,

    /// Minimum network latency in milliseconds.
    pub min_latency_ms: u64,

    /// Maximum network latency in milliseconds (for jitter).
    pub max_latency_ms: u64,

    /// Whether to enforce rate limits.
    pub enabled: bool,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            max_tx_per_second: 0, // Unlimited by default for fast tests
            block_time_ms: 0,     // No delay by default
            min_latency_ms: 0,
            max_latency_ms: 0,
            enabled: false,
        }
    }
}

impl RateLimiterConfig {
    /// Create a config with realistic Ethereum mainnet delays.
    pub fn mainnet_realistic() -> Self {
        Self {
            max_tx_per_second: 15,  // ~15 TPS on mainnet
            block_time_ms: 12000,   // 12 second blocks
            min_latency_ms: 50,
            max_latency_ms: 200,
            enabled: true,
        }
    }

    /// Create a config with realistic Polygon delays.
    pub fn polygon_realistic() -> Self {
        Self {
            max_tx_per_second: 30,  // Higher TPS
            block_time_ms: 2000,    // 2 second blocks
            min_latency_ms: 20,
            max_latency_ms: 100,
            enabled: true,
        }
    }

    /// Create a config for stress testing with moderate limits.
    pub fn stress_test() -> Self {
        Self {
            max_tx_per_second: 10,
            block_time_ms: 100,     // Fast blocks for testing
            min_latency_ms: 5,
            max_latency_ms: 20,
            enabled: true,
        }
    }
}

/// P3-04: Simple token bucket rate limiter.
struct RateLimiter {
    config: RateLimiterConfig,
    last_tx_time: RwLock<Instant>,
    tx_count_this_second: AtomicU64,
    second_start: RwLock<Instant>,
}

impl RateLimiter {
    fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            last_tx_time: RwLock::new(Instant::now()),
            tx_count_this_second: AtomicU64::new(0),
            second_start: RwLock::new(Instant::now()),
        }
    }

    /// Wait for rate limit and apply simulated latency.
    /// Returns the simulated latency applied.
    async fn wait_for_slot(&self) -> Duration {
        if !self.config.enabled {
            return Duration::ZERO;
        }

        // Check rate limit
        if self.config.max_tx_per_second > 0 {
            let now = Instant::now();
            let mut second_start = self.second_start.write().await;

            // Reset counter if we're in a new second
            if now.duration_since(*second_start) >= Duration::from_secs(1) {
                *second_start = now;
                self.tx_count_this_second.store(0, Ordering::SeqCst);
            }

            let count = self.tx_count_this_second.fetch_add(1, Ordering::SeqCst);
            if count >= self.config.max_tx_per_second as u64 {
                // Wait until next second
                let wait_time = Duration::from_secs(1) - now.duration_since(*second_start);
                if wait_time > Duration::ZERO {
                    sleep(wait_time).await;
                }
                *second_start = Instant::now();
                self.tx_count_this_second.store(1, Ordering::SeqCst);
            }
        }

        // Apply simulated latency
        let latency = if self.config.max_latency_ms > self.config.min_latency_ms {
            let range = self.config.max_latency_ms - self.config.min_latency_ms;
            let jitter = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as u64) % range;
            Duration::from_millis(self.config.min_latency_ms + jitter)
        } else {
            Duration::from_millis(self.config.min_latency_ms)
        };

        if latency > Duration::ZERO {
            sleep(latency).await;
        }

        // Update last tx time
        *self.last_tx_time.write().await = Instant::now();

        latency
    }

    /// Get simulated block time.
    fn block_time(&self) -> Duration {
        Duration::from_millis(self.config.block_time_ms)
    }
}

impl Default for MockEthereumClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEthereumClient {
    /// Create a new mock Ethereum client with no rate limiting (fast tests).
    pub fn new() -> Self {
        Self {
            tx_counter: AtomicU64::new(1),
            block_number: AtomicU64::new(1000000),
            payments: Arc::new(RwLock::new(Vec::new())),
            anchors: Arc::new(RwLock::new(Vec::new())),
            contributions: Arc::new(RwLock::new(Vec::new())),
            failure_config: Arc::new(RwLock::new(FailureConfig::default())),
            gas_price: 20_000_000_000, // 20 gwei
            rate_limiter: RateLimiter::new(RateLimiterConfig::default()),
        }
    }

    /// P3-04: Create a mock client with realistic delays for integration testing.
    ///
    /// Use preset configs like `RateLimiterConfig::polygon_realistic()` or
    /// `RateLimiterConfig::mainnet_realistic()`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = MockEthereumClient::with_realistic_delays(
    ///     RateLimiterConfig::polygon_realistic()
    /// );
    /// // Transactions will now have realistic delays and rate limits
    /// ```
    pub fn with_realistic_delays(config: RateLimiterConfig) -> Self {
        Self {
            tx_counter: AtomicU64::new(1),
            block_number: AtomicU64::new(1000000),
            payments: Arc::new(RwLock::new(Vec::new())),
            anchors: Arc::new(RwLock::new(Vec::new())),
            contributions: Arc::new(RwLock::new(Vec::new())),
            failure_config: Arc::new(RwLock::new(FailureConfig::default())),
            gas_price: 20_000_000_000,
            rate_limiter: RateLimiter::new(config),
        }
    }

    /// P3-04: Create a mock client configured for stress testing.
    pub fn for_stress_test() -> Self {
        Self::with_realistic_delays(RateLimiterConfig::stress_test())
    }

    /// Configure failures for testing error handling.
    pub async fn configure_failures(&self, config: FailureConfig) {
        *self.failure_config.write().await = config;
    }

    /// Get the current simulated block number.
    pub fn block_number(&self) -> u64 {
        self.block_number.load(Ordering::SeqCst)
    }

    /// Advance the simulated block number.
    pub fn advance_blocks(&self, count: u64) {
        self.block_number.fetch_add(count, Ordering::SeqCst);
    }

    /// Get current gas price.
    pub fn gas_price(&self) -> u64 {
        self.gas_price
    }

    /// Generate a mock transaction hash.
    fn next_tx_hash(&self) -> String {
        let id = self.tx_counter.fetch_add(1, Ordering::SeqCst);
        format!("0x{:064x}", id)
    }

    /// Generate a mock block hash.
    fn block_hash(&self) -> String {
        let block = self.block_number.load(Ordering::SeqCst);
        format!("0x{:064x}", block * 1000)
    }

    /// Distribute payment to contributors.
    ///
    /// P3-04: Now respects rate limiting and simulated delays when configured.
    pub async fn distribute_payment(
        &self,
        request: PaymentDistributionRequest,
    ) -> MockResult<PaymentDistributionResult> {
        // P3-04: Apply rate limiting and latency
        self.rate_limiter.wait_for_slot().await;

        // Check for configured failures
        {
            let mut config = self.failure_config.write().await;

            if config.fail_next_payments > 0 {
                config.fail_next_payments -= 1;
                return Err(MockError::SimulatedFailure(
                    "Configured to fail payment".to_string(),
                ));
            }

            if config.failing_model_ids.contains(&request.model_id) {
                return Err(MockError::SimulatedFailure(format!(
                    "Model {} configured to fail",
                    request.model_id
                )));
            }
        }

        // Validate splits sum to 10000
        let total_bps: u64 = request.splits.iter().map(|s| s.basis_points).sum();
        if total_bps != 10000 && !request.splits.is_empty() {
            return Err(MockError::SimulatedFailure(format!(
                "Splits must sum to 10000, got {}",
                total_bps
            )));
        }

        // Generate mock transaction
        let tx_hash = self.next_tx_hash();
        let block_number = self.block_number.fetch_add(1, Ordering::SeqCst);

        // Record the payment
        let recorded = RecordedPayment {
            model_id: request.model_id.clone(),
            round: request.round,
            total_amount_wei: request.total_amount_wei.clone(),
            splits: request.splits.clone(),
            tx_hash: tx_hash.clone(),
            block_number,
        };
        self.payments.write().await.push(recorded);

        Ok(PaymentDistributionResult {
            tx_hash,
            block_number: Some(block_number),
            gas_used: 150000,
            effective_gas_price: self.gas_price,
            status: true,
            error: None,
        })
    }

    /// Anchor reputation on-chain.
    ///
    /// P3-04: Now respects rate limiting and simulated delays when configured.
    pub async fn anchor_reputation(
        &self,
        update: ReputationUpdate,
    ) -> MockResult<AnchorResult> {
        // P3-04: Apply rate limiting and latency
        self.rate_limiter.wait_for_slot().await;

        // Check for configured failures
        {
            let mut config = self.failure_config.write().await;

            if config.fail_next_anchors > 0 {
                config.fail_next_anchors -= 1;
                return Err(MockError::SimulatedFailure(
                    "Configured to fail anchor".to_string(),
                ));
            }

            if config.failing_agent_ids.contains(&update.agent_id) {
                return Err(MockError::SimulatedFailure(format!(
                    "Agent {} configured to fail",
                    update.agent_id
                )));
            }
        }

        // Validate score bounds
        if update.score_bps > 10000 {
            return Err(MockError::SimulatedFailure(format!(
                "Score must be <= 10000, got {}",
                update.score_bps
            )));
        }

        // Generate mock transaction
        let tx_hash = self.next_tx_hash();
        let block_number = self.block_number.fetch_add(1, Ordering::SeqCst);

        // Record the anchor
        let recorded = RecordedAnchor {
            agent_id: update.agent_id.clone(),
            eth_address: update.eth_address.clone(),
            score_bps: update.score_bps,
            round: update.round,
            evidence_hash: update.evidence_hash.clone(),
            tx_hash: tx_hash.clone(),
            block_number,
        };
        self.anchors.write().await.push(recorded);

        Ok(AnchorResult {
            tx_hash,
            block_number,
            block_hash: self.block_hash(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            commitment_id: Some(format!("anchor-{}-{}", update.agent_id, update.round)),
        })
    }

    /// Record FL contributions on-chain.
    ///
    /// P3-04: Now respects rate limiting and simulated delays when configured.
    pub async fn record_contributions(
        &self,
        model_id: &str,
        round: u64,
        contributions: &[(String, u64)],
        _proof_hash: &str,
    ) -> MockResult<AnchorResult> {
        // P3-04: Apply rate limiting and latency
        self.rate_limiter.wait_for_slot().await;

        let tx_hash = self.next_tx_hash();
        let block_number = self.block_number.fetch_add(1, Ordering::SeqCst);

        // Record the contributions
        self.contributions.write().await.push((
            model_id.to_string(),
            round,
            contributions.to_vec(),
        ));

        Ok(AnchorResult {
            tx_hash,
            block_number,
            block_hash: self.block_hash(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            commitment_id: Some(format!("contrib-{}-{}", model_id, round)),
        })
    }

    // === Verification Methods ===

    /// Get all recorded payments.
    pub async fn get_payments(&self) -> Vec<RecordedPayment> {
        self.payments.read().await.clone()
    }

    /// Get payments for a specific model.
    pub async fn get_payments_for_model(&self, model_id: &str) -> Vec<RecordedPayment> {
        self.payments
            .read()
            .await
            .iter()
            .filter(|p| p.model_id == model_id)
            .cloned()
            .collect()
    }

    /// Get all recorded anchors.
    pub async fn get_anchors(&self) -> Vec<RecordedAnchor> {
        self.anchors.read().await.clone()
    }

    /// Get anchors for a specific agent.
    pub async fn get_anchors_for_agent(&self, agent_id: &str) -> Vec<RecordedAnchor> {
        self.anchors
            .read()
            .await
            .iter()
            .filter(|a| a.agent_id == agent_id)
            .cloned()
            .collect()
    }

    /// Get all recorded contributions.
    pub async fn get_contributions(&self) -> Vec<(String, u64, Vec<(String, u64)>)> {
        self.contributions.read().await.clone()
    }

    /// Get total number of transactions.
    pub fn total_transactions(&self) -> u64 {
        self.tx_counter.load(Ordering::SeqCst) - 1
    }

    /// Clear all recorded operations.
    pub async fn clear_records(&self) {
        self.payments.write().await.clear();
        self.anchors.write().await.clear();
        self.contributions.write().await.clear();
    }

    /// Assert a payment was made with expected values.
    pub async fn assert_payment_made(
        &self,
        model_id: &str,
        round: u64,
        expected_recipients: usize,
    ) -> bool {
        let payments = self.get_payments_for_model(model_id).await;
        payments.iter().any(|p| {
            p.round == round && p.splits.len() == expected_recipients
        })
    }

    /// Assert an anchor was made for an agent.
    pub async fn assert_anchor_made(
        &self,
        agent_id: &str,
        expected_score_bps: u64,
    ) -> bool {
        let anchors = self.get_anchors_for_agent(agent_id).await;
        anchors.iter().any(|a| a.score_bps == expected_score_bps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_payment_distribution() {
        let client = MockEthereumClient::new();

        let request = PaymentDistributionRequest {
            model_id: "test-model".to_string(),
            round: 1,
            total_amount_wei: "1000000000000000000".to_string(),
            splits: vec![
                PaymentSplit {
                    address: "0x123".to_string(),
                    basis_points: 6000,
                    node_id: "node1".to_string(),
                },
                PaymentSplit {
                    address: "0x456".to_string(),
                    basis_points: 4000,
                    node_id: "node2".to_string(),
                },
            ],
            platform_fee_bps: 0,
        };

        let result = client.distribute_payment(request).await.unwrap();
        assert!(result.status);
        assert!(result.tx_hash.starts_with("0x"));

        // Verify recorded
        let payments = client.get_payments().await;
        assert_eq!(payments.len(), 1);
        assert_eq!(payments[0].model_id, "test-model");
        assert_eq!(payments[0].splits.len(), 2);
    }

    #[tokio::test]
    async fn test_mock_anchor_reputation() {
        let client = MockEthereumClient::new();

        let update = ReputationUpdate {
            agent_id: "agent-1".to_string(),
            eth_address: "0xabc".to_string(),
            score_bps: 8500,
            round: 5,
            evidence_hash: Some("0xhash".to_string()),
        };

        let result = client.anchor_reputation(update).await.unwrap();
        assert!(result.tx_hash.starts_with("0x"));
        assert!(result.commitment_id.is_some());

        // Verify recorded
        let anchors = client.get_anchors().await;
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].agent_id, "agent-1");
        assert_eq!(anchors[0].score_bps, 8500);
    }

    #[tokio::test]
    async fn test_mock_configured_failure() {
        let client = MockEthereumClient::new();

        // Configure to fail next 2 payments
        client
            .configure_failures(FailureConfig {
                fail_next_payments: 2,
                ..Default::default()
            })
            .await;

        let request = PaymentDistributionRequest {
            model_id: "test".to_string(),
            round: 1,
            total_amount_wei: "1000".to_string(),
            splits: vec![],
            platform_fee_bps: 0,
        };

        // First two should fail
        assert!(client.distribute_payment(request.clone()).await.is_err());
        assert!(client.distribute_payment(request.clone()).await.is_err());

        // Third should succeed
        assert!(client.distribute_payment(request).await.is_ok());
    }

    #[tokio::test]
    async fn test_mock_invalid_splits() {
        let client = MockEthereumClient::new();

        let request = PaymentDistributionRequest {
            model_id: "test".to_string(),
            round: 1,
            total_amount_wei: "1000".to_string(),
            splits: vec![PaymentSplit {
                address: "0x1".to_string(),
                basis_points: 5000, // Only 50%, should be 100%
                node_id: "n1".to_string(),
            }],
            platform_fee_bps: 0,
        };

        let result = client.distribute_payment(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_assertion_helpers() {
        let client = MockEthereumClient::new();

        let request = PaymentDistributionRequest {
            model_id: "model-x".to_string(),
            round: 10,
            total_amount_wei: "1000".to_string(),
            splits: vec![
                PaymentSplit {
                    address: "0x1".to_string(),
                    basis_points: 10000,
                    node_id: "n1".to_string(),
                },
            ],
            platform_fee_bps: 0,
        };

        client.distribute_payment(request).await.unwrap();

        assert!(client.assert_payment_made("model-x", 10, 1).await);
        assert!(!client.assert_payment_made("model-x", 11, 1).await);
        assert!(!client.assert_payment_made("other", 10, 1).await);
    }

    // P3-04: Rate limiting tests
    #[tokio::test]
    async fn test_rate_limiter_config_presets() {
        let mainnet = RateLimiterConfig::mainnet_realistic();
        assert_eq!(mainnet.max_tx_per_second, 15);
        assert_eq!(mainnet.block_time_ms, 12000);
        assert!(mainnet.enabled);

        let polygon = RateLimiterConfig::polygon_realistic();
        assert_eq!(polygon.max_tx_per_second, 30);
        assert_eq!(polygon.block_time_ms, 2000);
        assert!(polygon.enabled);

        let stress = RateLimiterConfig::stress_test();
        assert_eq!(stress.max_tx_per_second, 10);
        assert!(stress.enabled);
    }

    #[tokio::test]
    async fn test_mock_with_realistic_delays() {
        // Create client with minimal delays for test speed
        let config = RateLimiterConfig {
            max_tx_per_second: 100, // High limit
            block_time_ms: 0,
            min_latency_ms: 1,
            max_latency_ms: 5,
            enabled: true,
        };

        let client = MockEthereumClient::with_realistic_delays(config);

        let request = PaymentDistributionRequest {
            model_id: "test".to_string(),
            round: 1,
            total_amount_wei: "1000".to_string(),
            splits: vec![PaymentSplit {
                address: "0x1".to_string(),
                basis_points: 10000,
                node_id: "n1".to_string(),
            }],
            platform_fee_bps: 0,
        };

        let start = std::time::Instant::now();
        let result = client.distribute_payment(request).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Should have some latency applied
        assert!(elapsed.as_millis() >= 1, "Expected at least 1ms delay");
    }

    #[tokio::test]
    async fn test_mock_for_stress_test_constructor() {
        let client = MockEthereumClient::for_stress_test();

        // Should work without panicking
        let update = ReputationUpdate {
            agent_id: "stress-test-agent".to_string(),
            eth_address: "0xtest".to_string(),
            score_bps: 5000,
            round: 1,
            evidence_hash: None,
        };

        let result = client.anchor_reputation(update).await;
        assert!(result.is_ok());
    }
}
