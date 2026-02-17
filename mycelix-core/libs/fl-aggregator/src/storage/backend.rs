//! Storage backend trait definition.

use async_trait::async_trait;
use thiserror::Error;

use super::types::*;

/// Storage backend errors.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Connection error.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Not connected.
    #[error("Not connected to storage backend")]
    NotConnected,

    /// Resource not found.
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Integrity check failed.
    #[error("Integrity error: {0}")]
    Integrity(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Query error.
    #[error("Query error: {0}")]
    Query(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Abstract storage backend trait.
///
/// All storage backends (LocalFile, PostgreSQL, Holochain) implement this trait
/// to provide a consistent interface to the FL coordinator.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Get backend type.
    fn backend_type(&self) -> BackendType;

    /// Check if connected.
    fn is_connected(&self) -> bool;

    // ==================== Connection Management ====================

    /// Connect to storage backend.
    async fn connect(&mut self) -> StorageResult<()>;

    /// Disconnect from storage backend.
    async fn disconnect(&mut self) -> StorageResult<()>;

    /// Check backend health.
    async fn health_check(&self) -> StorageResult<HealthStatus>;

    // ==================== Gradient Operations ====================

    /// Store a gradient record.
    ///
    /// Returns the gradient ID.
    async fn store_gradient(&self, record: &GradientRecord) -> StorageResult<String>;

    /// Get a gradient by ID.
    async fn get_gradient(&self, gradient_id: &str) -> StorageResult<Option<GradientRecord>>;

    /// Get all gradients for a round.
    async fn get_gradients_by_round(&self, round_num: u64) -> StorageResult<Vec<GradientRecord>>;

    /// Get gradients by node.
    async fn get_gradients_by_node(&self, node_id: &str) -> StorageResult<Vec<GradientRecord>>;

    /// Verify gradient integrity.
    ///
    /// For immutable backends (Holochain, Blockchain): Always returns true.
    /// For mutable backends (PostgreSQL, LocalFile): Verifies hash.
    async fn verify_gradient_integrity(&self, gradient_id: &str) -> StorageResult<bool>;

    // ==================== Credit Operations ====================

    /// Issue credits to a node.
    ///
    /// Returns the transaction ID.
    async fn issue_credit(
        &self,
        holder: &str,
        amount: u64,
        earned_from: &str,
    ) -> StorageResult<String>;

    /// Get credit balance for a node.
    async fn get_credit_balance(&self, node_id: &str) -> StorageResult<u64>;

    /// Get credit history for a node.
    async fn get_credit_history(&self, node_id: &str) -> StorageResult<Vec<CreditRecord>>;

    // ==================== Reputation Operations ====================

    /// Get reputation data for a node.
    async fn get_reputation(&self, node_id: &str) -> StorageResult<ReputationData>;

    /// Update node reputation.
    async fn update_reputation(
        &self,
        node_id: &str,
        score_delta: f32,
        reason: &str,
    ) -> StorageResult<()>;

    // ==================== Byzantine Event Logging ====================

    /// Log a Byzantine detection event.
    ///
    /// Returns the event ID.
    async fn log_byzantine_event(&self, event: &ByzantineEvent) -> StorageResult<String>;

    /// Get Byzantine events (optionally filtered).
    async fn get_byzantine_events(
        &self,
        node_id: Option<&str>,
        round_num: Option<u64>,
    ) -> StorageResult<Vec<ByzantineEvent>>;

    // ==================== Statistics ====================

    /// Get backend statistics.
    async fn get_stats(&self) -> StorageResult<BackendStats>;
}

/// Backend configuration options.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Operation timeout in seconds.
    pub operation_timeout_secs: u64,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Retry delay in milliseconds.
    pub retry_delay_ms: u64,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            connect_timeout_secs: 30,
            operation_timeout_secs: 10,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = StorageError::NotFound("gradient_123".to_string());
        assert!(err.to_string().contains("gradient_123"));
    }

    #[test]
    fn test_config_defaults() {
        let config = BackendConfig::default();
        assert_eq!(config.connect_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }
}
