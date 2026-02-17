//! Composite storage backend with intelligent routing.
//!
//! Routes operations to appropriate backends based on data type and age:
//! - Active round data → Primary (fast) backend
//! - Archived data → Secondary (durable) backend
//! - Queries → Checks both backends

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::backend::{StorageBackend, StorageError, StorageResult};
use super::types::*;

/// Routing strategy for composite backend.
#[derive(Clone, Copy, Debug, Default)]
pub enum RoutingStrategy {
    /// All writes go to primary, reads check both.
    #[default]
    PrimaryFirst,
    /// Writes replicate to both backends.
    Replicate,
    /// Route based on data age (rounds).
    Tiered {
        /// Rounds older than this go to archive.
        archive_threshold: u64,
    },
}

/// Composite storage backend.
///
/// Combines multiple backends with intelligent routing:
/// - Primary backend for hot/active data
/// - Archive backend for cold/historical data
/// - Automatic failover and replication options
pub struct CompositeBackend {
    /// Primary (fast) backend.
    primary: Arc<RwLock<Box<dyn StorageBackend>>>,
    /// Archive (durable) backend.
    archive: Option<Arc<RwLock<Box<dyn StorageBackend>>>>,
    /// Routing strategy.
    strategy: RoutingStrategy,
    /// Current round number (for tiered routing).
    current_round: Arc<RwLock<u64>>,
}

impl CompositeBackend {
    /// Create with primary backend only.
    pub fn new(primary: Box<dyn StorageBackend>) -> Self {
        Self {
            primary: Arc::new(RwLock::new(primary)),
            archive: None,
            strategy: RoutingStrategy::PrimaryFirst,
            current_round: Arc::new(RwLock::new(0)),
        }
    }

    /// Add archive backend.
    pub fn with_archive(mut self, archive: Box<dyn StorageBackend>) -> Self {
        self.archive = Some(Arc::new(RwLock::new(archive)));
        self
    }

    /// Set routing strategy.
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Update current round (for tiered routing).
    pub async fn set_current_round(&self, round: u64) {
        *self.current_round.write().await = round;
    }

    /// Get current round.
    pub async fn current_round(&self) -> u64 {
        *self.current_round.read().await
    }

    /// Check if round should go to archive.
    async fn should_archive(&self, round_num: u64) -> bool {
        match self.strategy {
            RoutingStrategy::Tiered { archive_threshold } => {
                let current = *self.current_round.read().await;
                current > round_num && (current - round_num) > archive_threshold
            }
            _ => false,
        }
    }

    /// Archive old rounds from primary to archive backend.
    pub async fn archive_old_rounds(&self, threshold_rounds: u64) -> StorageResult<u64> {
        let archive = self
            .archive
            .as_ref()
            .ok_or(StorageError::Config("No archive backend configured".to_string()))?;

        let current = *self.current_round.read().await;
        if current < threshold_rounds {
            return Ok(0);
        }

        let cutoff_round = current - threshold_rounds;
        let mut archived_count = 0u64;

        // Get gradients from primary that should be archived
        let primary = self.primary.read().await;

        for round in 0..=cutoff_round {
            let gradients = primary.get_gradients_by_round(round).await?;

            if !gradients.is_empty() {
                let archive = archive.write().await;
                for gradient in &gradients {
                    archive.store_gradient(gradient).await?;
                    archived_count += 1;
                }
                debug!(round = round, count = gradients.len(), "Archived round to secondary");
            }
        }

        info!(archived_count = archived_count, "Archive operation complete");
        Ok(archived_count)
    }
}

#[async_trait]
impl StorageBackend for CompositeBackend {
    fn backend_type(&self) -> BackendType {
        // Return primary's type since that's the main backend
        BackendType::LocalFile // Composite doesn't have its own type
    }

    fn is_connected(&self) -> bool {
        // Check synchronously - this is a limitation
        // In practice, call health_check for accurate status
        true
    }

    async fn connect(&mut self) -> StorageResult<()> {
        // Connect primary
        {
            let mut primary = self.primary.write().await;
            primary.connect().await?;
        }

        // Connect archive if present
        if let Some(archive) = &self.archive {
            let mut archive = archive.write().await;
            archive.connect().await?;
        }

        info!("Composite backend connected");
        Ok(())
    }

    async fn disconnect(&mut self) -> StorageResult<()> {
        // Disconnect primary
        {
            let mut primary = self.primary.write().await;
            primary.disconnect().await?;
        }

        // Disconnect archive if present
        if let Some(archive) = &self.archive {
            let mut archive = archive.write().await;
            archive.disconnect().await?;
        }

        info!("Composite backend disconnected");
        Ok(())
    }

    async fn health_check(&self) -> StorageResult<HealthStatus> {
        let primary = self.primary.read().await;
        let mut primary_health = primary.health_check().await?;

        // Check archive if present
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            let archive_health = archive.health_check().await?;

            primary_health.metadata.insert(
                "archive_healthy".to_string(),
                serde_json::json!(archive_health.healthy),
            );
            primary_health.metadata.insert(
                "archive_latency_ms".to_string(),
                serde_json::json!(archive_health.latency_ms),
            );
        }

        primary_health.metadata.insert(
            "strategy".to_string(),
            serde_json::json!(format!("{:?}", self.strategy)),
        );

        Ok(primary_health)
    }

    async fn store_gradient(&self, record: &GradientRecord) -> StorageResult<String> {
        match self.strategy {
            RoutingStrategy::Replicate => {
                // Write to both
                let id = {
                    let primary = self.primary.read().await;
                    primary.store_gradient(record).await?
                };

                if let Some(archive) = &self.archive {
                    let archive = archive.read().await;
                    if let Err(e) = archive.store_gradient(record).await {
                        warn!(error = %e, "Failed to replicate to archive");
                    }
                }

                Ok(id)
            }
            RoutingStrategy::Tiered { .. } => {
                // Check if should go to archive
                if self.should_archive(record.round_num).await {
                    if let Some(archive) = &self.archive {
                        let archive = archive.read().await;
                        return archive.store_gradient(record).await;
                    }
                }

                let primary = self.primary.read().await;
                primary.store_gradient(record).await
            }
            RoutingStrategy::PrimaryFirst => {
                let primary = self.primary.read().await;
                primary.store_gradient(record).await
            }
        }
    }

    async fn get_gradient(&self, gradient_id: &str) -> StorageResult<Option<GradientRecord>> {
        // Try primary first
        let primary = self.primary.read().await;
        if let Some(record) = primary.get_gradient(gradient_id).await? {
            return Ok(Some(record));
        }

        // Try archive if present
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            return archive.get_gradient(gradient_id).await;
        }

        Ok(None)
    }

    async fn get_gradients_by_round(&self, round_num: u64) -> StorageResult<Vec<GradientRecord>> {
        let mut results = Vec::new();

        // Get from primary
        {
            let primary = self.primary.read().await;
            results.extend(primary.get_gradients_by_round(round_num).await?);
        }

        // Get from archive if tiered and round is old
        if let Some(archive) = &self.archive {
            if self.should_archive(round_num).await || results.is_empty() {
                let archive = archive.read().await;
                let archive_results = archive.get_gradients_by_round(round_num).await?;

                // Merge, avoiding duplicates by ID
                let existing_ids: std::collections::HashSet<String> =
                    results.iter().map(|r| r.id.clone()).collect();

                for record in archive_results {
                    if !existing_ids.contains(&record.id) {
                        results.push(record);
                    }
                }
            }
        }

        // Sort by timestamp
        results.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        Ok(results)
    }

    async fn get_gradients_by_node(&self, node_id: &str) -> StorageResult<Vec<GradientRecord>> {
        let mut results = Vec::new();

        // Get from primary
        {
            let primary = self.primary.read().await;
            results.extend(primary.get_gradients_by_node(node_id).await?);
        }

        // Get from archive
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            let archive_results = archive.get_gradients_by_node(node_id).await?;

            let existing_ids: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            for record in archive_results {
                if !existing_ids.contains(&record.id) {
                    results.push(record);
                }
            }
        }

        results.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        Ok(results)
    }

    async fn verify_gradient_integrity(&self, gradient_id: &str) -> StorageResult<bool> {
        // Try primary
        let primary = self.primary.read().await;
        match primary.verify_gradient_integrity(gradient_id).await {
            Ok(result) => return Ok(result),
            Err(StorageError::NotFound(_)) => {}
            Err(e) => return Err(e),
        }

        // Try archive
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            return archive.verify_gradient_integrity(gradient_id).await;
        }

        Err(StorageError::NotFound(gradient_id.to_string()))
    }

    async fn issue_credit(
        &self,
        holder: &str,
        amount: u64,
        earned_from: &str,
    ) -> StorageResult<String> {
        // Credits always go to primary (ledger consistency)
        let primary = self.primary.read().await;
        let id = primary.issue_credit(holder, amount, earned_from).await?;

        // Replicate if configured
        if matches!(self.strategy, RoutingStrategy::Replicate) {
            if let Some(archive) = &self.archive {
                let archive = archive.read().await;
                let _ = archive.issue_credit(holder, amount, earned_from).await;
            }
        }

        Ok(id)
    }

    async fn get_credit_balance(&self, node_id: &str) -> StorageResult<u64> {
        // Credits from primary (authoritative)
        let primary = self.primary.read().await;
        primary.get_credit_balance(node_id).await
    }

    async fn get_credit_history(&self, node_id: &str) -> StorageResult<Vec<CreditRecord>> {
        let primary = self.primary.read().await;
        primary.get_credit_history(node_id).await
    }

    async fn get_reputation(&self, node_id: &str) -> StorageResult<ReputationData> {
        let primary = self.primary.read().await;
        primary.get_reputation(node_id).await
    }

    async fn update_reputation(
        &self,
        node_id: &str,
        score_delta: f32,
        reason: &str,
    ) -> StorageResult<()> {
        let primary = self.primary.read().await;
        primary.update_reputation(node_id, score_delta, reason).await
    }

    async fn log_byzantine_event(&self, event: &ByzantineEvent) -> StorageResult<String> {
        // Byzantine events go to primary
        let primary = self.primary.read().await;
        let id = primary.log_byzantine_event(event).await?;

        // Replicate if configured
        if matches!(self.strategy, RoutingStrategy::Replicate) {
            if let Some(archive) = &self.archive {
                let archive = archive.read().await;
                let _ = archive.log_byzantine_event(event).await;
            }
        }

        Ok(id)
    }

    async fn get_byzantine_events(
        &self,
        node_id: Option<&str>,
        round_num: Option<u64>,
    ) -> StorageResult<Vec<ByzantineEvent>> {
        let mut results = Vec::new();

        // Get from primary
        {
            let primary = self.primary.read().await;
            results.extend(primary.get_byzantine_events(node_id, round_num).await?);
        }

        // Get from archive for historical queries
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            let archive_results = archive.get_byzantine_events(node_id, round_num).await?;

            let existing_ids: std::collections::HashSet<String> =
                results.iter().map(|e| e.event_id.clone()).collect();

            for event in archive_results {
                if !existing_ids.contains(&event.event_id) {
                    results.push(event);
                }
            }
        }

        results.sort_by(|a, b| b.timestamp.partial_cmp(&a.timestamp).unwrap());
        Ok(results)
    }

    async fn get_stats(&self) -> StorageResult<BackendStats> {
        let primary = self.primary.read().await;
        let mut stats = primary.get_stats().await?;

        // Add archive stats if present
        if let Some(archive) = &self.archive {
            let archive = archive.read().await;
            if let Ok(archive_stats) = archive.get_stats().await {
                stats.metadata.insert(
                    "archive_gradients".to_string(),
                    serde_json::json!(archive_stats.total_gradients),
                );
                stats.metadata.insert(
                    "archive_storage_bytes".to_string(),
                    serde_json::json!(archive_stats.storage_size_bytes),
                );
            }
        }

        stats.metadata.insert(
            "composite".to_string(),
            serde_json::json!(true),
        );
        stats.metadata.insert(
            "strategy".to_string(),
            serde_json::json!(format!("{:?}", self.strategy)),
        );

        Ok(stats)
    }
}

#[cfg(test)]
#[cfg(feature = "storage-local")]
mod tests {
    use super::*;
    use crate::storage::LocalFileBackend;
    use tempfile::TempDir;

    async fn setup_composite() -> (CompositeBackend, TempDir, TempDir) {
        let primary_dir = TempDir::new().unwrap();
        let archive_dir = TempDir::new().unwrap();

        let primary = Box::new(LocalFileBackend::new(primary_dir.path()));
        let archive = Box::new(LocalFileBackend::new(archive_dir.path()));

        let mut composite = CompositeBackend::new(primary)
            .with_archive(archive)
            .with_strategy(RoutingStrategy::PrimaryFirst);

        composite.connect().await.unwrap();

        (composite, primary_dir, archive_dir)
    }

    #[tokio::test]
    async fn test_composite_basic_operations() {
        let (composite, _p, _a) = setup_composite().await;

        // Store gradient
        let record = GradientRecord::new("node_1", 1, vec![1.0, 2.0], "hash1");
        let id = composite.store_gradient(&record).await.unwrap();

        // Retrieve
        let retrieved = composite.get_gradient(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.node_id, "node_1");
    }

    #[tokio::test]
    async fn test_composite_replication() {
        let primary_dir = TempDir::new().unwrap();
        let archive_dir = TempDir::new().unwrap();

        let primary = Box::new(LocalFileBackend::new(primary_dir.path()));
        let archive = Box::new(LocalFileBackend::new(archive_dir.path()));

        let mut composite = CompositeBackend::new(primary)
            .with_archive(archive)
            .with_strategy(RoutingStrategy::Replicate);

        composite.connect().await.unwrap();

        // Store gradient
        let record = GradientRecord::new("node_1", 1, vec![1.0], "hash1");
        let id = composite.store_gradient(&record).await.unwrap();

        // Should be in both backends
        // Check primary directly
        let mut primary_check = LocalFileBackend::new(primary_dir.path());
        primary_check.connect().await.unwrap();
        assert!(primary_check.get_gradient(&id).await.unwrap().is_some());

        // Check archive directly
        let mut archive_check = LocalFileBackend::new(archive_dir.path());
        archive_check.connect().await.unwrap();
        assert!(archive_check.get_gradient(&id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_composite_health_check() {
        let (composite, _p, _a) = setup_composite().await;

        let health = composite.health_check().await.unwrap();
        assert!(health.healthy);
        assert!(health.metadata.contains_key("archive_healthy"));
    }

    #[tokio::test]
    async fn test_composite_stats() {
        let (composite, _p, _a) = setup_composite().await;

        // Store some data
        let record = GradientRecord::new("node_1", 1, vec![1.0], "hash1");
        composite.store_gradient(&record).await.unwrap();

        let stats = composite.get_stats().await.unwrap();
        assert!(stats.metadata.contains_key("composite"));
        assert_eq!(stats.metadata["composite"], serde_json::json!(true));
    }
}
