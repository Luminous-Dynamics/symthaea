//! LocalFile storage backend.
//!
//! Simple JSON-based file storage for testing and development.
//! No database infrastructure required.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

use super::backend::{BackendConfig, StorageBackend, StorageError, StorageResult};
use super::types::*;

/// LocalFile storage backend.
///
/// Features:
/// - No database required
/// - Human-readable JSON files
/// - Fast iteration for testing
/// - Easy inspection and debugging
pub struct LocalFileBackend {
    /// Base data directory.
    data_dir: PathBuf,
    /// Gradients directory.
    gradients_dir: PathBuf,
    /// Credits directory.
    credits_dir: PathBuf,
    /// Byzantine events directory.
    byzantine_dir: PathBuf,
    /// Reputation file path.
    reputation_file: PathBuf,
    /// Connection state.
    connected: AtomicBool,
    /// Connection timestamp.
    connection_time: Option<Instant>,
    /// Configuration.
    config: BackendConfig,
}

impl LocalFileBackend {
    /// Create a new LocalFile backend.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();

        Self {
            gradients_dir: data_dir.join("gradients"),
            credits_dir: data_dir.join("credits"),
            byzantine_dir: data_dir.join("byzantine_events"),
            reputation_file: data_dir.join("reputation.json"),
            data_dir,
            connected: AtomicBool::new(false),
            connection_time: None,
            config: BackendConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(data_dir: impl AsRef<Path>, config: BackendConfig) -> Self {
        let mut backend = Self::new(data_dir);
        backend.config = config;
        backend
    }

    /// Get current timestamp.
    fn current_timestamp() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    /// Generate a simple UUID.
    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        format!(
            "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
            (timestamp >> 96) as u32,
            (timestamp >> 80) as u16,
            (timestamp >> 64) as u16 & 0x0FFF,
            ((timestamp >> 48) as u16 & 0x3FFF) | 0x8000,
            timestamp as u64 & 0xFFFFFFFFFFFF
        )
    }

    /// Read JSON file.
    async fn read_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &Path,
    ) -> StorageResult<Option<T>> {
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(path).await?;
        serde_json::from_str(&content)
            .map(Some)
            .map_err(|e| StorageError::Serialization(e.to_string()))
    }

    /// Write JSON file.
    async fn write_json<T: serde::Serialize>(&self, path: &Path, data: &T) -> StorageResult<()> {
        let content = serde_json::to_string_pretty(data)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let mut file = fs::File::create(path).await?;
        file.write_all(content.as_bytes()).await?;
        file.sync_all().await?;

        Ok(())
    }

    /// List JSON files in directory.
    async fn list_json_files(&self, dir: &Path) -> StorageResult<Vec<PathBuf>> {
        let mut files = Vec::new();

        let mut entries = fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                files.push(path);
            }
        }

        Ok(files)
    }
}

#[async_trait]
impl StorageBackend for LocalFileBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::LocalFile
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    async fn connect(&mut self) -> StorageResult<()> {
        // Create directory structure
        fs::create_dir_all(&self.data_dir).await?;
        fs::create_dir_all(&self.gradients_dir).await?;
        fs::create_dir_all(&self.credits_dir).await?;
        fs::create_dir_all(&self.byzantine_dir).await?;

        // Initialize reputation file if doesn't exist
        if !self.reputation_file.exists() {
            let empty: HashMap<String, serde_json::Value> = HashMap::new();
            self.write_json(&self.reputation_file, &empty).await?;
        }

        self.connected.store(true, Ordering::Relaxed);
        self.connection_time = Some(Instant::now());

        info!(data_dir = %self.data_dir.display(), "LocalFile backend connected");
        Ok(())
    }

    async fn disconnect(&mut self) -> StorageResult<()> {
        self.connected.store(false, Ordering::Relaxed);
        info!("LocalFile backend disconnected");
        Ok(())
    }

    async fn health_check(&self) -> StorageResult<HealthStatus> {
        if !self.is_connected() {
            return Ok(HealthStatus {
                healthy: false,
                latency_ms: None,
                storage_available: false,
                metadata: [("error".to_string(), serde_json::json!("Not connected"))]
                    .into_iter()
                    .collect(),
            });
        }

        // Test write access
        let test_file = self.data_dir.join(".health_check");
        let start = Instant::now();

        match fs::write(&test_file, "test").await {
            Ok(_) => {
                let _ = fs::remove_file(&test_file).await;
                let latency = start.elapsed().as_secs_f64() * 1000.0;

                // Count files
                let gradient_count = self
                    .list_json_files(&self.gradients_dir)
                    .await
                    .map(|f| f.len())
                    .unwrap_or(0);
                let credit_count = self
                    .list_json_files(&self.credits_dir)
                    .await
                    .map(|f| f.len())
                    .unwrap_or(0);

                Ok(HealthStatus {
                    healthy: true,
                    latency_ms: Some(latency),
                    storage_available: true,
                    metadata: [
                        (
                            "data_dir".to_string(),
                            serde_json::json!(self.data_dir.display().to_string()),
                        ),
                        ("gradient_files".to_string(), serde_json::json!(gradient_count)),
                        ("credit_files".to_string(), serde_json::json!(credit_count)),
                        ("disk_writable".to_string(), serde_json::json!(true)),
                    ]
                    .into_iter()
                    .collect(),
                })
            }
            Err(e) => Ok(HealthStatus {
                healthy: false,
                latency_ms: None,
                storage_available: false,
                metadata: [("error".to_string(), serde_json::json!(e.to_string()))]
                    .into_iter()
                    .collect(),
            }),
        }
    }

    async fn store_gradient(&self, record: &GradientRecord) -> StorageResult<String> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let id = if record.id.is_empty() {
            Self::generate_id()
        } else {
            record.id.clone()
        };

        let mut record = record.clone();
        record.id = id.clone();

        let file_path = self.gradients_dir.join(format!("{}.json", id));
        self.write_json(&file_path, &record).await?;

        debug!(
            gradient_id = %id,
            node_id = %record.node_id,
            encrypted = record.encrypted,
            "Gradient stored"
        );

        Ok(id)
    }

    async fn get_gradient(&self, gradient_id: &str) -> StorageResult<Option<GradientRecord>> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let file_path = self.gradients_dir.join(format!("{}.json", gradient_id));
        self.read_json(&file_path).await
    }

    async fn get_gradients_by_round(&self, round_num: u64) -> StorageResult<Vec<GradientRecord>> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let mut gradients = Vec::new();

        for file_path in self.list_json_files(&self.gradients_dir).await? {
            if let Some(record) = self.read_json::<GradientRecord>(&file_path).await? {
                if record.round_num == round_num && record.validation_passed {
                    gradients.push(record);
                }
            }
        }

        gradients.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        Ok(gradients)
    }

    async fn get_gradients_by_node(&self, node_id: &str) -> StorageResult<Vec<GradientRecord>> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let mut gradients = Vec::new();

        for file_path in self.list_json_files(&self.gradients_dir).await? {
            if let Some(record) = self.read_json::<GradientRecord>(&file_path).await? {
                if record.node_id == node_id {
                    gradients.push(record);
                }
            }
        }

        gradients.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        Ok(gradients)
    }

    async fn verify_gradient_integrity(&self, gradient_id: &str) -> StorageResult<bool> {
        let gradient = self
            .get_gradient(gradient_id)
            .await?
            .ok_or_else(|| StorageError::NotFound(gradient_id.to_string()))?;

        // For raw gradients, verify hash
        if let GradientData::Raw(values) = &gradient.gradient {
            use sha2::{Digest, Sha256};

            let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            let computed_hash = format!("{:x}", Sha256::digest(&bytes));

            return Ok(computed_hash == gradient.gradient_hash);
        }

        // For encrypted gradients, we can't verify without decryption
        Ok(true)
    }

    async fn issue_credit(
        &self,
        holder: &str,
        amount: u64,
        earned_from: &str,
    ) -> StorageResult<String> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let record = CreditRecord {
            transaction_id: Self::generate_id(),
            holder: holder.to_string(),
            amount,
            earned_from: earned_from.to_string(),
            timestamp: Self::current_timestamp(),
            metadata: HashMap::new(),
        };

        let file_path = self
            .credits_dir
            .join(format!("{}.json", record.transaction_id));
        self.write_json(&file_path, &record).await?;

        debug!(
            transaction_id = %record.transaction_id,
            holder = %holder,
            amount = amount,
            "Credit issued"
        );

        Ok(record.transaction_id)
    }

    async fn get_credit_balance(&self, node_id: &str) -> StorageResult<u64> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let mut total = 0u64;

        for file_path in self.list_json_files(&self.credits_dir).await? {
            if let Some(record) = self.read_json::<CreditRecord>(&file_path).await? {
                if record.holder == node_id {
                    total += record.amount;
                }
            }
        }

        Ok(total)
    }

    async fn get_credit_history(&self, node_id: &str) -> StorageResult<Vec<CreditRecord>> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let mut credits = Vec::new();

        for file_path in self.list_json_files(&self.credits_dir).await? {
            if let Some(record) = self.read_json::<CreditRecord>(&file_path).await? {
                if record.holder == node_id {
                    credits.push(record);
                }
            }
        }

        credits.sort_by(|a, b| b.timestamp.partial_cmp(&a.timestamp).unwrap());
        Ok(credits)
    }

    async fn get_reputation(&self, node_id: &str) -> StorageResult<ReputationData> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        // Count gradients
        let mut total_submitted = 0u64;
        let mut accepted = 0u64;

        for file_path in self.list_json_files(&self.gradients_dir).await? {
            if let Some(record) = self.read_json::<GradientRecord>(&file_path).await? {
                if record.node_id == node_id {
                    total_submitted += 1;
                    if record.validation_passed {
                        accepted += 1;
                    }
                }
            }
        }

        // Count Byzantine events
        let mut byzantine_count = 0u64;
        for file_path in self.list_json_files(&self.byzantine_dir).await? {
            if let Some(event) = self.read_json::<ByzantineEvent>(&file_path).await? {
                if event.node_id == node_id {
                    byzantine_count += 1;
                }
            }
        }

        // Calculate score
        let base_score = if total_submitted > 0 {
            accepted as f32 / total_submitted as f32
        } else {
            0.5
        };
        let byzantine_penalty = (byzantine_count as f32 * 0.1).min(0.5);
        let score = (base_score - byzantine_penalty).max(0.0);

        // Get total credits
        let total_credits = self.get_credit_balance(node_id).await?;

        Ok(ReputationData {
            node_id: node_id.to_string(),
            score,
            gradients_submitted: total_submitted,
            gradients_accepted: accepted,
            byzantine_events: byzantine_count,
            total_credits,
        })
    }

    async fn update_reputation(
        &self,
        node_id: &str,
        score_delta: f32,
        reason: &str,
    ) -> StorageResult<()> {
        // LocalFile backend computes reputation from events
        // This is a no-op but could log to an audit file
        debug!(
            node_id = %node_id,
            delta = score_delta,
            reason = %reason,
            "Reputation update (tracked via events)"
        );
        Ok(())
    }

    async fn log_byzantine_event(&self, event: &ByzantineEvent) -> StorageResult<String> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let id = if event.event_id.is_empty() {
            Self::generate_id()
        } else {
            event.event_id.clone()
        };

        let mut event = event.clone();
        event.event_id = id.clone();

        let file_path = self.byzantine_dir.join(format!("{}.json", id));
        self.write_json(&file_path, &event).await?;

        info!(
            event_id = %id,
            node_id = %event.node_id,
            severity = %event.severity,
            "Byzantine event logged"
        );

        Ok(id)
    }

    async fn get_byzantine_events(
        &self,
        node_id: Option<&str>,
        round_num: Option<u64>,
    ) -> StorageResult<Vec<ByzantineEvent>> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let mut events = Vec::new();

        for file_path in self.list_json_files(&self.byzantine_dir).await? {
            if let Some(event) = self.read_json::<ByzantineEvent>(&file_path).await? {
                // Apply filters
                if let Some(nid) = node_id {
                    if event.node_id != nid {
                        continue;
                    }
                }
                if let Some(rn) = round_num {
                    if event.round_num != rn {
                        continue;
                    }
                }
                events.push(event);
            }
        }

        events.sort_by(|a, b| b.timestamp.partial_cmp(&a.timestamp).unwrap());
        Ok(events)
    }

    async fn get_stats(&self) -> StorageResult<BackendStats> {
        if !self.is_connected() {
            return Err(StorageError::NotConnected);
        }

        let gradient_count = self.list_json_files(&self.gradients_dir).await?.len() as u64;
        let credit_files = self.list_json_files(&self.credits_dir).await?;
        let byzantine_count = self.list_json_files(&self.byzantine_dir).await?.len() as u64;

        // Calculate total credits
        let mut total_credits = 0u64;
        for file_path in &credit_files {
            if let Some(record) = self.read_json::<CreditRecord>(file_path).await? {
                total_credits += record.amount;
            }
        }

        // Estimate storage size
        let mut storage_size = 0u64;
        for dir in [&self.gradients_dir, &self.credits_dir, &self.byzantine_dir] {
            for file in self.list_json_files(dir).await? {
                if let Ok(meta) = fs::metadata(&file).await {
                    storage_size += meta.len();
                }
            }
        }

        let uptime = self
            .connection_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        Ok(BackendStats {
            backend_type: BackendType::LocalFile,
            total_gradients: gradient_count,
            total_credits_issued: total_credits,
            total_byzantine_events: byzantine_count,
            storage_size_bytes: storage_size,
            uptime_seconds: uptime,
            metadata: [
                (
                    "data_dir".to_string(),
                    serde_json::json!(self.data_dir.display().to_string()),
                ),
                (
                    "gradient_files".to_string(),
                    serde_json::json!(gradient_count),
                ),
                (
                    "credit_files".to_string(),
                    serde_json::json!(credit_files.len()),
                ),
                (
                    "byzantine_files".to_string(),
                    serde_json::json!(byzantine_count),
                ),
            ]
            .into_iter()
            .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_backend() -> (LocalFileBackend, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = LocalFileBackend::new(temp_dir.path());
        backend.connect().await.unwrap();
        (backend, temp_dir)
    }

    #[tokio::test]
    async fn test_connect_creates_directories() {
        let temp_dir = TempDir::new().unwrap();
        let mut backend = LocalFileBackend::new(temp_dir.path());

        assert!(!backend.is_connected());
        backend.connect().await.unwrap();
        assert!(backend.is_connected());

        assert!(temp_dir.path().join("gradients").exists());
        assert!(temp_dir.path().join("credits").exists());
        assert!(temp_dir.path().join("byzantine_events").exists());
    }

    #[tokio::test]
    async fn test_health_check() {
        let (backend, _temp) = setup_backend().await;

        let health = backend.health_check().await.unwrap();
        assert!(health.healthy);
        assert!(health.storage_available);
        assert!(health.latency_ms.is_some());
    }

    #[tokio::test]
    async fn test_store_and_retrieve_gradient() {
        let (backend, _temp) = setup_backend().await;

        let record = GradientRecord::new("node_1", 1, vec![1.0, 2.0, 3.0], "hash123");

        let id = backend.store_gradient(&record).await.unwrap();
        let retrieved = backend.get_gradient(&id).await.unwrap().unwrap();

        assert_eq!(retrieved.node_id, "node_1");
        assert_eq!(retrieved.round_num, 1);
    }

    #[tokio::test]
    async fn test_get_gradients_by_round() {
        let (backend, _temp) = setup_backend().await;

        // Store gradients for different rounds
        let r1 = GradientRecord::new("node_1", 1, vec![1.0], "h1");
        let r2 = GradientRecord::new("node_2", 1, vec![2.0], "h2");
        let r3 = GradientRecord::new("node_1", 2, vec![3.0], "h3");

        backend.store_gradient(&r1).await.unwrap();
        backend.store_gradient(&r2).await.unwrap();
        backend.store_gradient(&r3).await.unwrap();

        let round_1 = backend.get_gradients_by_round(1).await.unwrap();
        assert_eq!(round_1.len(), 2);

        let round_2 = backend.get_gradients_by_round(2).await.unwrap();
        assert_eq!(round_2.len(), 1);
    }

    #[tokio::test]
    async fn test_credit_operations() {
        let (backend, _temp) = setup_backend().await;

        // Issue credits
        backend
            .issue_credit("node_1", 100, "gradient_quality")
            .await
            .unwrap();
        backend
            .issue_credit("node_1", 50, "peer_validation")
            .await
            .unwrap();
        backend
            .issue_credit("node_2", 75, "gradient_quality")
            .await
            .unwrap();

        // Check balances
        let balance_1 = backend.get_credit_balance("node_1").await.unwrap();
        let balance_2 = backend.get_credit_balance("node_2").await.unwrap();

        assert_eq!(balance_1, 150);
        assert_eq!(balance_2, 75);

        // Check history
        let history = backend.get_credit_history("node_1").await.unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn test_byzantine_event_logging() {
        let (backend, _temp) = setup_backend().await;

        let event = ByzantineEvent::new("node_1", 5, "pogq", ByzantineSeverity::High)
            .with_details(serde_json::json!({"score": 0.1}));

        let id = backend.log_byzantine_event(&event).await.unwrap();

        let events = backend.get_byzantine_events(None, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_id, id);

        // Filter by node
        let filtered = backend
            .get_byzantine_events(Some("node_1"), None)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);

        let empty = backend
            .get_byzantine_events(Some("node_2"), None)
            .await
            .unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn test_reputation() {
        let (backend, _temp) = setup_backend().await;

        // Store some gradients
        let g1 = GradientRecord::new("node_1", 1, vec![1.0], "h1").with_validation(true);
        let g2 = GradientRecord::new("node_1", 2, vec![2.0], "h2").with_validation(true);
        let g3 = GradientRecord::new("node_1", 3, vec![3.0], "h3").with_validation(false);

        backend.store_gradient(&g1).await.unwrap();
        backend.store_gradient(&g2).await.unwrap();
        backend.store_gradient(&g3).await.unwrap();

        // Issue some credits
        backend.issue_credit("node_1", 50, "test").await.unwrap();

        let rep = backend.get_reputation("node_1").await.unwrap();

        assert_eq!(rep.gradients_submitted, 3);
        assert_eq!(rep.gradients_accepted, 2);
        assert_eq!(rep.total_credits, 50);
        // Score should be 2/3 ≈ 0.667 (no Byzantine events)
        assert!(rep.score > 0.6 && rep.score < 0.7);
    }

    #[tokio::test]
    async fn test_stats() {
        let (backend, _temp) = setup_backend().await;

        // Store some data
        let g = GradientRecord::new("node_1", 1, vec![1.0], "h1");
        backend.store_gradient(&g).await.unwrap();
        backend.issue_credit("node_1", 100, "test").await.unwrap();

        let stats = backend.get_stats().await.unwrap();

        assert_eq!(stats.backend_type, BackendType::LocalFile);
        assert_eq!(stats.total_gradients, 1);
        assert_eq!(stats.total_credits_issued, 100);
        assert!(stats.uptime_seconds > 0.0);
    }
}
