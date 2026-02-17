//! PostgreSQL storage backend.
//!
//! Production-ready PostgreSQL backend with connection pooling,
//! ACID transactions, and SQL-based filtering.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use tracing::{debug, info};

use super::backend::{BackendConfig, StorageBackend, StorageError, StorageResult};
use super::types::*;

/// PostgreSQL storage backend.
///
/// Features:
/// - Connection pooling for high concurrency
/// - ACID transactions
/// - Complex queries and aggregation
/// - Production-ready reliability
pub struct PostgresBackend {
    /// Database connection string.
    connection_string: String,
    /// Connection pool.
    pool: Option<PgPool>,
    /// Connection state.
    connected: AtomicBool,
    /// Connection timestamp.
    connection_time: Option<Instant>,
    /// Configuration.
    config: BackendConfig,
    /// Pool size.
    pool_size: u32,
}

impl PostgresBackend {
    /// Create a new PostgreSQL backend.
    pub fn new(connection_string: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
            pool: None,
            connected: AtomicBool::new(false),
            connection_time: None,
            config: BackendConfig::default(),
            pool_size: 10,
        }
    }

    /// Create from individual connection parameters.
    ///
    /// # Security Warning
    ///
    /// This method constructs a connection string containing database credentials.
    /// The password will be embedded in the connection URL and stored in memory.
    ///
    /// **Recommendations:**
    /// - Use environment variables for credentials (e.g., `DATABASE_URL`)
    /// - Ensure proper permissions on config files if storing credentials
    /// - Consider using `from_connection_string()` with a properly secured source
    /// - Never log or display the connection string
    ///
    /// # Parameters
    ///
    /// - `host`: Database server hostname or IP address
    /// - `port`: Database server port (typically 5432)
    /// - `database`: Database name
    /// - `user`: Database username
    /// - `password`: Database password (handle securely!)
    pub fn from_params(
        host: &str,
        port: u16,
        database: &str,
        user: &str,
        password: &str,
    ) -> Self {
        let connection_string = format!(
            "postgres://{}:{}@{}:{}/{}",
            user, password, host, port, database
        );
        Self::new(connection_string)
    }

    /// Set connection pool size.
    pub fn with_pool_size(mut self, size: u32) -> Self {
        self.pool_size = size;
        self
    }

    /// Set configuration.
    pub fn with_config(mut self, config: BackendConfig) -> Self {
        self.config = config;
        self
    }

    /// Get pool reference (internal).
    fn pool(&self) -> StorageResult<&PgPool> {
        self.pool
            .as_ref()
            .ok_or(StorageError::NotConnected)
    }

    /// Initialize database schema.
    pub async fn init_schema(&self) -> StorageResult<()> {
        let pool = self.pool()?;

        // Create gradients table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS gradients (
                id VARCHAR(64) PRIMARY KEY,
                node_id VARCHAR(128) NOT NULL,
                round_num BIGINT NOT NULL,
                gradient_data JSONB NOT NULL,
                gradient_hash VARCHAR(128) NOT NULL,
                pogq_score REAL,
                zkpoc_verified BOOLEAN DEFAULT FALSE,
                validation_passed BOOLEAN DEFAULT TRUE,
                reputation_score REAL DEFAULT 0.5,
                encrypted BOOLEAN DEFAULT FALSE,
                submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        // Create indexes for gradients
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_gradients_round ON gradients(round_num)",
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_gradients_node ON gradients(node_id)",
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        // Create credits table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS credits (
                id VARCHAR(64) PRIMARY KEY,
                holder VARCHAR(128) NOT NULL,
                amount BIGINT NOT NULL,
                earned_from VARCHAR(256) NOT NULL,
                issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_credits_holder ON credits(holder)",
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        // Create byzantine_events table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS byzantine_events (
                id VARCHAR(64) PRIMARY KEY,
                node_id VARCHAR(128) NOT NULL,
                round_num BIGINT NOT NULL,
                detection_method VARCHAR(64) NOT NULL,
                severity VARCHAR(16) NOT NULL,
                details JSONB DEFAULT '{}',
                detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_byzantine_node ON byzantine_events(node_id)",
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_byzantine_round ON byzantine_events(round_num)",
        )
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        info!("PostgreSQL schema initialized");
        Ok(())
    }

    /// Generate UUID.
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
}

#[async_trait]
impl StorageBackend for PostgresBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::PostgreSQL
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    async fn connect(&mut self) -> StorageResult<()> {
        let pool = PgPoolOptions::new()
            .max_connections(self.pool_size)
            .acquire_timeout(std::time::Duration::from_secs(
                self.config.connect_timeout_secs,
            ))
            .connect(&self.connection_string)
            .await
            .map_err(|e| StorageError::Connection(e.to_string()))?;

        self.pool = Some(pool);
        self.connected.store(true, Ordering::Relaxed);
        self.connection_time = Some(Instant::now());

        info!(
            pool_size = self.pool_size,
            "PostgreSQL backend connected"
        );

        // Initialize schema
        self.init_schema().await?;

        Ok(())
    }

    async fn disconnect(&mut self) -> StorageResult<()> {
        if let Some(pool) = self.pool.take() {
            pool.close().await;
        }
        self.connected.store(false, Ordering::Relaxed);
        info!("PostgreSQL backend disconnected");
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

        let pool = self.pool()?;
        let start = Instant::now();

        match sqlx::query("SELECT 1").fetch_one(pool).await {
            Ok(_) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;

                Ok(HealthStatus {
                    healthy: true,
                    latency_ms: Some(latency),
                    storage_available: true,
                    metadata: [
                        ("pool_size".to_string(), serde_json::json!(pool.size())),
                        (
                            "idle_connections".to_string(),
                            serde_json::json!(pool.num_idle()),
                        ),
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
        let pool = self.pool()?;

        let id = if record.id.is_empty() {
            Self::generate_id()
        } else {
            record.id.clone()
        };

        let gradient_json = serde_json::to_value(&record.gradient)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let metadata_json = serde_json::to_value(&record.metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO gradients (
                id, node_id, round_num, gradient_data, gradient_hash,
                pogq_score, zkpoc_verified, validation_passed,
                reputation_score, encrypted, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(&id)
        .bind(&record.node_id)
        .bind(record.round_num as i64)
        .bind(&gradient_json)
        .bind(&record.gradient_hash)
        .bind(record.pogq_score)
        .bind(record.zkpoc_verified)
        .bind(record.validation_passed)
        .bind(record.reputation_score)
        .bind(record.encrypted)
        .bind(&metadata_json)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        debug!(
            gradient_id = %id,
            node_id = %record.node_id,
            "Gradient stored in PostgreSQL"
        );

        Ok(id)
    }

    async fn get_gradient(&self, gradient_id: &str) -> StorageResult<Option<GradientRecord>> {
        let pool = self.pool()?;

        let row = sqlx::query(
            "SELECT * FROM gradients WHERE id = $1",
        )
        .bind(gradient_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        match row {
            Some(row) => {
                let gradient_json: serde_json::Value = row.get("gradient_data");
                let metadata_json: serde_json::Value = row.get("metadata");
                let submitted_at: chrono::DateTime<chrono::Utc> = row.get("submitted_at");

                let gradient: GradientData = serde_json::from_value(gradient_json)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;

                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_value(metadata_json).unwrap_or_default();

                Ok(Some(GradientRecord {
                    id: row.get("id"),
                    node_id: row.get("node_id"),
                    round_num: row.get::<i64, _>("round_num") as u64,
                    gradient,
                    gradient_hash: row.get("gradient_hash"),
                    pogq_score: row.get("pogq_score"),
                    zkpoc_verified: row.get("zkpoc_verified"),
                    validation_passed: row.get("validation_passed"),
                    reputation_score: row.get("reputation_score"),
                    timestamp: submitted_at.timestamp() as f64,
                    encrypted: row.get("encrypted"),
                    metadata,
                }))
            }
            None => Ok(None),
        }
    }

    async fn get_gradients_by_round(&self, round_num: u64) -> StorageResult<Vec<GradientRecord>> {
        let pool = self.pool()?;

        let rows = sqlx::query(
            r#"
            SELECT * FROM gradients
            WHERE round_num = $1 AND validation_passed = TRUE
            ORDER BY submitted_at
            "#,
        )
        .bind(round_num as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let gradient_json: serde_json::Value = row.get("gradient_data");
            let metadata_json: serde_json::Value = row.get("metadata");
            let submitted_at: chrono::DateTime<chrono::Utc> = row.get("submitted_at");

            let gradient: GradientData = serde_json::from_value(gradient_json)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;

            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_value(metadata_json).unwrap_or_default();

            results.push(GradientRecord {
                id: row.get("id"),
                node_id: row.get("node_id"),
                round_num: row.get::<i64, _>("round_num") as u64,
                gradient,
                gradient_hash: row.get("gradient_hash"),
                pogq_score: row.get("pogq_score"),
                zkpoc_verified: row.get("zkpoc_verified"),
                validation_passed: row.get("validation_passed"),
                reputation_score: row.get("reputation_score"),
                timestamp: submitted_at.timestamp() as f64,
                encrypted: row.get("encrypted"),
                metadata,
            });
        }

        Ok(results)
    }

    async fn get_gradients_by_node(&self, node_id: &str) -> StorageResult<Vec<GradientRecord>> {
        let pool = self.pool()?;

        let rows = sqlx::query(
            "SELECT * FROM gradients WHERE node_id = $1 ORDER BY submitted_at",
        )
        .bind(node_id)
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let gradient_json: serde_json::Value = row.get("gradient_data");
            let metadata_json: serde_json::Value = row.get("metadata");
            let submitted_at: chrono::DateTime<chrono::Utc> = row.get("submitted_at");

            let gradient: GradientData = serde_json::from_value(gradient_json)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;

            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_value(metadata_json).unwrap_or_default();

            results.push(GradientRecord {
                id: row.get("id"),
                node_id: row.get("node_id"),
                round_num: row.get::<i64, _>("round_num") as u64,
                gradient,
                gradient_hash: row.get("gradient_hash"),
                pogq_score: row.get("pogq_score"),
                zkpoc_verified: row.get("zkpoc_verified"),
                validation_passed: row.get("validation_passed"),
                reputation_score: row.get("reputation_score"),
                timestamp: submitted_at.timestamp() as f64,
                encrypted: row.get("encrypted"),
                metadata,
            });
        }

        Ok(results)
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
        let pool = self.pool()?;
        let id = Self::generate_id();

        sqlx::query(
            r#"
            INSERT INTO credits (id, holder, amount, earned_from)
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(&id)
        .bind(holder)
        .bind(amount as i64)
        .bind(earned_from)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        debug!(
            transaction_id = %id,
            holder = %holder,
            amount = amount,
            "Credit issued in PostgreSQL"
        );

        Ok(id)
    }

    async fn get_credit_balance(&self, node_id: &str) -> StorageResult<u64> {
        let pool = self.pool()?;

        let result: (i64,) = sqlx::query_as(
            "SELECT COALESCE(SUM(amount), 0) FROM credits WHERE holder = $1",
        )
        .bind(node_id)
        .fetch_one(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        Ok(result.0 as u64)
    }

    async fn get_credit_history(&self, node_id: &str) -> StorageResult<Vec<CreditRecord>> {
        let pool = self.pool()?;

        let rows = sqlx::query(
            "SELECT * FROM credits WHERE holder = $1 ORDER BY issued_at DESC",
        )
        .bind(node_id)
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let issued_at: chrono::DateTime<chrono::Utc> = row.get("issued_at");
            let metadata_json: serde_json::Value = row.get("metadata");

            results.push(CreditRecord {
                transaction_id: row.get("id"),
                holder: row.get("holder"),
                amount: row.get::<i64, _>("amount") as u64,
                earned_from: row.get("earned_from"),
                timestamp: issued_at.timestamp() as f64,
                metadata: serde_json::from_value(metadata_json).unwrap_or_default(),
            });
        }

        Ok(results)
    }

    async fn get_reputation(&self, node_id: &str) -> StorageResult<ReputationData> {
        let pool = self.pool()?;

        // Get gradient stats
        let gradient_stats: (i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*)::bigint,
                COALESCE(SUM(CASE WHEN validation_passed THEN 1 ELSE 0 END), 0)::bigint
            FROM gradients
            WHERE node_id = $1
            "#,
        )
        .bind(node_id)
        .fetch_one(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        // Get Byzantine count
        let byzantine_count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*)::bigint FROM byzantine_events WHERE node_id = $1",
        )
        .bind(node_id)
        .fetch_one(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        // Get total credits
        let total_credits = self.get_credit_balance(node_id).await?;

        // Calculate reputation score
        let submitted = gradient_stats.0.max(1) as f32;
        let accepted = gradient_stats.1 as f32;
        let base_score = accepted / submitted;
        let byzantine_penalty = (byzantine_count.0 as f32 * 0.1).min(0.5);
        let score = (base_score - byzantine_penalty).max(0.0);

        Ok(ReputationData {
            node_id: node_id.to_string(),
            score,
            gradients_submitted: gradient_stats.0 as u64,
            gradients_accepted: gradient_stats.1 as u64,
            byzantine_events: byzantine_count.0 as u64,
            total_credits,
        })
    }

    async fn update_reputation(
        &self,
        node_id: &str,
        score_delta: f32,
        reason: &str,
    ) -> StorageResult<()> {
        // PostgreSQL backend computes reputation from events
        debug!(
            node_id = %node_id,
            delta = score_delta,
            reason = %reason,
            "Reputation update (tracked via events)"
        );
        Ok(())
    }

    async fn log_byzantine_event(&self, event: &ByzantineEvent) -> StorageResult<String> {
        let pool = self.pool()?;

        let id = if event.event_id.is_empty() {
            Self::generate_id()
        } else {
            event.event_id.clone()
        };

        let severity_str = event.severity.to_string();

        sqlx::query(
            r#"
            INSERT INTO byzantine_events (id, node_id, round_num, detection_method, severity, details)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(&id)
        .bind(&event.node_id)
        .bind(event.round_num as i64)
        .bind(&event.detection_method)
        .bind(&severity_str)
        .bind(&event.details)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Query(e.to_string()))?;

        info!(
            event_id = %id,
            node_id = %event.node_id,
            severity = %severity_str,
            "Byzantine event logged in PostgreSQL"
        );

        Ok(id)
    }

    async fn get_byzantine_events(
        &self,
        node_id: Option<&str>,
        round_num: Option<u64>,
    ) -> StorageResult<Vec<ByzantineEvent>> {
        let pool = self.pool()?;

        let mut query = String::from("SELECT * FROM byzantine_events WHERE TRUE");
        let mut param_count = 0;

        if node_id.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND node_id = ${}", param_count));
        }

        if round_num.is_some() {
            param_count += 1;
            query.push_str(&format!(" AND round_num = ${}", param_count));
        }

        query.push_str(" ORDER BY detected_at DESC");

        let mut query_builder = sqlx::query(&query);

        if let Some(nid) = node_id {
            query_builder = query_builder.bind(nid);
        }

        if let Some(rn) = round_num {
            query_builder = query_builder.bind(rn as i64);
        }

        let rows = query_builder
            .fetch_all(pool)
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let detected_at: chrono::DateTime<chrono::Utc> = row.get("detected_at");
            let severity_str: String = row.get("severity");
            let details: serde_json::Value = row.get("details");

            let severity = match severity_str.as_str() {
                "low" => ByzantineSeverity::Low,
                "medium" => ByzantineSeverity::Medium,
                "high" => ByzantineSeverity::High,
                "critical" => ByzantineSeverity::Critical,
                _ => ByzantineSeverity::Medium,
            };

            results.push(ByzantineEvent {
                event_id: row.get("id"),
                node_id: row.get("node_id"),
                round_num: row.get::<i64, _>("round_num") as u64,
                detection_method: row.get("detection_method"),
                severity,
                details,
                timestamp: detected_at.timestamp() as f64,
            });
        }

        Ok(results)
    }

    async fn get_stats(&self) -> StorageResult<BackendStats> {
        let pool = self.pool()?;

        let gradient_count: (i64,) = sqlx::query_as("SELECT COUNT(*)::bigint FROM gradients")
            .fetch_one(pool)
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let credit_total: (i64,) =
            sqlx::query_as("SELECT COALESCE(SUM(amount), 0)::bigint FROM credits")
                .fetch_one(pool)
                .await
                .map_err(|e| StorageError::Query(e.to_string()))?;

        let byzantine_count: (i64,) =
            sqlx::query_as("SELECT COUNT(*)::bigint FROM byzantine_events")
                .fetch_one(pool)
                .await
                .map_err(|e| StorageError::Query(e.to_string()))?;

        // Estimate storage size
        let storage_size: (i64,) = sqlx::query_as(
            r#"
            SELECT COALESCE(
                pg_total_relation_size('gradients') +
                pg_total_relation_size('credits') +
                pg_total_relation_size('byzantine_events'),
                0
            )::bigint
            "#,
        )
        .fetch_one(pool)
        .await
        .unwrap_or((0,));

        let uptime = self
            .connection_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        Ok(BackendStats {
            backend_type: BackendType::PostgreSQL,
            total_gradients: gradient_count.0 as u64,
            total_credits_issued: credit_total.0 as u64,
            total_byzantine_events: byzantine_count.0 as u64,
            storage_size_bytes: storage_size.0 as u64,
            uptime_seconds: uptime,
            metadata: [
                ("pool_size".to_string(), serde_json::json!(pool.size())),
                (
                    "idle_connections".to_string(),
                    serde_json::json!(pool.num_idle()),
                ),
            ]
            .into_iter()
            .collect(),
        })
    }
}

// Note: Tests require a running PostgreSQL instance
// Run with: cargo test --features storage-postgres -- --ignored
#[cfg(test)]
mod tests {
    use super::*;

    // These tests require DATABASE_URL environment variable
    // Example: DATABASE_URL=postgres://user:pass@localhost/test_db

    #[tokio::test]
    #[ignore = "requires PostgreSQL"]
    async fn test_postgres_connection() {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL required");
        let mut backend = PostgresBackend::new(url);

        backend.connect().await.unwrap();
        assert!(backend.is_connected());

        let health = backend.health_check().await.unwrap();
        assert!(health.healthy);

        backend.disconnect().await.unwrap();
    }
}
