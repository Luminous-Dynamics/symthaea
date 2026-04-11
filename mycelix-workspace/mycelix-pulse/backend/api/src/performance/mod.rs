// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance Optimization
//!
//! Query optimization, caching strategies, and performance monitoring

use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Query analyzer for performance optimization
pub struct QueryAnalyzer {
    pool: PgPool,
}

impl QueryAnalyzer {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Analyze query execution plan
    pub async fn explain(&self, query: &str) -> Result<QueryPlan, sqlx::Error> {
        let explain_query = format!("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}", query);

        let row: (serde_json::Value,) = sqlx::query_as(&explain_query)
            .fetch_one(&self.pool)
            .await?;

        let plan = serde_json::from_value(row.0).unwrap_or_default();
        Ok(plan)
    }

    /// Get slow queries from pg_stat_statements
    pub async fn get_slow_queries(&self, min_duration_ms: f64) -> Result<Vec<SlowQuery>, sqlx::Error> {
        let queries = sqlx::query_as::<_, SlowQuery>(
            r#"
            SELECT
                query,
                calls::bigint as calls,
                mean_exec_time as avg_time_ms,
                total_exec_time as total_time_ms,
                rows::bigint as rows_returned,
                shared_blks_hit::bigint as cache_hits,
                shared_blks_read::bigint as disk_reads
            FROM pg_stat_statements
            WHERE mean_exec_time > $1
            ORDER BY total_exec_time DESC
            LIMIT 50
            "#,
        )
        .bind(min_duration_ms)
        .fetch_all(&self.pool)
        .await?;

        Ok(queries)
    }

    /// Get missing index suggestions
    pub async fn suggest_indexes(&self) -> Result<Vec<IndexSuggestion>, sqlx::Error> {
        // Analyze sequential scans on large tables
        let suggestions = sqlx::query_as::<_, IndexSuggestion>(
            r#"
            SELECT
                schemaname || '.' || relname as table_name,
                seq_scan as sequential_scans,
                idx_scan as index_scans,
                seq_tup_read as rows_scanned,
                n_live_tup as table_rows,
                CASE
                    WHEN seq_scan > 0 THEN seq_tup_read::float / seq_scan
                    ELSE 0
                END as avg_rows_per_scan
            FROM pg_stat_user_tables
            WHERE seq_scan > 100
              AND n_live_tup > 10000
              AND (idx_scan IS NULL OR seq_scan > idx_scan * 10)
            ORDER BY seq_tup_read DESC
            LIMIT 20
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(suggestions)
    }

    /// Get table statistics
    pub async fn get_table_stats(&self) -> Result<Vec<TableStats>, sqlx::Error> {
        let stats = sqlx::query_as::<_, TableStats>(
            r#"
            SELECT
                schemaname || '.' || relname as table_name,
                n_live_tup as row_count,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                pg_size_pretty(pg_total_relation_size(relid)) as total_size
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(stats)
    }

    /// Get index usage statistics
    pub async fn get_index_usage(&self) -> Result<Vec<IndexUsage>, sqlx::Error> {
        let usage = sqlx::query_as::<_, IndexUsage>(
            r#"
            SELECT
                schemaname || '.' || relname as table_name,
                indexrelname as index_name,
                idx_scan as scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            LIMIT 50
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(usage)
    }

    /// Identify unused indexes
    pub async fn find_unused_indexes(&self) -> Result<Vec<UnusedIndex>, sqlx::Error> {
        let unused = sqlx::query_as::<_, UnusedIndex>(
            r#"
            SELECT
                schemaname || '.' || relname as table_name,
                indexrelname as index_name,
                idx_scan as scans,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            WHERE idx_scan < 50
              AND indexrelname NOT LIKE '%_pkey'
            ORDER BY pg_relation_size(indexrelid) DESC
            LIMIT 20
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(unused)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryPlan {
    #[serde(rename = "Plan")]
    pub plan: Option<PlanNode>,
    #[serde(rename = "Planning Time")]
    pub planning_time: Option<f64>,
    #[serde(rename = "Execution Time")]
    pub execution_time: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanNode {
    #[serde(rename = "Node Type")]
    pub node_type: Option<String>,
    #[serde(rename = "Actual Rows")]
    pub actual_rows: Option<i64>,
    #[serde(rename = "Actual Total Time")]
    pub actual_total_time: Option<f64>,
    #[serde(rename = "Shared Hit Blocks")]
    pub shared_hit_blocks: Option<i64>,
    #[serde(rename = "Shared Read Blocks")]
    pub shared_read_blocks: Option<i64>,
    #[serde(rename = "Plans")]
    pub plans: Option<Vec<PlanNode>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SlowQuery {
    pub query: String,
    pub calls: i64,
    pub avg_time_ms: f64,
    pub total_time_ms: f64,
    pub rows_returned: i64,
    pub cache_hits: i64,
    pub disk_reads: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct IndexSuggestion {
    pub table_name: String,
    pub sequential_scans: i64,
    pub index_scans: Option<i64>,
    pub rows_scanned: i64,
    pub table_rows: i64,
    pub avg_rows_per_scan: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TableStats {
    pub table_name: String,
    pub row_count: i64,
    pub dead_tuples: i64,
    pub last_vacuum: Option<chrono::DateTime<chrono::Utc>>,
    pub last_autovacuum: Option<chrono::DateTime<chrono::Utc>>,
    pub last_analyze: Option<chrono::DateTime<chrono::Utc>>,
    pub total_size: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct IndexUsage {
    pub table_name: String,
    pub index_name: String,
    pub scans: i64,
    pub tuples_read: i64,
    pub tuples_fetched: i64,
    pub index_size: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UnusedIndex {
    pub table_name: String,
    pub index_name: String,
    pub scans: i64,
    pub index_size: String,
}

/// Connection pool monitor
pub struct PoolMonitor {
    pool: PgPool,
}

impl PoolMonitor {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let size = self.pool.size();
        let idle = self.pool.num_idle();

        PoolStats {
            total_connections: size,
            idle_connections: idle as u32,
            active_connections: size - idle as u32,
            max_connections: self.pool.options().get_max_connections(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_connections: u32,
    pub idle_connections: u32,
    pub active_connections: u32,
    pub max_connections: u32,
}

/// Query timing middleware
pub struct QueryTimer {
    thresholds: QueryThresholds,
}

#[derive(Debug, Clone)]
pub struct QueryThresholds {
    pub warn_ms: u64,
    pub error_ms: u64,
}

impl Default for QueryThresholds {
    fn default() -> Self {
        Self {
            warn_ms: 100,
            error_ms: 1000,
        }
    }
}

impl QueryTimer {
    pub fn new(thresholds: QueryThresholds) -> Self {
        Self { thresholds }
    }

    pub fn time<F, T>(&self, query_name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        let ms = duration.as_millis() as u64;
        if ms >= self.thresholds.error_ms {
            warn!(
                query = query_name,
                duration_ms = ms,
                "SLOW QUERY (critical)"
            );
        } else if ms >= self.thresholds.warn_ms {
            info!(
                query = query_name,
                duration_ms = ms,
                "Slow query"
            );
        }

        result
    }
}

/// Recommended indexes for Mycelix Mail
pub const RECOMMENDED_INDEXES: &str = r#"
-- Email lookups by user and folder
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_emails_user_folder_date
    ON emails(user_id, folder_id, received_at DESC);

-- Full-text search on emails
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_emails_search
    ON emails USING gin(to_tsvector('english', subject || ' ' || body_text));

-- Trust score lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trust_scores_entity
    ON trust_scores(entity_id, context);

-- Attestation verification
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_attestations_subject_issuer
    ON attestations(subject_id, issuer_id, valid_until)
    WHERE revoked_at IS NULL;

-- Contact lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contacts_user_email
    ON contacts(user_id, email);

-- Scheduled emails processing
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduled_pending
    ON scheduled_emails(scheduled_at)
    WHERE status = 'pending';

-- Email queue processing
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_email_queue_pending
    ON email_queue(priority DESC, created_at)
    WHERE status = 'pending';

-- Webhook delivery retries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_webhook_retry
    ON webhook_deliveries(next_retry_at)
    WHERE status = 'retrying';

-- Calendar upcoming events
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calendar_upcoming
    ON calendar_events(user_id, start_time)
    WHERE status = 'confirmed' AND start_time > NOW();

-- Audit log queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_time
    ON audit_log(user_id, created_at DESC);

-- GDPR requests tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gdpr_pending
    ON gdpr_requests(status, created_at)
    WHERE status IN ('pending', 'processing');

-- Multi-tenant partitioning hint
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_emails_tenant
    ON emails(tenant_id, received_at DESC)
    WHERE tenant_id IS NOT NULL;
"#;

/// Cache configuration recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub email_list_ttl: Duration,
    pub email_detail_ttl: Duration,
    pub contact_ttl: Duration,
    pub trust_score_ttl: Duration,
    pub template_ttl: Duration,
    pub max_cached_emails: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            email_list_ttl: Duration::from_secs(30),
            email_detail_ttl: Duration::from_secs(300),
            contact_ttl: Duration::from_secs(600),
            trust_score_ttl: Duration::from_secs(60),
            template_ttl: Duration::from_secs(3600),
            max_cached_emails: 1000,
        }
    }
}

/// Performance report generator
pub async fn generate_performance_report(pool: &PgPool) -> PerformanceReport {
    let analyzer = QueryAnalyzer::new(pool.clone());
    let pool_monitor = PoolMonitor::new(pool.clone());

    PerformanceReport {
        pool_stats: pool_monitor.get_stats(),
        slow_queries: analyzer.get_slow_queries(100.0).await.unwrap_or_default(),
        index_suggestions: analyzer.suggest_indexes().await.unwrap_or_default(),
        table_stats: analyzer.get_table_stats().await.unwrap_or_default(),
        unused_indexes: analyzer.find_unused_indexes().await.unwrap_or_default(),
        generated_at: chrono::Utc::now(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub pool_stats: PoolStats,
    pub slow_queries: Vec<SlowQuery>,
    pub index_suggestions: Vec<IndexSuggestion>,
    pub table_stats: Vec<TableStats>,
    pub unused_indexes: Vec<UnusedIndex>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}
