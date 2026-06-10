// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LanceDB-Backed Persistent Memory Storage
//!
//! Columnar, async-native storage backend for Symthaea's consciousness system.
//! Uses Apache Arrow format for efficient I/O and metadata filtering.
//!
//! # Important: Similarity Search Strategy
//!
//! LanceDB's native ANN index only supports float vectors with cosine/L2/dot.
//! Since BinaryHV uses 16,384-bit Hamming distance, this backend loads candidate
//! records and computes similarity in-memory using `BinaryHV::similarity()`.
//!
//! # Feature Gate
//!
//! Requires the `lancedb-backend` feature flag:
//! ```toml
//! symthaea = { version = "0.5", features = ["lancedb-backend"] }
//! ```

use super::{
    ConsciousnessDatabase, DatabaseError, DatabaseStats, DbResult, MemoryRecord, MemoryType,
    SearchResult,
};
use arrow_array::{
    FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray, builder::FixedSizeBinaryBuilder,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::sync::Arc;
use symthaea_core::hdc::binary_hv::BinaryHV;

// ============================================================================
// Schema
// ============================================================================

/// Arrow schema for the memories table.
fn memories_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "encoding",
            DataType::FixedSizeBinary(BinaryHV::BYTES as i32),
            false,
        ),
        Field::new("timestamp_ms", DataType::Int64, false),
        Field::new("memory_type", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("valence", DataType::Float32, false),
        Field::new("arousal", DataType::Float32, false),
        Field::new("phi", DataType::Float64, false),
        Field::new("topics", DataType::Utf8, false),
        Field::new("metadata", DataType::Utf8, false),
        Field::new("consolidation_strength", DataType::Float64, false),
        Field::new("retrieval_count", DataType::Int32, false),
    ]))
}

// ============================================================================
// Conversion Helpers
// ============================================================================

fn memory_type_to_str(mt: MemoryType) -> &'static str {
    match mt {
        MemoryType::Episodic => "episodic",
        MemoryType::Semantic => "semantic",
        MemoryType::Procedural => "procedural",
        MemoryType::Working => "working",
    }
}

fn str_to_memory_type(s: &str) -> MemoryType {
    match s {
        "episodic" => MemoryType::Episodic,
        "semantic" => MemoryType::Semantic,
        "procedural" => MemoryType::Procedural,
        "working" => MemoryType::Working,
        _ => MemoryType::Episodic,
    }
}

/// Convert a single MemoryRecord to an Arrow RecordBatch.
fn record_to_batch(record: &MemoryRecord) -> DbResult<RecordBatch> {
    let schema = memories_schema();

    let mut encoding_builder = FixedSizeBinaryBuilder::new(BinaryHV::BYTES as i32);
    encoding_builder
        .append_value(record.encoding.0.as_slice())
        .map_err(|e| DatabaseError::InsertFailed(format!("Encoding serialization failed: {e}")))?;

    let topics_json = serde_json::to_string(&record.topics).unwrap_or_else(|e| {
        tracing::warn!(
            record_id = %record.id,
            error = %e,
            "Failed to serialize topics for LanceDB record — storing empty array"
        );
        "[]".to_string()
    });

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![record.id.as_str()])),
            Arc::new(encoding_builder.finish()),
            Arc::new(Int64Array::from(vec![record.timestamp_ms as i64])),
            Arc::new(StringArray::from(vec![memory_type_to_str(
                record.memory_type,
            )])),
            Arc::new(StringArray::from(vec![record.content.as_str()])),
            Arc::new(Float32Array::from(vec![record.valence])),
            Arc::new(Float32Array::from(vec![record.arousal])),
            Arc::new(Float64Array::from(vec![record.psi])),
            Arc::new(StringArray::from(vec![topics_json.as_str()])),
            Arc::new(StringArray::from(vec![record.metadata.as_str()])),
            Arc::new(Float64Array::from(vec![record.consolidation_strength])),
            Arc::new(Int32Array::from(vec![record.retrieval_count as i32])),
        ],
    )
    .map_err(|e| DatabaseError::InsertFailed(format!("RecordBatch creation failed: {e}")))
}

/// Convert an Arrow RecordBatch to a Vec of MemoryRecords.
fn batch_to_records(batch: &RecordBatch) -> Vec<MemoryRecord> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Vec::new();
    }

    // Downcast all columns; bail if any fails
    let Some(id_col) = batch.column(0).as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let Some(enc_col) = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
    else {
        return Vec::new();
    };
    let Some(ts_col) = batch.column(2).as_any().downcast_ref::<Int64Array>() else {
        return Vec::new();
    };
    let Some(mt_col) = batch.column(3).as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let Some(content_col) = batch.column(4).as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let Some(val_col) = batch.column(5).as_any().downcast_ref::<Float32Array>() else {
        return Vec::new();
    };
    let Some(ar_col) = batch.column(6).as_any().downcast_ref::<Float32Array>() else {
        return Vec::new();
    };
    let Some(phi_col) = batch.column(7).as_any().downcast_ref::<Float64Array>() else {
        return Vec::new();
    };
    let Some(topics_col) = batch.column(8).as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let Some(meta_col) = batch.column(9).as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    let Some(cs_col) = batch.column(10).as_any().downcast_ref::<Float64Array>() else {
        return Vec::new();
    };
    let Some(rc_col) = batch.column(11).as_any().downcast_ref::<Int32Array>() else {
        return Vec::new();
    };

    (0..num_rows)
        .map(|i| {
            let encoding_bytes = enc_col.value(i);
            let mut arr = [0u8; BinaryHV::BYTES];
            if encoding_bytes.len() >= BinaryHV::BYTES {
                arr.copy_from_slice(&encoding_bytes[..BinaryHV::BYTES]);
            }

            let topics: Vec<String> =
                serde_json::from_str(topics_col.value(i)).unwrap_or_else(|e| {
                    tracing::warn!(
                        row = i,
                        raw = topics_col.value(i),
                        error = %e,
                        "Corrupted topics JSON in LanceDB record — defaulting to empty"
                    );
                    Vec::new()
                });

            let ts = ts_col.value(i);

            MemoryRecord {
                id: id_col.value(i).to_string(),
                encoding: BinaryHV(arr),
                timestamp_ms: if ts < 0 { 0u64 } else { ts as u64 },
                memory_type: str_to_memory_type(mt_col.value(i)),
                content: content_col.value(i).to_string(),
                valence: val_col.value(i),
                arousal: ar_col.value(i),
                psi: phi_col.value(i),
                topics,
                metadata: meta_col.value(i).to_string(),
                consolidation_strength: cs_col.value(i),
                retrieval_count: rc_col.value(i) as u32,
            }
        })
        .collect()
}

// ============================================================================
// Raw Hamming Similarity (no BinaryHV construction)
// ============================================================================

/// Compute Hamming similarity directly from raw byte slices.
///
/// Avoids constructing a full BinaryHV for each candidate row,
/// saving ~2KB of allocation per comparison during search.
#[inline]
fn hamming_similarity_raw(a: &[u8], b: &[u8]) -> f32 {
    let total_bits = (a.len() * 8) as u32;
    if total_bits == 0 {
        return 0.0;
    }
    let mut xor_count = 0u32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        xor_count += (ai ^ bi).count_ones();
    }
    1.0 - (xor_count as f32 / total_bits as f32)
}

// ============================================================================
// LanceMemory
// ============================================================================

/// LanceDB-backed persistent memory database.
///
/// Uses Apache Arrow columnar format for efficient storage and retrieval.
/// Similarity search loads candidates and computes Hamming distance in-memory,
/// since LanceDB's native ANN index does not support binary vectors.
pub struct LanceMemory {
    db: lancedb::Connection,
    table: tokio::sync::RwLock<Option<lancedb::Table>>,
    path: String,
    /// Whether this instance owns a temp directory that should be cleaned up on drop.
    is_temp: bool,
}

impl Drop for LanceMemory {
    fn drop(&mut self) {
        // Release the table reference before the connection drops,
        // preventing panics from lancedb's async cleanup racing with
        // file handle closure.
        if let Ok(mut guard) = self.table.try_write() {
            *guard = None;
        }
        // Clean up temp directories after releasing lancedb handles.
        if self.is_temp {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

impl LanceMemory {
    /// Create a persistent LanceDB database at the given directory path.
    pub async fn new(path: &str) -> DbResult<Self> {
        let db = lancedb::connect(path)
            .execute()
            .await
            .map_err(|e| DatabaseError::ConnectionFailed(format!("LanceDB connect failed: {e}")))?;

        let mem = Self {
            db,
            table: tokio::sync::RwLock::new(None),
            path: path.to_string(),
            is_temp: false,
        };

        mem.ensure_table().await?;
        eprintln!("[LanceMemory] Initialized at: {path}");
        Ok(mem)
    }

    /// Create a LanceDB database in a temporary directory.
    ///
    /// The directory is created under `/tmp/symthaea_lance_{pid}_{random}`
    /// and persists until manually removed.
    pub async fn in_memory() -> DbResult<Self> {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_lance_{}_{:08x}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| {
            DatabaseError::ConnectionFailed(format!("Temp dir creation failed: {e}"))
        })?;

        let mut mem = Self::new(&dir.to_string_lossy()).await?;
        mem.is_temp = true;
        Ok(mem)
    }

    /// Ensure the "memories" table exists, creating it if necessary.
    async fn ensure_table(&self) -> DbResult<()> {
        {
            let guard = self.table.read().await;
            if guard.is_some() {
                return Ok(());
            }
        }

        let mut guard = self.table.write().await;
        // Double-check after acquiring write lock
        if guard.is_some() {
            return Ok(());
        }

        let table = match self.db.open_table("memories").execute().await {
            Ok(t) => t,
            Err(_) => self
                .db
                .create_empty_table("memories", memories_schema())
                .execute()
                .await
                .map_err(|e| {
                    DatabaseError::ConnectionFailed(format!("Create table failed: {e}"))
                })?,
        };

        *guard = Some(table);
        Ok(())
    }

    /// Get a clone of the table handle.
    async fn table(&self) -> DbResult<lancedb::Table> {
        self.ensure_table().await?;
        self.table
            .read()
            .await
            .clone()
            .ok_or_else(|| DatabaseError::ConnectionFailed("Table not initialized".into()))
    }

    /// Search with optional predicate pushdown.
    ///
    /// `filter` is a Lance SQL expression applied BEFORE loading records,
    /// reducing I/O for large tables.
    /// Example: `"memory_type = 'episodic' AND timestamp_ms > 1700000000"`
    pub async fn search_similar_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        self.search_similar_2pass(query, top_k, filter).await
    }

    /// 2-pass search: projection pushdown for large tables, single-pass for small ones.
    ///
    /// **Pass 1** (large tables, N > top_k * 10): Fetch only `id` + `encoding` columns,
    /// compute Hamming similarity from raw bytes, take top-k IDs.
    ///
    /// **Pass 2**: Fetch full records for top-k IDs only.
    ///
    /// For small tables (N <= top_k * 10), a single pass fetching all columns is cheaper
    /// than two roundtrips.
    async fn search_similar_2pass(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        let table = self.table().await?;
        let total = table
            .count_rows(filter.map(|f| f.to_string()))
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {e}")))?;

        // Small table: single-pass is more efficient than two roundtrips
        if total <= top_k * 10 {
            return self.search_single_pass(query, top_k, filter).await;
        }

        // ── Pass 1: Lightweight scan — id + encoding only ───────────────
        let query_bytes = &query.0[..];

        let mut q = table.query();
        if let Some(f) = filter {
            q = q.only_if(f);
        }
        let q = q.select(Select::Columns(vec![
            "id".to_string(),
            "encoding".to_string(),
        ]));

        let stream = q
            .execute()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Pass-1 query failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Pass-1 batch failed: {e}")))?;

        // Compute similarity from raw Arrow bytes (no MemoryRecord construction)
        let mut scored: Vec<(String, f32)> = Vec::with_capacity(total);
        for batch in &batches {
            let Some(id_col) = batch.column(0).as_any().downcast_ref::<StringArray>() else {
                continue;
            };
            let Some(enc_col) = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
            else {
                continue;
            };

            for i in 0..batch.num_rows() {
                let sim = hamming_similarity_raw(query_bytes, enc_col.value(i));
                scored.push((id_col.value(i).to_string(), sim));
            }
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);

        if scored.is_empty() {
            return Ok(Vec::new());
        }

        // ── Pass 2: Fetch full records for top-k IDs ────────────────────
        let similarities: std::collections::HashMap<String, f32> = scored.iter().cloned().collect();
        let quoted: Vec<String> = scored
            .iter()
            .map(|(id, _)| format!("'{}'", id.replace('\'', "''")))
            .collect();
        let predicate = format!("id IN ({})", quoted.join(", "));

        let stream = table
            .query()
            .only_if(predicate)
            .execute()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Pass-2 query failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Pass-2 batch failed: {e}")))?;

        let mut results: Vec<SearchResult> = batches
            .iter()
            .flat_map(batch_to_records)
            .map(|record| {
                let similarity = similarities
                    .get(&record.id)
                    .copied()
                    .unwrap_or_else(|| query.similarity(&record.encoding));
                SearchResult { record, similarity }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results.truncate(top_k);

        let returned_ids: Vec<String> = results.iter().map(|r| r.record.id.clone()).collect();
        if let Err(e) = self.bump_retrieval_stats(&returned_ids).await {
            tracing::warn!("Reconsolidation bump failed: {e}");
        }

        Ok(results)
    }

    /// Single-pass search: loads all columns, computes similarity, returns top-k.
    /// Used for small tables where two roundtrips are more expensive.
    async fn search_single_pass(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        let table = self.table().await?;
        let query = *query;

        let mut q = table.query();
        if let Some(f) = filter {
            q = q.only_if(f);
        }

        let stream = q
            .execute()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Single-pass search failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Batch collection failed: {e}")))?;

        let mut results: Vec<SearchResult> = batches
            .iter()
            .flat_map(batch_to_records)
            .map(|record| {
                let similarity = query.similarity(&record.encoding);
                SearchResult { record, similarity }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results.truncate(top_k);

        let returned_ids: Vec<String> = results.iter().map(|r| r.record.id.clone()).collect();
        if let Err(e) = self.bump_retrieval_stats(&returned_ids).await {
            tracing::warn!("Reconsolidation bump failed: {e}");
        }

        Ok(results)
    }

    /// Migrate all records from a SqliteMemory database to this LanceDB instance.
    ///
    /// Reads all records via `list_all()`, batches them into groups of 1000,
    /// and inserts into the Lance table. Returns the number of records migrated.
    pub async fn migrate_from_sqlite(
        &self,
        sqlite: &super::sqlite_client::SqliteMemory,
    ) -> DbResult<usize> {
        use super::ConsciousnessDatabase;

        let records = sqlite.list_all().await?;
        let total = records.len();
        if total == 0 {
            return Ok(0);
        }

        let table = self.table().await?;
        let schema = memories_schema();

        for chunk in records.chunks(1000) {
            let mut batches: Vec<Result<RecordBatch, ArrowError>> = Vec::with_capacity(chunk.len());
            for record in chunk {
                batches.push(record_to_batch(record).map_err(|e| {
                    ArrowError::ExternalError(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    )))
                }));
            }

            let reader = RecordBatchIterator::new(batches, schema.clone());
            table.add(reader).execute().await.map_err(|e| {
                DatabaseError::InsertFailed(format!("Migration batch insert failed: {e}"))
            })?;
        }

        tracing::info!(count = total, "Migrated records from SQLite to LanceDB");
        Ok(total)
    }

    /// Bump retrieval_count and consolidation_strength for returned search results.
    async fn bump_retrieval_stats(&self, ids: &[String]) -> DbResult<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let table = self.table().await?;
        let quoted: Vec<String> = ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        let predicate = format!("id IN ({})", quoted.join(", "));

        table
            .update()
            .only_if(&predicate)
            .column("retrieval_count", "retrieval_count + 1")
            .column("consolidation_strength", "consolidation_strength + 0.05")
            .execute()
            .await
            .map_err(|e| {
                DatabaseError::QueryFailed(format!("Reconsolidation update failed: {e}"))
            })?;

        Ok(())
    }
}

#[async_trait]
impl ConsciousnessDatabase for LanceMemory {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        let table = self.table().await?;
        let batch = record_to_batch(&record)?;
        let schema = memories_schema();

        let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(batch)];
        let reader = RecordBatchIterator::new(batches, schema);

        let mut builder = table.merge_insert(&["id"]);
        builder.when_matched_update_all(None);
        builder.when_not_matched_insert_all();
        builder
            .execute(Box::new(reader))
            .await
            .map_err(|e| DatabaseError::InsertFailed(format!("LanceDB upsert failed: {e}")))?;

        Ok(())
    }

    async fn search_similar(&self, query: &BinaryHV, top_k: usize) -> DbResult<Vec<SearchResult>> {
        self.search_similar_2pass(query, top_k, None).await
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        let table = self.table().await?;
        let escaped_id = id.replace('\'', "''");
        let filter = format!("id = '{escaped_id}'");

        let stream = table
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Get query failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Get batch failed: {e}")))?;

        let records: Vec<MemoryRecord> = batches.iter().flat_map(batch_to_records).collect();
        Ok(records.into_iter().next())
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        let table = self.table().await?;
        let escaped_id = id.replace('\'', "''");
        let predicate = format!("id = '{escaped_id}'");

        let count = table
            .count_rows(Some(predicate.clone()))
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Pre-delete count failed: {e}")))?;

        if count == 0 {
            return Ok(false);
        }

        table
            .delete(&predicate)
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Delete failed: {e}")))?;

        Ok(true)
    }

    async fn count(&self) -> DbResult<usize> {
        let table = self.table().await?;
        table
            .count_rows(None)
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {e}")))
    }

    async fn health_check(&self) -> DbResult<bool> {
        let table = self.table().await?;
        table
            .count_rows(None)
            .await
            .map(|_| true)
            .map_err(|e| DatabaseError::QueryFailed(format!("Health check failed: {e}")))
    }

    async fn search_similar_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        self.search_similar_filtered(query, top_k, filter).await
    }

    async fn list_all(&self) -> DbResult<Vec<MemoryRecord>> {
        let table = self.table().await?;

        let stream = table
            .query()
            .execute()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("list_all query failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("list_all batch failed: {e}")))?;

        Ok(batches.iter().flat_map(batch_to_records).collect())
    }

    async fn stats(&self) -> DbResult<DatabaseStats> {
        let table = self.table().await?;

        let total_records = table
            .count_rows(None)
            .await
            .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {e}")))?;

        // Memory type distribution
        let mut type_counts = Vec::new();
        for mt in &["episodic", "semantic", "procedural", "working"] {
            // unwrap_or(0): Stats are non-critical observability data; a per-type
            // count failure should not abort the entire stats() call.
            let count = table
                .count_rows(Some(format!("memory_type = '{mt}'")))
                .await
                .unwrap_or(0);
            if count > 0 {
                type_counts.push((mt.to_string(), count));
            }
        }

        // Scan phi and timestamp columns
        let (avg_phi, oldest_ts, newest_ts) = if total_records > 0 {
            let stream = table
                .query()
                .execute()
                .await
                .map_err(|e| DatabaseError::QueryFailed(format!("Stats scan failed: {e}")))?;

            let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
                DatabaseError::QueryFailed(format!("Stats batch collection failed: {e}"))
            })?;

            let mut phi_sum = 0.0f64;
            let mut phi_count = 0usize;
            let mut min_ts = i64::MAX;
            let mut max_ts = i64::MIN;

            for batch in &batches {
                // Column 7 = phi (Float64)
                if let Some(phi_arr) = batch.column(7).as_any().downcast_ref::<Float64Array>() {
                    for i in 0..phi_arr.len() {
                        phi_sum += phi_arr.value(i);
                        phi_count += 1;
                    }
                }
                // Column 2 = timestamp_ms (Int64)
                if let Some(ts_arr) = batch.column(2).as_any().downcast_ref::<Int64Array>() {
                    for i in 0..ts_arr.len() {
                        let v = ts_arr.value(i);
                        min_ts = min_ts.min(v);
                        max_ts = max_ts.max(v);
                    }
                }
            }

            let avg = if phi_count > 0 {
                phi_sum / phi_count as f64
            } else {
                0.0
            };
            let oldest = if min_ts == i64::MAX {
                0u64
            } else {
                min_ts.max(0) as u64
            };
            let newest = if max_ts == i64::MIN {
                0u64
            } else {
                max_ts.max(0) as u64
            };

            (avg, oldest, newest)
        } else {
            (0.0, 0, 0)
        };

        // unwrap_or(0): Table version is metadata for the status string; a failure
        // here should not abort the stats response.
        let version = table.version().await.unwrap_or(0);
        let database_size_bytes = dir_size(&self.path);
        let backend_status = format!("lance:v{version}");

        Ok(DatabaseStats {
            total_records,
            database_size_bytes,
            page_count: 0,
            page_size: 0,
            freelist_count: 0,
            cache_hit_ratio: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            avg_query_latency_us: 0,
            total_queries: 0,
            memory_type_counts: type_counts,
            avg_psi: avg_phi,
            oldest_timestamp_ms: oldest_ts,
            newest_timestamp_ms: newest_ts,
            backend_status,
        })
    }
}

/// Calculate total size of a directory in bytes.
fn dir_size(path: &str) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let meta = e.metadata().ok();
            let file_type = e.file_type().ok();
            match (meta, file_type) {
                (Some(m), Some(ft)) if ft.is_file() => m.len(),
                (_, Some(ft)) if ft.is_dir() => dir_size(&e.path().to_string_lossy()),
                _ => 0,
            }
        })
        .sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_record(id: &str, seed: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: BinaryHV::random(seed),
            timestamp_ms: 1234567890 + seed,
            memory_type: MemoryType::Episodic,
            content: format!("Test memory {id}"),
            valence: 0.8,
            arousal: 0.5,
            psi: 0.75,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        }
    }

    #[tokio::test]
    async fn test_lance_memory_basic() {
        let db = LanceMemory::in_memory().await.unwrap();

        // Store
        db.store(test_record("test-1", 42)).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 1);

        // Get by ID
        let retrieved = db.get("test-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Test memory test-1");
        assert!((retrieved.psi - 0.75).abs() < 0.001);

        // Search similar (same seed = identical encoding)
        let results = db.search_similar(&BinaryHV::random(42), 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);

        // Delete
        assert!(db.delete("test-1").await.unwrap());
        assert_eq!(db.count().await.unwrap(), 0);
        assert!(!db.delete("test-1").await.unwrap());
    }

    #[tokio::test]
    async fn test_lance_persistence() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_lance_persist_test_{:08x}",
            rand::random::<u32>()
        ));
        let path = dir.to_string_lossy().to_string();

        // Create and store
        {
            let db = LanceMemory::new(&path).await.unwrap();
            db.store(test_record("persist-1", 123)).await.unwrap();
            assert_eq!(db.count().await.unwrap(), 1);
        }

        // Reopen and verify
        {
            let db = LanceMemory::new(&path).await.unwrap();
            let record = db.get("persist-1").await.unwrap().unwrap();
            assert_eq!(record.content, "Test memory persist-1");
            assert_eq!(db.count().await.unwrap(), 1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_lance_stats() {
        let db = LanceMemory::in_memory().await.unwrap();

        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 0);
        assert!(stats.backend_status.starts_with("lance:v"));

        for i in 0..5u64 {
            let mut rec = test_record(&format!("stats-{i}"), i);
            rec.memory_type = if i % 2 == 0 {
                MemoryType::Episodic
            } else {
                MemoryType::Semantic
            };
            rec.psi = 0.5 + i as f64 * 0.1;
            rec.timestamp_ms = 1000000000 + i * 1000;
            db.store(rec).await.unwrap();
        }

        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 5);
        assert!(!stats.memory_type_counts.is_empty());

        let total_type_count: usize = stats.memory_type_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total_type_count, 5);

        // avg phi: (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5 = 0.7
        assert!((stats.avg_psi - 0.7).abs() < 0.01);
        assert_eq!(stats.oldest_timestamp_ms, 1000000000);
        assert_eq!(stats.newest_timestamp_ms, 1000004000);
    }

    #[tokio::test]
    async fn test_lance_search_similar() {
        let db = LanceMemory::in_memory().await.unwrap();

        for i in 0..10u64 {
            db.store(test_record(&format!("search-{i}"), i * 100))
                .await
                .unwrap();
        }

        // Search for the encoding from seed 0
        let query = BinaryHV::random(0);
        let results = db.search_similar(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        // First result should be the exact match
        assert!(results[0].similarity > 0.99);
        assert_eq!(results[0].record.id, "search-0");
        // Results should be sorted descending
        assert!(results[0].similarity >= results[1].similarity);
        assert!(results[1].similarity >= results[2].similarity);
    }

    #[tokio::test]
    async fn test_lance_search_similar_filtered() {
        let db = LanceMemory::in_memory().await.unwrap();

        for i in 0..5u64 {
            let mut rec = test_record(&format!("filter-{i}"), i);
            rec.memory_type = if i % 2 == 0 {
                MemoryType::Episodic
            } else {
                MemoryType::Semantic
            };
            db.store(rec).await.unwrap();
        }

        // Unfiltered: all 5
        let all = db
            .search_similar_filtered(&BinaryHV::random(0), 10, None)
            .await
            .unwrap();
        assert_eq!(all.len(), 5);

        // Filtered: only episodic (i=0,2,4)
        let episodic = db
            .search_similar_filtered(&BinaryHV::random(0), 10, Some("memory_type = 'episodic'"))
            .await
            .unwrap();
        assert_eq!(episodic.len(), 3);
        for r in &episodic {
            assert_eq!(r.record.memory_type, MemoryType::Episodic);
        }
    }

    #[tokio::test]
    async fn test_lance_list_all() {
        use super::ConsciousnessDatabase;

        let db = LanceMemory::in_memory().await.unwrap();

        for i in 0..3u64 {
            db.store(test_record(&format!("list-{i}"), i))
                .await
                .unwrap();
        }

        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_lance_upsert() {
        let db = LanceMemory::in_memory().await.unwrap();

        let mut rec = test_record("upsert-1", 42);
        rec.content = "Original content".to_string();
        db.store(rec).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 1);

        // Upsert with same ID, different content
        let mut rec2 = test_record("upsert-1", 42);
        rec2.content = "Updated content".to_string();
        db.store(rec2).await.unwrap();

        // Should still be 1 record, with updated content
        assert_eq!(db.count().await.unwrap(), 1);
        let retrieved = db.get("upsert-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
    }

    // ====================================================================
    // Pure helper function tests (no LanceDB connection required)
    // ====================================================================

    #[test]
    fn test_memory_type_to_str_roundtrip() {
        assert_eq!(memory_type_to_str(MemoryType::Episodic), "episodic");
        assert_eq!(memory_type_to_str(MemoryType::Semantic), "semantic");
        assert_eq!(memory_type_to_str(MemoryType::Procedural), "procedural");
        assert_eq!(memory_type_to_str(MemoryType::Working), "working");

        for mt in [
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
            MemoryType::Working,
        ] {
            assert_eq!(str_to_memory_type(memory_type_to_str(mt)), mt);
        }
    }

    #[test]
    fn test_str_to_memory_type_unknown_defaults_to_episodic() {
        assert_eq!(str_to_memory_type("unknown"), MemoryType::Episodic);
        assert_eq!(str_to_memory_type(""), MemoryType::Episodic);
        assert_eq!(str_to_memory_type("EPISODIC"), MemoryType::Episodic);
        assert_eq!(str_to_memory_type("Semantic"), MemoryType::Episodic);
    }

    #[test]
    fn test_hamming_similarity_raw_identical() {
        let a = [0xAA_u8; 64];
        let b = [0xAA_u8; 64];
        let sim = hamming_similarity_raw(&a, &b);
        assert!(
            (sim - 1.0).abs() < f32::EPSILON,
            "Identical slices should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_hamming_similarity_raw_opposite() {
        let a = [0xFF_u8; 64];
        let b = [0x00_u8; 64];
        let sim = hamming_similarity_raw(&a, &b);
        assert!(
            sim.abs() < f32::EPSILON,
            "Complementary slices should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_hamming_similarity_raw_half_matching() {
        // 0xF0 = 11110000, 0x00 = 00000000 → XOR = 11110000 → 4 bits differ per byte = half
        let a = [0xF0_u8; 64];
        let b = [0x00_u8; 64];
        let sim = hamming_similarity_raw(&a, &b);
        assert!(
            (sim - 0.5).abs() < f32::EPSILON,
            "Half-matching slices should have similarity 0.5, got {sim}"
        );
    }

    #[test]
    fn test_hamming_similarity_raw_empty_slices() {
        let a: [u8; 0] = [];
        let b: [u8; 0] = [];
        let sim = hamming_similarity_raw(&a, &b);
        assert!(
            sim.abs() < f32::EPSILON,
            "Empty slices should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_hamming_similarity_raw_single_bit_flip() {
        let a = [0x00_u8; BinaryHV::BYTES];
        let mut b = [0x00_u8; BinaryHV::BYTES];
        b[0] = 0x01;
        let sim = hamming_similarity_raw(&a, &b);
        let expected = 1.0 - (1.0 / (BinaryHV::BYTES as f32 * 8.0));
        assert!(
            (sim - expected).abs() < 1e-6,
            "Single bit flip: expected {expected}, got {sim}"
        );
    }

    #[test]
    fn test_hamming_similarity_raw_matches_binaryhv_similarity() {
        let hv_a = BinaryHV::random(100);
        let hv_b = BinaryHV::random(200);
        let raw_sim = hamming_similarity_raw(&hv_a.0, &hv_b.0);
        let hv_sim = hv_a.similarity(&hv_b);
        assert!(
            (raw_sim - hv_sim).abs() < 1e-5,
            "Raw similarity ({raw_sim}) should match BinaryHV::similarity ({hv_sim})"
        );
    }

    #[test]
    fn test_record_to_batch_and_back_roundtrip() {
        let record = MemoryRecord {
            id: "roundtrip-1".to_string(),
            encoding: BinaryHV::random(42),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Semantic,
            content: "Roundtrip test content".to_string(),
            valence: -0.3,
            arousal: 0.9,
            psi: 0.42,
            topics: vec!["alpha".to_string(), "beta".to_string()],
            metadata: r#"{"key": "value"}"#.to_string(),
            consolidation_strength: 0.55,
            retrieval_count: 7,
        };

        let batch = record_to_batch(&record).expect("record_to_batch should succeed");
        assert_eq!(batch.num_rows(), 1);

        let records = batch_to_records(&batch);
        assert_eq!(records.len(), 1);

        let recovered = &records[0];
        assert_eq!(recovered.id, record.id);
        assert_eq!(recovered.content, record.content);
        assert_eq!(recovered.memory_type, record.memory_type);
        assert_eq!(recovered.timestamp_ms, record.timestamp_ms);
        assert!((recovered.valence - record.valence).abs() < 1e-6);
        assert!((recovered.arousal - record.arousal).abs() < 1e-6);
        assert!((recovered.psi - record.psi).abs() < 1e-10);
        assert_eq!(recovered.topics, record.topics);
        assert_eq!(recovered.metadata, record.metadata);
        assert!((recovered.consolidation_strength - record.consolidation_strength).abs() < 1e-10);
        assert_eq!(recovered.retrieval_count, record.retrieval_count);
        assert_eq!(recovered.encoding.0, record.encoding.0);
    }

    #[test]
    fn test_record_to_batch_all_memory_types() {
        for (mt, expected_str) in [
            (MemoryType::Episodic, "episodic"),
            (MemoryType::Semantic, "semantic"),
            (MemoryType::Procedural, "procedural"),
            (MemoryType::Working, "working"),
        ] {
            let mut record = test_record("mt-test", 0);
            record.memory_type = mt;
            let batch = record_to_batch(&record).unwrap();
            let recovered = batch_to_records(&batch);
            assert_eq!(
                recovered[0].memory_type, mt,
                "Memory type {expected_str} should survive roundtrip"
            );
        }
    }

    #[test]
    fn test_batch_to_records_empty_batch() {
        let schema = memories_schema();
        let mut encoding_builder = FixedSizeBinaryBuilder::new(BinaryHV::BYTES as i32);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(encoding_builder.finish()),
                Arc::new(Int64Array::from(Vec::<i64>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(Float32Array::from(Vec::<f32>::new())),
                Arc::new(Float32Array::from(Vec::<f32>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(StringArray::from(Vec::<&str>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )
        .unwrap();

        let records = batch_to_records(&batch);
        assert!(records.is_empty(), "Empty batch should produce no records");
    }

    #[test]
    fn test_record_to_batch_empty_topics() {
        let mut record = test_record("empty-topics", 0);
        record.topics = Vec::new();
        let batch = record_to_batch(&record).unwrap();
        let recovered = batch_to_records(&batch);
        assert!(
            recovered[0].topics.is_empty(),
            "Empty topics should roundtrip"
        );
    }

    #[test]
    fn test_record_to_batch_special_characters_in_content() {
        let mut record = test_record("special-chars", 0);
        record.content = "Hello 'world' -- \"quoted\" & <tagged> \n newline".to_string();
        record.id = "id-with-'quote".to_string();
        let batch = record_to_batch(&record).unwrap();
        let recovered = batch_to_records(&batch);
        assert_eq!(recovered[0].content, record.content);
        assert_eq!(recovered[0].id, record.id);
    }

    #[test]
    fn test_memories_schema_field_count() {
        let schema = memories_schema();
        assert_eq!(schema.fields().len(), 12, "Schema should have 12 fields");
        assert!(schema.field_with_name("id").is_ok());
        assert!(schema.field_with_name("encoding").is_ok());
        assert!(schema.field_with_name("timestamp_ms").is_ok());
        assert!(schema.field_with_name("memory_type").is_ok());
        assert!(schema.field_with_name("phi").is_ok());
        assert!(schema.field_with_name("consolidation_strength").is_ok());
        assert!(schema.field_with_name("retrieval_count").is_ok());
    }

    #[test]
    fn test_dir_size_nonexistent_path() {
        assert_eq!(
            dir_size("/nonexistent/path/that/does/not/exist"),
            0,
            "Non-existent directory should return size 0"
        );
    }

    #[test]
    fn test_batch_to_records_negative_timestamp_clamped() {
        let record = test_record("neg-ts", 0);
        let schema = memories_schema();
        let mut encoding_builder = FixedSizeBinaryBuilder::new(BinaryHV::BYTES as i32);
        encoding_builder
            .append_value(record.encoding.0.as_slice())
            .unwrap();

        let batch_neg = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["neg-ts"])),
                Arc::new(encoding_builder.finish()),
                Arc::new(Int64Array::from(vec![-5000i64])),
                Arc::new(StringArray::from(vec!["episodic"])),
                Arc::new(StringArray::from(vec!["content"])),
                Arc::new(Float32Array::from(vec![0.0f32])),
                Arc::new(Float32Array::from(vec![0.0f32])),
                Arc::new(Float64Array::from(vec![0.0f64])),
                Arc::new(StringArray::from(vec!["[]"])),
                Arc::new(StringArray::from(vec!["{}"])),
                Arc::new(Float64Array::from(vec![0.0f64])),
                Arc::new(Int32Array::from(vec![0i32])),
            ],
        )
        .unwrap();

        let records = batch_to_records(&batch_neg);
        assert_eq!(
            records[0].timestamp_ms, 0,
            "Negative timestamp should be clamped to 0"
        );
    }
}
