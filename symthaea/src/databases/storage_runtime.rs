// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Asynchronous storage boundary for the cognitive loop.
//!
//! The cognitive cycle should not block on a database write. This module
//! provides a small write-behind worker that accepts durable memory operations
//! through a bounded channel and applies them outside the hot path.

use std::sync::Arc;

use tokio::sync::{mpsc, mpsc::error::TrySendError, oneshot};
use tokio::task::JoinHandle;

use super::{ConsciousnessDatabase, DatabaseError, DbResult, MemoryRecord};

/// Default number of pending storage operations allowed before backpressure.
pub const DEFAULT_STORAGE_QUEUE_CAPACITY: usize = 1024;

/// A queued storage operation.
#[derive(Debug)]
pub enum StorageOp {
    /// Store or replace a memory record.
    StoreMemory(MemoryRecord),
    /// Store or replace a batch of memory records.
    StoreMemoryBatch(Vec<MemoryRecord>),
    /// Delete a memory by ID.
    DeleteMemory(String),
    /// Ask the worker to flush all previously accepted operations.
    Flush(oneshot::Sender<DbResult<()>>),
    /// Stop the worker after processing prior operations.
    Shutdown(oneshot::Sender<DbResult<()>>),
}

/// Error returned when an operation cannot be queued.
#[derive(Debug)]
pub enum StorageRuntimeError {
    /// The worker has already shut down.
    Closed,
    /// The bounded queue is full.
    Full,
    /// The worker dropped a flush/shutdown acknowledgement.
    AckDropped,
    /// The backend returned an error.
    Backend(DatabaseError),
}

impl std::fmt::Display for StorageRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "storage runtime is closed"),
            Self::Full => write!(f, "storage runtime queue is full"),
            Self::AckDropped => write!(f, "storage runtime acknowledgement was dropped"),
            Self::Backend(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for StorageRuntimeError {}

impl From<DatabaseError> for StorageRuntimeError {
    fn from(value: DatabaseError) -> Self {
        Self::Backend(value)
    }
}

/// Cloneable handle used by cognitive code to enqueue durable writes.
#[derive(Clone)]
pub struct StorageRuntimeHandle {
    tx: mpsc::Sender<StorageOp>,
}

impl StorageRuntimeHandle {
    /// Try to enqueue a store without awaiting. Use this on hot paths.
    pub fn try_store_memory(&self, record: MemoryRecord) -> Result<(), StorageRuntimeError> {
        self.tx
            .try_send(StorageOp::StoreMemory(record))
            .map_err(|err| match err {
                TrySendError::Full(_) => StorageRuntimeError::Full,
                TrySendError::Closed(_) => StorageRuntimeError::Closed,
            })
    }

    /// Try to enqueue a batch store without awaiting. Use this on hot paths.
    pub fn try_store_memory_batch(
        &self,
        records: Vec<MemoryRecord>,
    ) -> Result<(), StorageRuntimeError> {
        if records.is_empty() {
            return Ok(());
        }
        self.tx
            .try_send(StorageOp::StoreMemoryBatch(records))
            .map_err(|err| match err {
                TrySendError::Full(_) => StorageRuntimeError::Full,
                TrySendError::Closed(_) => StorageRuntimeError::Closed,
            })
    }

    /// Try to enqueue a delete without awaiting. Use this on hot paths.
    pub fn try_delete_memory(&self, id: impl Into<String>) -> Result<(), StorageRuntimeError> {
        self.tx
            .try_send(StorageOp::DeleteMemory(id.into()))
            .map_err(|err| match err {
                TrySendError::Full(_) => StorageRuntimeError::Full,
                TrySendError::Closed(_) => StorageRuntimeError::Closed,
            })
    }

    /// Enqueue a store, awaiting queue capacity if needed.
    pub async fn store_memory(&self, record: MemoryRecord) -> Result<(), StorageRuntimeError> {
        self.tx
            .send(StorageOp::StoreMemory(record))
            .await
            .map_err(|_| StorageRuntimeError::Closed)
    }

    /// Enqueue a batch store, awaiting queue capacity if needed.
    pub async fn store_memory_batch(
        &self,
        records: Vec<MemoryRecord>,
    ) -> Result<(), StorageRuntimeError> {
        if records.is_empty() {
            return Ok(());
        }
        self.tx
            .send(StorageOp::StoreMemoryBatch(records))
            .await
            .map_err(|_| StorageRuntimeError::Closed)
    }

    /// Enqueue a delete, awaiting queue capacity if needed.
    pub async fn delete_memory(&self, id: impl Into<String>) -> Result<(), StorageRuntimeError> {
        self.tx
            .send(StorageOp::DeleteMemory(id.into()))
            .await
            .map_err(|_| StorageRuntimeError::Closed)
    }

    /// Wait until all previously accepted operations have been applied.
    pub async fn flush(&self) -> Result<(), StorageRuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(StorageOp::Flush(tx))
            .await
            .map_err(|_| StorageRuntimeError::Closed)?;
        rx.await.map_err(|_| StorageRuntimeError::AckDropped)??;
        Ok(())
    }

    /// Stop the worker after applying all previously accepted operations.
    pub async fn shutdown(&self) -> Result<(), StorageRuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(StorageOp::Shutdown(tx))
            .await
            .map_err(|_| StorageRuntimeError::Closed)?;
        rx.await.map_err(|_| StorageRuntimeError::AckDropped)??;
        Ok(())
    }
}

/// Spawn a write-behind storage worker.
pub fn spawn_storage_runtime(
    backend: Arc<dyn ConsciousnessDatabase>,
    capacity: usize,
) -> (StorageRuntimeHandle, JoinHandle<()>) {
    let capacity = capacity.max(1);
    let (tx, rx) = mpsc::channel(capacity);
    let handle = StorageRuntimeHandle { tx };
    let join = tokio::spawn(storage_worker(backend, rx));
    (handle, join)
}

async fn storage_worker(
    backend: Arc<dyn ConsciousnessDatabase>,
    mut rx: mpsc::Receiver<StorageOp>,
) {
    while let Some(op) = rx.recv().await {
        match op {
            StorageOp::StoreMemory(record) => {
                if let Err(err) = backend.store(record).await {
                    tracing::warn!(error = %err, "storage runtime failed to store memory");
                }
            }
            StorageOp::StoreMemoryBatch(records) => {
                if let Err(err) = backend.store_batch(records).await {
                    tracing::warn!(error = %err, "storage runtime failed to store memory batch");
                }
            }
            StorageOp::DeleteMemory(id) => {
                if let Err(err) = backend.delete(&id).await {
                    tracing::warn!(error = %err, id, "storage runtime failed to delete memory");
                }
            }
            StorageOp::Flush(ack) => {
                let _ = ack.send(Ok(()));
            }
            StorageOp::Shutdown(ack) => {
                let _ = ack.send(Ok(()));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::databases::{MemoryType, SqliteMemory};
    use symthaea_core::hdc::binary_hv::BinaryHV;

    fn record(id: &str, seed: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            memory_type: MemoryType::Episodic,
            encoding: BinaryHV::random(seed),
            content: format!("queued memory {id}"),
            timestamp_ms: seed,
            valence: 0.0,
            arousal: 0.0,
            psi: 0.0,
            topics: Vec::new(),
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        }
    }

    #[tokio::test]
    async fn storage_runtime_flushes_queued_writes() {
        let db = Arc::new(SqliteMemory::in_memory().unwrap());
        let backend: Arc<dyn ConsciousnessDatabase> = db.clone();
        let (runtime, worker) = spawn_storage_runtime(backend, 8);

        runtime.try_store_memory(record("a", 1)).unwrap();
        runtime.store_memory(record("b", 2)).await.unwrap();
        runtime.flush().await.unwrap();

        assert_eq!(db.count().await.unwrap(), 2);
        assert!(db.get("a").await.unwrap().is_some());
        assert!(db.get("b").await.unwrap().is_some());

        runtime.shutdown().await.unwrap();
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn storage_runtime_applies_deletes() {
        let db = Arc::new(SqliteMemory::in_memory().unwrap());
        db.store(record("delete-me", 3)).await.unwrap();

        let backend: Arc<dyn ConsciousnessDatabase> = db.clone();
        let (runtime, worker) = spawn_storage_runtime(backend, 8);

        runtime.try_delete_memory("delete-me").unwrap();
        runtime.flush().await.unwrap();

        assert!(db.get("delete-me").await.unwrap().is_none());

        runtime.shutdown().await.unwrap();
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn storage_runtime_flushes_batch_writes() {
        let db = Arc::new(SqliteMemory::in_memory().unwrap());
        let backend: Arc<dyn ConsciousnessDatabase> = db.clone();
        let (runtime, worker) = spawn_storage_runtime(backend, 8);

        runtime
            .try_store_memory_batch(vec![record("batch-a", 4), record("batch-b", 5)])
            .unwrap();
        runtime.flush().await.unwrap();

        assert_eq!(db.count().await.unwrap(), 2);
        assert!(db.get("batch-a").await.unwrap().is_some());
        assert!(db.get("batch-b").await.unwrap().is_some());

        runtime.shutdown().await.unwrap();
        worker.await.unwrap();
    }
}
