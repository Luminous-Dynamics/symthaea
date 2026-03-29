// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Application state shared across handlers

use mycelix_desci_core::{
    query::QueryEngine,
    storage::{MemoryStorage, StorageBackend},
    trust::TrustManager,
};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Application metrics
#[derive(Clone)]
pub struct Metrics {
    pub queries_executed: Arc<AtomicU64>,
    pub claims_created: Arc<AtomicU64>,
    pub verifications_added: Arc<AtomicU64>,
    response_times: Arc<RwLock<Vec<u64>>>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            queries_executed: Arc::new(AtomicU64::new(0)),
            claims_created: Arc::new(AtomicU64::new(0)),
            verifications_added: Arc::new(AtomicU64::new(0)),
            response_times: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn average_response_time_ms(&self) -> f64 {
        // This would be more sophisticated in production
        0.0
    }
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Storage backend
    pub storage: MemoryStorage,
    /// Query engine
    pub query_engine: Arc<RwLock<QueryEngine>>,
    /// Trust manager
    pub trust_manager: Arc<RwLock<TrustManager>>,
    /// Application metrics
    pub metrics: Metrics,
    /// Server start time
    start_time: Instant,
}

impl AppState {
    /// Create new application state
    pub async fn new() -> crate::error::Result<Self> {
        let storage = MemoryStorage::new();
        let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
        let query_engine = QueryEngine::new(storage_arc);
        let trust_manager = TrustManager::new();

        Ok(Self {
            storage,
            query_engine: Arc::new(RwLock::new(query_engine)),
            trust_manager: Arc::new(RwLock::new(trust_manager)),
            metrics: Metrics::new(),
            start_time: Instant::now(),
        })
    }

    /// Get uptime since server start
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}
