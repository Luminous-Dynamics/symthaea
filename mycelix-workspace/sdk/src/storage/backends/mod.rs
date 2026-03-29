// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UESS Storage Backends
//!
//! Backend implementations for different storage tiers:
//! - Memory: Ephemeral, in-memory (M0)
//! - Local: File-based persistent (M1)
//! - DHT: Holochain distributed (M2)
//! - IPFS: Content-addressed (M3)

mod dht;
mod ipfs;
mod local;
mod memory;

// Real IPFS HTTP client (requires std feature)
#[cfg(feature = "std")]
pub mod ipfs_client;

// Holochain DHT client infrastructure
pub mod dht_client;

pub use dht::{DHTBackend, DHTBackendConfig, DHTBackendStats};
pub use ipfs::{IPFSBackend, IPFSBackendConfig, IPFSBackendStats};
pub use local::{LocalBackend, LocalBackendConfig, LocalBackendStats};
pub use memory::{MemoryBackend, MemoryBackendConfig, MemoryBackendStats};

use crate::epistemic::EpistemicClassification;
use crate::storage::types::{SchemaIdentity, StorageMetadata, StoredData};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// Backend adapter trait for storage operations
///
/// All backends must implement this trait to be usable by the UESS.
pub trait StorageBackendAdapter: Send + Sync {
    /// Get data by key
    fn get<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Option<StoredData<T>>> + Send + '_>>;

    /// Store data
    fn set<T: Serialize + Send + Sync + 'static>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Option<StorageMetadata>> + Send + '_>>;

    /// Delete data
    fn delete(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>>;

    /// Check if key exists
    fn has(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>>;

    /// List keys matching pattern
    fn keys(&self, pattern: Option<&str>)
        -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>>;

    /// Clear all data
    fn clear(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Get storage statistics
    fn stats(&self) -> Pin<Box<dyn Future<Output = BackendStats> + Send + '_>>;
}

/// Generic backend statistics
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Number of items stored in the backend
    pub item_count: usize,
    /// Total bytes consumed across all items
    pub total_size_bytes: usize,
    /// Timestamp (ms) of the oldest item, if available
    pub oldest_item: Option<u64>,
    /// Timestamp (ms) of the newest item, if available
    pub newest_item: Option<u64>,
}

/// Synchronous backend adapter for simple backends like Memory
impl StorageBackendAdapter for MemoryBackend {
    fn get<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Option<StoredData<T>>> + Send + '_>> {
        let result = MemoryBackend::get(self, key);
        Box::pin(async move { result })
    }

    fn set<T: Serialize + Send + Sync + 'static>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Option<StorageMetadata>> + Send + '_>> {
        let result =
            MemoryBackend::set(self, key, data, classification, schema, created_by, ttl_ms);
        Box::pin(async move { result })
    }

    fn delete(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = MemoryBackend::delete(self, key);
        Box::pin(async move { result })
    }

    fn has(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = MemoryBackend::has(self, key);
        Box::pin(async move { result })
    }

    fn keys(
        &self,
        pattern: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        let result = MemoryBackend::keys(self, pattern);
        Box::pin(async move { result })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        MemoryBackend::clear(self);
        Box::pin(async {})
    }

    fn stats(&self) -> Pin<Box<dyn Future<Output = BackendStats> + Send + '_>> {
        let stats = MemoryBackend::stats(self);
        Box::pin(async move {
            BackendStats {
                item_count: stats.item_count,
                total_size_bytes: stats.total_size_bytes,
                oldest_item: None,
                newest_item: None,
            }
        })
    }
}

// LocalBackend adapter implementation
impl StorageBackendAdapter for LocalBackend {
    fn get<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Option<StoredData<T>>> + Send + '_>> {
        let result = LocalBackend::get(self, key);
        Box::pin(async move { result })
    }

    fn set<T: Serialize + Send + Sync + 'static>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Option<StorageMetadata>> + Send + '_>> {
        let result = LocalBackend::set(self, key, data, classification, schema, created_by, ttl_ms);
        Box::pin(async move { result })
    }

    fn delete(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = LocalBackend::delete(self, key);
        Box::pin(async move { result })
    }

    fn has(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = LocalBackend::has(self, key);
        Box::pin(async move { result })
    }

    fn keys(
        &self,
        pattern: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        let result = LocalBackend::keys(self, pattern);
        Box::pin(async move { result })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        LocalBackend::clear(self);
        Box::pin(async {})
    }

    fn stats(&self) -> Pin<Box<dyn Future<Output = BackendStats> + Send + '_>> {
        let stats = LocalBackend::stats(self);
        Box::pin(async move {
            BackendStats {
                item_count: stats.item_count,
                total_size_bytes: stats.total_size_bytes,
                oldest_item: None,
                newest_item: None,
            }
        })
    }
}

// IPFSBackend adapter implementation
impl StorageBackendAdapter for IPFSBackend {
    fn get<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Option<StoredData<T>>> + Send + '_>> {
        let result = IPFSBackend::get(self, key);
        Box::pin(async move { result })
    }

    fn set<T: Serialize + Send + Sync + 'static>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Option<StorageMetadata>> + Send + '_>> {
        let result = IPFSBackend::set(self, key, data, classification, schema, created_by, ttl_ms);
        Box::pin(async move { result })
    }

    fn delete(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = IPFSBackend::delete(self, key);
        Box::pin(async move { result })
    }

    fn has(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = IPFSBackend::has(self, key);
        Box::pin(async move { result })
    }

    fn keys(
        &self,
        pattern: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        let result = IPFSBackend::keys(self, pattern);
        Box::pin(async move { result })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        IPFSBackend::clear(self);
        Box::pin(async {})
    }

    fn stats(&self) -> Pin<Box<dyn Future<Output = BackendStats> + Send + '_>> {
        let stats = IPFSBackend::stats(self);
        Box::pin(async move {
            BackendStats {
                item_count: stats.item_count,
                total_size_bytes: stats.bytes_stored,
                oldest_item: None,
                newest_item: None,
            }
        })
    }
}

// DHTBackend adapter implementation
impl StorageBackendAdapter for DHTBackend {
    fn get<T: for<'de> Deserialize<'de> + Send + 'static>(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Option<StoredData<T>>> + Send + '_>> {
        let result = DHTBackend::get(self, key);
        Box::pin(async move { result })
    }

    fn set<T: Serialize + Send + Sync + 'static>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Option<StorageMetadata>> + Send + '_>> {
        let result = DHTBackend::set(self, key, data, classification, schema, created_by, ttl_ms);
        Box::pin(async move { result })
    }

    fn delete(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = DHTBackend::delete(self, key);
        Box::pin(async move { result })
    }

    fn has(&self, key: &str) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
        let result = DHTBackend::has(self, key);
        Box::pin(async move { result })
    }

    fn keys(
        &self,
        pattern: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        let result = DHTBackend::keys(self, pattern);
        Box::pin(async move { result })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        DHTBackend::clear(self);
        Box::pin(async {})
    }

    fn stats(&self) -> Pin<Box<dyn Future<Output = BackendStats> + Send + '_>> {
        let stats = DHTBackend::stats(self);
        Box::pin(async move {
            BackendStats {
                item_count: stats.entry_count,
                total_size_bytes: stats.bytes_stored,
                oldest_item: None,
                newest_item: None,
            }
        })
    }
}
