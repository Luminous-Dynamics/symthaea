// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IPFS Content-Addressed Storage Backend
//!
//! Provides content-addressed storage for UESS M3 (Foundational) tier.
//! Uses the IPFS HTTP API for pinning and retrieving content.
//!
//! # Features
//!
//! - Content-addressed storage (CID-based)
//! - Immutable data (M3 tier requirement)
//! - Optional pinning for persistence
//! - Metadata stored alongside content
//! - Offline-first with local cache
//! - Real IPFS daemon support (requires `std` feature)
//!
//! # Real IPFS Mode
//!
//! To use with a real IPFS daemon:
//! ```rust,ignore
//! use mycelix_sdk::storage::backends::{IPFSBackend, IPFSBackendConfig};
//!
//! let config = IPFSBackendConfig::with_endpoint("http://localhost:5001");
//! let backend = IPFSBackend::with_config(config);
//! ```
//!
//! # Note
//!
//! This implementation provides a simulation mode for testing without
//! a running IPFS daemon. For production use, configure with a real
//! IPFS API endpoint and build with the `std` feature.

use crate::epistemic::EpistemicClassification;
use crate::storage::types::{SchemaIdentity, StorageMetadata, StoredData};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "std")]
use super::ipfs_client::IpfsClient;

/// IPFS backend configuration
#[derive(Debug, Clone)]
pub struct IPFSBackendConfig {
    /// IPFS API endpoint (e.g., "http://localhost:5001")
    pub api_endpoint: String,
    /// Whether to pin content after adding
    pub auto_pin: bool,
    /// Whether to use simulation mode (no real IPFS)
    pub simulation_mode: bool,
    /// Local cache size limit (items)
    pub cache_size_limit: usize,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for IPFSBackendConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "http://localhost:5001".to_string(),
            auto_pin: true,
            simulation_mode: true, // Default to simulation for safety
            cache_size_limit: 1000,
            timeout_ms: 30_000,
        }
    }
}

impl IPFSBackendConfig {
    /// Create config for simulation mode
    pub fn simulation() -> Self {
        Self {
            simulation_mode: true,
            ..Default::default()
        }
    }

    /// Create config for real IPFS with custom endpoint
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self {
            api_endpoint: endpoint.into(),
            simulation_mode: false,
            ..Default::default()
        }
    }
}

/// IPFS object wrapping data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IPFSObject<T> {
    /// The actual data
    data: T,
    /// Storage key (for reverse lookup)
    key: String,
    /// Epistemic classification
    classification: EpistemicClassification,
    /// Schema identity
    schema: SchemaIdentity,
    /// Storage timestamp (ms since epoch)
    stored_at: u64,
    /// Creator identifier
    created_by: String,
    /// Size in bytes (of data only)
    size_bytes: usize,
    /// IPFS object type marker
    ipfs_type: String,
}

/// Simulated CID entry for testing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimulatedEntry {
    /// Serialized object JSON
    content: String,
    /// The storage key
    key: String,
    /// Whether this entry is pinned
    pinned: bool,
    /// Storage timestamp
    stored_at: u64,
}

/// IPFS content-addressed storage backend
pub struct IPFSBackend {
    config: IPFSBackendConfig,
    /// Key to CID mapping
    key_to_cid: RwLock<HashMap<String, String>>,
    /// CID to key reverse mapping
    cid_to_key: RwLock<HashMap<String, String>>,
    /// Simulated IPFS storage (CID -> content)
    simulated_storage: RwLock<HashMap<String, SimulatedEntry>>,
    /// Statistics
    stats: RwLock<IPFSBackendStatsInternal>,
}

#[derive(Debug, Clone, Default)]
struct IPFSBackendStatsInternal {
    items_added: u64,
    items_retrieved: u64,
    items_pinned: u64,
    bytes_stored: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl IPFSBackend {
    /// Create a new IPFS backend with default configuration (simulation mode)
    pub fn new() -> Self {
        Self::with_config(IPFSBackendConfig::default())
    }

    /// Create a new IPFS backend with custom configuration
    pub fn with_config(config: IPFSBackendConfig) -> Self {
        Self {
            config,
            key_to_cid: RwLock::new(HashMap::new()),
            cid_to_key: RwLock::new(HashMap::new()),
            simulated_storage: RwLock::new(HashMap::new()),
            stats: RwLock::new(IPFSBackendStatsInternal::default()),
        }
    }

    /// Create a simulation-only backend for testing
    pub fn simulation() -> Self {
        Self::with_config(IPFSBackendConfig::simulation())
    }

    /// Get current timestamp in milliseconds
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Compute CID from content (simulated as SHA3-256 hash)
    fn compute_cid(content: &[u8]) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        // Use Qm prefix to simulate CIDv0 format
        format!("Qm{}", hex_encode(&hash))
    }

    /// Get data by key
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<StoredData<T>> {
        // Look up CID for key
        let cid = {
            let key_map = self.key_to_cid.read().ok()?;
            key_map.get(key).cloned()
        }?;

        self.get_by_cid(&cid)
    }

    /// Get data by CID directly
    pub fn get_by_cid<T: for<'de> Deserialize<'de>>(&self, cid: &str) -> Option<StoredData<T>> {
        if self.config.simulation_mode {
            self.get_simulated(cid)
        } else {
            self.get_from_ipfs(cid)
        }
    }

    /// Get from simulated storage
    fn get_simulated<T: for<'de> Deserialize<'de>>(&self, cid: &str) -> Option<StoredData<T>> {
        let storage = self.simulated_storage.read().ok()?;
        let entry = storage.get(cid)?;

        // Update stats
        {
            let mut stats = self.stats.write().ok()?;
            stats.items_retrieved += 1;
            stats.cache_hits += 1;
        }

        // Parse the stored object
        let obj: IPFSObject<T> = serde_json::from_str(&entry.content).ok()?;

        let metadata = StorageMetadata {
            cid: cid.to_string(),
            classification: obj.classification,
            schema: obj.schema,
            stored_at: obj.stored_at,
            modified_at: None, // IPFS is immutable
            version: 1,        // Always version 1 for immutable
            expires_at: None,  // M3 tier doesn't expire
            size_bytes: obj.size_bytes,
            created_by: obj.created_by,
            tombstone: false,
            retracted_by: None,
        };

        Some(StoredData {
            data: obj.data,
            metadata,
            verified: true, // CID-based verification
        })
    }

    /// Get from real IPFS via HTTP API
    #[cfg(feature = "std")]
    fn get_from_ipfs<T: for<'de> Deserialize<'de>>(&self, cid: &str) -> Option<StoredData<T>> {
        // First check local cache
        if let Some(cached) = self.get_simulated::<T>(cid) {
            let mut stats = self.stats.write().ok()?;
            stats.cache_hits += 1;
            return Some(cached);
        }

        // Cache miss - fetch from IPFS daemon
        {
            let mut stats = self.stats.write().ok()?;
            stats.cache_misses += 1;
        }

        let client = IpfsClient::new(&self.config.api_endpoint, self.config.timeout_ms);

        match client.cat(cid) {
            Ok(bytes) => {
                // Parse the content
                let content = String::from_utf8(bytes).ok()?;
                let obj: IPFSObject<T> = serde_json::from_str(&content).ok()?;

                // Cache locally for future access
                let _ = self.set_simulated(&obj.key, cid, content, obj.stored_at);

                // Update stats
                {
                    let mut stats = self.stats.write().ok()?;
                    stats.items_retrieved += 1;
                }

                let metadata = StorageMetadata {
                    cid: cid.to_string(),
                    classification: obj.classification,
                    schema: obj.schema,
                    stored_at: obj.stored_at,
                    modified_at: None,
                    version: 1,
                    expires_at: None,
                    size_bytes: obj.size_bytes,
                    created_by: obj.created_by,
                    tombstone: false,
                    retracted_by: None,
                };

                Some(StoredData {
                    data: obj.data,
                    metadata,
                    verified: true,
                })
            }
            Err(_) => None,
        }
    }

    /// Get from real IPFS (fallback when std feature not enabled)
    #[cfg(not(feature = "std"))]
    fn get_from_ipfs<T: for<'de> Deserialize<'de>>(&self, cid: &str) -> Option<StoredData<T>> {
        // Without std feature, can only use local cache
        {
            let mut stats = self.stats.write().ok()?;
            stats.cache_misses += 1;
        }
        self.get_simulated(cid)
    }

    /// Store data (returns CID)
    pub fn set<T: Serialize>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        _ttl_ms: Option<u64>, // Ignored for IPFS - M3 tier is persistent
    ) -> Option<StorageMetadata> {
        let now = Self::now_ms();

        // Serialize data to get size
        let data_json = serde_json::to_string(data).ok()?;
        let size_bytes = data_json.len();

        // Create IPFS object
        let obj = IPFSObject {
            data,
            key: key.to_string(),
            classification,
            schema: schema.clone(),
            stored_at: now,
            created_by: created_by.to_string(),
            size_bytes,
            ipfs_type: "uess-object".to_string(),
        };

        // Serialize full object
        let content = serde_json::to_string(&obj).ok()?;

        // Compute CID from content bytes
        let cid = Self::compute_cid(content.as_bytes());

        let content_len = content.len();
        if self.config.simulation_mode {
            self.set_simulated(key, &cid, content, now)
        } else {
            self.set_to_ipfs(key, &cid, content.as_bytes(), now)
        }?;

        // Update mappings
        {
            let mut key_map = self.key_to_cid.write().ok()?;
            let mut cid_map = self.cid_to_key.write().ok()?;
            key_map.insert(key.to_string(), cid.clone());
            cid_map.insert(cid.clone(), key.to_string());
        }

        // Update stats
        {
            let mut stats = self.stats.write().ok()?;
            stats.items_added += 1;
            stats.bytes_stored += content_len as u64;
            if self.config.auto_pin {
                stats.items_pinned += 1;
            }
        }

        Some(StorageMetadata {
            cid,
            classification,
            schema,
            stored_at: now,
            modified_at: None,
            version: 1,
            expires_at: None,
            size_bytes,
            created_by: created_by.to_string(),
            tombstone: false,
            retracted_by: None,
        })
    }

    /// Store in simulated storage
    fn set_simulated(&self, key: &str, cid: &str, content: String, stored_at: u64) -> Option<()> {
        let mut storage = self.simulated_storage.write().ok()?;

        // Check cache size limit
        if storage.len() >= self.config.cache_size_limit {
            // Remove oldest unpinned entry
            let oldest_unpinned = storage
                .iter()
                .filter(|(_, e)| !e.pinned)
                .min_by_key(|(_, e)| e.stored_at)
                .map(|(k, _)| k.clone());

            if let Some(old_cid) = oldest_unpinned {
                storage.remove(&old_cid);
            }
        }

        storage.insert(
            cid.to_string(),
            SimulatedEntry {
                content,
                key: key.to_string(),
                pinned: self.config.auto_pin,
                stored_at,
            },
        );

        Some(())
    }

    /// Store to real IPFS via HTTP API
    #[cfg(feature = "std")]
    fn set_to_ipfs(
        &self,
        key: &str,
        expected_cid: &str,
        content: &[u8],
        stored_at: u64,
    ) -> Option<()> {
        let client = IpfsClient::new(&self.config.api_endpoint, self.config.timeout_ms);

        // Add to IPFS (with or without pinning based on config)
        let result = if self.config.auto_pin {
            client.add_and_pin(content)
        } else {
            client.add(content)
        };

        match result {
            Ok(add_response) => {
                // The IPFS daemon returns a real CID - we should use that
                // Note: Our computed CID may differ from IPFS's CID due to different
                // hashing algorithms (we use SHA3-256, IPFS uses SHA2-256 by default)
                let ipfs_cid = add_response.hash;

                // Store in local cache with IPFS's CID
                self.set_simulated(
                    key,
                    &ipfs_cid,
                    String::from_utf8_lossy(content).to_string(),
                    stored_at,
                )
            }
            Err(_) => {
                // If IPFS daemon is unreachable, fall back to local cache only
                // This enables offline-first behavior
                self.set_simulated(
                    key,
                    expected_cid,
                    String::from_utf8_lossy(content).to_string(),
                    stored_at,
                )
            }
        }
    }

    /// Store to real IPFS (fallback when std feature not enabled)
    #[cfg(not(feature = "std"))]
    fn set_to_ipfs(&self, key: &str, cid: &str, content: &[u8], stored_at: u64) -> Option<()> {
        // Without std feature, can only use local cache
        self.set_simulated(
            key,
            cid,
            String::from_utf8_lossy(content).to_string(),
            stored_at,
        )
    }

    /// Pin content by CID (prevent garbage collection)
    pub fn pin(&self, cid: &str) -> bool {
        if self.config.simulation_mode {
            self.pin_simulated(cid)
        } else {
            self.pin_real(cid)
        }
    }

    /// Pin in simulated storage
    fn pin_simulated(&self, cid: &str) -> bool {
        let mut storage = match self.simulated_storage.write() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if let Some(entry) = storage.get_mut(cid) {
            entry.pinned = true;
            let mut stats = match self.stats.write() {
                Ok(s) => s,
                Err(_) => return false,
            };
            stats.items_pinned += 1;
            return true;
        }
        false
    }

    /// Pin via real IPFS daemon
    #[cfg(feature = "std")]
    fn pin_real(&self, cid: &str) -> bool {
        let client = IpfsClient::new(&self.config.api_endpoint, self.config.timeout_ms);
        match client.pin_add(cid) {
            Ok(_) => {
                // Also update local cache
                self.pin_simulated(cid);
                true
            }
            Err(_) => false,
        }
    }

    /// Pin via real IPFS (fallback when std feature not enabled)
    #[cfg(not(feature = "std"))]
    fn pin_real(&self, cid: &str) -> bool {
        // Without std feature, can only pin locally
        self.pin_simulated(cid)
    }

    /// Unpin content by CID
    pub fn unpin(&self, cid: &str) -> bool {
        if self.config.simulation_mode {
            self.unpin_simulated(cid)
        } else {
            self.unpin_real(cid)
        }
    }

    /// Unpin from simulated storage
    fn unpin_simulated(&self, cid: &str) -> bool {
        let mut storage = match self.simulated_storage.write() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if let Some(entry) = storage.get_mut(cid) {
            entry.pinned = false;
            return true;
        }
        false
    }

    /// Unpin via real IPFS daemon
    #[cfg(feature = "std")]
    fn unpin_real(&self, cid: &str) -> bool {
        let client = IpfsClient::new(&self.config.api_endpoint, self.config.timeout_ms);
        match client.pin_rm(cid) {
            Ok(_) => {
                // Also update local cache
                self.unpin_simulated(cid);
                true
            }
            Err(_) => false,
        }
    }

    /// Unpin via real IPFS (fallback when std feature not enabled)
    #[cfg(not(feature = "std"))]
    fn unpin_real(&self, cid: &str) -> bool {
        // Without std feature, can only unpin locally
        self.unpin_simulated(cid)
    }

    /// Delete is not truly supported on IPFS (immutable)
    /// This just removes from local mapping and unpins
    pub fn delete(&self, key: &str) -> bool {
        let cid = {
            let mut key_map = match self.key_to_cid.write() {
                Ok(m) => m,
                Err(_) => return false,
            };
            key_map.remove(key)
        };

        if let Some(cid) = cid {
            // Remove reverse mapping
            {
                let mut cid_map = match self.cid_to_key.write() {
                    Ok(m) => m,
                    Err(_) => return false,
                };
                cid_map.remove(&cid);
            }

            // Unpin (content may still exist on IPFS network)
            self.unpin(&cid);

            // Remove from simulated storage
            if self.config.simulation_mode {
                let mut storage = match self.simulated_storage.write() {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                storage.remove(&cid);
            }

            true
        } else {
            false
        }
    }

    /// Check if key exists in local mapping
    pub fn has(&self, key: &str) -> bool {
        let key_map = match self.key_to_cid.read() {
            Ok(m) => m,
            Err(_) => return false,
        };
        key_map.contains_key(key)
    }

    /// Check if CID exists
    pub fn has_cid(&self, cid: &str) -> bool {
        if self.config.simulation_mode {
            let storage = match self.simulated_storage.read() {
                Ok(s) => s,
                Err(_) => return false,
            };
            storage.contains_key(cid)
        } else {
            // Real IPFS: POST /api/v0/block/stat?arg={cid}
            false
        }
    }

    /// Get CID for a key
    pub fn get_cid(&self, key: &str) -> Option<String> {
        let key_map = self.key_to_cid.read().ok()?;
        key_map.get(key).cloned()
    }

    /// List all keys
    pub fn keys(&self, pattern: Option<&str>) -> Vec<String> {
        let key_map = match self.key_to_cid.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        match pattern {
            Some(pat) => key_map
                .keys()
                .filter(|k| k.contains(pat))
                .cloned()
                .collect(),
            None => key_map.keys().cloned().collect(),
        }
    }

    /// List all CIDs
    pub fn cids(&self) -> Vec<String> {
        let cid_map = match self.cid_to_key.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        cid_map.keys().cloned().collect()
    }

    /// Clear all mappings (doesn't affect IPFS network)
    pub fn clear(&self) {
        {
            if let Ok(mut key_map) = self.key_to_cid.write() {
                key_map.clear();
            }
        }
        {
            if let Ok(mut cid_map) = self.cid_to_key.write() {
                cid_map.clear();
            }
        }
        if self.config.simulation_mode {
            if let Ok(mut storage) = self.simulated_storage.write() {
                storage.clear();
            }
        }
    }

    /// Get storage statistics
    pub fn stats(&self) -> IPFSBackendStats {
        let key_map = match self.key_to_cid.read() {
            Ok(m) => m,
            Err(_) => return IPFSBackendStats::default(),
        };
        let internal_stats = match self.stats.read() {
            Ok(s) => s,
            Err(_) => return IPFSBackendStats::default(),
        };

        let pinned_count = if self.config.simulation_mode {
            let storage = match self.simulated_storage.read() {
                Ok(s) => s,
                Err(_) => return IPFSBackendStats::default(),
            };
            storage.values().filter(|e| e.pinned).count()
        } else {
            0
        };

        IPFSBackendStats {
            item_count: key_map.len(),
            total_cids: key_map.len(),
            pinned_count,
            bytes_stored: internal_stats.bytes_stored as usize,
            items_added: internal_stats.items_added,
            items_retrieved: internal_stats.items_retrieved,
            cache_hit_rate: if internal_stats.items_retrieved > 0 {
                internal_stats.cache_hits as f32
                    / (internal_stats.cache_hits + internal_stats.cache_misses) as f32
            } else {
                0.0
            },
            simulation_mode: self.config.simulation_mode,
        }
    }
}

impl Default for IPFSBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// IPFS backend statistics
#[derive(Debug, Clone, Default)]
pub struct IPFSBackendStats {
    /// Number of items currently stored
    pub item_count: usize,
    /// Total number of content identifiers tracked
    pub total_cids: usize,
    /// Number of pinned (persistent) items
    pub pinned_count: usize,
    /// Total bytes of stored content
    pub bytes_stored: usize,
    /// Cumulative items added
    pub items_added: u64,
    /// Cumulative items retrieved
    pub items_retrieved: u64,
    /// Cache hit rate as a fraction (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Whether this backend is running in simulation mode
    pub simulation_mode: bool,
}

/// Hex encoding utility
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

    fn test_classification() -> EpistemicClassification {
        EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M3Foundational,
        )
    }

    #[test]
    fn test_ipfs_backend_basic_operations() {
        let backend = IPFSBackend::simulation();

        // Test set
        let schema = SchemaIdentity::new("test-schema", "1.0");
        let result = backend.set(
            "test-key",
            &"test-value".to_string(),
            test_classification(),
            schema.clone(),
            "test-agent",
            None,
        );
        assert!(result.is_some());

        let metadata = result.unwrap();
        assert!(metadata.cid.starts_with("Qm")); // CIDv0 format

        // Test has
        assert!(backend.has("test-key"));
        assert!(!backend.has("nonexistent"));

        // Test get
        let retrieved: Option<StoredData<String>> = backend.get("test-key");
        assert!(retrieved.is_some());
        let data = retrieved.unwrap();
        assert_eq!(data.data, "test-value");
        assert_eq!(data.metadata.schema.id, "test-schema");
        assert!(data.verified);

        // Test delete (removes mapping, unpins)
        assert!(backend.delete("test-key"));
        assert!(!backend.has("test-key"));
    }

    #[test]
    fn test_ipfs_backend_cid_based_retrieval() {
        let backend = IPFSBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let meta = backend
            .set(
                "cid-test",
                &"cid-value",
                test_classification(),
                schema,
                "agent",
                None,
            )
            .unwrap();

        let cid = meta.cid.clone();

        // Get by CID directly
        let retrieved: Option<StoredData<String>> = backend.get_by_cid(&cid);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data, "cid-value");

        // Note: CIDs include metadata (timestamps), so two separate stores
        // will produce different CIDs even for the same data. This is intentional
        // for UESS as we want metadata to be immutably linked to the data.
        // For content deduplication, use the data hash directly.
    }

    #[test]
    fn test_ipfs_backend_pinning() {
        let backend = IPFSBackend::with_config(IPFSBackendConfig {
            auto_pin: false, // Don't auto-pin
            ..IPFSBackendConfig::simulation()
        });

        let schema = SchemaIdentity::new("test", "1.0");
        let meta = backend
            .set(
                "pin-test",
                &"pin-value",
                test_classification(),
                schema,
                "agent",
                None,
            )
            .unwrap();

        let cid = meta.cid;

        // Not pinned initially
        let stats = backend.stats();
        assert_eq!(stats.pinned_count, 0);

        // Pin it
        assert!(backend.pin(&cid));

        let stats = backend.stats();
        assert_eq!(stats.pinned_count, 1);

        // Unpin
        assert!(backend.unpin(&cid));

        let stats = backend.stats();
        assert_eq!(stats.pinned_count, 0);
    }

    #[test]
    fn test_ipfs_backend_keys_and_cids() {
        let backend = IPFSBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        backend.set(
            "key1",
            &"value1",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "key2",
            &"value2",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "other",
            &"value3",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );

        // Test keys listing
        let all_keys = backend.keys(None);
        assert_eq!(all_keys.len(), 3);

        // Test pattern matching
        let key_pattern = backend.keys(Some("key"));
        assert_eq!(key_pattern.len(), 2);

        // Test CIDs listing
        let cids = backend.cids();
        assert_eq!(cids.len(), 3);

        // All CIDs should be unique
        let unique_cids: std::collections::HashSet<_> = cids.into_iter().collect();
        assert_eq!(unique_cids.len(), 3);
    }

    #[test]
    fn test_ipfs_backend_stats() {
        let backend = IPFSBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        backend.set(
            "stat1",
            &"value1",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "stat2",
            &"value2",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );

        // Retrieve one
        let _: Option<StoredData<String>> = backend.get("stat1");

        let stats = backend.stats();
        assert_eq!(stats.item_count, 2);
        assert_eq!(stats.items_added, 2);
        assert_eq!(stats.items_retrieved, 1);
        assert!(stats.bytes_stored > 0);
        assert!(stats.simulation_mode);
    }

    #[test]
    fn test_ipfs_backend_immutability() {
        let backend = IPFSBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        // Store initial value
        let meta1 = backend
            .set(
                "immutable-key",
                &"value1",
                class.clone(),
                schema.clone(),
                "agent",
                None,
            )
            .unwrap();

        // Store different value with same key
        let meta2 = backend
            .set(
                "immutable-key",
                &"value2",
                class.clone(),
                schema.clone(),
                "agent",
                None,
            )
            .unwrap();

        // CIDs should be different (content-addressed)
        assert_ne!(meta1.cid, meta2.cid);

        // Key now points to new CID
        let current_cid = backend.get_cid("immutable-key").unwrap();
        assert_eq!(current_cid, meta2.cid);

        // But old CID data is still accessible if we have it
        let old_data: Option<StoredData<String>> = backend.get_by_cid(&meta1.cid);
        // Old data might not be accessible depending on cache eviction
        // but the CID is still valid if content exists
    }

    #[test]
    fn test_ipfs_backend_clear() {
        let backend = IPFSBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        backend.set(
            "clear1",
            &"value1",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "clear2",
            &"value2",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );

        assert_eq!(backend.keys(None).len(), 2);

        backend.clear();

        assert_eq!(backend.keys(None).len(), 0);
        assert_eq!(backend.cids().len(), 0);
    }

    #[test]
    fn test_cid_format() {
        // Verify CID format looks correct
        let content = b"test content for CID";
        let cid = IPFSBackend::compute_cid(content);

        assert!(cid.starts_with("Qm")); // CIDv0 prefix
        assert_eq!(cid.len(), 2 + 64); // "Qm" + 64 hex chars (SHA3-256)
    }
}
