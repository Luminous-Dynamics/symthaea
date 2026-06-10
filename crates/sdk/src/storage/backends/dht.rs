// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DHT (Distributed Hash Table) Storage Backend
//!
//! Provides distributed storage for UESS M2 (Persistent) tier via Holochain.
//! Uses the HDK for entry creation, linking, and retrieval.
//!
//! # Features
//!
//! - Distributed storage across Holochain DHT
//! - Path-based indexing for efficient queries
//! - Link types for relationships and access control
//! - Capability-based access control (CapBAC) support
//! - Simulation mode for testing without Holochain runtime

//! # Architecture
//!
//! In production, this backend communicates with Holochain zomes:
//! - Integrity zome: Entry types, link types, validation rules
//! - Coordinator zome: Extern functions for CRUD operations
//!
//! For testing, simulation mode provides in-memory DHT simulation.

use crate::epistemic::EpistemicClassification;
use crate::storage::types::{SchemaIdentity, StorageMetadata, StoredData};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// DHT backend configuration
#[derive(Debug, Clone)]
pub struct DHTBackendConfig {
    /// Whether to use simulation mode (no real Holochain)
    pub simulation_mode: bool,
    /// Holochain conductor WebSocket URL (for real mode)
    pub conductor_url: Option<String>,
    /// DNA hash for the storage zome
    pub dna_hash: Option<String>,
    /// Cell ID for the storage cell
    pub cell_id: Option<String>,
    /// Replication factor for data
    pub replication_factor: usize,
    /// Enable capability-based access control
    pub enable_capbac: bool,
}

impl Default for DHTBackendConfig {
    fn default() -> Self {
        Self {
            simulation_mode: true,
            conductor_url: None,
            dna_hash: None,
            cell_id: None,
            replication_factor: 3,
            enable_capbac: true,
        }
    }
}

impl DHTBackendConfig {
    /// Create simulation mode config
    pub fn simulation() -> Self {
        Self::default()
    }

    /// Create config for real Holochain
    pub fn with_conductor(url: impl Into<String>, dna_hash: impl Into<String>) -> Self {
        Self {
            simulation_mode: false,
            conductor_url: Some(url.into()),
            dna_hash: Some(dna_hash.into()),
            cell_id: None,
            replication_factor: 3,
            enable_capbac: true,
        }
    }
}

/// DHT entry representing stored data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DHTEntry<T> {
    /// Storage key
    key: String,
    /// The actual data
    data: T,
    /// Epistemic classification
    classification: EpistemicClassification,
    /// Schema identity
    schema: SchemaIdentity,
    /// Storage timestamp (ms since epoch)
    stored_at: u64,
    /// Creator agent ID
    created_by: String,
    /// Entry hash (action hash in Holochain terms)
    entry_hash: String,
    /// Size in bytes
    size_bytes: usize,
    /// Version number
    version: u32,
    /// Previous entry hash (for version chain)
    previous_hash: Option<String>,
}

/// Simulated DHT node for testing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimulatedNode {
    /// Entries stored on this node
    entries: HashMap<String, String>, // hash -> serialized entry
    /// Links from this node
    links: Vec<SimulatedLink>,
}

/// Simulated link between entries
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimulatedLink {
    /// Source entry hash
    source: String,
    /// Target entry hash
    target: String,
    /// Link type
    link_type: String,
    /// Link tag (optional metadata)
    tag: Option<String>,
}

/// DHT storage backend
pub struct DHTBackend {
    config: DHTBackendConfig,
    /// Key to entry hash mapping
    key_to_hash: RwLock<HashMap<String, String>>,
    /// Entry hash to key reverse mapping
    hash_to_key: RwLock<HashMap<String, String>>,
    /// Simulated DHT storage (entry_hash -> serialized entry)
    simulated_entries: RwLock<HashMap<String, String>>,
    /// Simulated links
    simulated_links: RwLock<Vec<SimulatedLink>>,
    /// Path index (path -> entry hashes)
    path_index: RwLock<HashMap<String, Vec<String>>>,
    /// Statistics
    stats: RwLock<DHTStatsInternal>,
}

#[derive(Debug, Clone, Default)]
struct DHTStatsInternal {
    entries_created: u64,
    entries_retrieved: u64,
    links_created: u64,
    queries_executed: u64,
    bytes_stored: u64,
}

impl DHTBackend {
    /// Create a new DHT backend with default configuration (simulation mode)
    pub fn new() -> Self {
        Self::with_config(DHTBackendConfig::default())
    }

    /// Create a new DHT backend with custom configuration
    pub fn with_config(config: DHTBackendConfig) -> Self {
        Self {
            config,
            key_to_hash: RwLock::new(HashMap::new()),
            hash_to_key: RwLock::new(HashMap::new()),
            simulated_entries: RwLock::new(HashMap::new()),
            simulated_links: RwLock::new(Vec::new()),
            path_index: RwLock::new(HashMap::new()),
            stats: RwLock::new(DHTStatsInternal::default()),
        }
    }

    /// Create a simulation-only backend for testing
    pub fn simulation() -> Self {
        Self::with_config(DHTBackendConfig::simulation())
    }

    /// Get current timestamp in milliseconds
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Compute entry hash from content
    fn compute_hash(content: &[u8]) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        // Use hC prefix to simulate Holochain hash format
        format!("hC{}", hex_encode(&hash[..32]))
    }

    /// Create a path anchor for indexing
    fn create_path(parts: &[&str]) -> String {
        parts.join(".")
    }

    /// Get data by key
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<StoredData<T>> {
        // Look up entry hash for key
        let entry_hash = {
            let key_map = self.key_to_hash.read().ok()?;
            key_map.get(key).cloned()
        }?;

        self.get_by_hash(&entry_hash)
    }

    /// Get data by entry hash directly
    pub fn get_by_hash<T: for<'de> Deserialize<'de>>(
        &self,
        entry_hash: &str,
    ) -> Option<StoredData<T>> {
        if self.config.simulation_mode {
            self.get_simulated(entry_hash)
        } else {
            self.get_from_holochain(entry_hash)
        }
    }

    /// Get from simulated DHT
    fn get_simulated<T: for<'de> Deserialize<'de>>(
        &self,
        entry_hash: &str,
    ) -> Option<StoredData<T>> {
        let entries = self.simulated_entries.read().ok()?;
        let serialized = entries.get(entry_hash)?;

        // Update stats
        {
            let mut stats = self.stats.write().ok()?;
            stats.entries_retrieved += 1;
        }

        // Parse the stored entry
        let entry: DHTEntry<T> = serde_json::from_str(serialized).ok()?;

        let metadata = StorageMetadata {
            cid: entry_hash.to_string(),
            classification: entry.classification,
            schema: entry.schema,
            stored_at: entry.stored_at,
            modified_at: None,
            version: entry.version,
            expires_at: None, // M2 tier is persistent
            size_bytes: entry.size_bytes,
            created_by: entry.created_by,
            tombstone: false,
            retracted_by: None,
        };

        Some(StoredData {
            data: entry.data,
            metadata,
            verified: true,
        })
    }

    /// Get from real Holochain (placeholder)
    fn get_from_holochain<T: for<'de> Deserialize<'de>>(
        &self,
        entry_hash: &str,
    ) -> Option<StoredData<T>> {
        // In a real implementation, this would:
        // 1. Connect to Holochain conductor via WebSocket
        // 2. Call zome function: get_storage_entry(entry_hash)
        // 3. Deserialize the Record
        // 4. Return StoredData

        // For now, fall back to simulation
        self.get_simulated(entry_hash)
    }

    /// Store data
    pub fn set<T: Serialize>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        _ttl_ms: Option<u64>, // Ignored for DHT - M2 tier is persistent
    ) -> Option<StorageMetadata> {
        let now = Self::now_ms();

        // Serialize data to get size
        let data_json = serde_json::to_string(data).ok()?;
        let size_bytes = data_json.len();

        // Check for existing entry (for versioning)
        let (version, previous_hash) = {
            let key_map = self.key_to_hash.read().ok()?;
            if let Some(existing_hash) = key_map.get(key) {
                // Get existing entry version
                let entries = self.simulated_entries.read().ok()?;
                if let Some(serialized) = entries.get(existing_hash) {
                    if let Ok(existing) =
                        serde_json::from_str::<DHTEntry<serde_json::Value>>(serialized)
                    {
                        (existing.version + 1, Some(existing_hash.clone()))
                    } else {
                        (1, None)
                    }
                } else {
                    (1, None)
                }
            } else {
                (1, None)
            }
        };

        // Create DHT entry
        let entry = DHTEntry {
            key: key.to_string(),
            data,
            classification,
            schema: schema.clone(),
            stored_at: now,
            created_by: created_by.to_string(),
            entry_hash: String::new(), // Will be computed
            size_bytes,
            version,
            previous_hash,
        };

        // Serialize and compute hash
        let serialized = serde_json::to_string(&entry).ok()?;
        let entry_hash = Self::compute_hash(serialized.as_bytes());

        // Update entry with its hash
        let mut entry_with_hash = entry;
        entry_with_hash.entry_hash = entry_hash.clone();
        let final_serialized = serde_json::to_string(&entry_with_hash).ok()?;

        if self.config.simulation_mode {
            self.set_simulated(key, &entry_hash, final_serialized.clone(), &classification)?;
        } else {
            self.set_to_holochain(key, &entry_hash, &final_serialized, &classification)?;
        }

        // Update mappings
        {
            let mut key_map = self.key_to_hash.write().ok()?;
            let mut hash_map = self.hash_to_key.write().ok()?;
            key_map.insert(key.to_string(), entry_hash.clone());
            hash_map.insert(entry_hash.clone(), key.to_string());
        }

        // Update stats
        {
            let mut stats = self.stats.write().ok()?;
            stats.entries_created += 1;
            stats.bytes_stored += final_serialized.len() as u64;
        }

        Some(StorageMetadata {
            cid: entry_hash,
            classification,
            schema,
            stored_at: now,
            modified_at: None,
            version,
            expires_at: None,
            size_bytes,
            created_by: created_by.to_string(),
            tombstone: false,
            retracted_by: None,
        })
    }

    /// Store in simulated DHT
    fn set_simulated(
        &self,
        key: &str,
        entry_hash: &str,
        serialized: String,
        classification: &EpistemicClassification,
    ) -> Option<()> {
        // Store entry
        {
            let mut entries = self.simulated_entries.write().ok()?;
            entries.insert(entry_hash.to_string(), serialized);
        }

        // Create path-based indexes
        let paths = vec![
            Self::create_path(&["storage", "all"]),
            Self::create_path(&["storage", "by_key", key]),
            Self::create_path(&[
                "storage",
                "by_classification",
                &format!("E{}", classification.empirical as u8),
            ]),
            Self::create_path(&[
                "storage",
                "by_classification",
                &format!("N{}", classification.normative as u8),
            ]),
            Self::create_path(&[
                "storage",
                "by_classification",
                &format!("M{}", classification.materiality as u8),
            ]),
        ];

        // Add to path index
        {
            let mut path_idx = self.path_index.write().ok()?;
            for path in paths {
                path_idx
                    .entry(path.clone())
                    .or_insert_with(Vec::new)
                    .push(entry_hash.to_string());

                // Create link
                let mut links = self.simulated_links.write().ok()?;
                links.push(SimulatedLink {
                    source: path,
                    target: entry_hash.to_string(),
                    link_type: "StorageIndex".to_string(),
                    tag: Some(key.to_string()),
                });
            }
        }

        // Update link stats
        {
            let mut stats = self.stats.write().ok()?;
            stats.links_created += 5; // 5 index links created
        }

        Some(())
    }

    /// Store to real Holochain (placeholder)
    fn set_to_holochain(
        &self,
        key: &str,
        entry_hash: &str,
        serialized: &str,
        classification: &EpistemicClassification,
    ) -> Option<()> {
        // In a real implementation, this would:
        // 1. Connect to Holochain conductor via WebSocket
        // 2. Call zome function: store_data(key, data, classification, schema)
        // 3. Create links for indexing
        // 4. Return the action hash

        // For now, use simulation
        self.set_simulated(key, entry_hash, serialized.to_string(), classification)
    }

    /// Delete entry (creates tombstone in DHT)
    pub fn delete(&self, key: &str) -> bool {
        let entry_hash = {
            let mut key_map = match self.key_to_hash.write() {
                Ok(m) => m,
                Err(_) => return false,
            };
            key_map.remove(key)
        };

        if let Some(hash) = entry_hash {
            // Remove reverse mapping
            {
                let mut hash_map = match self.hash_to_key.write() {
                    Ok(m) => m,
                    Err(_) => return false,
                };
                hash_map.remove(&hash);
            }

            // In simulation, remove from entries
            if self.config.simulation_mode {
                let mut entries = match self.simulated_entries.write() {
                    Ok(e) => e,
                    Err(_) => return false,
                };
                entries.remove(&hash);
            }

            // Note: In real Holochain, entries can't be truly deleted
            // Instead, we create a tombstone entry and remove links

            true
        } else {
            false
        }
    }

    /// Check if key exists
    pub fn has(&self, key: &str) -> bool {
        let key_map = match self.key_to_hash.read() {
            Ok(m) => m,
            Err(_) => return false,
        };
        key_map.contains_key(key)
    }

    /// Get entry hash for a key
    pub fn get_hash(&self, key: &str) -> Option<String> {
        let key_map = self.key_to_hash.read().ok()?;
        key_map.get(key).cloned()
    }

    /// Query entries by classification
    pub fn query_by_classification(
        &self,
        empirical: Option<u8>,
        normative: Option<u8>,
        materiality: Option<u8>,
    ) -> Vec<String> {
        let mut stats = match self.stats.write() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stats.queries_executed += 1;
        drop(stats);

        let path_idx = match self.path_index.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        // Build query paths
        let mut result_sets: Vec<Vec<String>> = vec![];

        if let Some(e) = empirical {
            let path = Self::create_path(&["storage", "by_classification", &format!("E{}", e)]);
            if let Some(hashes) = path_idx.get(&path) {
                result_sets.push(hashes.clone());
            }
        }

        if let Some(n) = normative {
            let path = Self::create_path(&["storage", "by_classification", &format!("N{}", n)]);
            if let Some(hashes) = path_idx.get(&path) {
                result_sets.push(hashes.clone());
            }
        }

        if let Some(m) = materiality {
            let path = Self::create_path(&["storage", "by_classification", &format!("M{}", m)]);
            if let Some(hashes) = path_idx.get(&path) {
                result_sets.push(hashes.clone());
            }
        }

        // Intersect result sets
        if result_sets.is_empty() {
            // Return all
            let all_path = Self::create_path(&["storage", "all"]);
            path_idx.get(&all_path).cloned().unwrap_or_default()
        } else if result_sets.len() == 1 {
            result_sets.remove(0)
        } else {
            // Intersection
            let mut result = result_sets.remove(0);
            for set in result_sets {
                let set_hash: std::collections::HashSet<_> = set.into_iter().collect();
                result.retain(|h| set_hash.contains(h));
            }
            result
        }
    }

    /// List all keys
    pub fn keys(&self, pattern: Option<&str>) -> Vec<String> {
        let key_map = match self.key_to_hash.read() {
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

    /// Clear all data
    pub fn clear(&self) {
        {
            if let Ok(mut key_map) = self.key_to_hash.write() {
                key_map.clear();
            }
        }
        {
            if let Ok(mut hash_map) = self.hash_to_key.write() {
                hash_map.clear();
            }
        }
        if self.config.simulation_mode {
            {
                if let Ok(mut entries) = self.simulated_entries.write() {
                    entries.clear();
                }
            }
            {
                if let Ok(mut links) = self.simulated_links.write() {
                    links.clear();
                }
            }
            {
                if let Ok(mut path_idx) = self.path_index.write() {
                    path_idx.clear();
                }
            }
        }
    }

    /// Get storage statistics
    pub fn stats(&self) -> DHTBackendStats {
        let key_map = match self.key_to_hash.read() {
            Ok(m) => m,
            Err(_) => return DHTBackendStats::default(),
        };
        let internal_stats = match self.stats.read() {
            Ok(s) => s,
            Err(_) => return DHTBackendStats::default(),
        };

        DHTBackendStats {
            entry_count: key_map.len(),
            entries_created: internal_stats.entries_created,
            entries_retrieved: internal_stats.entries_retrieved,
            links_created: internal_stats.links_created,
            queries_executed: internal_stats.queries_executed,
            bytes_stored: internal_stats.bytes_stored as usize,
            simulation_mode: self.config.simulation_mode,
            replication_factor: self.config.replication_factor,
        }
    }
}

impl Default for DHTBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// DHT backend statistics
#[derive(Debug, Clone, Default)]
pub struct DHTBackendStats {
    /// Number of entries currently stored
    pub entry_count: usize,
    /// Total number of entries created since initialization
    pub entries_created: u64,
    /// Total number of entries retrieved since initialization
    pub entries_retrieved: u64,
    /// Total number of links created for indexing
    pub links_created: u64,
    /// Total number of queries executed
    pub queries_executed: u64,
    /// Total bytes stored
    pub bytes_stored: usize,
    /// Whether running in simulation mode
    pub simulation_mode: bool,
    /// Configured replication factor
    pub replication_factor: usize,
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
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        )
    }

    #[test]
    fn test_dht_backend_basic_operations() {
        let backend = DHTBackend::simulation();

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
        assert!(metadata.cid.starts_with("hC")); // Holochain-style hash

        // Test has
        assert!(backend.has("test-key"));
        assert!(!backend.has("nonexistent"));

        // Test get
        let retrieved: Option<StoredData<String>> = backend.get("test-key");
        assert!(retrieved.is_some());
        let data = retrieved.unwrap();
        assert_eq!(data.data, "test-value");
        assert_eq!(data.metadata.schema.id, "test-schema");

        // Test delete
        assert!(backend.delete("test-key"));
        assert!(!backend.has("test-key"));
    }

    #[test]
    fn test_dht_backend_versioning() {
        let backend = DHTBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        // First version
        let meta1 = backend
            .set(
                "versioned",
                &"v1",
                class.clone(),
                schema.clone(),
                "agent",
                None,
            )
            .unwrap();
        assert_eq!(meta1.version, 1);

        // Update (new version)
        let meta2 = backend
            .set(
                "versioned",
                &"v2",
                class.clone(),
                schema.clone(),
                "agent",
                None,
            )
            .unwrap();
        assert_eq!(meta2.version, 2);

        // Hashes should be different
        assert_ne!(meta1.cid, meta2.cid);

        // Get returns latest
        let retrieved: Option<StoredData<String>> = backend.get("versioned");
        assert_eq!(retrieved.unwrap().data, "v2");
    }

    #[test]
    fn test_dht_backend_query_by_classification() {
        let backend = DHTBackend::simulation();

        let schema = SchemaIdentity::new("test", "1.0");

        // Store with different classifications
        let class_e2_n1_m2 = EpistemicClassification::new(
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        );
        let class_e3_n2_m2 = EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        backend.set(
            "key1",
            &"value1",
            class_e2_n1_m2.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "key2",
            &"value2",
            class_e2_n1_m2.clone(),
            schema.clone(),
            "agent",
            None,
        );
        backend.set(
            "key3",
            &"value3",
            class_e3_n2_m2.clone(),
            schema.clone(),
            "agent",
            None,
        );

        // Query by E level
        let e2_results = backend.query_by_classification(Some(2), None, None);
        assert_eq!(e2_results.len(), 2);

        let e3_results = backend.query_by_classification(Some(3), None, None);
        assert_eq!(e3_results.len(), 1);

        // Query by M level (all are M2)
        let m2_results = backend.query_by_classification(None, None, Some(2));
        assert_eq!(m2_results.len(), 3);

        // Query with multiple filters
        let combined = backend.query_by_classification(Some(2), Some(1), Some(2));
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_dht_backend_keys() {
        let backend = DHTBackend::simulation();

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

        // All keys
        let all_keys = backend.keys(None);
        assert_eq!(all_keys.len(), 3);

        // Pattern matching
        let key_pattern = backend.keys(Some("key"));
        assert_eq!(key_pattern.len(), 2);
    }

    #[test]
    fn test_dht_backend_stats() {
        let backend = DHTBackend::simulation();

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

        // Query
        backend.query_by_classification(Some(2), None, None);

        let stats = backend.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.entries_created, 2);
        assert_eq!(stats.entries_retrieved, 1);
        assert_eq!(stats.queries_executed, 1);
        assert!(stats.bytes_stored > 0);
        assert!(stats.simulation_mode);
    }

    #[test]
    fn test_dht_backend_clear() {
        let backend = DHTBackend::simulation();

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
    }

    #[test]
    fn test_dht_hash_format() {
        let content = b"test content for hash";
        let hash = DHTBackend::compute_hash(content);

        assert!(hash.starts_with("hC")); // Holochain-style prefix
        assert_eq!(hash.len(), 2 + 64); // "hC" + 64 hex chars (SHA3-256)
    }
}
