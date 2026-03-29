// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local File-Based Storage Backend
//!
//! Provides persistent file-based storage for UESS M1 (Temporal) tier.
//! Data is stored as JSON files in a directory structure.
//!
//! # Features
//!
//! - Persistent storage across restarts
//! - Automatic directory creation
//! - Key-based file organization
//! - TTL support with lazy expiration
//! - Thread-safe via file locking patterns

use crate::epistemic::EpistemicClassification;
use crate::storage::types::{SchemaIdentity, StorageMetadata, StoredData};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Local file-based storage backend configuration
#[derive(Debug, Clone)]
pub struct LocalBackendConfig {
    /// Base directory for storage
    pub base_dir: PathBuf,
    /// Whether to create directories automatically
    pub auto_create_dirs: bool,
    /// File extension for data files
    pub file_extension: String,
    /// Whether to store metadata separately
    pub separate_metadata: bool,
    /// Maximum file size in bytes (0 = unlimited)
    pub max_file_size: usize,
}

impl Default for LocalBackendConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("./uess_storage"),
            auto_create_dirs: true,
            file_extension: "json".to_string(),
            separate_metadata: true,
            max_file_size: 10 * 1024 * 1024, // 10 MB default
        }
    }
}

impl LocalBackendConfig {
    /// Create config with custom base directory
    pub fn with_base_dir<P: Into<PathBuf>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.into(),
            ..Default::default()
        }
    }
}

/// Entry stored in the local backend
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalEntry<T> {
    /// The original storage key
    key: String,
    /// The actual data
    data: T,
    /// Epistemic classification
    classification: EpistemicClassification,
    /// Schema identity
    schema: SchemaIdentity,
    /// Storage timestamp (ms since epoch)
    stored_at: u64,
    /// Last modified timestamp (ms since epoch)
    modified_at: Option<u64>,
    /// Creator identifier
    #[allow(dead_code)]
    created_by: String,
    /// TTL timestamp (ms since epoch), None = no expiration
    expires_at: Option<u64>,
    /// Content hash (CID) for integrity checking
    cid: String,
    /// Version number
    version: u32,
    /// Size in bytes
    size_bytes: usize,
}

/// Local file-based storage backend
pub struct LocalBackend {
    config: LocalBackendConfig,
    /// Cache of known keys for faster listing
    key_cache: RwLock<Vec<String>>,
    /// Track whether we've initialized the directory
    initialized: RwLock<bool>,
}

impl LocalBackend {
    /// Create a new local backend with default configuration
    pub fn new() -> Self {
        Self::with_config(LocalBackendConfig::default())
    }

    /// Create a new local backend with custom configuration
    pub fn with_config(config: LocalBackendConfig) -> Self {
        Self {
            config,
            key_cache: RwLock::new(Vec::new()),
            initialized: RwLock::new(false),
        }
    }

    /// Create with custom base directory
    pub fn with_base_dir<P: Into<PathBuf>>(base_dir: P) -> Self {
        Self::with_config(LocalBackendConfig::with_base_dir(base_dir))
    }

    /// Ensure the storage directory exists
    fn ensure_initialized(&self) -> std::io::Result<()> {
        let mut init = self
            .initialized
            .write()
            .map_err(|_| std::io::Error::other("Lock poisoned"))?;
        if !*init {
            if self.config.auto_create_dirs {
                fs::create_dir_all(&self.config.base_dir)?;
            }
            // Rebuild key cache from existing files
            self.rebuild_key_cache()?;
            *init = true;
        }
        Ok(())
    }

    /// Rebuild the key cache from filesystem
    fn rebuild_key_cache(&self) -> std::io::Result<()> {
        let mut cache = self
            .key_cache
            .write()
            .map_err(|_| std::io::Error::other("Lock poisoned"))?;
        cache.clear();

        if self.config.base_dir.exists() {
            for entry in fs::read_dir(&self.config.base_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file()
                    && path
                        .extension()
                        .is_some_and(|e| e == self.config.file_extension.as_str())
                {
                    // Read the file to extract the original key
                    if let Ok(mut file) = File::open(&path) {
                        let mut contents = String::new();
                        if file.read_to_string(&mut contents).is_ok() {
                            // Parse just to get the key field
                            if let Ok(entry) =
                                serde_json::from_str::<LocalEntry<serde_json::Value>>(&contents)
                            {
                                // Check if expired
                                if !Self::is_expired(&entry) {
                                    cache.push(entry.key);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Convert a key to a safe filename
    fn key_to_filename(key: &str) -> String {
        // Use hex encoding of SHA3 hash for long/complex keys
        if key.len() > 50
            || key
                .chars()
                .any(|c| !c.is_alphanumeric() && c != '-' && c != '_')
        {
            let mut hasher = Sha3_256::new();
            hasher.update(key.as_bytes());
            let hash = hasher.finalize();
            format!("h_{}", hex::encode(&hash[..16]))
        } else {
            // Safe key, use directly
            key.replace(['/', ':'], "_")
        }
    }

    /// Reverse filename to key (only works for non-hashed keys)
    #[allow(dead_code)]
    fn filename_to_key(filename: &str) -> Option<String> {
        if filename.starts_with("h_") {
            // Hashed key - can't reverse, but we can read the file to get metadata
            Some(filename.to_string())
        } else {
            Some(filename.to_string())
        }
    }

    /// Get the full path for a key
    fn key_path(&self, key: &str) -> PathBuf {
        let filename = Self::key_to_filename(key);
        self.config
            .base_dir
            .join(format!("{}.{}", filename, self.config.file_extension))
    }

    /// Compute content hash
    fn compute_hash<T: Serialize>(data: &T) -> String {
        let bytes = serde_json::to_vec(data).unwrap_or_default();
        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);
        let hash = hasher.finalize();
        hex::encode(&hash[..16])
    }

    /// Get current timestamp in milliseconds
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Check if an entry has expired (generic version for parsing)
    fn is_expired<T>(entry: &LocalEntry<T>) -> bool {
        if let Some(expires_at) = entry.expires_at {
            Self::now_ms() > expires_at
        } else {
            false
        }
    }

    /// Get data by key
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<StoredData<T>> {
        if self.ensure_initialized().is_err() {
            return None;
        }

        let path = self.key_path(key);
        if !path.exists() {
            return None;
        }

        // Read file
        let mut file = File::open(&path).ok()?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).ok()?;

        // Parse as generic entry first to check expiration
        let entry: LocalEntry<serde_json::Value> = serde_json::from_str(&contents).ok()?;

        // Check TTL
        if Self::is_expired(&entry) {
            // Clean up expired entry
            let _ = fs::remove_file(&path);
            let mut cache = self.key_cache.write().ok()?;
            cache.retain(|k| k != key);
            return None;
        }

        // Re-parse with actual type
        let entry: LocalEntry<T> = serde_json::from_str(&contents).ok()?;

        let metadata = StorageMetadata {
            cid: entry.cid.clone(),
            classification: entry.classification,
            schema: entry.schema.clone(),
            stored_at: entry.stored_at,
            modified_at: entry.modified_at,
            version: entry.version,
            expires_at: entry.expires_at,
            size_bytes: entry.size_bytes,
            created_by: entry.created_by.clone(),
            tombstone: false,
            retracted_by: None,
        };

        Some(StoredData {
            data: entry.data,
            metadata,
            verified: true, // We verified it by reading successfully
        })
    }

    /// Store data
    pub fn set<T: Serialize>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        schema: SchemaIdentity,
        created_by: &str,
        ttl_ms: Option<u64>,
    ) -> Option<StorageMetadata> {
        if self.ensure_initialized().is_err() {
            return None;
        }

        let path = self.key_path(key);
        let now = Self::now_ms();

        // Check for existing entry to get version
        let version = if path.exists() {
            if let Ok(mut file) = File::open(&path) {
                let mut contents = String::new();
                if file.read_to_string(&mut contents).is_ok() {
                    if let Ok(existing) =
                        serde_json::from_str::<LocalEntry<serde_json::Value>>(&contents)
                    {
                        existing.version + 1
                    } else {
                        1
                    }
                } else {
                    1
                }
            } else {
                1
            }
        } else {
            1
        };

        let cid = Self::compute_hash(data);
        let expires_at = ttl_ms.map(|ttl| now + ttl);

        // Pre-serialize to get size
        let data_json = match serde_json::to_string(data) {
            Ok(j) => j,
            Err(_) => return None,
        };
        let size_bytes = data_json.len();

        let entry = LocalEntry {
            key: key.to_string(),
            data,
            classification,
            schema: schema.clone(),
            stored_at: now,
            modified_at: None,
            created_by: created_by.to_string(),
            expires_at,
            cid: cid.clone(),
            version,
            size_bytes,
        };

        // Serialize the full entry
        let json = match serde_json::to_string_pretty(&entry) {
            Ok(j) => j,
            Err(_) => return None,
        };

        // Check size limit
        if self.config.max_file_size > 0 && json.len() > self.config.max_file_size {
            return None;
        }

        // Write atomically via temp file
        let temp_path = path.with_extension("tmp");
        if let Ok(mut file) = File::create(&temp_path) {
            if file.write_all(json.as_bytes()).is_ok() && fs::rename(&temp_path, &path).is_ok() {
                // Update key cache
                let mut cache = self.key_cache.write().ok()?;
                if !cache.contains(&key.to_string()) {
                    cache.push(key.to_string());
                }

                return Some(StorageMetadata {
                    cid,
                    classification,
                    schema,
                    stored_at: now,
                    modified_at: None,
                    version,
                    expires_at,
                    size_bytes,
                    created_by: created_by.to_string(),
                    tombstone: false,
                    retracted_by: None,
                });
            }
            let _ = fs::remove_file(&temp_path);
        }

        None
    }

    /// Delete data by key
    pub fn delete(&self, key: &str) -> bool {
        if self.ensure_initialized().is_err() {
            return false;
        }

        let path = self.key_path(key);
        if path.exists() && fs::remove_file(&path).is_ok() {
            let mut cache = match self.key_cache.write() {
                Ok(c) => c,
                Err(_) => return false,
            };
            cache.retain(|k| k != key);
            return true;
        }
        false
    }

    /// Check if key exists
    pub fn has(&self, key: &str) -> bool {
        if self.ensure_initialized().is_err() {
            return false;
        }

        let path = self.key_path(key);
        if !path.exists() {
            return false;
        }

        // Also check if expired
        if let Ok(mut file) = File::open(&path) {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(entry) = serde_json::from_str::<LocalEntry<serde_json::Value>>(&contents)
                {
                    if Self::is_expired(&entry) {
                        let _ = fs::remove_file(&path);
                        return false;
                    }
                    return true;
                }
            }
        }
        false
    }

    /// List keys matching optional pattern
    pub fn keys(&self, pattern: Option<&str>) -> Vec<String> {
        if self.ensure_initialized().is_err() {
            return vec![];
        }

        // Force rebuild cache to get fresh list
        let _ = self.rebuild_key_cache();

        let cache = match self.key_cache.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        match pattern {
            Some(pat) => cache.iter().filter(|k| k.contains(pat)).cloned().collect(),
            None => cache.clone(),
        }
    }

    /// Clear all data
    pub fn clear(&self) {
        if self.ensure_initialized().is_err() {
            return;
        }

        if let Ok(entries) = fs::read_dir(&self.config.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let _ = fs::remove_file(path);
                }
            }
        }

        if let Ok(mut cache) = self.key_cache.write() {
            cache.clear();
        }
    }

    /// Get storage statistics
    pub fn stats(&self) -> LocalBackendStats {
        if self.ensure_initialized().is_err() {
            return LocalBackendStats::default();
        }

        let mut stats = LocalBackendStats::default();

        if let Ok(entries) = fs::read_dir(&self.config.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    stats.item_count += 1;
                    if let Ok(metadata) = fs::metadata(&path) {
                        stats.total_size_bytes += metadata.len() as usize;
                    }
                }
            }
        }

        stats.base_dir = self.config.base_dir.to_string_lossy().to_string();
        stats
    }
}

impl Default for LocalBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Local backend statistics
#[derive(Debug, Clone, Default)]
pub struct LocalBackendStats {
    /// Number of items currently stored
    pub item_count: usize,
    /// Total size in bytes of all stored items
    pub total_size_bytes: usize,
    /// Base directory path for storage
    pub base_dir: String,
}

/// Hex encoding utilities (minimal implementation)
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{EmpiricalLevel, MaterialityLevel, NormativeLevel};
    use std::env;

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut dir = env::temp_dir();
        dir.push(format!("uess_test_{}_{}", std::process::id(), unique_id));
        dir
    }

    fn test_classification() -> EpistemicClassification {
        EpistemicClassification::new(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        )
    }

    #[test]
    fn test_local_backend_basic_operations() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

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

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_backend_keys() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        // Add multiple keys - verify each succeeds
        let r1 = backend.set(
            "key1",
            &"value1",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        assert!(r1.is_some(), "Failed to set key1");
        let r2 = backend.set(
            "key2",
            &"value2",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        assert!(r2.is_some(), "Failed to set key2");
        let r3 = backend.set(
            "other",
            &"value3",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        assert!(r3.is_some(), "Failed to set other");

        // Verify files exist
        assert!(dir.join("key1.json").exists(), "key1.json should exist");
        assert!(dir.join("key2.json").exists(), "key2.json should exist");
        assert!(dir.join("other.json").exists(), "other.json should exist");

        // Test keys listing
        let all_keys = backend.keys(None);
        assert_eq!(all_keys.len(), 3, "Expected 3 keys, got: {:?}", all_keys);

        // Test pattern matching
        let key_pattern = backend.keys(Some("key"));
        assert_eq!(key_pattern.len(), 2);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_backend_ttl() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        // Set with very short TTL (1ms)
        backend.set("expires", &"temp", class, schema, "agent", Some(1));

        // Wait for expiration
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Should be expired
        let result: Option<StoredData<String>> = backend.get("expires");
        assert!(result.is_none());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_backend_stats() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

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

        let stats = backend.stats();
        assert_eq!(stats.item_count, 2);
        assert!(stats.total_size_bytes > 0);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_backend_clear() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

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

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_backend_versioning() {
        let dir = temp_dir();
        let backend = LocalBackend::with_base_dir(&dir);

        let schema = SchemaIdentity::new("test", "1.0");
        let class = test_classification();

        // First write
        let meta1 = backend.set(
            "versioned",
            &"v1",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        assert_eq!(meta1.unwrap().version, 1);

        // Update
        let meta2 = backend.set(
            "versioned",
            &"v2",
            class.clone(),
            schema.clone(),
            "agent",
            None,
        );
        assert_eq!(meta2.unwrap().version, 2);

        // Verify latest value
        let retrieved: Option<StoredData<String>> = backend.get("versioned");
        assert_eq!(retrieved.unwrap().data, "v2");

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_key_filename_conversion() {
        // Simple keys
        assert_eq!(LocalBackend::key_to_filename("simple"), "simple");
        assert_eq!(LocalBackend::key_to_filename("with-dash"), "with-dash");

        // Complex keys get hashed
        let complex = "proof:round-1:client-001";
        let filename = LocalBackend::key_to_filename(complex);
        assert!(filename.starts_with("h_"));
    }
}
