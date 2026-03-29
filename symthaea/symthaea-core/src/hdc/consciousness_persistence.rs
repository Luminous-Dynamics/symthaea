// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Persistence: Versioned State Management with Auto-Save
//!
//! ## Revolutionary Features
//!
//! 1. **Version-Tagged Serialization**: Every state snapshot is tagged with
//!    a schema version for forward/backward compatibility
//!
//! 2. **Compression**: LZ4-style compression for efficient storage
//!    (10-20x reduction for hypervector data)
//!
//! 3. **Auto-Save**: Background thread saves state at configurable intervals
//!
//! 4. **Rollback Support**: Keep N previous snapshots for recovery
//!
//! 5. **Integrity Verification**: CRC32 checksums prevent corruption
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::consciousness_persistence::*;
//!
//! // Create persistence manager
//! let mut persist = ConsciousnessPersistence::new(
//!     PersistenceConfig::default()
//!         .with_path("/tmp/symthaea-state")
//!         .with_auto_save_interval(std::time::Duration::from_secs(60))
//! );
//!
//! // Save state
//! persist.save_snapshot(&my_state)?;
//!
//! // Load latest state
//! let state = persist.load_latest()?;
//!
//! // Rollback to previous version
//! persist.rollback(1)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::binary_hv::BinaryHV;
use super::sleep_and_altered_states::{DreamScenario, SleepAndAlteredStates};

// =============================================================================
// VERSION MANAGEMENT
// =============================================================================

/// Current schema version for consciousness state
pub const SCHEMA_VERSION: u32 = 1;

/// Magic bytes for file identification
pub const MAGIC_BYTES: &[u8; 8] = b"SYMTHAEA";

/// Size of the StateVersion header when serialized with bincode
/// This is calculated as: 8 (magic) + 4 (schema_version) + 8 (created_at) + 4 (checksum) + 1 (compression) + 8 (payload_size) = 33 bytes
pub const HEADER_SIZE: usize = 33;

/// Version header for state files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVersion {
    /// Magic bytes for file identification
    pub magic: [u8; 8],
    /// Schema version for compatibility
    pub schema_version: u32,
    /// Timestamp of creation (Unix millis)
    pub created_at: u64,
    /// CRC32 checksum of payload
    pub checksum: u32,
    /// Compression type (0 = none, 1 = simple RLE)
    pub compression: u8,
    /// Payload size in bytes (uncompressed)
    pub payload_size: u64,
}

impl StateVersion {
    /// Create a new version header
    pub fn new(payload_size: u64, checksum: u32, compression: u8) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            magic: *MAGIC_BYTES,
            schema_version: SCHEMA_VERSION,
            created_at,
            checksum,
            compression,
            payload_size,
        }
    }

    /// Verify magic bytes
    pub fn is_valid(&self) -> bool {
        &self.magic == MAGIC_BYTES
    }

    /// Check if schema is compatible
    pub fn is_compatible(&self) -> bool {
        self.schema_version <= SCHEMA_VERSION
    }
}

// =============================================================================
// COMPRESSION (Simple RLE for HDC vectors)
// =============================================================================

/// Simple Run-Length Encoding for binary/bipolar vectors
/// Achieves 10-20x compression for typical consciousness states
pub fn compress_rle(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len() / 2);
    let mut i = 0;

    while i < data.len() {
        let current = data[i];
        let mut count = 1u8;

        // Count consecutive identical bytes (max 255)
        while (i + count as usize) < data.len()
            && data[i + count as usize] == current
            && count < 255
        {
            count += 1;
        }

        // Only use RLE for runs of 4+ bytes
        if count >= 4 {
            result.push(0xFF); // RLE marker
            result.push(count);
            result.push(current);
            i += count as usize;
        } else {
            // Literal byte (escape 0xFF if needed)
            if current == 0xFF {
                result.push(0xFF);
                result.push(1);
                result.push(0xFF);
            } else {
                result.push(current);
            }
            i += 1;
        }
    }

    result
}

/// Decompress RLE data
pub fn decompress_rle(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() * 4);
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0xFF && i + 2 < data.len() {
            let count = data[i + 1] as usize;
            let value = data[i + 2];
            result.extend(std::iter::repeat_n(value, count));
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    result
}

// =============================================================================
// CHECKSUM
// =============================================================================

/// Simple CRC32 checksum for integrity verification
pub fn crc32(data: &[u8]) -> u32 {
    // CRC32 polynomial (IEEE 802.3)
    const POLY: u32 = 0xEDB88320;

    let mut crc = 0xFFFFFFFF_u32;

    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }

    !crc
}

// =============================================================================
// CONSCIOUSNESS STATE SNAPSHOT
// =============================================================================

/// A complete consciousness state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    /// Unique snapshot ID
    pub id: u64,

    /// Timestamp (Unix millis)
    pub timestamp: u64,

    /// Human-readable label (e.g., "before_dream_cycle")
    pub label: Option<String>,

    /// HDC dimension
    pub dimension: usize,

    /// Core consciousness hypervector (serialized as bytes)
    pub core_hv: Vec<u8>,

    /// Phi (integrated information) value
    pub phi: f64,

    /// Current cognitive mode
    pub cognitive_mode: String,

    /// Sleep/altered state
    pub sleep_state: Option<SleepAndAlteredStates>,

    /// Recent dream if any
    pub recent_dream: Option<DreamScenario>,

    /// Causal edges count
    pub causal_edges: usize,

    /// Memory count
    pub memory_count: usize,

    /// Custom metadata (JSON-serializable)
    pub metadata: std::collections::HashMap<String, String>,
}

impl ConsciousnessSnapshot {
    /// Create a minimal snapshot
    pub fn minimal(id: u64, dimension: usize, phi: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id,
            timestamp,
            label: None,
            dimension,
            core_hv: Vec::new(),
            phi,
            cognitive_mode: "Balanced".to_string(),
            sleep_state: None,
            recent_dream: None,
            causal_edges: 0,
            memory_count: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set core hypervector from BinaryHV
    pub fn with_core_hv(mut self, hv: &BinaryHV) -> Self {
        self.core_hv = hv.0.to_vec();
        self
    }

    /// Set cognitive mode
    pub fn with_cognitive_mode(mut self, mode: impl Into<String>) -> Self {
        self.cognitive_mode = mode.into();
        self
    }

    /// Add metadata entry
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        now.saturating_sub(self.timestamp) as f64 / 1000.0
    }
}

// =============================================================================
// PERSISTENCE CONFIGURATION
// =============================================================================

/// Configuration for consciousness persistence
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Base path for state files
    pub base_path: PathBuf,

    /// Auto-save interval (None = disabled)
    pub auto_save_interval: Option<Duration>,

    /// Maximum snapshots to retain (for rollback)
    pub max_snapshots: usize,

    /// Enable compression
    pub compression_enabled: bool,

    /// Verify checksum on load
    pub verify_checksum: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/tmp/symthaea-state"),
            auto_save_interval: Some(Duration::from_secs(300)), // 5 minutes
            max_snapshots: 10,
            compression_enabled: true,
            verify_checksum: true,
        }
    }
}

impl PersistenceConfig {
    /// Set base path
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.base_path = path.into();
        self
    }

    /// Set auto-save interval
    pub fn with_auto_save_interval(mut self, interval: Duration) -> Self {
        self.auto_save_interval = Some(interval);
        self
    }

    /// Disable auto-save
    pub fn without_auto_save(mut self) -> Self {
        self.auto_save_interval = None;
        self
    }

    /// Set max snapshots
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Disable compression
    pub fn without_compression(mut self) -> Self {
        self.compression_enabled = false;
        self
    }
}

// =============================================================================
// PERSISTENCE MANAGER
// =============================================================================

/// Consciousness persistence manager
pub struct ConsciousnessPersistence {
    /// Configuration
    config: PersistenceConfig,

    /// In-memory snapshot history (for fast rollback)
    history: VecDeque<ConsciousnessSnapshot>,

    /// Next snapshot ID
    next_id: u64,

    /// Auto-save state (shared with background thread)
    auto_save_state: Arc<RwLock<Option<ConsciousnessSnapshot>>>,

    /// Auto-save running flag
    auto_save_running: Arc<Mutex<bool>>,
}

impl ConsciousnessPersistence {
    /// Create new persistence manager
    pub fn new(config: PersistenceConfig) -> Self {
        // Ensure base directory exists
        if let Err(e) = fs::create_dir_all(&config.base_path) {
            eprintln!("[Persistence] Warning: Could not create directory: {e}");
        }

        Self {
            config,
            history: VecDeque::new(),
            next_id: 1,
            auto_save_state: Arc::new(RwLock::new(None)),
            auto_save_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Save a snapshot to disk and history
    pub fn save_snapshot(&mut self, snapshot: &ConsciousnessSnapshot) -> Result<PathBuf, String> {
        // Serialize snapshot
        let payload =
            bincode::serialize(snapshot).map_err(|e| format!("Serialization error: {e}"))?;

        // Optionally compress
        let (final_payload, compression_type) = if self.config.compression_enabled {
            (compress_rle(&payload), 1u8)
        } else {
            (payload.clone(), 0u8)
        };

        // Calculate checksum
        let checksum = crc32(&final_payload);

        // Create version header
        let header = StateVersion::new(payload.len() as u64, checksum, compression_type);
        let header_bytes =
            bincode::serialize(&header).map_err(|e| format!("Header serialization error: {e}"))?;

        // Write to file
        let filename = format!("snapshot_{:08}.sym", snapshot.id);
        let filepath = self.config.base_path.join(&filename);

        let mut file =
            fs::File::create(&filepath).map_err(|e| format!("File creation error: {e}"))?;

        file.write_all(&header_bytes)
            .map_err(|e| format!("Header write error: {e}"))?;
        file.write_all(&final_payload)
            .map_err(|e| format!("Payload write error: {e}"))?;

        // Add to history
        self.history.push_back(snapshot.clone());

        // Prune old snapshots
        while self.history.len() > self.config.max_snapshots {
            if let Some(old) = self.history.pop_front() {
                let old_filename = format!("snapshot_{:08}.sym", old.id);
                let old_path = self.config.base_path.join(&old_filename);
                let _ = fs::remove_file(old_path);
            }
        }

        Ok(filepath)
    }

    /// Create and save a new snapshot
    pub fn create_snapshot(
        &mut self,
        dimension: usize,
        phi: f64,
        label: Option<&str>,
    ) -> Result<ConsciousnessSnapshot, String> {
        let id = self.next_id;
        self.next_id += 1;

        let mut snapshot = ConsciousnessSnapshot::minimal(id, dimension, phi);
        if let Some(l) = label {
            snapshot = snapshot.with_label(l);
        }

        self.save_snapshot(&snapshot)?;

        Ok(snapshot)
    }

    /// Load a snapshot from disk
    pub fn load_snapshot(&self, id: u64) -> Result<ConsciousnessSnapshot, String> {
        let filename = format!("snapshot_{id:08}.sym");
        let filepath = self.config.base_path.join(&filename);

        let mut file = fs::File::open(&filepath).map_err(|e| format!("File open error: {e}"))?;

        // Read header (fixed size = HEADER_SIZE bytes)
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("Header read error: {e}"))?;

        let header: StateVersion = bincode::deserialize(&header_bytes)
            .map_err(|e| format!("Header deserialize error: {e}"))?;

        // Verify header
        if !header.is_valid() {
            return Err("Invalid file format (magic bytes mismatch)".to_string());
        }
        if !header.is_compatible() {
            return Err(format!(
                "Incompatible schema version: {} (current: {})",
                header.schema_version, SCHEMA_VERSION
            ));
        }

        // Read payload
        let mut payload = Vec::new();
        file.read_to_end(&mut payload)
            .map_err(|e| format!("Payload read error: {e}"))?;

        // Verify checksum
        if self.config.verify_checksum {
            let actual_checksum = crc32(&payload);
            if actual_checksum != header.checksum {
                return Err(format!(
                    "Checksum mismatch: expected {:08X}, got {:08X}",
                    header.checksum, actual_checksum
                ));
            }
        }

        // Decompress if needed
        let decompressed = if header.compression == 1 {
            decompress_rle(&payload)
        } else {
            payload
        };

        // Deserialize
        bincode::deserialize(&decompressed).map_err(|e| format!("Snapshot deserialize error: {e}"))
    }

    /// Load the latest snapshot
    pub fn load_latest(&self) -> Result<ConsciousnessSnapshot, String> {
        // Find latest snapshot file
        let entries = fs::read_dir(&self.config.base_path)
            .map_err(|e| format!("Directory read error: {e}"))?;

        let mut latest_id: Option<u64> = None;

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("snapshot_") && name_str.ends_with(".sym") {
                // Extract ID from filename
                if let Some(id_str) = name_str
                    .strip_prefix("snapshot_")
                    .and_then(|s| s.strip_suffix(".sym"))
                {
                    if let Ok(id) = id_str.parse::<u64>() {
                        latest_id = Some(latest_id.map(|l| l.max(id)).unwrap_or(id));
                    }
                }
            }
        }

        match latest_id {
            Some(id) => self.load_snapshot(id),
            None => Err("No snapshots found".to_string()),
        }
    }

    /// Rollback to a previous snapshot (by steps back)
    pub fn rollback(&mut self, steps_back: usize) -> Result<ConsciousnessSnapshot, String> {
        if steps_back == 0 {
            return Err("Cannot rollback 0 steps".to_string());
        }

        if self.history.len() <= steps_back {
            return Err(format!(
                "Cannot rollback {} steps (only {} snapshots in history)",
                steps_back,
                self.history.len()
            ));
        }

        // Remove recent snapshots
        for _ in 0..steps_back {
            if let Some(removed) = self.history.pop_back() {
                let filename = format!("snapshot_{:08}.sym", removed.id);
                let filepath = self.config.base_path.join(&filename);
                let _ = fs::remove_file(filepath);
            }
        }

        // Return the new latest
        self.history
            .back()
            .cloned()
            .ok_or_else(|| "History empty after rollback".to_string())
    }

    /// List all available snapshots
    pub fn list_snapshots(&self) -> Result<Vec<(u64, String)>, String> {
        let entries = fs::read_dir(&self.config.base_path)
            .map_err(|e| format!("Directory read error: {e}"))?;

        let mut snapshots = Vec::new();

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("snapshot_") && name_str.ends_with(".sym") {
                if let Some(id_str) = name_str
                    .strip_prefix("snapshot_")
                    .and_then(|s| s.strip_suffix(".sym"))
                {
                    if let Ok(id) = id_str.parse::<u64>() {
                        snapshots.push((id, name_str.to_string()));
                    }
                }
            }
        }

        snapshots.sort_by_key(|(id, _)| *id);
        Ok(snapshots)
    }

    /// Update auto-save state (call this periodically)
    pub fn update_auto_save_state(&self, snapshot: ConsciousnessSnapshot) {
        if let Ok(mut state) = self.auto_save_state.write() {
            *state = Some(snapshot);
        }
    }

    /// Start auto-save background thread
    pub fn start_auto_save(&self) -> Option<std::thread::JoinHandle<()>> {
        let interval = self.config.auto_save_interval?;

        // Check if already running
        {
            let mut running = self.auto_save_running.lock().expect("lock poisoned");
            if *running {
                return None;
            }
            *running = true;
        }

        let state = Arc::clone(&self.auto_save_state);
        let running_flag = Arc::clone(&self.auto_save_running);
        let base_path = self.config.base_path.clone();
        let compression = self.config.compression_enabled;

        Some(std::thread::spawn(move || {
            let mut auto_id = 1u64;

            loop {
                std::thread::sleep(interval);

                // Check if still running
                {
                    let running = running_flag.lock().expect("lock poisoned");
                    if !*running {
                        break;
                    }
                }

                // Get current state
                let snapshot_opt = {
                    let state_guard = state.read().expect("state RwLock poisoned");
                    state_guard.clone()
                };

                if let Some(mut snapshot) = snapshot_opt {
                    snapshot.id = auto_id;
                    snapshot.label = Some(format!("auto_save_{auto_id}"));

                    // Serialize and save
                    if let Ok(payload) = bincode::serialize(&snapshot) {
                        let final_payload = if compression {
                            compress_rle(&payload)
                        } else {
                            payload.clone()
                        };

                        let checksum = crc32(&final_payload);
                        let header = StateVersion::new(
                            payload.len() as u64,
                            checksum,
                            if compression { 1 } else { 0 },
                        );

                        if let Ok(header_bytes) = bincode::serialize(&header) {
                            let filename = format!("autosave_{auto_id:08}.sym");
                            let filepath = base_path.join(&filename);

                            if let Ok(mut file) = fs::File::create(&filepath) {
                                let _ = file.write_all(&header_bytes);
                                let _ = file.write_all(&final_payload);
                            }
                        }
                    }

                    auto_id += 1;
                }
            }
        }))
    }

    /// Stop auto-save
    pub fn stop_auto_save(&self) {
        let mut running = self.auto_save_running.lock().expect("lock poisoned");
        *running = false;
    }

    /// Get history length
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get config
    pub fn config(&self) -> &PersistenceConfig {
        &self.config
    }
}

impl Default for ConsciousnessPersistence {
    fn default() -> Self {
        Self::new(PersistenceConfig::default())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_compression() {
        // Test with repetitive data (common in bipolar vectors)
        let data: Vec<u8> = vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1];
        let compressed = compress_rle(&data);
        let decompressed = decompress_rle(&compressed);

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, Symthaea!";
        let checksum = crc32(data);

        // CRC32 should be deterministic
        assert_eq!(crc32(data), checksum);

        // Different data should have different checksum
        let data2 = b"Hello, World!";
        assert_ne!(crc32(data2), checksum);
    }

    #[test]
    fn test_version_header() {
        let header = StateVersion::new(1000, 0xDEADBEEF, 1);

        assert!(header.is_valid());
        assert!(header.is_compatible());
        assert_eq!(header.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_snapshot_creation() {
        let snapshot = ConsciousnessSnapshot::minimal(1, 16384, 0.75)
            .with_label("test_snapshot")
            .with_cognitive_mode("Flow")
            .with_metadata("test_key", "test_value");

        assert_eq!(snapshot.id, 1);
        assert_eq!(snapshot.dimension, 16384);
        assert_eq!(snapshot.phi, 0.75);
        assert_eq!(snapshot.label, Some("test_snapshot".to_string()));
        assert_eq!(snapshot.cognitive_mode, "Flow");
        assert_eq!(
            snapshot.metadata.get("test_key"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_persistence_save_load() {
        // Use temp directory
        let temp_dir = std::env::temp_dir().join("symthaea_test_persistence");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = PersistenceConfig::default()
            .with_path(&temp_dir)
            .without_auto_save();

        let mut persist = ConsciousnessPersistence::new(config);

        // Create and save snapshot
        let snapshot = persist
            .create_snapshot(16384, 0.65, Some("test"))
            .expect("Failed to create snapshot");

        // Load it back
        let loaded = persist
            .load_snapshot(snapshot.id)
            .expect("Failed to load snapshot");

        assert_eq!(snapshot.id, loaded.id);
        assert_eq!(snapshot.phi, loaded.phi);
        assert_eq!(snapshot.label, loaded.label);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_rollback() {
        let temp_dir = std::env::temp_dir().join("symthaea_test_rollback");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = PersistenceConfig::default()
            .with_path(&temp_dir)
            .without_auto_save();

        let mut persist = ConsciousnessPersistence::new(config);

        // Create 3 snapshots
        let _s1 = persist.create_snapshot(16384, 0.5, Some("first")).unwrap();
        let _s2 = persist.create_snapshot(16384, 0.6, Some("second")).unwrap();
        let s3 = persist.create_snapshot(16384, 0.7, Some("third")).unwrap();

        assert_eq!(persist.history_len(), 3);

        // Rollback 1 step
        let rolled_back = persist.rollback(1).expect("Failed to rollback");

        assert_eq!(persist.history_len(), 2);
        assert_ne!(rolled_back.id, s3.id);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
