// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spore persistence — checkpoint/restore for the consciousness kernel.
//!
//! Provides a storage trait and in-memory backend for testing.
//! Platform-specific backends (IndexedDB for WASM, file-based for native)
//! are provided via feature-gated modules.

use std::collections::HashMap;

/// Storage trait for Spore persistence.
///
/// Implementations provide key-value storage for serialized checkpoints.
pub trait SporeStorage {
    /// Load a value by key. Returns None if not found.
    fn load(&self, key: &str) -> Option<Vec<u8>>;
    /// Save a value by key. Returns true on success.
    fn save(&mut self, key: &str, data: &[u8]) -> bool;
    /// Delete a key.
    fn delete(&mut self, key: &str);
    /// List all keys.
    fn keys(&self) -> Vec<String>;
}

/// In-memory storage backend for testing.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    data: HashMap<String, Vec<u8>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SporeStorage for InMemoryStorage {
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.data.get(key).cloned()
    }

    fn save(&mut self, key: &str, data: &[u8]) -> bool {
        self.data.insert(key.to_string(), data.to_vec());
        true
    }

    fn delete(&mut self, key: &str) {
        self.data.remove(key);
    }

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}

/// File-based storage backend for native platforms (Android/iOS/desktop).
///
/// Stores checkpoints as files in a directory. Each key becomes a file.
pub struct FileStorage {
    base_path: String,
}

impl FileStorage {
    pub fn new(path: &str) -> Self {
        // Ensure directory exists
        let _ = std::fs::create_dir_all(path);
        Self {
            base_path: path.to_string(),
        }
    }

    fn key_path(&self, key: &str) -> std::path::PathBuf {
        std::path::Path::new(&self.base_path).join(format!("{key}.bin"))
    }
}

impl SporeStorage for FileStorage {
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        std::fs::read(self.key_path(key)).ok()
    }

    fn save(&mut self, key: &str, data: &[u8]) -> bool {
        std::fs::write(self.key_path(key), data).is_ok()
    }

    fn delete(&mut self, key: &str) {
        let _ = std::fs::remove_file(self.key_path(key));
    }

    fn keys(&self) -> Vec<String> {
        std::fs::read_dir(&self.base_path)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.strip_suffix(".bin").map(|s| s.to_string())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ============================================================================
// TREND HISTORY
// ============================================================================

/// A snapshot of consciousness trends at a point in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrendSnapshot {
    pub cycle: u64,
    pub consciousness_level: f32,
    pub phi: f32,
    pub harmony_alignment: f32,
    pub neuromod_balance: f32,
}

/// Quality-of-life snapshot with richer fields (used by engine.rs).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QolSnapshot {
    pub cycle: u64,
    pub timestamp_secs: u64,
    pub consciousness_level: f32,
    pub harmony_alignment: f32,
    pub metacog_accuracy: f32,
    pub allostatic_load: f32,
    pub dream_wisdom_count: usize,
    pub coherence_score: f32,
    pub safety_level: u8,
}

/// Summary statistics over trend history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrendSummary {
    pub snapshot_count: usize,
    pub consciousness_mean: f32,
    pub consciousness_stability: f32,
    pub phi_mean: f32,
    pub harmony_trend: f32,
}

/// Tracks consciousness trends over time for QOL analysis.
#[derive(Debug, Clone)]
pub struct TrendHistory {
    snapshots: Vec<TrendSnapshot>,
    max_snapshots: usize,
}

impl TrendHistory {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots: 1000,
        }
    }

    pub fn record(&mut self, snapshot: TrendSnapshot) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    /// Record a QolSnapshot by converting to TrendSnapshot. Called every cycle.
    pub fn maybe_record(&mut self, qol: QolSnapshot) {
        // Only record every 100 cycles to avoid bloat
        if !qol.cycle.is_multiple_of(100) && self.snapshots.len() > 10 {
            return;
        }
        self.record(TrendSnapshot {
            cycle: qol.cycle,
            consciousness_level: qol.consciousness_level,
            phi: qol.coherence_score,
            harmony_alignment: qol.harmony_alignment,
            neuromod_balance: 0.0, // Not tracked in QolSnapshot
        });
    }

    pub fn snapshots(&self) -> &[TrendSnapshot] {
        &self.snapshots
    }

    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn trend_summary(&self) -> TrendSummary {
        if self.snapshots.is_empty() {
            return TrendSummary {
                snapshot_count: 0,
                consciousness_mean: 0.0,
                consciousness_stability: 1.0,
                phi_mean: 0.0,
                harmony_trend: 0.0,
            };
        }
        let n = self.snapshots.len() as f32;
        let c_mean: f32 = self
            .snapshots
            .iter()
            .map(|s| s.consciousness_level)
            .sum::<f32>()
            / n;
        let c_var: f32 = self
            .snapshots
            .iter()
            .map(|s| (s.consciousness_level - c_mean).powi(2))
            .sum::<f32>()
            / n;
        let phi_mean: f32 = self.snapshots.iter().map(|s| s.phi).sum::<f32>() / n;
        let harmony_trend = if self.snapshots.len() >= 2 {
            let last = self.snapshots.last().unwrap().harmony_alignment;
            let first = self.snapshots.first().unwrap().harmony_alignment;
            last - first
        } else {
            0.0
        };
        TrendSummary {
            snapshot_count: self.snapshots.len(),
            consciousness_mean: c_mean,
            consciousness_stability: 1.0 - c_var.sqrt().min(1.0),
            phi_mean,
            harmony_trend,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.snapshots).unwrap_or_else(|_| "[]".to_string())
    }
}

impl Default for TrendHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// A serializable snapshot of the Spore engine state.
///
/// Contains the minimal state needed to restore the engine after restart.
#[derive(Debug, Clone)]
pub struct SporeCheckpoint {
    /// Current cycle count.
    pub cycle: u64,
    /// Consciousness level at checkpoint time.
    pub consciousness_level: f32,
    /// Neuromodulator levels [DA, NE, 5-HT, Oxytocin].
    pub neuromodulators: [f32; 4],
    /// Serialized semantic memory entries (key, encoded bytes).
    pub semantic_entries: Vec<(String, Vec<u8>)>,
    /// Serialized episodic memory entries.
    pub episodic_entries: Vec<Vec<u8>>,
    /// Trend snapshots for QOL history.
    pub trend_snapshots: Vec<TrendSnapshot>,
    /// Format version for forward compatibility.
    pub format_version: u32,
}

impl SporeCheckpoint {
    /// Current checkpoint format version.
    pub const FORMAT_VERSION: u32 = 1;

    /// Serialize the checkpoint to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic + version
        buf.extend_from_slice(b"SPCK");
        buf.extend_from_slice(&self.format_version.to_le_bytes());

        // Cycle
        buf.extend_from_slice(&self.cycle.to_le_bytes());

        // Consciousness level
        buf.extend_from_slice(&self.consciousness_level.to_le_bytes());

        // Neuromodulators
        for n in &self.neuromodulators {
            buf.extend_from_slice(&n.to_le_bytes());
        }

        // Semantic entries count + data
        buf.extend_from_slice(&(self.semantic_entries.len() as u32).to_le_bytes());
        for (key, data) in &self.semantic_entries {
            let key_bytes = key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
        }

        // Episodic entries count + data
        buf.extend_from_slice(&(self.episodic_entries.len() as u32).to_le_bytes());
        for entry in &self.episodic_entries {
            buf.extend_from_slice(&(entry.len() as u32).to_le_bytes());
            buf.extend_from_slice(entry);
        }

        buf
    }

    /// Deserialize a checkpoint from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        // Minimum size: magic(4) + version(4) + cycle(8) + consciousness(4) + neuromod(16) = 36
        if data.len() < 36 {
            return None;
        }

        let mut pos = 0;

        // Magic
        if &data[pos..pos + 4] != b"SPCK" {
            return None;
        }
        pos += 4;

        // Version
        let format_version = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;

        if format_version != Self::FORMAT_VERSION {
            return None; // Version mismatch
        }

        // Cycle
        let cycle = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;

        // Consciousness level
        let consciousness_level = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;

        // Neuromodulators
        let mut neuromodulators = [0.0f32; 4];
        for nm in &mut neuromodulators {
            *nm = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
            pos += 4;
        }

        // Semantic entries
        if pos + 4 > data.len() {
            return None;
        }
        let sem_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut semantic_entries = Vec::with_capacity(sem_count);
        for _ in 0..sem_count {
            if pos + 4 > data.len() {
                return None;
            }
            let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pos + key_len > data.len() {
                return None;
            }
            let key = String::from_utf8(data[pos..pos + key_len].to_vec()).ok()?;
            pos += key_len;
            if pos + 4 > data.len() {
                return None;
            }
            let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pos + data_len > data.len() {
                return None;
            }
            let entry_data = data[pos..pos + data_len].to_vec();
            pos += data_len;
            semantic_entries.push((key, entry_data));
        }

        // Episodic entries
        if pos + 4 > data.len() {
            return None;
        }
        let epi_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut episodic_entries = Vec::with_capacity(epi_count);
        for _ in 0..epi_count {
            if pos + 4 > data.len() {
                return None;
            }
            let entry_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pos + entry_len > data.len() {
                return None;
            }
            episodic_entries.push(data[pos..pos + entry_len].to_vec());
            pos += entry_len;
        }

        Some(Self {
            cycle,
            consciousness_level,
            neuromodulators,
            semantic_entries,
            episodic_entries,
            trend_snapshots: vec![], // Trend snapshots not yet serialized in binary format
            format_version,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_storage_crud() {
        let mut storage = InMemoryStorage::new();

        // Save
        assert!(storage.save("key1", b"hello"));
        assert!(storage.save("key2", b"world"));

        // Load
        assert_eq!(storage.load("key1"), Some(b"hello".to_vec()));
        assert_eq!(storage.load("key2"), Some(b"world".to_vec()));
        assert_eq!(storage.load("key3"), None);

        // Keys
        let mut keys = storage.keys();
        keys.sort();
        assert_eq!(keys, vec!["key1", "key2"]);

        // Delete
        storage.delete("key1");
        assert_eq!(storage.load("key1"), None);
        assert_eq!(storage.keys().len(), 1);
    }

    #[test]
    fn checkpoint_roundtrip() {
        let checkpoint = SporeCheckpoint {
            cycle: 42,
            consciousness_level: 0.73,
            neuromodulators: [0.5, 0.6, 0.7, 0.4],
            semantic_entries: vec![
                ("concept1".to_string(), vec![1, 2, 3]),
                ("concept2".to_string(), vec![4, 5, 6, 7]),
            ],
            episodic_entries: vec![vec![10, 20, 30], vec![40, 50]],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        };

        let bytes = checkpoint.to_bytes();
        let restored = SporeCheckpoint::from_bytes(&bytes).unwrap();

        assert_eq!(restored.cycle, 42);
        assert!((restored.consciousness_level - 0.73).abs() < 1e-6);
        assert_eq!(restored.neuromodulators, [0.5, 0.6, 0.7, 0.4]);
        assert_eq!(restored.semantic_entries.len(), 2);
        assert_eq!(restored.semantic_entries[0].0, "concept1");
        assert_eq!(restored.semantic_entries[0].1, vec![1, 2, 3]);
        assert_eq!(restored.episodic_entries.len(), 2);
    }

    #[test]
    fn checkpoint_empty() {
        let checkpoint = SporeCheckpoint {
            cycle: 0,
            consciousness_level: 0.0,
            neuromodulators: [0.0; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        };

        let bytes = checkpoint.to_bytes();
        let restored = SporeCheckpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.cycle, 0);
        assert!(restored.semantic_entries.is_empty());
        assert!(restored.episodic_entries.is_empty());
    }

    #[test]
    fn checkpoint_bad_magic_returns_none() {
        let mut bytes = SporeCheckpoint {
            cycle: 0,
            consciousness_level: 0.0,
            neuromodulators: [0.0; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        }
        .to_bytes();

        bytes[0] = b'X'; // corrupt magic
        assert!(SporeCheckpoint::from_bytes(&bytes).is_none());
    }

    #[test]
    fn checkpoint_bad_version_returns_none() {
        let mut bytes = SporeCheckpoint {
            cycle: 0,
            consciousness_level: 0.0,
            neuromodulators: [0.0; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        }
        .to_bytes();

        // Corrupt version field (bytes 4-7)
        bytes[4] = 99;
        assert!(SporeCheckpoint::from_bytes(&bytes).is_none());
    }

    #[test]
    fn checkpoint_truncated_returns_none() {
        assert!(SporeCheckpoint::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn checkpoint_save_and_restore_via_storage() {
        let mut storage = InMemoryStorage::new();

        let checkpoint = SporeCheckpoint {
            cycle: 100,
            consciousness_level: 0.85,
            neuromodulators: [0.6, 0.5, 0.7, 0.3],
            semantic_entries: vec![("test".to_string(), vec![42])],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        };

        let bytes = checkpoint.to_bytes();
        storage.save("checkpoint", &bytes);

        let loaded = storage.load("checkpoint").unwrap();
        let restored = SporeCheckpoint::from_bytes(&loaded).unwrap();
        assert_eq!(restored.cycle, 100);
        assert!((restored.consciousness_level - 0.85).abs() < 1e-6);
    }
}
