// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spore persistence — checkpoint/restore for the consciousness kernel.
//!
//! Provides a storage trait and in-memory backend for testing.
//! Platform-specific backends (IndexedDB for WASM, file-based for native)
//! are provided via feature-gated modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;

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

// ===========================================================================
// QOL Trend Tracking
// ===========================================================================

/// Maximum number of trend snapshots to retain.
pub const TREND_HISTORY_CAP: usize = 200;

/// Sampling interval: record one snapshot every N cycles.
pub const TREND_SAMPLE_INTERVAL: u64 = 600; // ~30 sec at 20Hz

/// A single QOL measurement snapshot for trend tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QolSnapshot {
    /// Cycle count when this snapshot was taken.
    pub cycle: u64,
    /// Unix timestamp (seconds) when captured.
    pub timestamp_secs: u64,
    /// Overall consciousness level [0, 1].
    pub consciousness_level: f32,
    /// Eight Harmonies aggregate alignment [0, 1].
    pub harmony_alignment: f32,
    /// Meta-cognitive self-assessment accuracy [0, 1].
    pub metacog_accuracy: f32,
    /// Allostatic stress load [0, 1].
    pub allostatic_load: f32,
    /// Number of dream wisdom entries accumulated.
    pub dream_wisdom_count: u32,
    /// Temporal coherence score [0, 1].
    pub coherence_score: f32,
    /// Safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_level: u8,
}

/// Ring-buffer history of QOL trend snapshots.
///
/// Maintains a capped deque of recent QOL measurements for personal
/// trend visualization and drift detection.
#[derive(Debug, Clone)]
pub struct TrendHistory {
    snapshots: VecDeque<QolSnapshot>,
    /// Last cycle at which a snapshot was taken.
    last_sample_cycle: u64,
}

impl TrendHistory {
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::with_capacity(TREND_HISTORY_CAP),
            last_sample_cycle: 0,
        }
    }

    /// Record a snapshot if enough cycles have elapsed since the last sample.
    ///
    /// Returns true if a snapshot was recorded.
    pub fn maybe_record(&mut self, snapshot: QolSnapshot) -> bool {
        if !self.snapshots.is_empty()
            && snapshot.cycle < self.last_sample_cycle + TREND_SAMPLE_INTERVAL as u64
        {
            return false;
        }
        self.last_sample_cycle = snapshot.cycle;
        if self.snapshots.len() >= TREND_HISTORY_CAP {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
        true
    }

    /// Force-record a snapshot regardless of interval (for manual triggers).
    pub fn record(&mut self, snapshot: QolSnapshot) {
        self.last_sample_cycle = snapshot.cycle;
        if self.snapshots.len() >= TREND_HISTORY_CAP {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Get all snapshots as a slice.
    pub fn snapshots(&self) -> &VecDeque<QolSnapshot> {
        &self.snapshots
    }

    /// Number of snapshots stored.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get all snapshots as JSON.
    pub fn to_json(&self) -> String {
        let v: Vec<&QolSnapshot> = self.snapshots.iter().collect();
        serde_json::to_string(&v).unwrap_or_else(|_| "[]".to_string())
    }

    /// Compute a simple trend summary.
    pub fn trend_summary(&self) -> TrendSummary {
        if self.snapshots.len() < 2 {
            return TrendSummary::default();
        }
        let first = self.snapshots.front().unwrap();
        let last = self.snapshots.back().unwrap();
        let n = self.snapshots.len() as f32;

        let consciousness_delta = last.consciousness_level - first.consciousness_level;
        let harmony_delta = last.harmony_alignment - first.harmony_alignment;

        let consciousness_mean: f32 = self
            .snapshots
            .iter()
            .map(|s| s.consciousness_level)
            .sum::<f32>()
            / n;
        let harmony_mean: f32 = self
            .snapshots
            .iter()
            .map(|s| s.harmony_alignment)
            .sum::<f32>()
            / n;

        // Variance for stability assessment
        let consciousness_var: f32 = self
            .snapshots
            .iter()
            .map(|s| (s.consciousness_level - consciousness_mean).powi(2))
            .sum::<f32>()
            / n;

        TrendSummary {
            consciousness_delta,
            harmony_delta,
            consciousness_mean,
            harmony_mean,
            consciousness_stability: 1.0 - consciousness_var.sqrt().min(1.0),
            sample_count: self.snapshots.len() as u32,
            span_seconds: last.timestamp_secs.saturating_sub(first.timestamp_secs),
        }
    }

    /// Serialize trend snapshots to bytes for checkpoint embedding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let json =
            serde_json::to_vec(&self.snapshots.iter().collect::<Vec<_>>()).unwrap_or_default();
        let mut buf = Vec::with_capacity(12 + json.len());
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.last_sample_cycle.to_le_bytes());
        buf.extend_from_slice(&json);
        buf
    }

    /// Deserialize trend snapshots from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let json_len = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let last_sample_cycle = u64::from_le_bytes(data[4..12].try_into().ok()?);
        if data.len() < 12 + json_len {
            return None;
        }
        let snapshots: Vec<QolSnapshot> = serde_json::from_slice(&data[12..12 + json_len]).ok()?;
        let mut deque = VecDeque::with_capacity(TREND_HISTORY_CAP);
        for s in snapshots.into_iter().rev().take(TREND_HISTORY_CAP) {
            deque.push_front(s);
        }
        Some(Self {
            snapshots: deque,
            last_sample_cycle,
        })
    }
}

impl Default for TrendHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for QOL trends.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrendSummary {
    /// Change in consciousness from first to last snapshot.
    pub consciousness_delta: f32,
    /// Change in harmony alignment from first to last snapshot.
    pub harmony_delta: f32,
    /// Mean consciousness level.
    pub consciousness_mean: f32,
    /// Mean harmony alignment.
    pub harmony_mean: f32,
    /// Stability score [0, 1] — higher means less variance.
    pub consciousness_stability: f32,
    /// Number of samples in the trend.
    pub sample_count: u32,
    /// Time span in seconds from first to last sample.
    pub span_seconds: u64,
}

// ===========================================================================
// Checkpoint
// ===========================================================================

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
    /// QOL trend history (v2+).
    pub trend_snapshots: Vec<QolSnapshot>,
    /// Format version for forward compatibility.
    pub format_version: u32,
}

impl SporeCheckpoint {
    /// Current checkpoint format version.
    pub const FORMAT_VERSION: u32 = 2;

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

        // v2: QOL trend snapshots (JSON-encoded for flexibility)
        if self.format_version >= 2 {
            let trend_json = serde_json::to_vec(&self.trend_snapshots).unwrap_or_default();
            buf.extend_from_slice(&(trend_json.len() as u32).to_le_bytes());
            buf.extend_from_slice(&trend_json);
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

        if format_version > Self::FORMAT_VERSION {
            return None; // Future version we can't read
        }
        // Versions <= FORMAT_VERSION are forward-compatible (we read what we can,
        // missing fields get defaults, extra trailing data is ignored).

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

        // v2: QOL trend snapshots
        let trend_snapshots = if format_version >= 2 && pos + 4 <= data.len() {
            let trend_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pos + trend_len <= data.len() {
                serde_json::from_slice(&data[pos..pos + trend_len]).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let _ = pos; // suppress unused warning

        Some(Self {
            cycle,
            consciousness_level,
            neuromodulators,
            semantic_entries,
            episodic_entries,
            trend_snapshots,
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
            trend_snapshots: vec![QolSnapshot {
                cycle: 40,
                timestamp_secs: 1000,
                consciousness_level: 0.7,
                harmony_alignment: 0.6,
                metacog_accuracy: 0.5,
                allostatic_load: 0.2,
                dream_wisdom_count: 3,
                coherence_score: 0.8,
                safety_level: 0,
            }],
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
        assert_eq!(restored.trend_snapshots.len(), 1);
        assert!((restored.trend_snapshots[0].consciousness_level - 0.7).abs() < 1e-6);
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
    fn checkpoint_future_version_returns_none() {
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

        // Set version to a future version (> FORMAT_VERSION)
        let future_version = (SporeCheckpoint::FORMAT_VERSION + 10).to_le_bytes();
        bytes[4..8].copy_from_slice(&future_version);
        assert!(
            SporeCheckpoint::from_bytes(&bytes).is_none(),
            "future version should be rejected"
        );
    }

    #[test]
    fn checkpoint_older_version_accepted() {
        // Version 0 should be readable (forward-compatible: same layout, older version tag)
        let mut bytes = SporeCheckpoint {
            cycle: 42,
            consciousness_level: 0.5,
            neuromodulators: [0.1; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: SporeCheckpoint::FORMAT_VERSION,
        }
        .to_bytes();

        // Downgrade version to 0
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        let restored = SporeCheckpoint::from_bytes(&bytes);
        assert!(
            restored.is_some(),
            "older version should be forward-compatible"
        );
        assert_eq!(restored.unwrap().cycle, 42);
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

    // =====================================================================
    // QOL Trend History tests
    // =====================================================================

    fn make_snapshot(cycle: u64, consciousness: f32) -> QolSnapshot {
        QolSnapshot {
            cycle,
            timestamp_secs: 1000 + cycle,
            consciousness_level: consciousness,
            harmony_alignment: 0.5,
            metacog_accuracy: 0.5,
            allostatic_load: 0.1,
            dream_wisdom_count: 0,
            coherence_score: 0.8,
            safety_level: 0,
        }
    }

    #[test]
    fn trend_history_respects_interval() {
        let mut history = TrendHistory::new();
        assert!(history.maybe_record(make_snapshot(0, 0.5)));
        // Too soon — should be rejected
        assert!(!history.maybe_record(make_snapshot(100, 0.6)));
        // At interval boundary — should be accepted
        assert!(history.maybe_record(make_snapshot(600, 0.7)));
        assert_eq!(history.count(), 2);
    }

    #[test]
    fn trend_history_cap_enforced() {
        let mut history = TrendHistory::new();
        for i in 0..TREND_HISTORY_CAP + 50 {
            let cycle = (i as u64) * TREND_SAMPLE_INTERVAL;
            history.maybe_record(make_snapshot(cycle, 0.5));
        }
        assert_eq!(history.count(), TREND_HISTORY_CAP);
    }

    #[test]
    fn trend_history_force_record() {
        let mut history = TrendHistory::new();
        history.record(make_snapshot(0, 0.5));
        history.record(make_snapshot(1, 0.6)); // Should succeed despite interval
        assert_eq!(history.count(), 2);
    }

    #[test]
    fn trend_summary_delta() {
        let mut history = TrendHistory::new();
        history.record(make_snapshot(0, 0.3));
        history.record(make_snapshot(600, 0.7));
        let summary = history.trend_summary();
        assert!((summary.consciousness_delta - 0.4).abs() < 1e-6);
        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.span_seconds, 600);
    }

    #[test]
    fn trend_summary_stability() {
        let mut history = TrendHistory::new();
        // All same consciousness = perfect stability
        for i in 0..10 {
            history.record(make_snapshot(i * 600, 0.5));
        }
        let summary = history.trend_summary();
        assert!((summary.consciousness_stability - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trend_history_serialization_roundtrip() {
        let mut history = TrendHistory::new();
        history.record(make_snapshot(0, 0.3));
        history.record(make_snapshot(600, 0.7));
        let bytes = history.to_bytes();
        let restored = TrendHistory::from_bytes(&bytes).unwrap();
        assert_eq!(restored.count(), 2);
        assert!((restored.snapshots().back().unwrap().consciousness_level - 0.7).abs() < 1e-6);
    }

    #[test]
    fn trend_json_roundtrip() {
        let mut history = TrendHistory::new();
        history.record(make_snapshot(0, 0.5));
        let json = history.to_json();
        assert!(json.contains("consciousness_level"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn v1_checkpoint_backward_compat_no_trends() {
        // Simulate a v1 checkpoint (no trend data appended)
        let v1_checkpoint = SporeCheckpoint {
            cycle: 42,
            consciousness_level: 0.5,
            neuromodulators: [0.5; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: vec![],
            format_version: 1, // explicitly v1
        };
        // Serialize as v1 (won't include trend section)
        let bytes = v1_checkpoint.to_bytes();

        // Should deserialize successfully with empty trends
        let restored = SporeCheckpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.cycle, 42);
        assert!(restored.trend_snapshots.is_empty());
    }

    #[test]
    fn v2_checkpoint_with_trends() {
        let trends = vec![
            make_snapshot(100, 0.4),
            make_snapshot(700, 0.6),
            make_snapshot(1300, 0.8),
        ];
        let checkpoint = SporeCheckpoint {
            cycle: 1300,
            consciousness_level: 0.8,
            neuromodulators: [0.5; 4],
            semantic_entries: vec![],
            episodic_entries: vec![],
            trend_snapshots: trends,
            format_version: SporeCheckpoint::FORMAT_VERSION,
        };
        let bytes = checkpoint.to_bytes();
        let restored = SporeCheckpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.trend_snapshots.len(), 3);
        assert!((restored.trend_snapshots[2].consciousness_level - 0.8).abs() < 1e-6);
    }
}
