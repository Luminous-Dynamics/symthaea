// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Working Memory — 7-Item Context with Activation Decay
//!
//! Maintains the current focus of attention during NixOS management.
//! Items have activation levels that decay over time; when capacity is
//! exceeded, the lowest-activation item is evicted (biological constraint
//! from Miller's 7±2 law).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// Default capacity (Miller's number).
const DEFAULT_CAPACITY: usize = 7;

/// Activation decay rate per step.
const DECAY_RATE: f64 = 0.9;

/// Source of a working memory item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// From user input text.
    UserInput,
    /// From system observation.
    SystemObservation,
    /// From action execution result.
    ActionResult,
    /// From causal reasoning insight.
    CausalInsight,
    /// From goal inference.
    GoalState,
}

/// A single item in working memory.
#[derive(Debug, Clone)]
pub struct MemoryItem {
    /// HDC content vector.
    pub content: ContinuousHV,
    /// Current activation level (0.0–1.0). Decays over time.
    pub activation: f64,
    /// Where this item came from.
    pub source: MemorySource,
    /// Human-readable label for debugging.
    pub label: String,
    /// Step at which this item was added.
    pub added_at: u64,
    /// Number of decay cycles this item has survived since being added.
    /// Used for graduation: items that persist long enough in WM
    /// are candidates for promotion to episodic memory.
    pub steps_survived: u64,
}

/// Working memory with capacity-limited, activation-gated storage.
pub struct WorkingMemory {
    /// Active items, sorted by activation (highest first).
    items: Vec<MemoryItem>,
    /// Maximum number of items.
    capacity: usize,
    /// Current time step (increments with each operation).
    step: u64,
    /// Last item evicted by push(), available until next push().
    /// Callers can use [`take_evicted`](Self::take_evicted) to retrieve it for graduation
    /// to episodic memory.
    last_evicted: Option<MemoryItem>,
}

impl WorkingMemory {
    /// Create with default capacity (7).
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            capacity: DEFAULT_CAPACITY,
            step: 0,
            last_evicted: None,
        }
    }

    /// Create with custom capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::new(),
            capacity: capacity.max(1),
            step: 0,
            last_evicted: None,
        }
    }

    /// Add an item to working memory.
    ///
    /// If capacity is exceeded, the lowest-activation item is evicted.
    /// New items start with activation = 1.0.
    pub fn push(&mut self, content: ContinuousHV, source: MemorySource, label: String) {
        self.step += 1;
        self.last_evicted = None;

        // Decay existing items and track survival
        for item in &mut self.items {
            item.activation *= DECAY_RATE;
            item.steps_survived += 1;
        }

        let item = MemoryItem {
            content,
            activation: 1.0,
            source,
            label,
            added_at: self.step,
            steps_survived: 0,
        };

        self.items.push(item);

        // Evict lowest-activation if over capacity
        if self.items.len() > self.capacity {
            // Find index of minimum activation
            let Some(min_idx) = self
                .items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.activation.total_cmp(&b.activation))
                .map(|(i, _)| i)
            else {
                return;
            };
            let evicted = self.items.remove(min_idx);
            self.last_evicted = Some(evicted);
        }

        // Sort by activation (highest first)
        self.items
            .sort_by(|a, b| b.activation.total_cmp(&a.activation));
    }

    /// Take the last evicted item, if any.
    ///
    /// Returns the item that was evicted by the most recent [`push`](Self::push) call.
    /// This item is a candidate for graduation to episodic memory if its
    /// `steps_survived` exceeds the minimum threshold.
    ///
    /// Returns `None` if no item was evicted or if the evicted item was
    /// already taken.
    pub fn take_evicted(&mut self) -> Option<MemoryItem> {
        self.last_evicted.take()
    }

    /// Boost activation of items similar to the query.
    ///
    /// Items with high similarity get their activation refreshed,
    /// implementing associative retrieval.
    pub fn attend(&mut self, query: &ContinuousHV, boost: f64) {
        for item in &mut self.items {
            let sim = item.content.similarity(query).max(0.0) as f64;
            if sim > 0.3 {
                item.activation = (item.activation + sim * boost).min(1.0);
            }
        }
        self.items
            .sort_by(|a, b| b.activation.total_cmp(&a.activation));
    }

    /// Get the bundled context vector — weighted bundle of all items.
    ///
    /// This is the "gist" of current working memory, used for goal
    /// inference and action selection.
    pub fn context_vector(&self) -> ContinuousHV {
        if self.items.is_empty() {
            return ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION);
        }

        let refs: Vec<&ContinuousHV> = self.items.iter().map(|i| &i.content).collect();
        let weights: Vec<f32> = self.items.iter().map(|i| i.activation as f32).collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Retrieve the most similar item to a query.
    pub fn retrieve(&self, query: &ContinuousHV) -> Option<&MemoryItem> {
        self.items.iter().max_by(|a, b| {
            let sim_a = a.content.similarity(query);
            let sim_b = b.content.similarity(query);
            sim_a.total_cmp(&sim_b)
        })
    }

    /// Get all current items (highest activation first).
    pub fn items(&self) -> &[MemoryItem] {
        &self.items
    }

    /// Current number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether working memory is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Current time step.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Average activation across all items.
    pub fn mean_activation(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.items.iter().map(|i| i.activation).sum();
        sum / self.items.len() as f64
    }

    /// Save working memory labels and metadata to JSON.
    ///
    /// HDC vectors are NOT saved (they're reconstructed from the codebook on load).
    /// Only labels, activation levels, sources, and step counts are persisted.
    pub fn save(&self) -> SavedWorkingMemory {
        SavedWorkingMemory {
            step: self.step,
            items: self
                .items
                .iter()
                .map(|item| SavedItem {
                    label: item.label.clone(),
                    activation: item.activation,
                    source: item.source.clone(),
                    added_at: item.added_at,
                    steps_survived: item.steps_survived,
                })
                .collect(),
        }
    }

    /// Restore working memory from saved state, reconstructing HDC vectors
    /// from the codebook.
    pub fn load(saved: &SavedWorkingMemory, codebook: &mut crate::encoding::NixCodebook) -> Self {
        let mut wm = Self::with_capacity(DEFAULT_CAPACITY);
        wm.step = saved.step;

        for saved_item in &saved.items {
            let content = codebook.get_or_create(&saved_item.label).clone();
            wm.items.push(MemoryItem {
                content,
                activation: saved_item.activation,
                source: saved_item.source.clone(),
                label: saved_item.label.clone(),
                added_at: saved_item.added_at,
                steps_survived: saved_item.steps_survived,
            });
        }

        // Sort by activation (highest first)
        wm.items.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        wm
    }
}

/// Serializable representation of a working memory item (no HDC vectors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedItem {
    pub label: String,
    pub activation: f64,
    pub source: MemorySource,
    pub added_at: u64,
    /// Number of decay cycles survived (for graduation tracking)
    #[serde(default)]
    pub steps_survived: u64,
}

/// Serializable snapshot of working memory state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedWorkingMemory {
    pub step: u64,
    pub items: Vec<SavedItem>,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(1024, seed) // Smaller dim for test speed
    }

    #[test]
    fn test_capacity_eviction() {
        let mut wm = WorkingMemory::with_capacity(3);
        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        wm.push(make_hv(2), MemorySource::UserInput, "b".into());
        wm.push(make_hv(3), MemorySource::UserInput, "c".into());
        assert_eq!(wm.len(), 3);

        // Adding 4th should evict the lowest-activation (which is "a" due to decay)
        wm.push(make_hv(4), MemorySource::UserInput, "d".into());
        assert_eq!(wm.len(), 3);

        // "a" should have been evicted (lowest activation after 3 rounds of decay)
        let labels: Vec<&str> = wm.items().iter().map(|i| i.label.as_str()).collect();
        assert!(!labels.contains(&"a"), "Oldest item should be evicted");
    }

    #[test]
    fn test_activation_decay() {
        let mut wm = WorkingMemory::with_capacity(7);
        wm.push(make_hv(1), MemorySource::UserInput, "first".into());

        let initial = wm.items()[0].activation;
        assert!((initial - 1.0).abs() < 1e-6);

        // Adding another item causes decay on the first
        wm.push(make_hv(2), MemorySource::UserInput, "second".into());

        let first_activation = wm
            .items()
            .iter()
            .find(|i| i.label == "first")
            .unwrap()
            .activation;
        assert!(first_activation < 1.0, "First item should have decayed");
        assert!(
            first_activation > 0.8,
            "Should not have decayed too much: {}",
            first_activation
        );
    }

    #[test]
    fn test_attend_boosts_similar() {
        let mut wm = WorkingMemory::with_capacity(7);
        let hv1 = make_hv(1);
        let hv2 = make_hv(2);
        wm.push(hv1.clone(), MemorySource::UserInput, "target".into());
        wm.push(hv2, MemorySource::UserInput, "other".into());

        // Attend to something similar to hv1
        let activation_before = wm
            .items()
            .iter()
            .find(|i| i.label == "target")
            .unwrap()
            .activation;

        wm.attend(&hv1, 0.5);

        let activation_after = wm
            .items()
            .iter()
            .find(|i| i.label == "target")
            .unwrap()
            .activation;

        assert!(activation_after >= activation_before);
    }

    #[test]
    fn test_context_vector() {
        let mut wm = WorkingMemory::with_capacity(7);
        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        wm.push(make_hv(2), MemorySource::SystemObservation, "b".into());

        let ctx = wm.context_vector();
        assert!(ctx.norm() > 0.0);
    }

    #[test]
    fn test_empty_context() {
        let wm = WorkingMemory::new();
        let ctx = wm.context_vector();
        assert!(ctx.norm() < 1e-6, "Empty WM should produce zero context");
    }

    #[test]
    fn test_save_and_load() {
        use crate::encoding::NixCodebook;

        let mut wm = WorkingMemory::with_capacity(7);
        wm.push(make_hv(1), MemorySource::UserInput, "service-failed".into());
        wm.push(
            make_hv(2),
            MemorySource::SystemObservation,
            "store-growth".into(),
        );

        let saved = wm.save();
        assert_eq!(saved.items.len(), 2);

        // Serialize and deserialize
        let json = serde_json::to_string(&saved).unwrap();
        let restored_saved: SavedWorkingMemory = serde_json::from_str(&json).unwrap();

        let mut cb = NixCodebook::new();
        let loaded = WorkingMemory::load(&restored_saved, &mut cb);
        assert_eq!(loaded.len(), 2);

        // Check labels preserved
        let labels: Vec<&str> = loaded.items().iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"service-failed"));
        assert!(labels.contains(&"store-growth"));

        // Check activation preserved
        let svc_activation = loaded
            .items()
            .iter()
            .find(|i| i.label == "service-failed")
            .unwrap()
            .activation;
        assert!(
            svc_activation < 1.0,
            "Should preserve decayed activation: {}",
            svc_activation
        );
    }

    #[test]
    fn test_retrieve_finds_most_similar() {
        let mut wm = WorkingMemory::with_capacity(7);
        let target = make_hv(1);
        wm.push(target.clone(), MemorySource::UserInput, "target".into());
        wm.push(make_hv(100), MemorySource::UserInput, "other1".into());
        wm.push(make_hv(200), MemorySource::UserInput, "other2".into());

        let result = wm.retrieve(&target);
        assert!(result.is_some());
        assert_eq!(result.unwrap().label, "target");
    }

    #[test]
    fn test_retrieve_empty_returns_none() {
        let wm = WorkingMemory::new();
        let query = make_hv(1);
        assert!(wm.retrieve(&query).is_none());
    }

    #[test]
    fn test_mean_activation() {
        let mut wm = WorkingMemory::with_capacity(7);
        assert!(
            (wm.mean_activation() - 0.0).abs() < 1e-6,
            "Empty WM mean = 0"
        );

        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        assert!(
            (wm.mean_activation() - 1.0).abs() < 1e-6,
            "Single item at 1.0"
        );

        // Second push decays first to DECAY_RATE, new item is 1.0
        wm.push(make_hv(2), MemorySource::UserInput, "b".into());
        let expected = (DECAY_RATE + 1.0) / 2.0;
        assert!(
            (wm.mean_activation() - expected).abs() < 1e-6,
            "Mean should be {:.3}, got {:.3}",
            expected,
            wm.mean_activation()
        );
    }

    #[test]
    fn test_take_evicted_lifecycle() {
        let mut wm = WorkingMemory::with_capacity(2);
        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        wm.push(make_hv(2), MemorySource::UserInput, "b".into());

        // No eviction yet
        assert!(wm.take_evicted().is_none());

        // 3rd push evicts lowest-activation item
        wm.push(make_hv(3), MemorySource::UserInput, "c".into());
        let evicted = wm.take_evicted();
        assert!(evicted.is_some(), "Should have evicted an item");
        assert_eq!(evicted.unwrap().label, "a"); // oldest, most decayed

        // Second take should be None
        assert!(wm.take_evicted().is_none());
    }

    #[test]
    fn test_evicted_steps_survived() {
        let mut wm = WorkingMemory::with_capacity(2);
        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        wm.push(make_hv(2), MemorySource::UserInput, "b".into());
        wm.push(make_hv(3), MemorySource::UserInput, "c".into());

        let evicted = wm.take_evicted().unwrap();
        // "a" survived 2 decay cycles (push of b, push of c)
        assert_eq!(evicted.steps_survived, 2);
    }

    #[test]
    fn test_step_increments() {
        let mut wm = WorkingMemory::new();
        assert_eq!(wm.step(), 0);
        wm.push(make_hv(1), MemorySource::UserInput, "a".into());
        assert_eq!(wm.step(), 1);
        wm.push(make_hv(2), MemorySource::UserInput, "b".into());
        assert_eq!(wm.step(), 2);
    }
}
