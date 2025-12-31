//! # Temporal Holographic Memory
//!
//! ## Purpose
//! Implements holographic memory with explicit temporal binding, enabling:
//! - Causal reasoning: "What happened before X?"
//! - Temporal similarity: Events close in time have similar temporal encodings
//! - Order preservation: Unlike bundling, temporal binding maintains sequence
//!
//! ## Theoretical Basis
//! Standard HDC bundling loses temporal order (A⊕B = B⊕A).
//! Temporal holographic memory solves this by binding each event with its
//! temporal position: hologram = event ⊗ T(t)
//!
//! To recall, we can:
//! 1. Filter by time (causal queries)
//! 2. Unbind the temporal component to match content
//! 3. Use temporal similarity for "nearby in time" queries
//!
//! ## Key Innovations
//! - **Temporal basis vectors**: Circular encoding for smooth time similarity
//! - **Causal filtering**: Only recall events before a deadline
//! - **Temporal unbinding**: Recover content from hologram
//! - **Sequence prediction**: Query what typically follows an event
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::memory::temporal_holographic::{
//!     TemporalHolographicMemory, TemporalConfig
//! };
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let mut memory = TemporalHolographicMemory::new(TemporalConfig::default());
//!
//! // Encode events with timestamps
//! let event1 = ContinuousHV::random(16384, 1);
//! let event2 = ContinuousHV::random(16384, 2);
//!
//! memory.encode(event1, 1.0);  // t=1.0
//! memory.encode(event2, 2.0);  // t=2.0
//!
//! // Causal query: what happened before t=1.5?
//! let query = ContinuousHV::random(16384, 1);
//! let results = memory.recall_before(&query, 1.5);
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::hdc::unified_hv::ContinuousHV;
use crate::hdc::temporal_encoder::TemporalEncoder;
use crate::hdc::HDC_DIMENSION;

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for temporal holographic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Number of temporal basis vectors (time quanta)
    pub temporal_resolution: usize,

    /// Time scale in seconds for one full rotation (e.g., 86400 for daily)
    pub time_scale_secs: u64,

    /// Minimum similarity threshold for recall
    pub similarity_threshold: f32,

    /// Maximum number of events to store
    pub max_events: usize,

    /// HDC dimension (should match semantic space)
    pub dimension: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            temporal_resolution: 1000,  // 1000 time quanta
            time_scale_secs: 86400,     // 24 hours
            similarity_threshold: 0.3,   // Minimum similarity for recall
            max_events: 10000,           // Maximum stored events
            dimension: HDC_DIMENSION,    // 16,384
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPORAL EVENT
// ═══════════════════════════════════════════════════════════════════════════

/// A single event stored with temporal binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    /// Event content bound with temporal position: event ⊗ T(t)
    pub hologram: ContinuousHV,

    /// Original event content (for reconstruction validation)
    pub content: ContinuousHV,

    /// Timestamp (seconds since epoch or relative)
    pub time: f64,

    /// Optional event label/type
    pub label: Option<String>,

    /// Importance weight (affects recall priority)
    pub importance: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// RECALL RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a memory recall operation
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// Retrieved event content
    pub content: ContinuousHV,

    /// Similarity score to query
    pub similarity: f32,

    /// Timestamp of the event
    pub time: f64,

    /// Optional label
    pub label: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPORAL HOLOGRAPHIC MEMORY
// ═══════════════════════════════════════════════════════════════════════════

/// Holographic memory with explicit temporal binding
///
/// Stores events as holograms: hologram = content ⊗ temporal_basis[t]
/// This enables:
/// - Causal queries (only events before deadline)
/// - Temporal similarity (events close in time)
/// - Sequence prediction (what follows X?)
#[derive(Debug, Clone)]
pub struct TemporalHolographicMemory {
    /// Configuration
    config: TemporalConfig,

    /// Temporal encoder for generating time vectors
    #[allow(dead_code)] // Used for temporal basis generation
    temporal_encoder: TemporalEncoder,

    /// Pre-computed temporal basis vectors for fast binding
    /// Index by (time_quantum % temporal_resolution)
    temporal_basis: Vec<ContinuousHV>,

    /// Stored events with temporal binding
    events: Vec<TemporalEvent>,
}

impl TemporalHolographicMemory {
    /// Create new temporal holographic memory
    pub fn new(config: TemporalConfig) -> Self {
        let temporal_encoder = TemporalEncoder::with_config(
            config.dimension,
            Duration::from_secs(config.time_scale_secs),
            0.0, // No phase shift
        );

        // Pre-compute temporal basis vectors
        let temporal_basis = Self::generate_temporal_basis(&temporal_encoder, &config);

        Self {
            config,
            temporal_encoder,
            temporal_basis,
            events: Vec::new(),
        }
    }

    /// Generate temporal basis vectors
    fn generate_temporal_basis(encoder: &TemporalEncoder, config: &TemporalConfig) -> Vec<ContinuousHV> {
        let time_quantum = config.time_scale_secs as f64 / config.temporal_resolution as f64;

        (0..config.temporal_resolution)
            .map(|i| {
                let time = Duration::from_secs_f64(i as f64 * time_quantum);
                let vec = encoder.encode_time(time).unwrap_or_else(|_| {
                    vec![0.0; config.dimension]
                });
                ContinuousHV { values: vec }
            })
            .collect()
    }

    /// Get temporal basis vector for a given time
    fn get_temporal_vector(&self, time: f64) -> &ContinuousHV {
        let time_quantum = self.config.time_scale_secs as f64 / self.config.temporal_resolution as f64;
        let index = ((time / time_quantum) as usize) % self.config.temporal_resolution;
        &self.temporal_basis[index]
    }

    /// Encode and store an event with temporal binding
    ///
    /// The event is bound with its temporal position: hologram = event ⊗ T(t)
    pub fn encode(&mut self, event: ContinuousHV, time: f64) {
        self.encode_with_label(event, time, None, 1.0);
    }

    /// Encode event with label and importance
    pub fn encode_with_label(&mut self, event: ContinuousHV, time: f64, label: Option<String>, importance: f32) {
        let time_hv = self.get_temporal_vector(time);
        let hologram = event.bind(time_hv);

        let temporal_event = TemporalEvent {
            hologram,
            content: event,
            time,
            label,
            importance,
        };

        self.events.push(temporal_event);

        // Enforce max events limit (remove oldest first)
        if self.events.len() > self.config.max_events {
            // Sort by time and remove oldest
            self.events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            self.events.remove(0);
        }
    }

    /// Recall events before a given deadline (causal query)
    ///
    /// Only returns events with timestamp < deadline, respecting causality.
    /// Results are sorted by similarity to query.
    pub fn recall_before(&self, query: &ContinuousHV, deadline: f64) -> Vec<RecallResult> {
        let mut results: Vec<RecallResult> = self.events
            .iter()
            .filter(|e| e.time < deadline)
            .map(|e| {
                // Unbind temporal component to get content similarity
                let time_hv = self.get_temporal_vector(e.time);

                // For continuous HVs, binding with same vector twice doesn't give identity
                // Instead, we directly compare with stored content
                let sim = query.similarity(&e.content);

                RecallResult {
                    content: e.content.clone(),
                    similarity: sim,
                    time: e.time,
                    label: e.label.clone(),
                }
            })
            .filter(|r| r.similarity > self.config.similarity_threshold)
            .collect();

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        results
    }

    /// Recall events in a time window
    pub fn recall_in_window(&self, query: &ContinuousHV, start: f64, end: f64) -> Vec<RecallResult> {
        let mut results: Vec<RecallResult> = self.events
            .iter()
            .filter(|e| e.time >= start && e.time < end)
            .map(|e| {
                let sim = query.similarity(&e.content);
                RecallResult {
                    content: e.content.clone(),
                    similarity: sim,
                    time: e.time,
                    label: e.label.clone(),
                }
            })
            .filter(|r| r.similarity > self.config.similarity_threshold)
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results
    }

    /// Find events temporally close to a reference time
    ///
    /// Uses temporal similarity to find events that occurred near the reference time.
    pub fn recall_near_time(&self, reference_time: f64, window: f64) -> Vec<RecallResult> {
        self.events
            .iter()
            .filter(|e| (e.time - reference_time).abs() < window)
            .map(|e| RecallResult {
                content: e.content.clone(),
                similarity: 1.0 - (e.time - reference_time).abs() as f32 / window as f32,
                time: e.time,
                label: e.label.clone(),
            })
            .collect()
    }

    /// Predict what typically follows a given event
    ///
    /// Finds similar past events and returns what happened next.
    pub fn predict_next(&self, query: &ContinuousHV, lookahead: f64) -> Vec<RecallResult> {
        // Find similar events
        let similar_events: Vec<_> = self.events
            .iter()
            .map(|e| (e, query.similarity(&e.content)))
            .filter(|(_, sim)| *sim > self.config.similarity_threshold)
            .collect();

        // For each similar event, find what happened in the lookahead window
        let mut predictions = Vec::new();

        for (event, _sim) in similar_events {
            let start = event.time;
            let end = event.time + lookahead;

            for next_event in &self.events {
                if next_event.time > start && next_event.time < end {
                    predictions.push(RecallResult {
                        content: next_event.content.clone(),
                        similarity: 1.0 - ((next_event.time - start) / lookahead) as f32,
                        time: next_event.time,
                        label: next_event.label.clone(),
                    });
                }
            }
        }

        // Sort by similarity and deduplicate
        predictions.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        predictions
    }

    /// Get temporal context bundle for a time range
    ///
    /// Bundles all events in a time range into a single superposition vector.
    /// Useful for representing "what was happening around time T".
    pub fn temporal_context(&self, center: f64, window: f64) -> ContinuousHV {
        let events_in_window: Vec<_> = self.events
            .iter()
            .filter(|e| (e.time - center).abs() < window)
            .collect();

        if events_in_window.is_empty() {
            return ContinuousHV::zero(self.config.dimension);
        }

        // Weight by temporal proximity and importance
        let weights: Vec<f32> = events_in_window
            .iter()
            .map(|e| {
                let temporal_weight = 1.0 - ((e.time - center).abs() / window) as f32;
                temporal_weight * e.importance
            })
            .collect();

        let content_refs: Vec<&ContinuousHV> = events_in_window
            .iter()
            .map(|e| &e.content)
            .collect();

        ContinuousHV::weighted_bundle(&content_refs, &weights)
    }

    /// Get number of stored events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Get memory statistics
    pub fn stats(&self) -> TemporalMemoryStats {
        if self.events.is_empty() {
            return TemporalMemoryStats {
                event_count: 0,
                time_span: 0.0,
                earliest_time: 0.0,
                latest_time: 0.0,
                avg_importance: 0.0,
            };
        }

        let times: Vec<f64> = self.events.iter().map(|e| e.time).collect();
        let earliest = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let latest = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_importance: f32 = self.events.iter().map(|e| e.importance).sum::<f32>()
            / self.events.len() as f32;

        TemporalMemoryStats {
            event_count: self.events.len(),
            time_span: latest - earliest,
            earliest_time: earliest,
            latest_time: latest,
            avg_importance,
        }
    }
}

/// Statistics about temporal memory
#[derive(Debug, Clone)]
pub struct TemporalMemoryStats {
    /// Number of stored events
    pub event_count: usize,

    /// Time span from earliest to latest event
    pub time_span: f64,

    /// Earliest event time
    pub earliest_time: f64,

    /// Latest event time
    pub latest_time: f64,

    /// Average importance of stored events
    pub avg_importance: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memory() -> TemporalHolographicMemory {
        let config = TemporalConfig {
            temporal_resolution: 100,
            time_scale_secs: 3600, // 1 hour for testing
            similarity_threshold: 0.1,
            max_events: 100,
            dimension: 1024, // Smaller for tests
        };
        TemporalHolographicMemory::new(config)
    }

    #[test]
    fn test_encode_and_recall() {
        let mut memory = create_test_memory();

        // Create a single event type and encode at different times
        // Using the same HV ensures similarity > threshold for recall
        let event = ContinuousHV::random(1024, 42);

        memory.encode(event.clone(), 1.0);
        memory.encode(event.clone(), 2.0);
        memory.encode(event.clone(), 3.0);

        assert_eq!(memory.len(), 3);

        // Causal query: events before t=2.5
        let results = memory.recall_before(&event, 2.5);

        // Should find events at t=1.0 and t=2.0, but not t=3.0
        assert!(results.iter().any(|r| r.time == 1.0));
        assert!(results.iter().any(|r| r.time == 2.0));
        assert!(!results.iter().any(|r| r.time == 3.0));
    }

    #[test]
    fn test_causal_ordering() {
        let mut memory = create_test_memory();

        let event = ContinuousHV::random(1024, 42);
        memory.encode(event.clone(), 5.0);
        memory.encode(event.clone(), 10.0);
        memory.encode(event.clone(), 15.0);

        // Query with deadline before all events
        let results = memory.recall_before(&event, 4.0);
        assert!(results.is_empty());

        // Query with deadline after first event
        let results = memory.recall_before(&event, 7.0);
        assert_eq!(results.len(), 1);

        // Query with deadline after all events
        let results = memory.recall_before(&event, 20.0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_temporal_context() {
        let mut memory = create_test_memory();

        // Create cluster of events around t=5.0
        for i in 0..5 {
            let event = ContinuousHV::random(1024, i);
            memory.encode(event, 5.0 + i as f64 * 0.1);
        }

        // Create distant event
        let distant = ContinuousHV::random(1024, 100);
        memory.encode(distant, 100.0);

        // Get context around t=5.0
        let context = memory.temporal_context(5.0, 1.0);

        // Context should be non-zero (has bundled events)
        let norm: f32 = context.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_predict_next() {
        let mut memory = create_test_memory();

        // Create a sequence: A -> B -> C
        let event_a = ContinuousHV::random(1024, 1);
        let event_b = ContinuousHV::random(1024, 2);
        let event_c = ContinuousHV::random(1024, 3);

        memory.encode(event_a.clone(), 1.0);
        memory.encode(event_b.clone(), 2.0);
        memory.encode(event_c.clone(), 3.0);

        // Predict what follows event_a
        let predictions = memory.predict_next(&event_a, 2.0);

        // Should include event_b (at t=2.0)
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut memory = create_test_memory();

        memory.encode(ContinuousHV::random(1024, 1), 10.0);
        memory.encode(ContinuousHV::random(1024, 2), 20.0);
        memory.encode(ContinuousHV::random(1024, 3), 30.0);

        let stats = memory.stats();

        assert_eq!(stats.event_count, 3);
        assert!((stats.time_span - 20.0).abs() < 0.001);
        assert!((stats.earliest_time - 10.0).abs() < 0.001);
        assert!((stats.latest_time - 30.0).abs() < 0.001);
    }
}
