#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test Data Builders
//!
//! Factory functions and builder patterns for creating test data.
//! These provide consistent, reproducible test data across all integration tests.

use symthaea::databases::{MemoryRecord, MemoryType};
use symthaea::hdc::binary_hv::BinaryHV;

// ============================================================================
// Memory Record Builder
// ============================================================================

/// Builder for creating MemoryRecord instances with sensible defaults.
///
/// # Example
/// ```rust,ignore
/// let record = MemoryRecordBuilder::new("test-1")
///     .with_type(MemoryType::Episodic)
///     .with_content("Test content")
///     .with_psi(0.75)
///     .build();
/// ```
#[derive(Clone)]
pub struct MemoryRecordBuilder {
    id: String,
    seed: u64,
    memory_type: MemoryType,
    content: String,
    valence: f32,
    arousal: f32,
    psi: f64,
    topics: Vec<String>,
    metadata: String,
    timestamp_ms: u64,
}

impl MemoryRecordBuilder {
    /// Create a new builder with the given ID
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            seed: 42,
            memory_type: MemoryType::Working,
            content: format!("Test content for {}", id),
            valence: 0.5,
            arousal: 0.3,
            psi: 0.65,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
            timestamp_ms: 1700000000000,
        }
    }

    /// Set the random seed for encoding generation
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.timestamp_ms = 1700000000000 + seed;
        self
    }

    /// Set the memory type
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type;
        self
    }

    /// Set the content
    pub fn with_content(mut self, content: &str) -> Self {
        self.content = content.to_string();
        self
    }

    /// Set the valence (-1.0 to 1.0)
    pub fn with_valence(mut self, valence: f32) -> Self {
        self.valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Set the arousal (0.0 to 1.0)
    pub fn with_arousal(mut self, arousal: f32) -> Self {
        self.arousal = arousal.clamp(0.0, 1.0);
        self
    }

    /// Set the psi value (0.0 to 1.0)
    pub fn with_psi(mut self, psi: f64) -> Self {
        self.psi = psi.clamp(0.0, 1.0);
        self
    }

    /// Set the topics
    pub fn with_topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }

    /// Set metadata as JSON string
    pub fn with_metadata(mut self, metadata: &str) -> Self {
        self.metadata = metadata.to_string();
        self
    }

    /// Set the timestamp
    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }

    /// Build the MemoryRecord
    pub fn build(self) -> MemoryRecord {
        MemoryRecord {
            id: self.id,
            encoding: BinaryHV::random(self.seed),
            timestamp_ms: self.timestamp_ms,
            memory_type: self.memory_type,
            content: self.content,
            valence: self.valence,
            arousal: self.arousal,
            psi: self.psi,
            topics: self.topics,
            metadata: self.metadata,
            consolidation_strength: 0.0,
            retrieval_count: 0,
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a simple test memory record (shorthand)
///
/// # Example
/// ```rust,ignore
/// let record = create_test_memory("test-1", 42, MemoryType::Working);
/// ```
pub fn create_test_memory(id: &str, seed: u64, memory_type: MemoryType) -> MemoryRecord {
    MemoryRecordBuilder::new(id)
        .with_seed(seed)
        .with_type(memory_type)
        .build()
}

/// Create a test memory with custom content
pub fn create_memory_with_content(
    id: &str,
    seed: u64,
    memory_type: MemoryType,
    content: &str,
) -> MemoryRecord {
    MemoryRecordBuilder::new(id)
        .with_seed(seed)
        .with_type(memory_type)
        .with_content(content)
        .build()
}

/// Create multiple test memories with sequential seeds
///
/// # Example
/// ```rust,ignore
/// let memories = create_memory_batch("batch", 100, 5, MemoryType::Episodic);
/// // Creates batch-0, batch-1, batch-2, batch-3, batch-4
/// ```
pub fn create_memory_batch(
    prefix: &str,
    start_seed: u64,
    count: usize,
    memory_type: MemoryType,
) -> Vec<MemoryRecord> {
    (0..count)
        .map(|i| {
            let id = format!("{}-{}", prefix, i);
            create_test_memory(&id, start_seed + i as u64, memory_type)
        })
        .collect()
}

// ============================================================================
// Test Data Constants
// ============================================================================

/// Standard test content strings
pub mod test_content {
    pub const SHORT: &str = "Short test content";
    pub const MEDIUM: &str =
        "This is a medium-length test content string for integration testing purposes.";
    pub const LONG: &str = "This is a longer test content string that spans multiple sentences. \
        It contains various words and phrases to test content handling. \
        The content should be processed correctly by all database operations. \
        Testing edge cases with moderately long strings is important.";
    pub const UNICODE: &str = "Unicode test: 日本語 emoji 🧠💡 special chars ñ ü ö";
    pub const EMPTY: &str = "";
}

/// Standard test metadata JSON strings
pub mod test_metadata {
    pub const EMPTY: &str = "{}";
    pub const SIMPLE: &str = r#"{"key": "value"}"#;
    pub const NESTED: &str = r#"{"outer": {"inner": "value"}, "array": [1, 2, 3]}"#;
    pub const LARGE: &str = r#"{"field1": "value1", "field2": "value2", "field3": "value3", "field4": "value4", "field5": "value5"}"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let record = MemoryRecordBuilder::new("test").build();
        assert_eq!(record.id, "test");
        assert_eq!(record.memory_type, MemoryType::Working);
        assert!((record.psi - 0.65).abs() < 0.001);
    }

    #[test]
    fn test_builder_chain() {
        let record = MemoryRecordBuilder::new("chain-test")
            .with_seed(123)
            .with_type(MemoryType::Episodic)
            .with_psi(0.9)
            .with_content("Custom content")
            .build();

        assert_eq!(record.id, "chain-test");
        assert_eq!(record.memory_type, MemoryType::Episodic);
        assert!((record.psi - 0.9).abs() < 0.001);
        assert_eq!(record.content, "Custom content");
    }

    #[test]
    fn test_create_batch() {
        let batch = create_memory_batch("test", 0, 3, MemoryType::Working);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].id, "test-0");
        assert_eq!(batch[1].id, "test-1");
        assert_eq!(batch[2].id, "test-2");
    }
}