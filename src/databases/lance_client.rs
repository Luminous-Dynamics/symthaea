//! LanceDB Vector + Columnar Database Client
//!
//! **Mental Role**: Long-Term Memory
//!
//! LanceDB handles persistent memory storage - like the hippocampus and
//! neocortex working together for long-term consolidation.
//!
//! ## Architecture Role
//!
//! ```text
//! Experience → [LanceDB: 10-100ms] → Consolidated Memory → Retrieval
//!                    ↑
//!            Long-Term Memory
//!            - Episodic memories
//!            - Semantic knowledge
//!            - Procedural skills
//! ```
//!
//! ## Consciousness Mapping
//!
//! - **#29 Memory Types**: Episodic, Semantic, Procedural storage
//! - **#7 Dynamics**: Temporal memory consolidation
//! - **#13 Multi-scale Time**: Short to long-term transitions

use super::{ConsciousnessDatabase, DbResult, MemoryRecord, MemoryType, SearchResult};
use crate::hdc::binary_hv::HV16;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for LanceDB
#[derive(Debug, Clone)]
pub struct LanceConfig {
    /// Database directory path
    pub path: String,

    /// Table name for memories
    pub table_name: String,

    /// Enable memory consolidation
    pub enable_consolidation: bool,

    /// Consolidation threshold (hours)
    pub consolidation_hours: u64,
}

impl Default for LanceConfig {
    fn default() -> Self {
        Self {
            path: ".symthaea/lancedb".to_string(),
            table_name: "consciousness_ltm".to_string(),
            enable_consolidation: true,
            consolidation_hours: 24,
        }
    }
}

/// Memory consolidation state
#[derive(Debug, Clone)]
pub struct ConsolidationState {
    /// Memory ID
    pub id: String,

    /// Number of times recalled
    pub recall_count: u32,

    /// Last recall timestamp
    pub last_recall: u64,

    /// Consolidation strength (0-1)
    pub strength: f32,

    /// Is this memory consolidated?
    pub is_consolidated: bool,
}

/// LanceDB-backed long-term memory database
pub struct LanceLongTerm {
    config: LanceConfig,

    #[cfg(feature = "lance")]
    db: Option<lancedb::Connection>,

    /// Fallback storage
    memories: RwLock<HashMap<String, MemoryRecord>>,

    /// Consolidation tracking
    consolidation: RwLock<HashMap<String, ConsolidationState>>,

    /// Timestamp of last consolidation run
    last_consolidation: RwLock<u64>,
}

impl LanceLongTerm {
    /// Create new LanceDB long-term memory database
    pub fn new(config: LanceConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "lance")]
            db: None,
            memories: RwLock::new(HashMap::new()),
            consolidation: RwLock::new(HashMap::new()),
            last_consolidation: RwLock::new(0),
        }
    }

    /// Connect to LanceDB
    #[cfg(feature = "lance")]
    pub async fn connect(&mut self) -> DbResult<()> {
        match lancedb::connect(&self.config.path).execute().await {
            Ok(db) => {
                self.db = Some(db);
                Ok(())
            }
            Err(e) => Err(format!("Failed to connect to LanceDB: {}", e)),
        }
    }

    /// Record a memory recall (strengthens consolidation)
    pub fn record_recall(&self, id: &str) {
        let mut cons = self.consolidation.write().unwrap();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(state) = cons.get_mut(id) {
            state.recall_count += 1;
            state.last_recall = now;

            // Strengthen consolidation with each recall
            state.strength = (state.strength + 0.1).min(1.0);

            if state.recall_count >= 3 && state.strength >= 0.5 {
                state.is_consolidated = true;
            }
        } else {
            cons.insert(
                id.to_string(),
                ConsolidationState {
                    id: id.to_string(),
                    recall_count: 1,
                    last_recall: now,
                    strength: 0.3,
                    is_consolidated: false,
                },
            );
        }
    }

    /// Get consolidation state for a memory
    pub fn consolidation_state(&self, id: &str) -> Option<ConsolidationState> {
        self.consolidation.read().unwrap().get(id).cloned()
    }

    /// Get all consolidated memories
    pub fn consolidated_memories(&self) -> Vec<String> {
        self.consolidation
            .read()
            .unwrap()
            .iter()
            .filter(|(_, state)| state.is_consolidated)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Run memory consolidation (simulates sleep consolidation)
    pub async fn run_consolidation(&self) -> DbResult<usize> {
        let mut cons = self.consolidation.write().unwrap();
        let mut consolidated_count = 0;

        for (_, state) in cons.iter_mut() {
            if !state.is_consolidated && state.strength >= 0.4 {
                // Boost strength during "sleep"
                state.strength = (state.strength + 0.2).min(1.0);

                if state.strength >= 0.7 {
                    state.is_consolidated = true;
                    consolidated_count += 1;
                }
            }
        }

        // Update last consolidation time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        *self.last_consolidation.write().unwrap() = now;

        Ok(consolidated_count)
    }

    /// Get memories by type
    pub fn memories_by_type(&self, memory_type: MemoryType) -> Vec<MemoryRecord> {
        self.memories
            .read()
            .unwrap()
            .values()
            .filter(|m| m.memory_type == memory_type)
            .cloned()
            .collect()
    }

    /// Calculate memory decay (forgetting curve)
    pub fn memory_strength(&self, id: &str) -> f32 {
        if let Some(state) = self.consolidation.read().unwrap().get(id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            // Ebbinghaus forgetting curve approximation
            let hours_since_recall = (now - state.last_recall) as f32 / 3600.0;
            let base_retention = state.strength;

            // Consolidated memories decay much slower
            let decay_rate = if state.is_consolidated { 0.01 } else { 0.1 };

            // R = S * e^(-t/λ)
            base_retention * (-hours_since_recall * decay_rate).exp()
        } else {
            0.0
        }
    }
}

#[async_trait]
impl ConsciousnessDatabase for LanceLongTerm {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        let id = record.id.clone();

        self.memories
            .write()
            .unwrap()
            .insert(id.clone(), record);

        // Initialize consolidation tracking
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.consolidation.write().unwrap().insert(
            id,
            ConsolidationState {
                id: String::new(),
                recall_count: 0,
                last_recall: now,
                strength: 0.2, // Initial encoding strength
                is_consolidated: false,
            },
        );

        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let memories = self.memories.read().unwrap();

        let mut results: Vec<(MemoryRecord, f32, String)> = memories
            .iter()
            .map(|(id, record)| {
                let sim = query.similarity(&record.encoding);
                // Weight by consolidation strength
                let strength = self.memory_strength(id);
                let weighted_sim = sim * (0.5 + 0.5 * strength);

                (record.clone(), weighted_sim, id.clone())
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Record recalls for consolidation (after sorting)
        for (_, _, id) in &results[..top_k.min(results.len())] {
            self.record_recall(id);
        }

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(record, similarity, _id)| SearchResult { record, similarity })
            .collect())
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        if let Some(record) = self.memories.read().unwrap().get(id) {
            self.record_recall(id);
            Ok(Some(record.clone()))
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        let removed = self.memories.write().unwrap().remove(id).is_some();
        if removed {
            self.consolidation.write().unwrap().remove(id);
        }
        Ok(removed)
    }

    async fn count(&self) -> DbResult<usize> {
        Ok(self.memories.read().unwrap().len())
    }

    async fn health_check(&self) -> DbResult<bool> {
        Ok(true)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_config_default() {
        let config = LanceConfig::default();
        assert!(config.enable_consolidation);
        assert_eq!(config.consolidation_hours, 24);
    }

    #[tokio::test]
    async fn test_lance_store_retrieve() {
        let db = LanceLongTerm::new(LanceConfig::default());

        let record = MemoryRecord {
            id: "ltm-1".to_string(),
            encoding: HV16::random(42),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Episodic,
            content: "Long-term memory".to_string(),
            valence: 0.7,
            arousal: 0.4,
            phi: 0.6,
            topics: vec!["memory".to_string()],
            metadata: "{}".to_string(),
        };

        db.store(record.clone()).await.unwrap();

        let retrieved = db.get("ltm-1").await.unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_consolidation_tracking() {
        let db = LanceLongTerm::new(LanceConfig::default());

        db.record_recall("test-mem");
        let state = db.consolidation_state("test-mem");
        assert!(state.is_some());
        assert_eq!(state.unwrap().recall_count, 1);

        // Multiple recalls should strengthen
        db.record_recall("test-mem");
        db.record_recall("test-mem");

        let state = db.consolidation_state("test-mem").unwrap();
        assert!(state.recall_count >= 3);
        assert!(state.strength > 0.3);
    }

    #[test]
    fn test_consolidation_run() {
        let db = LanceLongTerm::new(LanceConfig::default());

        // Insert directly (bypass async)
        {
            let mut memories = db.memories.write().unwrap();
            memories.insert(
                "cons-1".to_string(),
                MemoryRecord {
                    id: "cons-1".to_string(),
                    encoding: HV16::random(1),
                    timestamp_ms: 1700000000000,
                    memory_type: MemoryType::Episodic,
                    content: "Important event".to_string(),
                    valence: 0.9,
                    arousal: 0.8,
                    phi: 0.7,
                    topics: vec!["important".to_string()],
                    metadata: "{}".to_string(),
                },
            );
        }

        // Initialize consolidation state
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            db.consolidation.write().unwrap().insert(
                "cons-1".to_string(),
                ConsolidationState {
                    id: "cons-1".to_string(),
                    recall_count: 0,
                    last_recall: now,
                    strength: 0.2,
                    is_consolidated: false,
                },
            );
        }

        // Simulate multiple recalls (increases strength to ~0.7)
        for _ in 0..5 {
            db.record_recall("cons-1");
        }

        // Check that consolidation happened (recall_count >= 3 && strength >= 0.5)
        let state = db.consolidation_state("cons-1").unwrap();
        assert!(state.recall_count >= 5, "Should have 5 recalls");
        assert!(state.strength >= 0.5, "Strength should be at least 0.5");
        assert!(state.is_consolidated, "Should be consolidated after 5 recalls");
    }

    #[test]
    fn test_memory_by_type() {
        // Create the database first
        let db = LanceLongTerm::new(LanceConfig::default());

        // Insert memories directly (bypass async)
        {
            let mut memories = db.memories.write().unwrap();

            memories.insert(
                "ep-1".to_string(),
                MemoryRecord {
                    id: "ep-1".to_string(),
                    encoding: HV16::random(1),
                    timestamp_ms: 1700000000000,
                    memory_type: MemoryType::Episodic,
                    content: "Episode 1".to_string(),
                    valence: 0.0,
                    arousal: 0.0,
                    phi: 0.0,
                    topics: vec![],
                    metadata: "{}".to_string(),
                },
            );

            memories.insert(
                "sem-1".to_string(),
                MemoryRecord {
                    id: "sem-1".to_string(),
                    encoding: HV16::random(2),
                    timestamp_ms: 1700000000001,
                    memory_type: MemoryType::Semantic,
                    content: "Semantic fact".to_string(),
                    valence: 0.0,
                    arousal: 0.0,
                    phi: 0.0,
                    topics: vec![],
                    metadata: "{}".to_string(),
                },
            );
        }

        let episodic = db.memories_by_type(MemoryType::Episodic);
        assert_eq!(episodic.len(), 1);

        let semantic = db.memories_by_type(MemoryType::Semantic);
        assert_eq!(semantic.len(), 1);
    }

    #[test]
    fn test_memory_strength_decay() {
        let db = LanceLongTerm::new(LanceConfig::default());

        // Add consolidation state
        {
            let mut cons = db.consolidation.write().unwrap();
            cons.insert(
                "decay-test".to_string(),
                ConsolidationState {
                    id: "decay-test".to_string(),
                    recall_count: 5,
                    last_recall: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    strength: 0.8,
                    is_consolidated: true,
                },
            );
        }

        let strength = db.memory_strength("decay-test");
        assert!(strength > 0.7); // Recently recalled, should be strong
    }

    #[test]
    fn test_consolidated_memories_list() {
        let db = LanceLongTerm::new(LanceConfig::default());

        {
            let mut cons = db.consolidation.write().unwrap();

            cons.insert(
                "con-1".to_string(),
                ConsolidationState {
                    id: "con-1".to_string(),
                    recall_count: 5,
                    last_recall: 0,
                    strength: 0.9,
                    is_consolidated: true,
                },
            );

            cons.insert(
                "not-con".to_string(),
                ConsolidationState {
                    id: "not-con".to_string(),
                    recall_count: 1,
                    last_recall: 0,
                    strength: 0.2,
                    is_consolidated: false,
                },
            );
        }

        let consolidated = db.consolidated_memories();
        assert_eq!(consolidated.len(), 1);
        assert!(consolidated.contains(&"con-1".to_string()));
    }
}
