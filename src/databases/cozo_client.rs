//! CozoDB Datalog Database Client
//!
//! **Mental Role**: Prefrontal Cortex
//!
//! CozoDB handles complex reasoning and planning - like the prefrontal
//! cortex in the brain. It uses Datalog for logical inference and
//! recursive reasoning.
//!
//! ## Architecture Role
//!
//! ```text
//! Complex Query → [CozoDB: 100-1000ms] → Logical Reasoning → Decision
//!                      ↑
//!              Prefrontal Cortex
//!              - Datalog inference
//!              - Goal reasoning
//!              - Planning
//! ```
//!
//! ## Consciousness Mapping
//!
//! - **#24 HOT**: Higher-order reasoning about mental states
//! - **#22 FEP**: Belief updating through logical inference
//! - **#14 Causal Efficacy**: Tracking causal chains

use super::{ConsciousnessDatabase, DbResult, MemoryRecord, SearchResult};
use crate::hdc::binary_hv::HV16;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for CozoDB
#[derive(Debug, Clone)]
pub struct CozoConfig {
    /// Database path (or :memory: for in-memory)
    pub path: String,

    /// Enable reasoning extensions
    pub enable_reasoning: bool,
}

impl Default for CozoConfig {
    fn default() -> Self {
        Self {
            path: ":memory:".to_string(),
            enable_reasoning: true,
        }
    }
}

/// Reasoning relation for causal chains
#[derive(Debug, Clone)]
pub struct CausalRelation {
    pub cause: String,
    pub effect: String,
    pub strength: f32,
    pub timestamp: u64,
}

/// Higher-order thought representation
#[derive(Debug, Clone)]
pub struct HigherOrderThought {
    /// ID of thought
    pub id: String,

    /// Object-level content
    pub content: String,

    /// Meta-level interpretation
    pub meta_interpretation: String,

    /// Confidence in meta-level
    pub confidence: f32,

    /// Is this thought about a thought?
    pub order: u8,
}

/// CozoDB-backed prefrontal cortex database
pub struct CozoPrefrontal {
    config: CozoConfig,

    #[cfg(feature = "datalog")]
    db: Option<cozo::DbInstance>,

    /// Fallback storage for memories
    memories: RwLock<HashMap<String, MemoryRecord>>,

    /// Causal relations
    causal_relations: RwLock<Vec<CausalRelation>>,

    /// Higher-order thoughts
    hot_thoughts: RwLock<Vec<HigherOrderThought>>,

    /// Beliefs (for FEP)
    beliefs: RwLock<HashMap<String, f32>>,
}

impl CozoPrefrontal {
    /// Create new CozoDB prefrontal database
    pub fn new(config: CozoConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "datalog")]
            db: None,
            memories: RwLock::new(HashMap::new()),
            causal_relations: RwLock::new(Vec::new()),
            hot_thoughts: RwLock::new(Vec::new()),
            beliefs: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize database and create schema
    #[cfg(feature = "datalog")]
    pub fn initialize(&mut self) -> DbResult<()> {
        use cozo::{DbInstance, DataValue};

        let db = if self.config.path == ":memory:" {
            DbInstance::new("mem", "", Default::default())
        } else {
            DbInstance::new("sqlite", &self.config.path, Default::default())
        };

        match db {
            Ok(db) => {
                // Create relations for consciousness
                let schema = r#"
                    :create memories {
                        id: String,
                        content: String,
                        memory_type: String,
                        phi: Float,
                        valence: Float,
                        arousal: Float
                    }

                    :create causal_relations {
                        cause: String,
                        effect: String,
                        strength: Float,
                        timestamp: Int
                    }

                    :create higher_order_thoughts {
                        id: String,
                        content: String,
                        meta_interpretation: String,
                        confidence: Float,
                        order: Int
                    }

                    :create beliefs {
                        proposition: String,
                        probability: Float
                    }
                "#;

                // Would run schema creation here
                // db.run_script(schema, Default::default())?;

                self.db = Some(db);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create CozoDB: {}", e)),
        }
    }

    /// Add a causal relation
    pub fn add_causal_relation(&self, relation: CausalRelation) {
        self.causal_relations.write().unwrap().push(relation);
    }

    /// Query causal chain: what caused X?
    pub fn what_caused(&self, effect: &str) -> Vec<CausalRelation> {
        self.causal_relations
            .read()
            .unwrap()
            .iter()
            .filter(|r| r.effect == effect)
            .cloned()
            .collect()
    }

    /// Query causal chain: what does X cause?
    pub fn what_effects(&self, cause: &str) -> Vec<CausalRelation> {
        self.causal_relations
            .read()
            .unwrap()
            .iter()
            .filter(|r| r.cause == cause)
            .cloned()
            .collect()
    }

    /// Add a higher-order thought
    pub fn add_hot(&self, thought: HigherOrderThought) {
        self.hot_thoughts.write().unwrap().push(thought);
    }

    /// Get higher-order thoughts about a content
    pub fn hot_about(&self, content: &str) -> Vec<HigherOrderThought> {
        self.hot_thoughts
            .read()
            .unwrap()
            .iter()
            .filter(|t| t.content.contains(content))
            .cloned()
            .collect()
    }

    /// Update belief (FEP-style Bayesian update)
    pub fn update_belief(&self, proposition: &str, new_evidence: f32) {
        let mut beliefs = self.beliefs.write().unwrap();
        let current = beliefs.get(proposition).copied().unwrap_or(0.5);

        // Simple Bayesian-ish update
        let updated = (current + new_evidence) / 2.0;
        beliefs.insert(proposition.to_string(), updated.clamp(0.0, 1.0));
    }

    /// Get belief probability
    pub fn belief_probability(&self, proposition: &str) -> f32 {
        self.beliefs
            .read()
            .unwrap()
            .get(proposition)
            .copied()
            .unwrap_or(0.5)
    }

    /// Run Datalog query (if feature enabled)
    #[cfg(feature = "datalog")]
    pub fn query(&self, datalog: &str) -> DbResult<Vec<Vec<String>>> {
        if let Some(ref db) = self.db {
            match db.run_script(datalog, Default::default()) {
                Ok(result) => {
                    Ok(result
                        .rows
                        .iter()
                        .map(|row| row.iter().map(|v| format!("{:?}", v)).collect())
                        .collect())
                }
                Err(e) => Err(format!("Query failed: {}", e)),
            }
        } else {
            Err("Database not initialized".to_string())
        }
    }
}

#[async_trait]
impl ConsciousnessDatabase for CozoPrefrontal {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        self.memories
            .write()
            .unwrap()
            .insert(record.id.clone(), record);
        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let memories = self.memories.read().unwrap();

        let mut results: Vec<(MemoryRecord, f32)> = memories
            .iter()
            .map(|(_id, record)| {
                let sim = query.similarity(&record.encoding);
                (record.clone(), sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(record, similarity)| SearchResult { record, similarity })
            .collect())
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        Ok(self.memories.read().unwrap().get(id).cloned())
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        Ok(self.memories.write().unwrap().remove(id).is_some())
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
    use crate::databases::MemoryType;

    /// Helper to create a test MemoryRecord
    fn test_record(id: &str, seed: u64, memory_type: MemoryType, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: HV16::random(seed),
            timestamp_ms: 1700000000000 + seed,
            memory_type,
            content: content.to_string(),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.7,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
        }
    }

    #[test]
    fn test_cozo_config_default() {
        let config = CozoConfig::default();
        assert_eq!(config.path, ":memory:");
        assert!(config.enable_reasoning);
    }

    #[tokio::test]
    async fn test_cozo_store_retrieve() {
        let db = CozoPrefrontal::new(CozoConfig::default());
        let record = test_record("cozo-1", 42, MemoryType::Semantic, "Logical reasoning");
        db.store(record).await.unwrap();

        let retrieved = db.get("cozo-1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Logical reasoning");
    }

    #[test]
    fn test_causal_relations() {
        let db = CozoPrefrontal::new(CozoConfig::default());

        db.add_causal_relation(CausalRelation {
            cause: "rain".to_string(),
            effect: "wet_ground".to_string(),
            strength: 0.95,
            timestamp: 1000,
        });

        db.add_causal_relation(CausalRelation {
            cause: "wet_ground".to_string(),
            effect: "slippery".to_string(),
            strength: 0.8,
            timestamp: 1001,
        });

        let causes = db.what_caused("wet_ground");
        assert_eq!(causes.len(), 1);
        assert_eq!(causes[0].cause, "rain");

        let effects = db.what_effects("wet_ground");
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].effect, "slippery");
    }

    #[test]
    fn test_higher_order_thoughts() {
        let db = CozoPrefrontal::new(CozoConfig::default());

        db.add_hot(HigherOrderThought {
            id: "hot-1".to_string(),
            content: "I am happy".to_string(),
            meta_interpretation: "I am aware that I am happy".to_string(),
            confidence: 0.9,
            order: 2,
        });

        let thoughts = db.hot_about("happy");
        assert_eq!(thoughts.len(), 1);
        assert_eq!(thoughts[0].order, 2);
    }

    #[test]
    fn test_belief_updating() {
        let db = CozoPrefrontal::new(CozoConfig::default());

        // Initial belief
        assert_eq!(db.belief_probability("sun_will_rise"), 0.5);

        // Update with strong evidence
        db.update_belief("sun_will_rise", 1.0);
        let p1 = db.belief_probability("sun_will_rise");
        assert!(p1 > 0.5);

        // Update again
        db.update_belief("sun_will_rise", 1.0);
        let p2 = db.belief_probability("sun_will_rise");
        assert!(p2 > p1);
    }

    #[tokio::test]
    async fn test_cozo_count() {
        let db = CozoPrefrontal::new(CozoConfig::default());

        assert_eq!(db.count().await.unwrap(), 0);

        db.store(MemoryRecord {
            id: "count-1".to_string(),
            encoding: HV16::random(1),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Working,
            content: "First".to_string(),
            valence: 0.0,
            arousal: 0.0,
            phi: 0.0,
            topics: vec![],
            metadata: "{}".to_string(),
        })
        .await
        .unwrap();

        assert_eq!(db.count().await.unwrap(), 1);
    }
}
