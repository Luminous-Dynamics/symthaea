//! Persistent Coding Experience Store: HDC-encoded coding experiences that survive across sessions.
//!
//! Replaces the in-memory `error_pattern_memory` and `code_generation_cache` on the
//! `Symthaea` facade with a persistent SQLite-backed store. Error patterns and successful
//! code generations are encoded as BinaryHVs for fast similarity retrieval.
//!
//! ## Architecture
//!
//! Each coding experience is stored as a `MemoryRecord` in the `ConsciousnessDatabase`
//! with `MemoryType::Procedural` (how-to knowledge). The encoding is a BinaryHV of the
//! task description, enabling similarity search: "fix borrow checker" retrieves prior
//! experiences with borrow-related fixes.
//!
//! ## Valence Encoding
//!
//! Experiences carry emotional valence:
//! - **Positive** (valence > 0): successful generations, fixes that worked
//! - **Negative** (valence < 0): compilation failures, test failures
//! - **Arousal**: proportional to prediction error (surprising outcomes = high arousal)

use crate::databases::{
    ConsciousnessDatabase, DatabaseConfig, MemoryRecord, MemoryType, SearchResult,
};
use std::sync::atomic::{AtomicU64, Ordering};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Monotonic counter to ensure unique IDs even within the same millisecond.
static EXPERIENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A coding experience record suitable for storage and retrieval.
#[derive(Debug, Clone)]
pub struct CodingExperience {
    /// The task description that produced this experience.
    pub task: String,
    /// The error pattern (for failures) or generated code summary (for successes).
    pub detail: String,
    /// Whether this experience was a success (code compiled/tests passed).
    pub success: bool,
    /// The backend tier used (Native/LocalLLM/CloudLLM).
    pub tier: String,
    /// Fix hint: what worked to resolve the error (for failures that were later fixed).
    pub fix_hint: Option<String>,
}

/// Persistent coding experience store backed by ConsciousnessDatabase.
///
/// Stores error patterns and successful generations as HDC-encoded procedural
/// memories. Survives across sessions (SQLite-backed) and supports similarity
/// retrieval for injecting relevant hints into future generation prompts.
pub struct CodingExperienceStore {
    /// The underlying database (typically SqliteMemory).
    db: Box<dyn ConsciousnessDatabase>,
    /// In-memory cache of recent error hints: (pattern, fix_hint).
    /// Hydrated from DB on startup, updated on each store.
    error_hints_cache: Vec<(String, String)>,
    /// In-memory cache of recent successful generations: (task, code_summary).
    success_cache: Vec<(String, String)>,
    /// Maximum cache size (oldest entries evicted).
    max_cache_size: usize,
}

impl CodingExperienceStore {
    /// Create a new experience store with an in-memory SQLite database.
    pub async fn new() -> Result<Self, crate::databases::DatabaseError> {
        let config = DatabaseConfig::default(); // in-memory SQLite
        let db = crate::databases::create_database(&config).await?;
        Ok(Self {
            db,
            error_hints_cache: Vec::new(),
            success_cache: Vec::new(),
            max_cache_size: 128,
        })
    }

    /// Create a store with a persistent SQLite database at the given path.
    pub async fn persistent(path: &str) -> Result<Self, crate::databases::DatabaseError> {
        let config = DatabaseConfig {
            path: Some(path.to_string()),
            ..Default::default()
        };
        let db = crate::databases::create_database(&config).await?;

        let mut store = Self {
            db,
            error_hints_cache: Vec::new(),
            success_cache: Vec::new(),
            max_cache_size: 128,
        };

        // Hydrate caches from DB
        store.hydrate_caches().await;
        Ok(store)
    }

    /// Encode text into a BinaryHV using deterministic hashing.
    ///
    /// Each word gets a seed-based random HV, and the sentence encoding is the
    /// majority vote (bundling) of word HVs — standard HDC text encoding.
    fn encode_text(text: &str) -> BinaryHV {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return BinaryHV::random(0);
        }

        let word_hvs: Vec<BinaryHV> = words
            .iter()
            .map(|w| {
                let seed = Self::hash_word(w);
                BinaryHV::random(seed)
            })
            .collect();

        // Bundle (majority vote) all word HVs
        BinaryHV::bundle(&word_hvs)
    }

    /// Hash a word to a u64 seed for deterministic BinaryHV generation.
    fn hash_word(word: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.to_lowercase().hash(&mut hasher);
        hasher.finish()
    }

    /// Hydrate in-memory caches from the database on startup.
    async fn hydrate_caches(&mut self) {
        // Load recent error experiences (negative valence)
        if let Ok(records) = self.db.list_all().await {
            for record in records.iter().rev().take(self.max_cache_size) {
                if record.valence < 0.0 {
                    // Error experience: content = error pattern, metadata = fix hint
                    let fix_hint = if record.metadata.is_empty() || record.metadata == "{}" {
                        record.content.clone()
                    } else {
                        record.metadata.clone()
                    };
                    self.error_hints_cache
                        .push((record.content.clone(), fix_hint));
                } else {
                    // Success experience: content = task, topics[0] = code summary
                    let summary = record.topics.first().cloned().unwrap_or_default();
                    self.success_cache.push((record.content.clone(), summary));
                }
            }
        }
    }

    /// Store a coding experience.
    pub async fn store(&mut self, experience: CodingExperience) {
        let encoding = Self::encode_text(&experience.task);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let valence = if experience.success { 0.7 } else { -0.7 };
        let seq = EXPERIENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = format!("coding_exp_{}_{}", now_ms, seq);

        let record = MemoryRecord {
            id,
            memory_type: MemoryType::Procedural,
            encoding,
            content: if experience.success {
                experience.task.clone()
            } else {
                experience.detail.clone()
            },
            timestamp_ms: now_ms,
            valence,
            arousal: if experience.success { 0.3 } else { 0.8 },
            psi: 0.5,
            topics: vec![
                experience.tier.clone(),
                if experience.success {
                    "success".to_string()
                } else {
                    "failure".to_string()
                },
            ],
            metadata: experience
                .fix_hint
                .clone()
                .unwrap_or_else(|| experience.task.clone()),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        if let Err(e) = self.db.store(record).await {
            tracing::warn!(
                target: "symthaea::coding_experience",
                error = %e,
                "Failed to store coding experience"
            );
            return;
        }

        // Update caches
        if experience.success {
            self.success_cache
                .push((experience.task, experience.detail));
            if self.success_cache.len() > self.max_cache_size {
                self.success_cache.remove(0);
            }
        } else {
            let hint = experience
                .fix_hint
                .unwrap_or_else(|| experience.detail.clone());
            self.error_hints_cache.push((experience.detail, hint));
            if self.error_hints_cache.len() > self.max_cache_size {
                self.error_hints_cache.remove(0);
            }
        }
    }

    /// Query for similar coding experiences by task description.
    ///
    /// Returns up to `top_k` similar experiences, sorted by similarity.
    pub async fn query_similar(&self, task: &str, top_k: usize) -> Vec<SearchResult> {
        let query_hv = Self::encode_text(task);
        self.db
            .search_similar(&query_hv, top_k)
            .await
            .unwrap_or_default()
    }

    /// Get error hints relevant to a task, from both cache and similarity search.
    ///
    /// Returns `(error_pattern, fix_hint)` pairs. The cache provides fast access
    /// to recent errors, while similarity search finds older relevant patterns.
    pub async fn error_hints_for(&self, task: &str, max_hints: usize) -> Vec<(String, String)> {
        let mut hints = Vec::new();

        // 1. Check cache for matching patterns (fast, substring match)
        let task_lower = task.to_lowercase();
        for (pattern, hint) in &self.error_hints_cache {
            if task_lower.contains(&pattern.to_lowercase().chars().take(30).collect::<String>())
                || pattern
                    .to_lowercase()
                    .contains(&task_lower.chars().take(30).collect::<String>())
            {
                hints.push((pattern.clone(), hint.clone()));
                if hints.len() >= max_hints {
                    return hints;
                }
            }
        }

        // 2. Similarity search for deeper matches
        let results = self.query_similar(task, max_hints * 2).await;
        for result in results {
            if result.record.valence < 0.0 && result.similarity > 0.55 {
                let hint = if result.record.metadata.is_empty() || result.record.metadata == "{}" {
                    result.record.content.clone()
                } else {
                    result.record.metadata.clone()
                };
                hints.push((result.record.content, hint));
                if hints.len() >= max_hints {
                    break;
                }
            }
        }

        hints
    }

    /// Get the in-memory error hints cache (for injection into CodingAgent).
    pub fn cached_error_hints(&self) -> &[(String, String)] {
        &self.error_hints_cache
    }

    /// Get the in-memory success cache.
    pub fn cached_successes(&self) -> &[(String, String)] {
        &self.success_cache
    }

    /// Get the success rate for a recipe pattern (e.g., "CargoCheck" or "WriteFile→CargoCheck").
    ///
    /// Searches caches for experiences matching the recipe pattern.
    /// For successes: task field starts with "recipe:<pattern>".
    /// For failures: detail field contains the atoms pattern (since error cache stores detail, not task).
    /// Returns `(successes, total)` — caller can compute rate. Returns (0, 0) if no data.
    pub fn recipe_success_rate(&self, recipe_key: &str) -> (usize, usize) {
        let prefix = format!("recipe:{}", recipe_key);

        let mut successes = 0usize;
        let mut total = 0usize;

        // Success cache: (task, detail) — task starts with "recipe:..."
        for (task, _) in &self.success_cache {
            if task.starts_with(&prefix) {
                successes += 1;
                total += 1;
            }
        }

        // Error cache: (detail, hint) — detail contains "atoms=[...]" with the recipe key
        for (detail, _) in &self.error_hints_cache {
            if detail.contains(recipe_key) {
                total += 1;
            }
        }

        (successes, total)
    }

    /// Get success rates for multiple recipe patterns at once.
    /// Returns a map from recipe_key to success_rate (0.0-1.0, with 0.5 prior for unseen recipes).
    pub fn recipe_success_rates(&self, recipe_keys: &[&str]) -> Vec<f32> {
        recipe_keys
            .iter()
            .map(|key| {
                let (successes, total) = self.recipe_success_rate(key);
                if total == 0 {
                    0.5 // uninformative prior
                } else {
                    // Bayesian: (successes + 1) / (total + 2) — Laplace smoothing
                    (successes as f32 + 1.0) / (total as f32 + 2.0)
                }
            })
            .collect()
    }

    /// Total number of stored experiences.
    pub async fn count(&self) -> usize {
        self.db.count().await.unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve_error() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store
            .store(CodingExperience {
                task: "add fibonacci function".to_string(),
                detail: "error[E0412]: cannot find type `Vec`".to_string(),
                success: false,
                tier: "Native".to_string(),
                fix_hint: Some("use std::vec::Vec".to_string()),
            })
            .await;

        assert_eq!(store.count().await, 1);
        assert_eq!(store.cached_error_hints().len(), 1);
        assert!(store.cached_error_hints()[0].0.contains("E0412"));
        assert!(store.cached_error_hints()[0].1.contains("std::vec::Vec"));
    }

    #[tokio::test]
    async fn test_store_and_retrieve_success() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store
            .store(CodingExperience {
                task: "add sort function".to_string(),
                detail: "pub fn sort(arr: &mut [i32]) { ... }".to_string(),
                success: true,
                tier: "LocalLLM".to_string(),
                fix_hint: None,
            })
            .await;

        assert_eq!(store.count().await, 1);
        assert_eq!(store.cached_successes().len(), 1);
        assert!(store.cached_successes()[0].0.contains("sort"));
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        // Store a fibonacci-related error
        store
            .store(CodingExperience {
                task: "implement fibonacci sequence".to_string(),
                detail: "overflow in fibonacci calculation".to_string(),
                success: false,
                tier: "Native".to_string(),
                fix_hint: Some("use saturating_add".to_string()),
            })
            .await;

        // Query with similar task
        let results = store.query_similar("add fibonacci function", 5).await;
        assert!(
            !results.is_empty(),
            "Should find similar fibonacci experience"
        );
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let mut store = CodingExperienceStore::new().await.unwrap();
        store.max_cache_size = 3;

        for i in 0..5 {
            store
                .store(CodingExperience {
                    task: format!("task {i}"),
                    detail: format!("error {i}"),
                    success: false,
                    tier: "Native".to_string(),
                    fix_hint: Some(format!("fix {i}")),
                })
                .await;
        }

        // Cache should be capped at 3
        assert_eq!(store.cached_error_hints().len(), 3);
        // DB should have all 5
        assert_eq!(store.count().await, 5);
    }

    #[tokio::test]
    async fn test_error_hints_for_task() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store
            .store(CodingExperience {
                task: "parse JSON input".to_string(),
                detail: "serde deserialization failed".to_string(),
                success: false,
                tier: "LocalLLM".to_string(),
                fix_hint: Some("add #[derive(Deserialize)]".to_string()),
            })
            .await;

        store
            .store(CodingExperience {
                task: "sort numbers".to_string(),
                detail: "index out of bounds".to_string(),
                success: false,
                tier: "Native".to_string(),
                fix_hint: Some("check array length first".to_string()),
            })
            .await;

        let hints = store.error_hints_for("parse JSON data", 5).await;
        // Should find the JSON-related error, not the sort error
        assert!(
            !hints.is_empty(),
            "Should find hints for JSON task: {:?}",
            hints
        );
    }

    #[tokio::test]
    async fn test_mixed_success_and_failure() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store
            .store(CodingExperience {
                task: "add greeting".to_string(),
                detail: "pub fn hello() -> &str".to_string(),
                success: true,
                tier: "Native".to_string(),
                fix_hint: None,
            })
            .await;

        store
            .store(CodingExperience {
                task: "add greeting".to_string(),
                detail: "lifetime error".to_string(),
                success: false,
                tier: "LocalLLM".to_string(),
                fix_hint: Some("use &'static str".to_string()),
            })
            .await;

        assert_eq!(store.cached_successes().len(), 1);
        assert_eq!(store.cached_error_hints().len(), 1);
        assert_eq!(store.count().await, 2);
    }

    #[tokio::test]
    async fn test_recipe_success_rate_no_data() {
        let store = CodingExperienceStore::new().await.unwrap();
        let (successes, total) = store.recipe_success_rate("CargoCheck");
        assert_eq!(successes, 0);
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_recipe_success_rate_with_data() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        // Store a successful recipe trace (task=recipe:..., detail=summary)
        store
            .store(CodingExperience {
                task: "recipe:CargoCheck".to_string(),
                detail: "energy=3.0, steps=1, atoms=[CargoCheck]".to_string(),
                success: true,
                tier: "MoleculeExecutor".to_string(),
                fix_hint: None,
            })
            .await;

        // Store a failed recipe trace
        // For failures, error_hints_cache stores (detail, hint), so detail
        // must contain the recipe key for matching.
        store
            .store(CodingExperience {
                task: "recipe:CargoCheck".to_string(),
                detail: "energy=3.0, steps=1, atoms=[CargoCheck]".to_string(),
                success: false,
                tier: "MoleculeExecutor".to_string(),
                fix_hint: None,
            })
            .await;

        let (successes, total) = store.recipe_success_rate("CargoCheck");
        assert_eq!(successes, 1);
        assert_eq!(total, 2);
    }

    #[tokio::test]
    async fn test_recipe_success_rates_bayesian() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        // Two successes for WriteFile→CargoCheck
        for _ in 0..2 {
            store
                .store(CodingExperience {
                    task: "recipe:WriteFile→CargoCheck".to_string(),
                    detail: "energy=4.0".to_string(),
                    success: true,
                    tier: "MoleculeExecutor".to_string(),
                    fix_hint: None,
                })
                .await;
        }

        let rates = store.recipe_success_rates(&["WriteFile→CargoCheck", "UnseenRecipe"]);
        assert_eq!(rates.len(), 2);
        // WriteFile→CargoCheck: (2+1)/(2+2) = 0.75
        assert!((rates[0] - 0.75).abs() < 0.01);
        // UnseenRecipe: 0.5 (uninformative prior)
        assert!((rates[1] - 0.5).abs() < f32::EPSILON);
    }
}
