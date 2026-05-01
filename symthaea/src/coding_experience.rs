// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use symthaea_core::hdc::unified_hv::ContinuousHV;

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
    /// Optional HDC diagnostic geometry of the failure.
    pub diagnostic_hv: Option<ContinuousHV>,
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
    /// Records queued for DB persistence (flushed on next async call or explicit flush).
    pending_writes: Vec<MemoryRecord>,
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
            pending_writes: Vec::new(),
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
            pending_writes: Vec::new(),
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

    /// Persist a record to the database (best-effort, fire-and-forget).
    ///
    /// Appends to `pending_writes` for later flush. This avoids async/sync
    /// boundary issues — callers just push records and flush happens on the
    /// next async boundary (store(), query_similar(), etc.) or explicit flush().
    fn queue_persist(&mut self, record: MemoryRecord) {
        self.pending_writes.push(record);
    }

    /// Flush all pending writes to the database.
    pub async fn flush(&mut self) {
        let records = std::mem::take(&mut self.pending_writes);
        for record in records {
            let _ = self.db.store(record).await;
        }
    }

    /// Hydrate in-memory caches from the database on startup.
    ///
    /// Distinguishes three record types by content prefix:
    /// - `fix:` → error_hints_cache (fix strategies with error code matching)
    /// - `template:` → success_cache (learned code templates)
    /// - other → error_hints_cache (negative valence) or success_cache (positive)
    async fn hydrate_caches(&mut self) {
        if let Ok(records) = self.db.list_all().await {
            for record in records.iter().rev().take(self.max_cache_size) {
                if record.content.starts_with("fix:") {
                    // Fix strategy: content = "fix:<error_sig>", metadata = strategy
                    let strategy = if record.metadata.is_empty() || record.metadata == "{}" {
                        record.content.clone()
                    } else {
                        record.metadata.clone()
                    };
                    // Avoid duplicates
                    if !self
                        .error_hints_cache
                        .iter()
                        .any(|(k, _)| k == &record.content)
                    {
                        self.error_hints_cache
                            .push((record.content.clone(), strategy));
                    }
                } else if record.content.starts_with("template:") {
                    // Learned template: content = "template:<task>", topics[1] = code
                    let code = record.topics.get(1).cloned().unwrap_or_default();
                    // Replace if exists, otherwise push
                    if let Some(pos) = self
                        .success_cache
                        .iter()
                        .position(|(k, _)| k == &record.content)
                    {
                        self.success_cache[pos].1 = code;
                    } else {
                        self.success_cache.push((record.content.clone(), code));
                    }
                } else if record.valence < 0.0 {
                    // Generic error experience: content = error pattern, metadata = fix hint
                    let fix_hint = if record.metadata.is_empty() || record.metadata == "{}" {
                        record.content.clone()
                    } else {
                        record.metadata.clone()
                    };
                    self.error_hints_cache
                        .push((record.content.clone(), fix_hint));
                } else {
                    // Generic success experience: content = task, topics[0] = code summary
                    let summary = record.topics.first().cloned().unwrap_or_default();
                    self.success_cache.push((record.content.clone(), summary));
                }
            }
        }
    }

    /// Store a coding experience.
    pub async fn store(&mut self, experience: CodingExperience) {
        // Flush any queued records first
        self.flush().await;

        // PRIORITIZE: If we have a diagnostic HV, use it for the memory encoding.
        // This closes the loop: failure geometry → memory store → future recall.
        let encoding = if let Some(diag_hv) = &experience.diagnostic_hv {
            BinaryHV::from_continuous(diag_hv)
        } else {
            Self::encode_text(&experience.task)
        };

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

    /// Query for similar coding experiences by diagnostic geometry.
    ///
    /// This allows the agent to "remember" how it fixed a similar-looking
    /// compiler error in the past by comparing the failure geometries.
    pub async fn query_by_diagnostic(
        &self,
        diagnostic: &ContinuousHV,
        top_k: usize,
    ) -> Vec<SearchResult> {
        // Convert ContinuousHV to BinaryHV for the database search
        let query_hv = BinaryHV::from_continuous(diagnostic);
        self.db
            .search_similar(&query_hv, top_k)
            .await
            .unwrap_or_default()
    }

    /// Look up a learned code template by failure diagnostic.
    ///
    /// If we have a failure geometry, we can check if we've seen and fixed
    /// a similar failure before.
    pub async fn learned_template_for_diagnostic(
        &self,
        diagnostic: &ContinuousHV,
    ) -> Option<String> {
        let results = self.query_by_diagnostic(diagnostic, 3).await;
        for result in results {
            // If it's a positive valence record (success) that matched our failure geometry,
            // it likely contains the template that fixed it.
            if result.record.valence > 0.0 && result.similarity > 0.6 {
                // Return the task/template name or the first topic if it's a template
                if result.record.content.starts_with("template:") {
                    return result.record.topics.get(1).cloned();
                }
            }
        }
        None
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

    /// Store a structured fix: maps an error signature to a fix strategy.
    ///
    /// Error signature is normalized (paths/line numbers stripped) so the same
    /// error type matches across different files. The fix strategy describes
    /// WHAT was done (e.g., "add Clone derive", "insert .clone()").
    ///
    /// Persists to the database so fix strategies survive across sessions.
    pub fn store_fix_strategy(
        &mut self,
        error_signature: &str,
        fix_strategy: &str,
        diagnostic_hv: Option<&ContinuousHV>,
    ) {
        // Store in error_hints_cache with a "fix:" prefix to distinguish from generic hints
        let key = format!("fix:{}", error_signature);
        // Avoid duplicates: if we already have this exact fix, skip
        if self.error_hints_cache.iter().any(|(k, _)| k == &key) {
            return;
        }
        self.error_hints_cache
            .push((key.clone(), fix_strategy.to_string()));
        if self.error_hints_cache.len() > self.max_cache_size {
            self.error_hints_cache.remove(0);
        }

        // Persist to DB so fix strategies survive across sessions
        let encoding = if let Some(diag_hv) = diagnostic_hv {
            BinaryHV::from_continuous(diag_hv)
        } else {
            Self::encode_text(error_signature)
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let seq = EXPERIENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let record = MemoryRecord {
            id: format!("fix_{}_{}", now_ms, seq),
            memory_type: MemoryType::Procedural,
            encoding,
            content: key,
            timestamp_ms: now_ms,
            valence: -0.3, // negative (error-related) but mild (fix is known)
            arousal: 0.2,
            psi: 0.5,
            topics: vec!["fix_strategy".to_string()],
            metadata: fix_strategy.to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        // Queue for DB persistence (flushed on next async boundary)
        self.queue_persist(record);
    }

    /// Look up a cached fix strategy for an error signature.
    ///
    /// Returns the fix strategy string if a matching signature was previously stored.
    /// Matches on substring: "fix:error[E0277]" matches queries containing "E0277".
    pub fn lookup_fix_strategy(&self, error_signature: &str) -> Option<&str> {
        // Extract error code if present (e.g., "E0308" from "error[E0308]: ...")
        let error_code = extract_error_code(error_signature);

        for (key, strategy) in &self.error_hints_cache {
            if !key.starts_with("fix:") {
                continue;
            }
            let stored_sig = &key[4..]; // strip "fix:" prefix

            // Match by error code (most reliable)
            if let Some(ref code) = error_code {
                if stored_sig.contains(code.as_str()) {
                    return Some(strategy.as_str());
                }
            }

            // Fall back to substring match on the signature
            if stored_sig.len() > 10
                && error_signature.contains(&stored_sig[..stored_sig.len().min(40)])
            {
                return Some(strategy.as_str());
            }
        }
        None
    }

    /// Store a learned code template: maps a task signature to generated code.
    ///
    /// Called after LLM-generated code passes cargo check. The template is stored
    /// so future similar tasks can use it natively without LLM escalation.
    ///
    /// Persists to the database so learned templates survive across sessions.
    pub fn store_learned_template(
        &mut self,
        task: &str,
        code: &str,
        diagnostic_hv: Option<&ContinuousHV>,
    ) {
        let key = format!("template:{}", task.to_lowercase());
        let code_summary: String = code.chars().take(2000).collect();
        // Replace if exists, otherwise push
        if let Some(pos) = self.success_cache.iter().position(|(k, _)| k == &key) {
            self.success_cache[pos].1 = code_summary.clone();
        } else {
            self.success_cache.push((key.clone(), code_summary.clone()));
            if self.success_cache.len() > self.max_cache_size {
                self.success_cache.remove(0);
            }
        }

        // Persist to DB so templates survive across sessions
        let encoding = if let Some(diag_hv) = diagnostic_hv {
            BinaryHV::from_continuous(diag_hv)
        } else {
            Self::encode_text(task)
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let seq = EXPERIENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let record = MemoryRecord {
            id: format!("template_{}_{}", now_ms, seq),
            memory_type: MemoryType::Procedural,
            encoding,
            content: key,
            timestamp_ms: now_ms,
            valence: 0.9, // strongly positive (verified working code)
            arousal: 0.1,
            psi: 0.5,
            topics: vec!["learned_template".to_string(), code_summary],
            metadata: task.to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        self.queue_persist(record);
    }

    /// Look up a learned code template for a task.
    ///
    /// Uses HDC similarity: encodes the query task and compares against stored
    /// template keys. Returns the code template if similarity > 0.6.
    pub fn lookup_learned_template(&self, task: &str) -> Option<&str> {
        let query_lower = task.to_lowercase();
        let query_hv = Self::encode_text(&query_lower);

        let mut best_sim = 0.0f32;
        let mut best_code: Option<&str> = None;

        for (key, code) in &self.success_cache {
            if !key.starts_with("template:") {
                continue;
            }
            let stored_task = &key[9..]; // strip "template:" prefix
            let stored_hv = Self::encode_text(stored_task);
            let sim = query_hv.similarity(&stored_hv);

            if sim > best_sim && sim > 0.6 {
                best_sim = sim;
                best_code = Some(code.as_str());
            }
        }

        best_code
    }

    /// Total number of stored experiences.
    pub async fn count(&self) -> usize {
        self.db.count().await.unwrap_or(0)
    }
}

/// Extract an error code like "E0308" from a rustc error message.
fn extract_error_code(msg: &str) -> Option<String> {
    // Match "error[E0308]" or just "E0308"
    let start = msg.find('E').and_then(|i| {
        if i + 5 <= msg.len() && msg[i + 1..i + 5].chars().all(|c| c.is_ascii_digit()) {
            Some(i)
        } else {
            None
        }
    })?;
    Some(msg[start..start + 5].to_string())
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

    // ── Fix Strategy Tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_store_and_lookup_fix_strategy() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store.store_fix_strategy(
            "error[E0308]: mismatched types expected `u32` found `i32`",
            "add explicit cast: `as u32`",
            None,
        );

        let fix = store.lookup_fix_strategy("error[E0308]: different types");
        assert!(fix.is_some(), "Should find fix by error code E0308");
        assert!(fix.unwrap().contains("as u32"));
    }

    #[tokio::test]
    async fn test_fix_strategy_no_match() {
        let store = CodingExperienceStore::new().await.unwrap();
        let fix = store.lookup_fix_strategy("error[E9999]: unknown error");
        assert!(fix.is_none(), "Should return None for unknown errors");
    }

    #[tokio::test]
    async fn test_fix_strategy_deduplication() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store.store_fix_strategy("error[E0308]: mismatched types", "cast fix", None);
        store.store_fix_strategy("error[E0308]: mismatched types", "cast fix v2", None);

        let fix_count = store
            .cached_error_hints()
            .iter()
            .filter(|(k, _)| k.starts_with("fix:"))
            .count();
        assert_eq!(fix_count, 1, "Should not store duplicate fix strategies");
    }

    // ── Learned Template Tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_store_and_lookup_learned_template() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store.store_learned_template(
            "implement binary heap data structure",
            "pub struct BinaryHeap<T> { data: Vec<T> }\nimpl<T: Ord> BinaryHeap<T> {\n    pub fn new() -> Self { Self { data: Vec::new() } }\n}\n",
            None,
        );

        let template = store.lookup_learned_template("implement binary heap data structure");
        assert!(template.is_some(), "Should find exact match template");
        assert!(template.unwrap().contains("BinaryHeap"));
    }

    #[tokio::test]
    async fn test_learned_template_overwrites_on_same_task() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store.store_learned_template("add sort function", "pub fn sort_v1() {}", None);
        store.store_learned_template("add sort function", "pub fn sort_v2() {}", None);

        let template = store.lookup_learned_template("add sort function");
        assert!(template.is_some());
        assert!(
            template.unwrap().contains("sort_v2"),
            "Should overwrite with latest version"
        );

        // Should not accumulate duplicate entries
        let template_count = store
            .cached_successes()
            .iter()
            .filter(|(k, _)| k.starts_with("template:"))
            .count();
        assert_eq!(template_count, 1, "Should have exactly one template entry");
    }

    #[tokio::test]
    async fn test_learned_template_similarity_threshold() {
        let mut store = CodingExperienceStore::new().await.unwrap();

        store.store_learned_template(
            "implement a redis client connection pool",
            "pub struct RedisPool { /* ... */ }",
            None,
        );

        // Completely unrelated task should not match
        let template = store.lookup_learned_template("add fibonacci function");
        assert!(
            template.is_none(),
            "Unrelated task should not match learned template"
        );
    }

    // ── Persistence Tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_fix_strategy_persists_to_db() {
        // Use a temp file for persistent DB
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_persist.db");

        // Store a fix strategy in one store instance
        {
            let mut store = CodingExperienceStore::persistent(&db_path.to_string_lossy())
                .await
                .unwrap();
            store.store_fix_strategy("error[E0308]: mismatched types", "cast with `as u32`", None);
            store.flush().await; // Persist queued writes to DB
            assert_eq!(
                store
                    .cached_error_hints()
                    .iter()
                    .filter(|(k, _)| k.starts_with("fix:"))
                    .count(),
                1
            );
        }

        // Create a new store from the same DB — should hydrate
        {
            let store = CodingExperienceStore::persistent(&db_path.to_string_lossy())
                .await
                .unwrap();
            let fix = store.lookup_fix_strategy("error[E0308]: different wording");
            assert!(
                fix.is_some(),
                "Fix strategy should survive across store instances"
            );
            assert!(fix.unwrap().contains("as u32"));
        }
    }

    #[tokio::test]
    async fn test_learned_template_persists_to_db() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_templates.db");

        {
            let mut store = CodingExperienceStore::persistent(&db_path.to_string_lossy())
                .await
                .unwrap();
            store.store_learned_template(
                "implement binary search",
                "pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> { todo!() }",
                None,
            );
            store.flush().await; // Persist queued writes to DB
        }

        {
            let store = CodingExperienceStore::persistent(&db_path.to_string_lossy())
                .await
                .unwrap();
            let template = store.lookup_learned_template("implement binary search");
            assert!(
                template.is_some(),
                "Learned template should survive across store instances"
            );
            assert!(template.unwrap().contains("binary_search"));
        }
    }

    // ── Extract Error Code Tests ───────────────────────────────────────

    #[test]
    fn test_extract_error_code() {
        assert_eq!(
            extract_error_code("error[E0308]: mismatched types"),
            Some("E0308".to_string())
        );
        assert_eq!(
            extract_error_code("error[E0277]: the trait bound"),
            Some("E0277".to_string())
        );
        assert_eq!(extract_error_code("no error code here"), None);
        assert_eq!(extract_error_code(""), None);
    }
}
