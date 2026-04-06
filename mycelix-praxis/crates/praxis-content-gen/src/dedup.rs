// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Semantic Deduplication (SemDeDup)
//!
//! Detects and removes conceptually duplicate content from generated batches
//! using embedding-based cosine similarity. Keyword/lexical deduplication misses
//! paraphrased duplicates — this catches them via semantic vector space.
//!
//! ## Algorithm
//!
//! 1. Extract representative text from each content item (explanation + first problem)
//! 2. Embed via Ollama embeddinggemma:300m (HTTP API on localhost:11434)
//! 3. Pairwise cosine similarity on 256D embedding vectors
//! 4. Threshold at 0.90 — mark later-generated item as duplicate
//! 5. Return kept items + removal log
//!
//! ## Usage
//!
//! ```ignore
//! let config = SemDeDupConfig::default();
//! let result = dedup_texts(&texts, &config).await?;
//! println!("Kept: {}, Removed: {}", result.kept_indices.len(), result.removed.len());
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for semantic deduplication.
#[derive(Clone, Debug)]
pub struct SemDeDupConfig {
    /// Cosine similarity threshold above which items are considered duplicates.
    /// Default: 0.90 (very high similarity = near-identical meaning)
    pub similarity_threshold: f32,

    /// Ollama API endpoint (default: http://127.0.0.1:11434)
    pub ollama_url: String,

    /// Embedding model to use (default: embeddinggemma:300m)
    pub model: String,
}

impl Default for SemDeDupConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.90,
            ollama_url: "http://127.0.0.1:11434".to_string(),
            model: "embeddinggemma:300m".to_string(),
        }
    }
}

/// Result of semantic deduplication.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DedupResult {
    /// Indices of items kept (not duplicates)
    pub kept_indices: Vec<usize>,
    /// Removal log: (removed_index, duplicate_of_index, cosine_similarity)
    pub removed: Vec<(usize, usize, f32)>,
    /// Statistics
    pub stats: DedupStats,
}

/// Deduplication statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DedupStats {
    pub total_items: usize,
    pub items_kept: usize,
    pub items_removed: usize,
    pub avg_similarity: f32,
    pub max_similarity: f32,
    pub embedding_failures: usize,
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Deduplicate a set of texts using pre-computed embeddings.
///
/// This is the core algorithm — provider-agnostic. Pass embeddings from any
/// source (Ollama, Qwen3, etc.).
pub fn dedup_with_embeddings(
    embeddings: &[Vec<f32>],
    config: &SemDeDupConfig,
) -> DedupResult {
    let n = embeddings.len();
    let mut kept = vec![true; n];
    let mut removed = Vec::new();
    let mut similarities = Vec::new();

    // Pairwise comparison — O(n²) but fine for batches < 500
    for i in 0..n {
        if !kept[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !kept[j] {
                continue;
            }
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            similarities.push(sim);
            if sim >= config.similarity_threshold {
                // Mark the later item as duplicate
                kept[j] = false;
                removed.push((j, i, sim));
            }
        }
    }

    let kept_indices: Vec<usize> = (0..n).filter(|i| kept[*i]).collect();
    let avg_sim = if similarities.is_empty() {
        0.0
    } else {
        similarities.iter().sum::<f32>() / similarities.len() as f32
    };
    let max_sim = similarities.iter().cloned().fold(0.0_f32, f32::max);

    DedupResult {
        kept_indices: kept_indices.clone(),
        removed: removed.clone(),
        stats: DedupStats {
            total_items: n,
            items_kept: kept_indices.len(),
            items_removed: removed.len(),
            avg_similarity: avg_sim,
            max_similarity: max_sim,
            embedding_failures: 0,
        },
    }
}

/// Embed texts via Ollama HTTP API and deduplicate.
///
/// Requires the `dedup` feature and Ollama running on localhost:11434.
#[cfg(feature = "dedup")]
pub async fn dedup_texts(
    texts: &[String],
    config: &SemDeDupConfig,
) -> Result<DedupResult, String> {
    let client = reqwest::Client::new();
    let mut embeddings = Vec::with_capacity(texts.len());
    let mut failures = 0usize;

    for text in texts {
        match embed_via_ollama(&client, text, config).await {
            Ok(emb) => embeddings.push(emb),
            Err(_) => {
                // On failure, use a zero vector (will not match anything)
                embeddings.push(vec![0.0; 256]);
                failures += 1;
            }
        }
    }

    let mut result = dedup_with_embeddings(&embeddings, config);
    result.stats.embedding_failures = failures;
    Ok(result)
}

/// Call Ollama embedding API for a single text.
#[cfg(feature = "dedup")]
async fn embed_via_ollama(
    client: &reqwest::Client,
    text: &str,
    config: &SemDeDupConfig,
) -> Result<Vec<f32>, String> {
    #[derive(Serialize)]
    struct OllamaEmbedRequest<'a> {
        model: &'a str,
        input: &'a str,
    }

    #[derive(Deserialize)]
    struct OllamaEmbedResponse {
        embeddings: Vec<Vec<f32>>,
    }

    let resp = client
        .post(format!("{}/api/embed", config.ollama_url))
        .json(&OllamaEmbedRequest {
            model: &config.model,
            input: text,
        })
        .send()
        .await
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Ollama returned {}", resp.status()));
    }

    let body: OllamaEmbedResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

    body.embeddings
        .into_iter()
        .next()
        .ok_or_else(|| "No embeddings returned".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_identical_vectors() {
        let emb = vec![vec![1.0, 2.0, 3.0]; 5]; // 5 identical vectors
        let config = SemDeDupConfig {
            similarity_threshold: 0.90,
            ..Default::default()
        };
        let result = dedup_with_embeddings(&emb, &config);
        assert_eq!(result.stats.items_kept, 1);
        assert_eq!(result.stats.items_removed, 4);
    }

    #[test]
    fn test_dedup_all_unique() {
        let emb = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let config = SemDeDupConfig {
            similarity_threshold: 0.90,
            ..Default::default()
        };
        let result = dedup_with_embeddings(&emb, &config);
        assert_eq!(result.stats.items_kept, 3);
        assert_eq!(result.stats.items_removed, 0);
    }

    #[test]
    fn test_dedup_mixed() {
        let emb = vec![
            vec![1.0, 0.0, 0.0],      // unique
            vec![0.99, 0.01, 0.0],     // near-duplicate of [0]
            vec![0.0, 1.0, 0.0],      // unique
            vec![0.0, 0.98, 0.02],     // near-duplicate of [2]
        ];
        let config = SemDeDupConfig {
            similarity_threshold: 0.99,
            ..Default::default()
        };
        let result = dedup_with_embeddings(&emb, &config);
        // With threshold 0.99, the near-duplicates (0.99+ similarity) should be caught
        // but the orthogonal pairs should not
        assert!(result.stats.items_kept >= 2);
    }

    #[test]
    fn test_dedup_empty() {
        let emb: Vec<Vec<f32>> = vec![];
        let config = SemDeDupConfig::default();
        let result = dedup_with_embeddings(&emb, &config);
        assert_eq!(result.stats.items_kept, 0);
        assert_eq!(result.stats.items_removed, 0);
    }
}
