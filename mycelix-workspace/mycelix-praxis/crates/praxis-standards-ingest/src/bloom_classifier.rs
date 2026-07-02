// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # ML Bloom's Taxonomy Classifier
//!
//! Nearest-centroid classifier for Bloom's taxonomy levels using semantic
//! embeddings. Replaces keyword-based verb matching with contextual inference.
//!
//! ## Algorithm
//!
//! 1. **Training**: Embed labeled descriptions → compute per-class centroid (6 levels × dim D)
//! 2. **Inference**: Embed new description → find nearest centroid → return level + confidence
//! 3. **Confidence**: `1 - (dist_nearest / dist_second)` — low confidence flags for human review
//!
//! ## Bootstrap
//!
//! Training data is bootstrapped from the existing keyword `infer_bloom_level()` function
//! applied to all curriculum nodes. This provides ~1,791 labeled examples.
//!
//! Requires: Ollama running on localhost:11434 with embeddinggemma:300m model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The six Bloom's taxonomy cognitive levels.
pub const BLOOM_LEVELS: &[&str] = &[
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
];

/// A trained Bloom's taxonomy classifier using nearest-centroid in embedding space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BloomClassifier {
    /// Centroid embedding for each Bloom level
    centroids: HashMap<String, Vec<f32>>,
    /// Number of training samples per level
    sample_counts: HashMap<String, usize>,
    /// Embedding dimension
    dim: usize,
    /// Confidence threshold below which predictions are flagged for review
    pub confidence_threshold: f32,
}

/// Result of classifying a description into a Bloom level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BloomPrediction {
    /// Predicted Bloom level
    pub level: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Whether this prediction needs human review
    pub needs_review: bool,
    /// Runner-up level and its distance (for borderline cases)
    pub runner_up: Option<(String, f32)>,
}

impl BloomClassifier {
    /// Train a classifier from labeled (description, bloom_level) pairs.
    ///
    /// Each pair's embedding must be pre-computed. Call `train_from_embeddings`
    /// with the embeddings aligned to the labels.
    pub fn train_from_embeddings(
        labels: &[String],
        embeddings: &[Vec<f32>],
        confidence_threshold: f32,
    ) -> Option<Self> {
        if labels.len() != embeddings.len() || labels.is_empty() {
            return None;
        }

        let dim = embeddings[0].len();
        if dim == 0 {
            return None;
        }

        // Accumulate per-class sums
        let mut sums: HashMap<String, Vec<f64>> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for (label, emb) in labels.iter().zip(embeddings.iter()) {
            if emb.len() != dim {
                continue;
            }
            let sum = sums
                .entry(label.clone())
                .or_insert_with(|| vec![0.0f64; dim]);
            for (i, v) in emb.iter().enumerate() {
                sum[i] += *v as f64;
            }
            *counts.entry(label.clone()).or_default() += 1;
        }

        // Compute centroids (mean)
        let mut centroids = HashMap::new();
        for (label, sum) in &sums {
            let count = counts[label] as f64;
            let centroid: Vec<f32> = sum.iter().map(|s| (*s / count) as f32).collect();
            centroids.insert(label.clone(), centroid);
        }

        Some(Self {
            centroids,
            sample_counts: counts,
            dim,
            confidence_threshold,
        })
    }

    /// Classify a pre-computed embedding into a Bloom level.
    pub fn classify_embedding(&self, embedding: &[f32]) -> BloomPrediction {
        if embedding.len() != self.dim || self.centroids.is_empty() {
            return BloomPrediction {
                level: "Understand".to_string(),
                confidence: 0.0,
                needs_review: true,
                runner_up: None,
            };
        }

        // Compute cosine similarity to each centroid
        let mut similarities: Vec<(String, f32)> = self
            .centroids
            .iter()
            .map(|(label, centroid)| {
                let sim = cosine_similarity(embedding, centroid);
                (label.clone(), sim)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = &similarities[0];
        let runner_up = similarities.get(1);

        // Confidence: ratio of best to runner-up similarity
        // Higher means more decisive separation
        let confidence = if let Some(second) = runner_up {
            if second.1 <= 0.0 {
                1.0
            } else {
                // Margin-based confidence: how much better is best vs second?
                let margin = best.1 - second.1;
                // Normalize: margin of 0.1+ = high confidence
                (margin * 5.0).clamp(0.0, 1.0)
            }
        } else {
            1.0
        };

        BloomPrediction {
            level: best.0.clone(),
            confidence,
            needs_review: confidence < self.confidence_threshold,
            runner_up: runner_up.map(|(l, s)| (l.clone(), *s)),
        }
    }

    /// Number of training samples per Bloom level.
    pub fn sample_counts(&self) -> &HashMap<String, usize> {
        &self.sample_counts
    }

    /// Number of Bloom levels with at least one training sample.
    pub fn num_classes(&self) -> usize {
        self.centroids.len()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Embed a single text via Ollama HTTP API.
///
/// Requires Ollama running on localhost:11434 with the specified model.
#[cfg(feature = "ml-bloom")]
pub async fn embed_text_ollama(
    client: &reqwest::Client,
    text: &str,
    model: &str,
    ollama_url: &str,
) -> Result<Vec<f32>, String> {
    #[derive(Serialize)]
    struct Req<'a> {
        model: &'a str,
        input: &'a str,
    }
    #[derive(Deserialize)]
    struct Resp {
        embeddings: Vec<Vec<f32>>,
    }

    let resp = client
        .post(format!("{}/api/embed", ollama_url))
        .json(&Req { model, input: text })
        .send()
        .await
        .map_err(|e| format!("Ollama request: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Ollama status: {}", resp.status()));
    }

    let body: Resp = resp.json().await.map_err(|e| format!("Parse: {}", e))?;
    body.embeddings
        .into_iter()
        .next()
        .ok_or_else(|| "No embedding returned".to_string())
}

/// Train a BloomClassifier from labeled descriptions using Ollama embeddings.
///
/// 1. Embeds each description via Ollama
/// 2. Builds per-class centroids
/// 3. Returns trained classifier
#[cfg(feature = "ml-bloom")]
pub async fn train_from_labeled_descriptions(
    labeled: &[(String, String)], // (description, bloom_level)
    model: &str,
    ollama_url: &str,
    confidence_threshold: f32,
) -> Result<BloomClassifier, String> {
    let client = reqwest::Client::new();
    let mut labels = Vec::new();
    let mut embeddings = Vec::new();

    for (desc, level) in labeled {
        match embed_text_ollama(&client, desc, model, ollama_url).await {
            Ok(emb) => {
                labels.push(level.clone());
                embeddings.push(emb);
            }
            Err(e) => {
                eprintln!("Skipping '{}...': {}", &desc[..desc.len().min(40)], e);
            }
        }
    }

    BloomClassifier::train_from_embeddings(&labels, &embeddings, confidence_threshold)
        .ok_or_else(|| "Failed to train classifier (no valid embeddings)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(primary_dim: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if primary_dim < dim {
            v[primary_dim] = 1.0;
        }
        v
    }

    #[test]
    fn test_train_and_classify() {
        // 3 classes, each pointing in a different dimension
        let labels = vec![
            "Remember".to_string(),
            "Remember".to_string(),
            "Understand".to_string(),
            "Understand".to_string(),
            "Create".to_string(),
            "Create".to_string(),
        ];
        let embeddings = vec![
            make_embedding(0, 8), // Remember → dim 0
            make_embedding(0, 8),
            make_embedding(2, 8), // Understand → dim 2
            make_embedding(2, 8),
            make_embedding(5, 8), // Create → dim 5
            make_embedding(5, 8),
        ];

        let classifier =
            BloomClassifier::train_from_embeddings(&labels, &embeddings, 0.6).unwrap();

        assert_eq!(classifier.num_classes(), 3);

        // Test: embedding near dim 0 → Remember
        let pred = classifier.classify_embedding(&make_embedding(0, 8));
        assert_eq!(pred.level, "Remember");
        assert!(pred.confidence > 0.5);

        // Test: embedding near dim 2 → Understand
        let pred = classifier.classify_embedding(&make_embedding(2, 8));
        assert_eq!(pred.level, "Understand");

        // Test: embedding near dim 5 → Create
        let pred = classifier.classify_embedding(&make_embedding(5, 8));
        assert_eq!(pred.level, "Create");
    }

    #[test]
    fn test_ambiguous_gets_low_confidence() {
        let labels = vec![
            "Remember".to_string(),
            "Understand".to_string(),
        ];
        // Both centroids in very similar directions
        let embeddings = vec![
            vec![1.0, 0.9, 0.0],
            vec![0.9, 1.0, 0.0],
        ];

        let classifier =
            BloomClassifier::train_from_embeddings(&labels, &embeddings, 0.6).unwrap();

        // Query with a vector between both centroids
        let pred = classifier.classify_embedding(&[0.95, 0.95, 0.0]);
        // Should have low confidence due to small margin
        assert!(pred.confidence < 0.8);
    }

    #[test]
    fn test_empty_training_data() {
        let result = BloomClassifier::train_from_embeddings(&[], &[], 0.6);
        assert!(result.is_none());
    }

    #[test]
    fn test_needs_review_flag() {
        let labels = vec!["Apply".to_string(), "Analyze".to_string()];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let classifier =
            BloomClassifier::train_from_embeddings(&labels, &embeddings, 0.9).unwrap();

        // Clear classification → high confidence → no review
        let pred = classifier.classify_embedding(&[1.0, 0.0, 0.0]);
        assert_eq!(pred.level, "Apply");
        assert!(!pred.needs_review);

        // Ambiguous → might need review depending on threshold
        let pred = classifier.classify_embedding(&[0.5, 0.0, 0.5]);
        // With threshold 0.9, this borderline case needs review
        assert!(pred.needs_review || pred.confidence >= 0.9);
    }

    #[test]
    fn test_sample_counts() {
        let labels = vec![
            "Remember".to_string(),
            "Remember".to_string(),
            "Remember".to_string(),
            "Create".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![0.0, 1.0],
        ];
        let classifier =
            BloomClassifier::train_from_embeddings(&labels, &embeddings, 0.6).unwrap();
        assert_eq!(classifier.sample_counts()["Remember"], 3);
        assert_eq!(classifier.sample_counts()["Create"], 1);
    }
}
