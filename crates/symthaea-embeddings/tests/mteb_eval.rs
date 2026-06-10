// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MTEB-subset evaluation — Retrieval (MRR@10, Recall@5) and Classification tasks.
//!
//! Tier 1 (always): Simulated embeddings — verifies pipeline produces finite metrics.
//! Tier 2 (`burn-hub`, `#[ignore]`): Real model — asserts quality thresholds.

// ── Metric helpers ─────────────────────────────────────────────────────

/// Mean Reciprocal Rank at k: 1/rank of the first relevant document in top-k,
/// or 0.0 if none are relevant.
fn compute_mrr_at_k(ranked_relevance: &[bool], k: usize) -> f32 {
    for (i, &relevant) in ranked_relevance.iter().take(k).enumerate() {
        if relevant {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

/// Recall at k: fraction of total relevant documents appearing in top-k.
fn compute_recall_at_k(ranked_relevance: &[bool], k: usize, total_relevant: usize) -> f32 {
    if total_relevant == 0 {
        return 0.0;
    }
    let found = ranked_relevance.iter().take(k).filter(|&&r| r).count();
    found as f32 / total_relevant as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

// ── Retrieval dataset ──────────────────────────────────────────────────

struct RetrievalQuery {
    query: &'static str,
    /// Indices into the document corpus that are relevant.
    relevant_doc_indices: Vec<usize>,
}

fn retrieval_corpus() -> Vec<&'static str> {
    vec![
        // Science (0-3)
        "Photosynthesis converts sunlight into chemical energy in plants",
        "DNA carries the genetic instructions for all living organisms",
        "The theory of relativity describes gravity as spacetime curvature",
        "Quantum entanglement links particles regardless of distance",
        // Technology (4-7)
        "Machine learning models learn patterns from training data",
        "HTTPS encrypts web traffic between browser and server",
        "Containerization packages applications with their dependencies",
        "Version control systems track changes to source code over time",
        // History (8-11)
        "The Renaissance began in Italy during the 14th century",
        "Ancient Rome developed extensive road networks across Europe",
        "The Industrial Revolution started in 18th century Britain",
        "The printing press revolutionized the spread of information",
        // Food (12-15)
        "Sourdough bread uses wild yeast for fermentation",
        "Sushi originated in Southeast Asia as a preservation method",
        "Chocolate is made from roasted cacao beans",
        "Pasta has been a staple food in Italian cuisine for centuries",
        // Geography (16-19)
        "The Sahara is the largest hot desert in the world",
        "Mount Everest is the highest peak above sea level",
        "The Amazon River carries more water than any other river",
        "Antarctica contains about 70 percent of Earth's fresh water",
    ]
}

fn retrieval_queries() -> Vec<RetrievalQuery> {
    vec![
        RetrievalQuery {
            query: "How do plants make food from sunlight?",
            relevant_doc_indices: vec![0],
        },
        RetrievalQuery {
            query: "What is genetic material made of?",
            relevant_doc_indices: vec![1],
        },
        RetrievalQuery {
            query: "How do neural networks work?",
            relevant_doc_indices: vec![4],
        },
        RetrievalQuery {
            query: "What happened during the European cultural rebirth?",
            relevant_doc_indices: vec![8],
        },
        RetrievalQuery {
            query: "What are the tallest mountains on Earth?",
            relevant_doc_indices: vec![17],
        },
    ]
}

// ── Classification dataset ─────────────────────────────────────────────

struct ClassificationData {
    texts: Vec<&'static str>,
    labels: Vec<&'static str>,
}

fn classification_dataset() -> ClassificationData {
    let texts = vec![
        // Science (4 texts)
        "The mitochondria is the powerhouse of the cell",
        "Black holes form when massive stars collapse",
        "Enzymes catalyze biochemical reactions in organisms",
        "The periodic table organizes elements by atomic number",
        // Sports (4 texts)
        "The World Cup is the biggest football tournament",
        "Basketball was invented by James Naismith in 1891",
        "Olympic athletes train for years before competing",
        "Tennis requires both physical fitness and mental focus",
        // Technology (4 texts)
        "Artificial intelligence can recognize images and speech",
        "Blockchain provides a decentralized transaction ledger",
        "Cloud computing delivers services over the internet",
        "Cybersecurity protects systems from digital attacks",
        // Nature (4 texts)
        "Coral reefs support diverse marine ecosystems",
        "Migration patterns of birds span thousands of miles",
        "Rainforests produce a significant portion of oxygen",
        "Volcanoes shape landscapes through eruption and erosion",
        // Music (4 texts)
        "Jazz originated in New Orleans in the early 20th century",
        "Classical symphonies typically have four movements",
        "The guitar is one of the most popular instruments worldwide",
        "Hip hop culture includes rapping, DJing, and breakdancing",
    ];

    let labels = vec![
        "science",
        "science",
        "science",
        "science",
        "sports",
        "sports",
        "sports",
        "sports",
        "technology",
        "technology",
        "technology",
        "technology",
        "nature",
        "nature",
        "nature",
        "nature",
        "music",
        "music",
        "music",
        "music",
    ];

    ClassificationData { texts, labels }
}

/// Nearest-centroid classifier: computes category centroids from training set,
/// then classifies test items by cosine similarity to centroids.
fn nearest_centroid_classify(
    train_embeddings: &[Vec<f32>],
    train_labels: &[&str],
    test_embeddings: &[Vec<f32>],
) -> Vec<String> {
    use std::collections::HashMap;

    let dim = train_embeddings[0].len();

    // Build centroids per label
    let mut centroid_sum: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut centroid_count: HashMap<&str, usize> = HashMap::new();

    for (emb, &label) in train_embeddings.iter().zip(train_labels) {
        let entry = centroid_sum.entry(label).or_insert_with(|| vec![0.0; dim]);
        for (s, &e) in entry.iter_mut().zip(emb) {
            *s += e;
        }
        *centroid_count.entry(label).or_insert(0) += 1;
    }

    // Average
    let centroids: HashMap<&str, Vec<f32>> = centroid_sum
        .into_iter()
        .map(|(label, mut sum)| {
            let count = centroid_count[label] as f32;
            for v in &mut sum {
                *v /= count;
            }
            (label, sum)
        })
        .collect();

    // Classify test items
    test_embeddings
        .iter()
        .map(|emb| {
            centroids
                .iter()
                .map(|(label, centroid)| (*label, cosine(emb, centroid)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(label, _)| label.to_string())
                .unwrap_or_default()
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_retrieval_mrr_known_values() {
    // Perfect ranking: first doc is relevant
    assert!((compute_mrr_at_k(&[true, false, false], 10) - 1.0).abs() < 1e-6);
    // Second position
    assert!((compute_mrr_at_k(&[false, true, false], 10) - 0.5).abs() < 1e-6);
    // Third position
    assert!((compute_mrr_at_k(&[false, false, true], 10) - 1.0 / 3.0).abs() < 1e-6);
    // No relevant in top-k
    assert_eq!(compute_mrr_at_k(&[false, false, false], 3), 0.0);
    // k limits search
    assert_eq!(compute_mrr_at_k(&[false, false, true], 2), 0.0);

    // Recall
    assert!((compute_recall_at_k(&[true, false, true], 3, 2) - 1.0).abs() < 1e-6);
    assert!((compute_recall_at_k(&[true, false, false], 3, 2) - 0.5).abs() < 1e-6);
    assert_eq!(compute_recall_at_k(&[false, false, false], 3, 0), 0.0);
}

#[test]
fn test_retrieval_pipeline_simulated() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let mut embedder = Qwen3Embedder::new(Qwen3Config::simulated()).unwrap();
    let corpus = retrieval_corpus();
    let queries = retrieval_queries();

    // Embed all documents
    let doc_embeddings: Vec<Vec<f32>> = corpus
        .iter()
        .map(|doc| embedder.embed(doc).unwrap().embedding)
        .collect();

    let mut mrr_sum = 0.0f32;
    let mut recall_sum = 0.0f32;

    for q in &queries {
        let query_emb = embedder.embed(q.query).unwrap().embedding;

        // Rank documents by cosine similarity (descending)
        let mut scored: Vec<(usize, f32)> = doc_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| (i, cosine(&query_emb, doc_emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ranked_relevance: Vec<bool> = scored
            .iter()
            .map(|(idx, _)| q.relevant_doc_indices.contains(idx))
            .collect();

        let mrr = compute_mrr_at_k(&ranked_relevance, 10);
        let recall = compute_recall_at_k(&ranked_relevance, 5, q.relevant_doc_indices.len());
        mrr_sum += mrr;
        recall_sum += recall;
    }

    let avg_mrr = mrr_sum / queries.len() as f32;
    let avg_recall = recall_sum / queries.len() as f32;

    eprintln!("Retrieval (simulated): MRR@10={avg_mrr:.4}, Recall@5={avg_recall:.4}");

    // Pipeline correctness: metrics should be finite and in valid range
    assert!(avg_mrr.is_finite(), "MRR@10 should be finite");
    assert!(avg_recall.is_finite(), "Recall@5 should be finite");
    assert!(
        avg_mrr >= 0.0 && avg_mrr <= 1.0,
        "MRR@10 out of range: {avg_mrr}"
    );
    assert!(
        avg_recall >= 0.0 && avg_recall <= 1.0,
        "Recall@5 out of range: {avg_recall}"
    );
}

#[test]
fn test_classification_pipeline_simulated() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let mut embedder = Qwen3Embedder::new(Qwen3Config::simulated()).unwrap();
    let dataset = classification_dataset();

    // Embed all texts
    let embeddings: Vec<Vec<f32>> = dataset
        .texts
        .iter()
        .map(|t| embedder.embed(t).unwrap().embedding)
        .collect();

    // Split: first 3 per category = train (indices 0,1,2,4,5,6,...), last = test
    let mut train_embs = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_embs = Vec::new();
    let mut test_labels = Vec::new();

    let categories = ["science", "sports", "technology", "nature", "music"];
    for (cat_idx, _cat) in categories.iter().enumerate() {
        let base = cat_idx * 4;
        for offset in 0..3 {
            train_embs.push(embeddings[base + offset].clone());
            train_labels.push(dataset.labels[base + offset]);
        }
        test_embs.push(embeddings[base + 3].clone());
        test_labels.push(dataset.labels[base + 3]);
    }

    let predictions = nearest_centroid_classify(&train_embs, &train_labels, &test_embs);

    let correct = predictions
        .iter()
        .zip(test_labels.iter())
        .filter(|(pred, gold)| pred.as_str() == **gold)
        .count();
    let accuracy = correct as f32 / test_labels.len() as f32;

    eprintln!(
        "Classification (simulated): accuracy={accuracy:.2} ({correct}/{} correct)",
        test_labels.len()
    );

    // Pipeline correctness: accuracy should be finite and in valid range
    assert!(accuracy.is_finite(), "Accuracy should be finite");
    assert!(
        accuracy >= 0.0 && accuracy <= 1.0,
        "Accuracy out of range: {accuracy}"
    );
}

#[cfg(feature = "burn-hub")]
#[test]
#[ignore]
fn test_retrieval_real_model() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
    let mut embedder = Qwen3Embedder::new(config).unwrap();
    assert!(
        embedder.is_model_loaded(),
        "Model should be loaded from hub"
    );

    let corpus = retrieval_corpus();
    let queries = retrieval_queries();

    let doc_embeddings: Vec<Vec<f32>> = corpus
        .iter()
        .map(|doc| embedder.embed(doc).unwrap().embedding)
        .collect();

    let mut mrr_sum = 0.0f32;

    for q in &queries {
        let query_emb = embedder.embed(q.query).unwrap().embedding;

        let mut scored: Vec<(usize, f32)> = doc_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| (i, cosine(&query_emb, doc_emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ranked_relevance: Vec<bool> = scored
            .iter()
            .map(|(idx, _)| q.relevant_doc_indices.contains(idx))
            .collect();

        let mrr = compute_mrr_at_k(&ranked_relevance, 10);
        mrr_sum += mrr;
        eprintln!("  Q: \"{}\" → MRR@10={mrr:.4}", q.query);
    }

    let avg_mrr = mrr_sum / queries.len() as f32;
    eprintln!("Retrieval (real model): MRR@10={avg_mrr:.4}");

    assert!(
        avg_mrr > 0.5,
        "Real model should achieve MRR@10 > 0.5, got {avg_mrr:.4}"
    );
}

#[cfg(feature = "burn-hub")]
#[test]
#[ignore]
fn test_classification_real_model() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
    let mut embedder = Qwen3Embedder::new(config).unwrap();
    assert!(
        embedder.is_model_loaded(),
        "Model should be loaded from hub"
    );

    let dataset = classification_dataset();

    let embeddings: Vec<Vec<f32>> = dataset
        .texts
        .iter()
        .map(|t| embedder.embed(t).unwrap().embedding)
        .collect();

    let mut train_embs = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_embs = Vec::new();
    let mut test_labels = Vec::new();

    let categories = ["science", "sports", "technology", "nature", "music"];
    for (cat_idx, _cat) in categories.iter().enumerate() {
        let base = cat_idx * 4;
        for offset in 0..3 {
            train_embs.push(embeddings[base + offset].clone());
            train_labels.push(dataset.labels[base + offset]);
        }
        test_embs.push(embeddings[base + 3].clone());
        test_labels.push(dataset.labels[base + 3]);
    }

    let predictions = nearest_centroid_classify(&train_embs, &train_labels, &test_embs);

    let correct = predictions
        .iter()
        .zip(test_labels.iter())
        .filter(|(pred, gold)| pred.as_str() == **gold)
        .count();
    let accuracy = correct as f32 / test_labels.len() as f32;

    eprintln!("Classification (real model): accuracy={accuracy:.2}");

    assert!(
        accuracy > 0.6,
        "Real model should achieve accuracy > 60%, got {accuracy:.2}"
    );
}
