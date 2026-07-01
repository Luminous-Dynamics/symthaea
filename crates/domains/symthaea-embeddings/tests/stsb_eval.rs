// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STS-B quality evaluation — measures embedding quality via sentence similarity
//! and Spearman rank correlation.
//!
//! Tier 1 (always available): 15 hand-crafted sentence pairs with gold scores.
//! Tier 2 (`burn-hub`, `#[ignore]`): Same pairs with real model, assert rho > 0.6.

/// Compute Spearman rank correlation coefficient.
fn spearman_rho(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f32;
    let rank_x = rank(x);
    let rank_y = rank(y);
    let d_sq: f32 = rank_x
        .iter()
        .zip(&rank_y)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    1.0 - (6.0 * d_sq) / (n * (n * n - 1.0))
}

/// Assign ranks to values (average rank for ties).
fn rank(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all tied values
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        // Average rank for the tied group (1-based)
        let avg_rank = (i + 1 + j) as f32 / 2.0;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

/// Hand-crafted sentence pairs with gold similarity scores (0.0-5.0 scale).
fn sentence_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        // High similarity (>4.0)
        (
            "A dog runs in the park",
            "A puppy runs through the park",
            4.5,
        ),
        ("A woman is playing piano", "A lady plays the piano", 4.3),
        (
            "The cat is sleeping on the couch",
            "A cat sleeps on the sofa",
            4.6,
        ),
        ("A man is riding a bicycle", "A person is cycling", 4.2),
        ("Children are playing outside", "Kids play outdoors", 4.4),
        // Medium similarity (2.0-3.5)
        ("A woman is cooking dinner", "A man is eating food", 2.5),
        ("The dog is in the garden", "A cat sits on the fence", 2.0),
        (
            "A man reads a newspaper",
            "A person watches television",
            2.2,
        ),
        (
            "It is raining heavily outside",
            "The weather is cold today",
            2.8,
        ),
        (
            "A plane flies over the city",
            "A bird soars above the trees",
            3.0,
        ),
        // Low similarity (<1.5)
        ("A dog is running", "The stock market crashed", 0.5),
        (
            "She is singing a song",
            "The bridge is under construction",
            0.3,
        ),
        (
            "A boy plays with a ball",
            "The scientific method requires evidence",
            0.4,
        ),
        (
            "The flowers bloom in spring",
            "A computer program has a bug",
            0.2,
        ),
        (
            "A fish swims in the ocean",
            "The political debate was heated",
            0.6,
        ),
    ]
}

#[test]
fn test_spearman_known_values() {
    // Perfect positive correlation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let rho = spearman_rho(&x, &y);
    assert!(
        (rho - 1.0).abs() < 1e-6,
        "Perfect positive should give rho=1.0, got {rho}"
    );

    // Perfect negative correlation
    let y_inv = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let rho_inv = spearman_rho(&x, &y_inv);
    assert!(
        (rho_inv - (-1.0)).abs() < 1e-6,
        "Perfect negative should give rho=-1.0, got {rho_inv}"
    );

    // No correlation (well-known example)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![3.0, 1.0, 5.0, 2.0, 4.0];
    let rho_mixed = spearman_rho(&a, &b);
    assert!(
        rho_mixed.abs() < 0.9,
        "Mixed values should have low correlation, got {rho_mixed}"
    );
}

#[test]
fn test_stsb_simulated_pipeline() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let mut embedder = Qwen3Embedder::new(Qwen3Config::simulated()).unwrap();
    let pairs = sentence_pairs();

    let mut predicted_sims = Vec::with_capacity(pairs.len());
    let mut gold_scores = Vec::with_capacity(pairs.len());

    for (text_a, text_b, gold) in &pairs {
        let emb_a = embedder.embed(text_a).unwrap();
        let emb_b = embedder.embed(text_b).unwrap();
        let sim = cosine(&emb_a.embedding, &emb_b.embedding);
        predicted_sims.push(sim);
        gold_scores.push(*gold);
    }

    let rho = spearman_rho(&predicted_sims, &gold_scores);
    eprintln!("STS-B simulated pipeline: Spearman rho = {rho:.4}");

    // Simulated embeddings won't achieve high correlation — just verify
    // the pipeline produces a finite, reasonable value
    assert!(rho.is_finite(), "Spearman rho should be finite");
    assert!(
        rho > -1.1 && rho < 1.1,
        "Spearman rho should be in [-1, 1], got {rho}"
    );
}

#[cfg(feature = "burn-hub")]
#[test]
#[ignore]
fn test_stsb_real_model_quality() {
    use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

    let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
    let mut embedder = Qwen3Embedder::new(config).unwrap();
    assert!(
        embedder.is_model_loaded(),
        "Model should be loaded from hub"
    );

    let pairs = sentence_pairs();

    let mut predicted_sims = Vec::with_capacity(pairs.len());
    let mut gold_scores = Vec::with_capacity(pairs.len());

    for (text_a, text_b, gold) in &pairs {
        let emb_a = embedder.embed(text_a).unwrap();
        let emb_b = embedder.embed(text_b).unwrap();
        let sim = cosine(&emb_a.embedding, &emb_b.embedding);
        predicted_sims.push(sim);
        gold_scores.push(*gold);
        eprintln!("  \"{text_a}\" vs \"{text_b}\": sim={sim:.4}, gold={gold}");
    }

    let rho = spearman_rho(&predicted_sims, &gold_scores);
    eprintln!("STS-B real model: Spearman rho = {rho:.4}");

    assert!(
        rho > 0.6,
        "Real model should achieve Spearman rho > 0.6, got {rho:.4}"
    );
}
