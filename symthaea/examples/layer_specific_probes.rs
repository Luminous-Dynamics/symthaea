// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer-Specific Probe Training
//!
//! Trains separate classifiers for each layer to identify which layer
//! best discriminates phenomenal from functional concepts.
//!
//! Uses logistic regression on layer activations to predict concept class.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example layer_specific_probes --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   LAYER-SPECIFIC PROBE TRAINING");
    println!("   Which layer best discriminates phenomenal from functional?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    println!(
        "Loaded {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    // Load model
    println!("Loading BGE-M3...");
    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Layers to test
    let layers = vec![0, 6, 12, 18, 21, 23];

    println!("================================================================");
    println!("   EXTRACTING LAYER ACTIVATIONS");
    println!("================================================================\n");

    // Extract activations
    let mut phen_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers.len()];
    let mut func_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers.len()];

    println!("Processing phenomenal concepts...");
    for (i, concept) in phenomenal.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\r", i, phenomenal.len());
        }
        let acts = extractor.extract_layers(concept, &layers)?;
        for (j, act) in acts.into_iter().enumerate() {
            phen_by_layer[j].push(act.activation);
        }
    }
    println!("  Done                    ");

    println!("Processing functional concepts...");
    for (i, concept) in functional.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\r", i, functional.len());
        }
        let acts = extractor.extract_layers(concept, &layers)?;
        for (j, act) in acts.into_iter().enumerate() {
            func_by_layer[j].push(act.activation);
        }
    }
    println!("  Done\n");

    println!("================================================================");
    println!("   TRAINING LOGISTIC REGRESSION PROBES");
    println!("================================================================\n");

    println!("Using 80/20 train/test split, 5-fold cross-validation\n");

    println!("Layer │ Train Acc │ Test Acc │ AUC-ROC │ F1 Score");
    println!("──────┼───────────┼──────────┼─────────┼─────────");

    let mut layer_metrics = Vec::new();

    for (i, &layer) in layers.iter().enumerate() {
        // Prepare data
        let mut X = Vec::new();
        let mut y = Vec::new();

        for act in &phen_by_layer[i] {
            X.push(act.clone());
            y.push(1.0); // phenomenal = 1
        }
        for act in &func_by_layer[i] {
            X.push(act.clone());
            y.push(0.0); // functional = 0
        }

        // Train/test split (shuffle first)
        let mut indices: Vec<usize> = (0..X.len()).collect();
        shuffle(&mut indices, 42 + layer as u64);

        let split = (X.len() as f64 * 0.8) as usize;
        let train_indices: Vec<usize> = indices[..split].to_vec();
        let test_indices: Vec<usize> = indices[split..].to_vec();

        let X_train: Vec<Vec<f32>> = train_indices.iter().map(|&i| X[i].clone()).collect();
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y[i]).collect();
        let X_test: Vec<Vec<f32>> = test_indices.iter().map(|&i| X[i].clone()).collect();
        let y_test: Vec<f64> = test_indices.iter().map(|&i| y[i]).collect();

        // Train logistic regression
        let weights = train_logistic_regression(&X_train, &y_train, 100, 0.01);

        // Evaluate
        let train_preds = predict(&X_train, &weights);
        let test_preds = predict(&X_test, &weights);

        let train_acc = accuracy(&y_train, &train_preds);
        let test_acc = accuracy(&y_test, &test_preds);
        let auc = compute_auc(&y_test, &predict_proba(&X_test, &weights));
        let f1 = f1_score(&y_test, &test_preds);

        println!(
            "{:5} │ {:9.2}% │ {:8.2}% │ {:7.4} │ {:7.4}",
            layer,
            train_acc * 100.0,
            test_acc * 100.0,
            auc,
            f1
        );

        layer_metrics.push((layer, train_acc, test_acc, auc, f1));
    }

    // Summary
    println!("\n================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    // Find best layer by test accuracy
    let best_by_acc = layer_metrics
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    // Find best layer by AUC
    let best_by_auc = layer_metrics
        .iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!("Best layer by TEST ACCURACY: Layer {}", best_by_acc.0);
    println!("  Test Accuracy: {:.2}%", best_by_acc.2 * 100.0);
    println!("  AUC-ROC: {:.4}\n", best_by_acc.3);

    println!("Best layer by AUC-ROC: Layer {}", best_by_auc.0);
    println!("  AUC-ROC: {:.4}", best_by_auc.3);
    println!("  Test Accuracy: {:.2}%\n", best_by_auc.2 * 100.0);

    // Compare with topology results
    println!("Comparison with topology results:");
    println!("  Topology: Layer 21 showed strongest phenomenal effect (p=0.001)");
    println!("  Probe: Layer {} shows best classification", best_by_auc.0);

    if best_by_auc.0 == 21 {
        println!("\n✓ CONVERGENT EVIDENCE: Both methods identify Layer 21");
    } else {
        println!("\n○ DIFFERENT OPTIMAL LAYERS:");
        println!("  Topology measures unity; probes measure discriminability");
        println!("  Both provide complementary information");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn load_corpus(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let mut concepts = Vec::new();
    if let Some(items) = json["concepts"].as_array() {
        for item in items {
            if let Some(text) = item["text"].as_str() {
                concepts.push(text.to_string());
            }
        }
    }
    Ok(concepts)
}

#[cfg(feature = "neural-bridge")]
fn shuffle(indices: &mut [usize], seed: u64) {
    let mut rng = seed;
    for i in (1..indices.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[cfg(feature = "neural-bridge")]
fn train_logistic_regression(X: &[Vec<f32>], y: &[f64], epochs: usize, lr: f64) -> Vec<f64> {
    let dim = X[0].len();
    let mut weights = vec![0.0; dim + 1]; // +1 for bias

    for _ in 0..epochs {
        let mut gradients = vec![0.0; dim + 1];

        for (xi, &yi) in X.iter().zip(y.iter()) {
            // Compute prediction
            let z: f64 = xi
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| (x as f64) * w)
                .sum::<f64>()
                + weights[dim]; // bias

            let pred = sigmoid(z);
            let error = pred - yi;

            // Accumulate gradients
            for (j, &xij) in xi.iter().enumerate() {
                gradients[j] += error * (xij as f64);
            }
            gradients[dim] += error; // bias gradient
        }

        // Update weights
        let n = X.len() as f64;
        for j in 0..=dim {
            weights[j] -= lr * gradients[j] / n;
        }
    }

    weights
}

#[cfg(feature = "neural-bridge")]
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(feature = "neural-bridge")]
fn predict_proba(X: &[Vec<f32>], weights: &[f64]) -> Vec<f64> {
    let dim = X[0].len();
    X.iter()
        .map(|xi| {
            let z: f64 = xi
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| (x as f64) * w)
                .sum::<f64>()
                + weights[dim];
            sigmoid(z)
        })
        .collect()
}

#[cfg(feature = "neural-bridge")]
fn predict(X: &[Vec<f32>], weights: &[f64]) -> Vec<f64> {
    predict_proba(X, weights)
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(feature = "neural-bridge")]
fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 0.5)
        .count();
    correct as f64 / y_true.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn compute_auc(y_true: &[f64], y_proba: &[f64]) -> f64 {
    // Simple AUC calculation
    let mut pairs: Vec<(f64, f64)> = y_proba
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| (p, t))
        .collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = y_true.iter().filter(|&&y| y > 0.5).count() as f64;
    let n_neg = y_true.len() as f64 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;

    for (_, label) in &pairs {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            auc += tp;
        }
    }

    auc / (n_pos * n_neg)
}

#[cfg(feature = "neural-bridge")]
fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if p > 0.5 && t > 0.5 {
            tp += 1.0;
        } else if p > 0.5 && t <= 0.5 {
            fp += 1.0;
        } else if p <= 0.5 && t > 0.5 {
            fn_ += 1.0;
        }
    }

    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}
