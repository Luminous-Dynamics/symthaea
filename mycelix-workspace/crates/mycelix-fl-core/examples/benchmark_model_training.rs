// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real Model Training Benchmark
//!
//! Trains a linear softmax classifier (784->10, 7,850 params) across
//! federated nodes using SGD, with synthetic MNIST-like clustered data.
//!
//! Proves the unified pipeline works for actual ML model training,
//! not just raw gradient aggregation.
//!
//! Run: `cargo run --example benchmark_model_training --release`

use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const INPUT_DIM: usize = 784;
const NUM_CLASSES: usize = 10;
const PARAM_COUNT: usize = INPUT_DIM * NUM_CLASSES + NUM_CLASSES; // 7,850

/// Simple softmax classifier: y = softmax(Wx + b)
struct SoftmaxModel {
    weights: Vec<f32>, // flattened: [W (784*10), b (10)]
}

impl SoftmaxModel {
    fn new(rng: &mut StdRng) -> Self {
        let mut weights = Vec::with_capacity(PARAM_COUNT);
        let scale = (2.0 / INPUT_DIM as f32).sqrt(); // He initialization
        for _ in 0..INPUT_DIM * NUM_CLASSES {
            weights.push(rng.gen_range(-scale..scale));
        }
        for _ in 0..NUM_CLASSES {
            weights.push(0.0); // bias init to zero
        }
        Self { weights }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), INPUT_DIM);
        let mut logits = vec![0.0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            let w_offset = c * INPUT_DIM;
            let mut sum = self.weights[INPUT_DIM * NUM_CLASSES + c]; // bias
            for i in 0..INPUT_DIM {
                sum += self.weights[w_offset + i] * input[i];
            }
            logits[c] = sum;
        }
        softmax(&logits)
    }

    fn compute_gradients(&self, input: &[f32], label: usize) -> Vec<f32> {
        let probs = self.forward(input);
        let mut gradients = vec![0.0f32; PARAM_COUNT];

        // dL/dW[c][i] = (p[c] - (c == label)) * input[i]
        // dL/db[c] = p[c] - (c == label)
        for c in 0..NUM_CLASSES {
            let error = probs[c] - if c == label { 1.0 } else { 0.0 };
            let w_offset = c * INPUT_DIM;
            for i in 0..INPUT_DIM {
                gradients[w_offset + i] = error * input[i];
            }
            gradients[INPUT_DIM * NUM_CLASSES + c] = error;
        }
        gradients
    }

    fn apply_gradients(&mut self, aggregated: &[f32], lr: f32) {
        assert_eq!(aggregated.len(), self.weights.len());
        for (w, &g) in self.weights.iter_mut().zip(aggregated.iter()) {
            *w -= lr * g;
        }
    }

    fn predict(&self, input: &[f32]) -> usize {
        let probs = self.forward(input);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    fn loss(&self, input: &[f32], label: usize) -> f32 {
        let probs = self.forward(input);
        -probs[label].max(1e-10).ln()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Generate synthetic clustered data (MNIST-like).
/// Each class has a centroid in 784-dim space; samples are Gaussian around centroids.
fn generate_clustered_data(n_per_class: usize, rng: &mut StdRng) -> Vec<(Vec<f32>, usize)> {
    let mut data = Vec::new();
    let noise_scale = 0.3;

    // Generate class centroids
    let mut centroids = Vec::new();
    for c in 0..NUM_CLASSES {
        let mut centroid = vec![0.0f32; INPUT_DIM];
        // Each class has active features in a different region
        let start = c * (INPUT_DIM / NUM_CLASSES);
        let end = start + (INPUT_DIM / NUM_CLASSES);
        for i in start..end {
            centroid[i] = 1.0;
        }
        centroids.push(centroid);
    }

    for c in 0..NUM_CLASSES {
        for _ in 0..n_per_class {
            let mut sample = centroids[c].clone();
            for val in sample.iter_mut() {
                *val += rng.gen_range(-noise_scale..noise_scale);
            }
            data.push((sample, c));
        }
    }

    data
}

/// Partition data non-IID: each node gets 2-3 classes
fn partition_non_iid(
    data: &[(Vec<f32>, usize)],
    n_nodes: usize,
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    let mut partitions: Vec<Vec<(Vec<f32>, usize)>> = vec![Vec::new(); n_nodes];

    // Assign 2-3 classes per node
    for node in 0..n_nodes {
        let primary_class = node % NUM_CLASSES;
        let secondary_class = (node + 1) % NUM_CLASSES;

        for (sample, label) in data {
            if *label == primary_class || *label == secondary_class {
                partitions[node].push((sample.clone(), *label));
            }
        }

        // Shuffle
        let n = partitions[node].len();
        for i in 0..n {
            let j = rng.gen_range(0..n);
            partitions[node].swap(i, j);
        }
    }

    partitions
}

fn evaluate(model: &SoftmaxModel, test_data: &[(Vec<f32>, usize)]) -> (f32, f32) {
    let mut correct = 0;
    let mut total_loss = 0.0;
    for (input, label) in test_data {
        if model.predict(input) == *label {
            correct += 1;
        }
        total_loss += model.loss(input, *label);
    }
    let accuracy = correct as f32 / test_data.len() as f32;
    let avg_loss = total_loss / test_data.len() as f32;
    (accuracy, avg_loss)
}

fn main() {
    println!("=== FL Model Training Benchmark ===\n");

    let mut rng = StdRng::seed_from_u64(42);
    let n_nodes = 10;
    let n_rounds = 20;
    let lr = 0.1;
    let local_epochs = 1;
    let samples_per_class = 50;

    // Generate data
    println!(
        "Generating synthetic data ({} samples/class)...",
        samples_per_class
    );
    let train_data = generate_clustered_data(samples_per_class, &mut rng);
    let test_data = generate_clustered_data(20, &mut rng);
    let partitions = partition_non_iid(&train_data, n_nodes, &mut rng);

    // Initialize global model
    let mut global_model = SoftmaxModel::new(&mut rng);

    // Reputations
    let mut reputations: HashMap<String, f32> = HashMap::new();
    for i in 0..n_nodes {
        reputations.insert(format!("node_{}", i), 0.85);
    }

    // Pipeline
    let config = PipelineConfig {
        min_reputation: 0.3,
        multi_signal_detection: false, // disable for speed
        ..Default::default()
    };
    let mut pipeline = UnifiedPipeline::new(config);

    let mut loss_history = Vec::new();
    let (init_acc, init_loss) = evaluate(&global_model, &test_data);
    println!(
        "Initial: accuracy={:.1}%, loss={:.3}",
        init_acc * 100.0,
        init_loss
    );

    // Training loop
    for round in 0..n_rounds {
        let mut contributions = Vec::new();

        for node_id in 0..n_nodes {
            // Local training
            let node_data = &partitions[node_id];
            if node_data.is_empty() {
                continue;
            }

            let mut local_grads = vec![0.0f32; PARAM_COUNT];
            let samples = node_data.len().min(50); // cap local samples
            let mut total_loss = 0.0;

            for _epoch in 0..local_epochs {
                for s in 0..samples {
                    let (input, label) = &node_data[s % node_data.len()];
                    let grads = global_model.compute_gradients(input, *label);
                    for (lg, g) in local_grads.iter_mut().zip(grads.iter()) {
                        *lg += g / samples as f32;
                    }
                    total_loss += global_model.loss(input, *label);
                }
            }

            let avg_loss = total_loss / (samples * local_epochs) as f32;
            contributions.push(GradientUpdate::new(
                format!("node_{}", node_id),
                round as u64 + 1,
                local_grads,
                samples as u32,
                avg_loss,
            ));
        }

        // Aggregate via pipeline
        let result = pipeline.aggregate(&contributions, &reputations).unwrap();
        global_model.apply_gradients(&result.aggregated.gradients, lr);

        let (acc, loss) = evaluate(&global_model, &test_data);
        loss_history.push(loss);
        if round % 5 == 0 || round == n_rounds - 1 {
            println!(
                "Round {:2}: accuracy={:.1}%, loss={:.3}, participants={}",
                round + 1,
                acc * 100.0,
                loss,
                result.aggregated.participant_count
            );
        }
    }

    let (final_acc, final_loss) = evaluate(&global_model, &test_data);

    // === TESTS ===
    println!("\n=== Verification ===");
    let mut passed = 0;
    let mut total = 0;

    // Test 1: Convergence above 60%
    total += 1;
    if final_acc > 0.60 {
        println!("[PASS] Test 1: Accuracy > 60%: {:.1}%", final_acc * 100.0);
        passed += 1;
    } else {
        println!("[FAIL] Test 1: Accuracy > 60%: {:.1}%", final_acc * 100.0);
    }

    // Test 2: Loss decreased
    total += 1;
    if final_loss < init_loss {
        println!(
            "[PASS] Test 2: Loss decreased: {:.3} -> {:.3}",
            init_loss, final_loss
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 2: Loss decreased: {:.3} -> {:.3}",
            init_loss, final_loss
        );
    }

    // Test 3: Loss monotonic (smoothed over 3-round window)
    total += 1;
    let mut smoothed_monotonic = true;
    if loss_history.len() >= 6 {
        for i in 3..loss_history.len() {
            let prev_avg: f32 = loss_history[i - 3..i].iter().sum::<f32>() / 3.0;
            let curr_avg: f32 = loss_history[i.saturating_sub(2)..=i].iter().sum::<f32>() / 3.0;
            if curr_avg > prev_avg * 1.1 {
                // Allow 10% tolerance
                smoothed_monotonic = false;
                break;
            }
        }
    }
    if smoothed_monotonic {
        println!("[PASS] Test 3: Loss monotonically decreasing (smoothed)");
        passed += 1;
    } else {
        println!("[FAIL] Test 3: Loss not monotonically decreasing");
    }

    // Test 4: Byzantine resilience
    total += 1;
    println!("\nTest 4: Byzantine resilience...");
    let mut byz_model = SoftmaxModel::new(&mut StdRng::seed_from_u64(42));
    let mut byz_reps = reputations.clone();
    for i in 0..2 {
        byz_reps.insert(format!("byz_{}", i), 0.15);
    }
    let byz_config = PipelineConfig {
        min_reputation: 0.3,
        multi_signal_detection: false,
        ..Default::default()
    };
    let mut byz_pipeline = UnifiedPipeline::new(byz_config);

    for round in 0..n_rounds {
        let mut contribs = Vec::new();
        for node_id in 0..n_nodes {
            let node_data = &partitions[node_id];
            if node_data.is_empty() {
                continue;
            }
            let mut grads = vec![0.0f32; PARAM_COUNT];
            let samples = node_data.len().min(50);
            for s in 0..samples {
                let (input, label) = &node_data[s % node_data.len()];
                let g = byz_model.compute_gradients(input, *label);
                for (lg, gg) in grads.iter_mut().zip(g.iter()) {
                    *lg += gg / samples as f32;
                }
            }
            contribs.push(GradientUpdate::new(
                format!("node_{}", node_id),
                round as u64 + 1,
                grads,
                samples as u32,
                0.5,
            ));
        }
        // Add 2 Byzantine nodes (20%)
        for i in 0..2 {
            let byz_grads: Vec<f32> = (0..PARAM_COUNT)
                .map(|j| if j % 2 == 0 { 50.0 } else { -50.0 })
                .collect();
            contribs.push(GradientUpdate::new(
                format!("byz_{}", i),
                round as u64 + 1,
                byz_grads,
                100,
                0.5,
            ));
        }
        let result = byz_pipeline.aggregate(&contribs, &byz_reps).unwrap();
        byz_model.apply_gradients(&result.aggregated.gradients, lr);
    }
    let (byz_acc, _) = evaluate(&byz_model, &test_data);
    if byz_acc > 0.40 {
        println!(
            "[PASS] Test 4: Byzantine accuracy > 40%: {:.1}%",
            byz_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 4: Byzantine accuracy > 40%: {:.1}%",
            byz_acc * 100.0
        );
    }

    // Test 5: DP doesn't destroy accuracy
    total += 1;
    println!("\nTest 5: DP resilience...");
    let mut dp_model = SoftmaxModel::new(&mut StdRng::seed_from_u64(42));
    let dp_config = PipelineConfig {
        dp_config: Some(DifferentialPrivacyConfig::low_privacy()),
        min_reputation: 0.3,
        multi_signal_detection: false,
        ..Default::default()
    };
    let mut dp_pipeline = UnifiedPipeline::new(dp_config);

    for round in 0..n_rounds {
        let mut contribs = Vec::new();
        for node_id in 0..n_nodes {
            let node_data = &partitions[node_id];
            if node_data.is_empty() {
                continue;
            }
            let mut grads = vec![0.0f32; PARAM_COUNT];
            let samples = node_data.len().min(50);
            for s in 0..samples {
                let (input, label) = &node_data[s % node_data.len()];
                let g = dp_model.compute_gradients(input, *label);
                for (lg, gg) in grads.iter_mut().zip(g.iter()) {
                    *lg += gg / samples as f32;
                }
            }
            contribs.push(GradientUpdate::new(
                format!("node_{}", node_id),
                round as u64 + 1,
                grads,
                samples as u32,
                0.5,
            ));
        }
        let result = dp_pipeline.aggregate(&contribs, &reputations).unwrap();
        dp_model.apply_gradients(&result.aggregated.gradients, lr);
    }
    let (dp_acc, _) = evaluate(&dp_model, &test_data);
    if dp_acc > 0.30 {
        println!("[PASS] Test 5: DP accuracy > 30%: {:.1}%", dp_acc * 100.0);
        passed += 1;
    } else {
        println!("[FAIL] Test 5: DP accuracy > 30%: {:.1}%", dp_acc * 100.0);
    }

    println!("\n=== RESULTS: {}/{} passed ===", passed, total);
    if passed < total {
        std::process::exit(1);
    }
}
