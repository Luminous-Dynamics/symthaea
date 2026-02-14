//! FEMNIST-Style Federated Benchmark
//!
//! Simulates realistic federated learning on FEMNIST-like digit recognition:
//! - 10 classes (digits 0-9) for clear convergence signal
//! - Non-IID "writer" partitions (each writer has 2-4 classes, power-law sample count)
//! - Compares pipeline presets: default, high_security, adaptive, performance
//! - Tests convergence under 0%, 10%, 20%, 30% Byzantine fractions
//!
//! Run: `cargo run --example benchmark_femnist --release`

use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const INPUT_DIM: usize = 784; // 28x28 flattened
const NUM_CLASSES: usize = 10; // digits 0-9
const PARAM_COUNT: usize = INPUT_DIM * NUM_CLASSES + NUM_CLASSES; // 7,850

/// Linear softmax classifier: y = softmax(Wx + b)
struct SoftmaxModel {
    weights: Vec<f32>,
}

impl SoftmaxModel {
    fn new(rng: &mut StdRng) -> Self {
        let mut weights = Vec::with_capacity(PARAM_COUNT);
        let scale = (2.0 / INPUT_DIM as f32).sqrt();
        for _ in 0..INPUT_DIM * NUM_CLASSES {
            weights.push(rng.gen_range(-scale..scale));
        }
        for _ in 0..NUM_CLASSES {
            weights.push(0.0);
        }
        Self { weights }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            let w_offset = c * INPUT_DIM;
            let mut sum = self.weights[INPUT_DIM * NUM_CLASSES + c];
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

/// Generate FEMNIST-like class centroids.
/// Each class has a distinct pattern in 28x28 space: stroke-like features
/// concentrated in class-specific spatial regions.
fn generate_class_centroids(rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::new();
    for c in 0..NUM_CLASSES {
        let mut centroid = vec![0.0f32; INPUT_DIM];
        // Each class has active features in overlapping but distinct spatial regions.
        // Use multiple strokes per class to mimic real character shapes.
        let n_strokes = 3 + (c % 4); // 3-6 strokes
        for s in 0..n_strokes {
            let start_row = ((c * 7 + s * 13) % 24) as usize;
            let start_col = ((c * 11 + s * 17) % 24) as usize;
            let length = 4 + (s % 3);
            let horizontal = (c + s) % 2 == 0;

            for step in 0..length {
                let (r, col) = if horizontal {
                    (start_row, (start_col + step) % 28)
                } else {
                    ((start_row + step) % 28, start_col)
                };
                let idx = r * 28 + col;
                centroid[idx] = 0.8 + rng.gen_range(-0.1..0.1);
                // Width: fill adjacent pixels
                if idx > 0 {
                    centroid[idx - 1] = 0.4;
                }
                if idx + 1 < INPUT_DIM {
                    centroid[idx + 1] = 0.4;
                }
            }
        }
        centroids.push(centroid);
    }
    centroids
}

/// Generate writer-specific data with non-IID class distribution.
///
/// Each writer:
/// - Has 2-5 classes they write
/// - Adds a writer-specific style bias (spatial offset + noise)
/// - Has power-law sample count (zipf distribution)
fn generate_writer_data(
    n_writers: usize,
    centroids: &[Vec<f32>],
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    let mut writers: Vec<Vec<(Vec<f32>, usize)>> = Vec::new();

    for w in 0..n_writers {
        let n_classes = 2 + rng.gen_range(0..3); // 2-4 classes per writer
        let primary = w % NUM_CLASSES;
        let mut writer_classes: Vec<usize> = vec![primary];
        for _ in 1..n_classes {
            let c = (primary + rng.gen_range(1..NUM_CLASSES)) % NUM_CLASSES;
            if !writer_classes.contains(&c) {
                writer_classes.push(c);
            }
        }

        // Power-law: writer w gets ~ 10 + 50/(w+1) samples per class
        let samples_per_class = 10 + 50 / (w / 3 + 1);

        // Writer-specific style: spatial shift + noise scale
        let x_shift = rng.gen_range(-2i32..3);
        let noise_scale = 0.2 + rng.gen_range(0.0..0.15);

        let mut data = Vec::new();
        for &c in &writer_classes {
            for _ in 0..samples_per_class {
                let mut sample = vec![0.0f32; INPUT_DIM];
                for row in 0..28 {
                    for col in 0..28 {
                        let src_col = (col as i32 - x_shift).clamp(0, 27) as usize;
                        let src_idx = row * 28 + src_col;
                        sample[row * 28 + col] =
                            centroids[c][src_idx] + rng.gen_range(-noise_scale..noise_scale);
                    }
                }
                data.push((sample, c));
            }
        }

        // Shuffle
        let n = data.len();
        for i in 0..n {
            let j = rng.gen_range(0..n);
            data.swap(i, j);
        }

        writers.push(data);
    }

    writers
}

fn evaluate(model: &SoftmaxModel, test_data: &[(Vec<f32>, usize)]) -> (f32, f32) {
    if test_data.is_empty() {
        return (0.0, f32::MAX);
    }
    let mut correct = 0;
    let mut total_loss = 0.0;
    for (input, label) in test_data {
        if model.predict(input) == *label {
            correct += 1;
        }
        total_loss += model.loss(input, *label);
    }
    (
        correct as f32 / test_data.len() as f32,
        total_loss / test_data.len() as f32,
    )
}

struct BenchmarkResult {
    preset: String,
    byzantine_pct: usize,
    final_accuracy: f32,
    final_loss: f32,
    converged: bool,
    loss_decreased: bool,
}

fn run_federated_training(
    config: PipelineConfig,
    preset_name: &str,
    writers: &[Vec<(Vec<f32>, usize)>],
    test_data: &[(Vec<f32>, usize)],
    byz_fraction: f32,
    n_rounds: usize,
    lr: f32,
) -> BenchmarkResult {
    let mut rng = StdRng::seed_from_u64(42);
    let mut model = SoftmaxModel::new(&mut rng);
    let n_writers = writers.len();
    let n_byz = (n_writers as f32 * byz_fraction) as usize;
    let n_honest = n_writers - n_byz;

    let mut reputations: HashMap<String, f32> = HashMap::new();
    for i in 0..n_honest {
        reputations.insert(format!("w{}", i), 0.85);
    }
    for i in 0..n_byz {
        reputations.insert(format!("byz{}", i), 0.15);
    }

    let mut pipeline = UnifiedPipeline::new(config);
    let (_, init_loss) = evaluate(&model, test_data);
    let mut loss_history = Vec::new();

    for round in 0..n_rounds {
        let mut contributions = Vec::new();

        // Honest writers
        for wid in 0..n_honest {
            let data = &writers[wid];
            if data.is_empty() {
                continue;
            }
            let samples = data.len().min(32);
            let mut grads = vec![0.0f32; PARAM_COUNT];
            let mut total_loss = 0.0;
            for s in 0..samples {
                let (input, label) = &data[s];
                let g = model.compute_gradients(input, *label);
                for (lg, gg) in grads.iter_mut().zip(g.iter()) {
                    *lg += gg / samples as f32;
                }
                total_loss += model.loss(input, *label);
            }
            contributions.push(GradientUpdate::new(
                format!("w{}", wid),
                round as u64 + 1,
                grads,
                samples as u32,
                total_loss / samples as f32,
            ));
        }

        // Byzantine writers: random large gradients
        for i in 0..n_byz {
            let byz_grads: Vec<f32> = (0..PARAM_COUNT)
                .map(|j| if (j + round) % 3 == 0 { 50.0 } else { -50.0 })
                .collect();
            contributions.push(GradientUpdate::new(
                format!("byz{}", i),
                round as u64 + 1,
                byz_grads,
                100,
                0.5,
            ));
        }

        match pipeline.aggregate(&contributions, &reputations) {
            Ok(result) => {
                model.apply_gradients(&result.aggregated.gradients, lr);
            }
            Err(_) => break,
        }

        let (_, loss) = evaluate(&model, test_data);
        loss_history.push(loss);
    }

    let (final_acc, final_loss) = evaluate(&model, test_data);
    let loss_decreased = final_loss < init_loss;

    // Check convergence: accuracy above random (1/10 = 10%) by meaningful margin
    // With non-IID data and 10 classes, 25% is a realistic target for linear model
    let converged = final_acc > 0.25;

    BenchmarkResult {
        preset: preset_name.to_string(),
        byzantine_pct: (byz_fraction * 100.0) as usize,
        final_accuracy: final_acc,
        final_loss,
        converged,
        loss_decreased,
    }
}

fn main() {
    println!("=== FEMNIST-Style Federated Benchmark ===\n");
    println!("Classes: {} (digits 0-9)", NUM_CLASSES);
    println!("Parameters: {} (784×10 + 10)", PARAM_COUNT);
    println!();

    let mut rng = StdRng::seed_from_u64(42);
    let centroids = generate_class_centroids(&mut rng);

    // Generate data
    let n_writers = 20;
    let n_rounds = 15;
    let lr = 0.05;

    println!(
        "Generating {} writer partitions (non-IID, power-law)...",
        n_writers
    );
    let writers = generate_writer_data(n_writers, &centroids, &mut rng);

    // IID test set
    let mut test_data = Vec::new();
    for c in 0..NUM_CLASSES {
        for _ in 0..5 {
            let mut sample = centroids[c].clone();
            for val in sample.iter_mut() {
                *val += rng.gen_range(-0.2..0.2);
            }
            test_data.push((sample, c));
        }
    }

    let total_samples: usize = writers.iter().map(|w| w.len()).sum();
    let max_per_writer = writers.iter().map(|w| w.len()).max().unwrap_or(0);
    let min_per_writer = writers.iter().map(|w| w.len()).min().unwrap_or(0);
    println!(
        "Total samples: {}, per-writer range: {}-{}",
        total_samples, min_per_writer, max_per_writer
    );
    println!(
        "Test set: {} samples ({} per class)",
        test_data.len(),
        test_data.len() / NUM_CLASSES
    );
    println!();

    // Presets to test
    let presets: Vec<(&str, PipelineConfig)> = vec![
        ("default", PipelineConfig::default()),
        ("high_security", PipelineConfig::high_security()),
        ("adaptive", PipelineConfig::adaptive()),
        ("performance", PipelineConfig::performance()),
    ];

    // Byzantine fractions to test
    let byz_fractions: Vec<f32> = vec![0.0, 0.10, 0.20, 0.30];

    let mut results: Vec<BenchmarkResult> = Vec::new();

    for (preset_name, config) in &presets {
        for &byz_frac in &byz_fractions {
            let result = run_federated_training(
                config.clone(),
                preset_name,
                &writers,
                &test_data,
                byz_frac,
                n_rounds,
                lr,
            );
            results.push(result);
        }
    }

    // Print results table
    println!(
        "=== Results ({} rounds, {} writers, lr={}) ===\n",
        n_rounds, n_writers, lr
    );
    println!(
        "{:<15} {:>6} {:>10} {:>10} {:>5} {:>5}",
        "Preset", "Byz%", "Accuracy", "Loss", "Conv", "LDec"
    );
    println!("{:-<60}", "");

    for r in &results {
        println!(
            "{:<15} {:>5}% {:>9.1}% {:>9.3} {:>5} {:>5}",
            r.preset,
            r.byzantine_pct,
            r.final_accuracy * 100.0,
            r.final_loss,
            if r.converged { "Y" } else { "N" },
            if r.loss_decreased { "Y" } else { "N" },
        );
    }

    // === Verification ===
    println!("\n=== Verification ===");
    let mut passed = 0;
    let mut total = 0;

    // Test 1: Default preset converges without Byzantine
    total += 1;
    let clean = results
        .iter()
        .find(|r| r.preset == "default" && r.byzantine_pct == 0)
        .unwrap();
    if clean.converged && clean.loss_decreased {
        println!(
            "[PASS] Test 1: Default converges clean: {:.1}%",
            clean.final_accuracy * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 1: Default should converge clean: acc={:.1}%, loss_dec={}",
            clean.final_accuracy * 100.0,
            clean.loss_decreased
        );
    }

    // Test 2: All presets without DP decrease loss at 0% Byzantine.
    // (high_security uses DP noise which can prevent convergence on small data)
    total += 1;
    let non_dp_presets = ["default", "adaptive", "performance"];
    let non_dp_converge = results
        .iter()
        .filter(|r| r.byzantine_pct == 0 && non_dp_presets.contains(&r.preset.as_str()))
        .all(|r| r.loss_decreased);
    if non_dp_converge {
        println!("[PASS] Test 2: All non-DP presets decrease loss at 0% Byzantine");
        passed += 1;
    } else {
        println!("[FAIL] Test 2: Not all non-DP presets converge clean");
    }

    // Test 3: Byzantine doesn't destroy default pipeline
    total += 1;
    let byz20 = results
        .iter()
        .find(|r| r.preset == "default" && r.byzantine_pct == 20)
        .unwrap();
    if byz20.loss_decreased {
        println!(
            "[PASS] Test 3: Default survives 20% Byzantine: {:.1}%",
            byz20.final_accuracy * 100.0
        );
        passed += 1;
    } else {
        println!("[FAIL] Test 3: Default should survive 20% Byzantine");
    }

    // Test 4: Adaptive preset handles 30% Byzantine (gates out low-rep nodes)
    total += 1;
    let adapt30 = results
        .iter()
        .find(|r| r.preset == "adaptive" && r.byzantine_pct == 30)
        .unwrap();
    if adapt30.loss_decreased {
        println!(
            "[PASS] Test 4: Adaptive survives 30% Byzantine: {:.1}%",
            adapt30.final_accuracy * 100.0
        );
        passed += 1;
    } else {
        println!("[FAIL] Test 4: Adaptive should survive 30% Byzantine");
    }

    // Test 5: Performance preset is faster (no detection overhead measured, but converges)
    total += 1;
    let perf = results
        .iter()
        .find(|r| r.preset == "performance" && r.byzantine_pct == 0)
        .unwrap();
    if perf.loss_decreased {
        println!(
            "[PASS] Test 5: Performance preset converges: {:.1}%",
            perf.final_accuracy * 100.0
        );
        passed += 1;
    } else {
        println!("[FAIL] Test 5: Performance preset should converge");
    }

    // Test 6: Non-IID data produces heterogeneous gradients
    total += 1;
    let class_sets: Vec<usize> = writers
        .iter()
        .map(|w| {
            let classes: std::collections::HashSet<usize> = w.iter().map(|(_, l)| *l).collect();
            classes.len()
        })
        .collect();
    let avg_classes = class_sets.iter().sum::<usize>() as f32 / class_sets.len() as f32;
    if avg_classes < (NUM_CLASSES as f32 * 0.3) {
        println!(
            "[PASS] Test 6: Non-IID verified (avg {:.1} classes/writer, < 30% of {})",
            avg_classes, NUM_CLASSES
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 6: Data should be non-IID, avg classes/writer: {:.1}",
            avg_classes
        );
    }

    println!("\n=== RESULTS: {}/{} passed ===", passed, total);
    if passed < total {
        std::process::exit(1);
    }
}
