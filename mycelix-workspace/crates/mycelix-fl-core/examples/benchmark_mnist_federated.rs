//! Federated MNIST Benchmark
//!
//! Trains a softmax classifier (784→10, 7,850 params) across federated nodes
//! using structured digit data. Compares:
//! - IID vs Non-IID data partitioning
//! - Clean vs Byzantine contamination (10%, 20%, 30%)
//! - Impact of DP noise on convergence
//! - Pipeline preset comparison
//!
//! Run: `cargo run --example benchmark_mnist_federated --release`

use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const DIM: usize = 784;
const NUM_CLASSES: usize = 10;
const PARAMS: usize = DIM * NUM_CLASSES + NUM_CLASSES; // 7,850

struct SoftmaxModel {
    weights: Vec<f32>,
}

impl SoftmaxModel {
    fn new(rng: &mut StdRng) -> Self {
        let scale = (2.0 / DIM as f32).sqrt();
        let weights: Vec<f32> = (0..PARAMS).map(|_| rng.gen_range(-scale..scale)).collect();
        Self { weights }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            let offset = c * DIM;
            logits[c] = self.weights[DIM * NUM_CLASSES + c];
            for i in 0..DIM {
                logits[c] += self.weights[offset + i] * input[i];
            }
        }
        softmax(&logits)
    }

    fn compute_gradients(&self, input: &[f32], label: usize) -> Vec<f32> {
        let probs = self.forward(input);
        let mut grads = vec![0.0f32; PARAMS];
        for c in 0..NUM_CLASSES {
            let err = probs[c] - if c == label { 1.0 } else { 0.0 };
            let offset = c * DIM;
            for i in 0..DIM {
                grads[offset + i] = err * input[i];
            }
            grads[DIM * NUM_CLASSES + c] = err;
        }
        grads
    }

    fn apply(&mut self, aggregated: &[f32], lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(aggregated) {
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
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Generate structured digit prototypes (deterministic).
/// Each digit has a distinctive spatial pattern in 28×28.
fn digit_prototypes(rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut protos = Vec::new();
    for d in 0..NUM_CLASSES {
        let mut img = vec![0.0f32; DIM];
        // Each digit gets distinct stroke patterns
        let n_strokes = 3 + d % 3;
        for s in 0..n_strokes {
            let row = ((d * 5 + s * 11) % 22 + 3) as usize;
            let col = ((d * 7 + s * 13) % 22 + 3) as usize;
            let len = 3 + (d + s) % 5;
            let horiz = (d + s) % 2 == 0;
            for step in 0..len {
                let (r, c) = if horiz {
                    (row, (col + step).min(27))
                } else {
                    ((row + step).min(27), col)
                };
                let idx = r * 28 + c;
                img[idx] = 0.9 + rng.gen_range(-0.05..0.05);
                // thickness
                if idx > 0 {
                    img[idx - 1] = (img[idx - 1] + 0.4).min(1.0);
                }
                if idx + 1 < DIM {
                    img[idx + 1] = (img[idx + 1] + 0.4).min(1.0);
                }
                if r > 0 {
                    img[(r - 1) * 28 + c] = (img[(r - 1) * 28 + c] + 0.3).min(1.0);
                }
                if r < 27 {
                    img[(r + 1) * 28 + c] = (img[(r + 1) * 28 + c] + 0.3).min(1.0);
                }
            }
        }
        protos.push(img);
    }
    protos
}

/// Generate samples for a digit with noise
fn sample_digit(proto: &[f32], rng: &mut StdRng, noise: f32) -> Vec<f32> {
    proto
        .iter()
        .map(|&v| (v + rng.gen_range(-noise..noise)).clamp(0.0, 1.0))
        .collect()
}

/// Partition data IID across nodes
fn partition_iid(
    protos: &[Vec<f32>],
    n_nodes: usize,
    samples_per_node: usize,
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    let mut nodes = vec![Vec::new(); n_nodes];
    let per_class = samples_per_node / NUM_CLASSES;
    for node in 0..n_nodes {
        for class in 0..NUM_CLASSES {
            for _ in 0..per_class {
                nodes[node].push((sample_digit(&protos[class], rng, 0.25), class));
            }
        }
        // Shuffle
        let n = nodes[node].len();
        for i in 0..n {
            let j = rng.gen_range(0..n);
            nodes[node].swap(i, j);
        }
    }
    nodes
}

/// Partition data non-IID: each node gets only 2-3 classes
fn partition_noniid(
    protos: &[Vec<f32>],
    n_nodes: usize,
    samples_per_node: usize,
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    let mut nodes = vec![Vec::new(); n_nodes];
    for node in 0..n_nodes {
        let n_classes = 2 + rng.gen_range(0..2); // 2-3 classes
        let primary = node % NUM_CLASSES;
        let mut classes = vec![primary];
        for _ in 1..n_classes {
            let c = (primary + rng.gen_range(1..NUM_CLASSES)) % NUM_CLASSES;
            if !classes.contains(&c) {
                classes.push(c);
            }
        }
        let per_class = samples_per_node / classes.len();
        for &class in &classes {
            for _ in 0..per_class {
                nodes[node].push((sample_digit(&protos[class], rng, 0.25), class));
            }
        }
        let n = nodes[node].len();
        for i in 0..n {
            let j = rng.gen_range(0..n);
            nodes[node].swap(i, j);
        }
    }
    nodes
}

fn evaluate(model: &SoftmaxModel, data: &[(Vec<f32>, usize)]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, f32::MAX);
    }
    let mut correct = 0;
    let mut total_loss = 0.0;
    for (x, y) in data {
        if model.predict(x) == *y {
            correct += 1;
        }
        total_loss += model.loss(x, *y);
    }
    (
        correct as f32 / data.len() as f32,
        total_loss / data.len() as f32,
    )
}

struct TrainResult {
    label: String,
    final_acc: f32,
    final_loss: f32,
    acc_history: Vec<f32>,
}

fn train_federated(
    label: &str,
    node_data: &[Vec<(Vec<f32>, usize)>],
    test_data: &[(Vec<f32>, usize)],
    config: PipelineConfig,
    n_rounds: usize,
    lr: f32,
    n_byz: usize,
) -> TrainResult {
    let mut rng = StdRng::seed_from_u64(42);
    let mut model = SoftmaxModel::new(&mut rng);
    let n_honest = node_data.len();

    let mut reputations: HashMap<String, f32> = HashMap::new();
    for i in 0..n_honest {
        reputations.insert(format!("n{}", i), 0.85);
    }
    for i in 0..n_byz {
        reputations.insert(format!("byz{}", i), 0.15);
    }

    let mut pipeline = UnifiedPipeline::new(config);
    let mut acc_history = Vec::new();

    for round in 0..n_rounds {
        let mut contributions = Vec::new();

        // Honest nodes
        for (nid, data) in node_data.iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            let batch = data.len().min(32);
            let mut grads = vec![0.0f32; PARAMS];
            let mut total_loss = 0.0;
            for s in 0..batch {
                let (x, y) = &data[s % data.len()];
                let g = model.compute_gradients(x, *y);
                for (lg, gg) in grads.iter_mut().zip(g.iter()) {
                    *lg += gg / batch as f32;
                }
                total_loss += model.loss(x, *y);
            }
            contributions.push(GradientUpdate::new(
                format!("n{}", nid),
                round as u64 + 1,
                grads,
                batch as u32,
                total_loss / batch as f32,
            ));
        }

        // Byzantine nodes
        for i in 0..n_byz {
            let byz_grads: Vec<f32> = (0..PARAMS)
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

        if let Ok(result) = pipeline.aggregate(&contributions, &reputations) {
            model.apply(&result.aggregated.gradients, lr);
        }

        let (acc, _) = evaluate(&model, test_data);
        acc_history.push(acc);
    }

    let (final_acc, final_loss) = evaluate(&model, test_data);
    TrainResult {
        label: label.into(),
        final_acc,
        final_loss,
        acc_history,
    }
}

fn main() {
    println!("=== Federated MNIST Benchmark ===\n");
    println!("Model: Linear Softmax (784→10, {} params)", PARAMS);

    let mut rng = StdRng::seed_from_u64(42);
    let protos = digit_prototypes(&mut rng);

    let n_nodes = 15;
    let samples_per_node = 100;
    let n_rounds = 20;
    let lr = 0.05;

    // Generate partitions
    let iid_data = partition_iid(&protos, n_nodes, samples_per_node, &mut rng);
    let noniid_data = partition_noniid(&protos, n_nodes, samples_per_node, &mut rng);

    // Test set (IID, 10 per class)
    let mut test_data = Vec::new();
    for c in 0..NUM_CLASSES {
        for _ in 0..10 {
            test_data.push((sample_digit(&protos[c], &mut rng, 0.2), c));
        }
    }

    println!(
        "Nodes: {}, Samples/node: {}, Rounds: {}, LR: {}",
        n_nodes, samples_per_node, n_rounds, lr
    );
    println!("Test set: {} samples\n", test_data.len());

    let mut results = Vec::new();

    // === Experiment 1: IID vs Non-IID ===
    println!("--- Experiment 1: IID vs Non-IID (0% Byzantine) ---\n");
    results.push(train_federated(
        "IID-clean",
        &iid_data,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        0,
    ));
    results.push(train_federated(
        "NonIID-clean",
        &noniid_data,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        0,
    ));

    // === Experiment 2: Byzantine Resistance ===
    println!("--- Experiment 2: Byzantine Resistance (IID) ---\n");
    for byz_pct in [10, 20, 30] {
        let n_byz = (n_nodes as f32 * byz_pct as f32 / 100.0) as usize;
        let label = format!("IID-byz{}%", byz_pct);
        results.push(train_federated(
            &label,
            &iid_data,
            &test_data,
            PipelineConfig::default(),
            n_rounds,
            lr,
            n_byz,
        ));
    }

    // === Experiment 3: Non-IID + Byzantine ===
    println!("--- Experiment 3: Non-IID + Byzantine ---\n");
    for byz_pct in [10, 20] {
        let n_byz = (n_nodes as f32 * byz_pct as f32 / 100.0) as usize;
        let label = format!("NonIID-byz{}%", byz_pct);
        results.push(train_federated(
            &label,
            &noniid_data,
            &test_data,
            PipelineConfig::default(),
            n_rounds,
            lr,
            n_byz,
        ));
    }

    // === Experiment 4: DP Impact ===
    println!("--- Experiment 4: DP Impact (IID, 0% Byzantine) ---\n");
    let dp_configs = [
        ("IID-dp-low", Some(DifferentialPrivacyConfig::low_privacy())),
        (
            "IID-dp-mod",
            Some(DifferentialPrivacyConfig::moderate_privacy()),
        ),
        (
            "IID-dp-high",
            Some(DifferentialPrivacyConfig::high_privacy()),
        ),
    ];
    for (label, dp) in dp_configs {
        let config = PipelineConfig {
            dp_config: dp,
            ..PipelineConfig::default()
        };
        results.push(train_federated(
            label, &iid_data, &test_data, config, n_rounds, lr, 0,
        ));
    }

    // === Results Table ===
    println!("=== Results ({} rounds) ===\n", n_rounds);
    println!(
        "{:<20} {:>10} {:>10} {:>12} {:>12}",
        "Experiment", "Accuracy", "Loss", "Acc@R5", "Acc@R10"
    );
    println!("{:-<68}", "");

    for r in &results {
        let acc_r5 = r.acc_history.get(4).copied().unwrap_or(0.0);
        let acc_r10 = r.acc_history.get(9).copied().unwrap_or(0.0);
        println!(
            "{:<20} {:>9.1}% {:>10.3} {:>11.1}% {:>11.1}%",
            r.label,
            r.final_acc * 100.0,
            r.final_loss,
            acc_r5 * 100.0,
            acc_r10 * 100.0
        );
    }

    // === Convergence Curves (ASCII sparkline) ===
    println!("\n--- Convergence Curves ---\n");
    for r in &results {
        let spark: String = r
            .acc_history
            .iter()
            .map(|&a| match (a * 100.0) as u32 {
                0..=10 => '▁',
                11..=20 => '▂',
                21..=30 => '▃',
                31..=40 => '▄',
                41..=50 => '▅',
                51..=60 => '▆',
                61..=70 => '▇',
                _ => '█',
            })
            .collect();
        println!("{:<20} {} {:.1}%", r.label, spark, r.final_acc * 100.0);
    }

    // === Verification ===
    println!("\n=== Verification ===");
    let mut passed = 0;
    let total_tests = 6;

    // Test 1: IID clean converges above 30%
    let iid_clean = results.iter().find(|r| r.label == "IID-clean").unwrap();
    if iid_clean.final_acc > 0.30 {
        println!(
            "[PASS] Test 1: IID clean converges: {:.1}%",
            iid_clean.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 1: IID clean too low: {:.1}%",
            iid_clean.final_acc * 100.0
        );
    }

    // Test 2: Non-IID clean converges (may be lower than IID)
    let noniid_clean = results.iter().find(|r| r.label == "NonIID-clean").unwrap();
    if noniid_clean.final_acc > 0.20 {
        println!(
            "[PASS] Test 2: Non-IID clean converges: {:.1}%",
            noniid_clean.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 2: Non-IID clean too low: {:.1}%",
            noniid_clean.final_acc * 100.0
        );
    }

    // Test 3: IID survives 20% Byzantine
    let iid_byz20 = results.iter().find(|r| r.label == "IID-byz20%").unwrap();
    if iid_byz20.final_acc > 0.20 {
        println!(
            "[PASS] Test 3: IID survives 20% Byzantine: {:.1}%",
            iid_byz20.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 3: IID-byz20% too low: {:.1}%",
            iid_byz20.final_acc * 100.0
        );
    }

    // Test 4: Accuracy decreases with more Byzantine
    let iid_byz10 = results.iter().find(|r| r.label == "IID-byz10%").unwrap();
    let iid_byz30 = results.iter().find(|r| r.label == "IID-byz30%").unwrap();
    if iid_byz10.final_acc >= iid_byz30.final_acc {
        println!(
            "[PASS] Test 4: More Byzantine → lower accuracy ({:.1}% ≥ {:.1}%)",
            iid_byz10.final_acc * 100.0,
            iid_byz30.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 4: 10% byz ({:.1}%) < 30% byz ({:.1}%)",
            iid_byz10.final_acc * 100.0,
            iid_byz30.final_acc * 100.0
        );
    }

    // Test 5: Low DP preserves most accuracy
    let iid_dp_low = results.iter().find(|r| r.label == "IID-dp-low").unwrap();
    let acc_drop = iid_clean.final_acc - iid_dp_low.final_acc;
    if acc_drop < 0.20 {
        println!(
            "[PASS] Test 5: Low DP accuracy drop < 20%: {:.1}%",
            acc_drop * 100.0
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 5: Low DP drops too much: {:.1}%",
            acc_drop * 100.0
        );
    }

    // Test 6: Convergence improves over time (acc@R5 < acc@R20)
    if iid_clean.acc_history.len() >= 20 {
        let early = iid_clean.acc_history[4];
        let late = iid_clean.final_acc;
        if late >= early {
            println!(
                "[PASS] Test 6: Accuracy improves R5→R20: {:.1}% → {:.1}%",
                early * 100.0,
                late * 100.0
            );
            passed += 1;
        } else {
            println!(
                "[FAIL] Test 6: Accuracy regresses: {:.1}% → {:.1}%",
                early * 100.0,
                late * 100.0
            );
        }
    } else {
        println!("[FAIL] Test 6: Not enough rounds");
    }

    println!("\n=== RESULTS: {}/{} passed ===", passed, total_tests);
    if passed < total_tests {
        std::process::exit(1);
    }
}
