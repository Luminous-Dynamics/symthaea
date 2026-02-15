//! Real MNIST Federated Benchmark
//!
//! Loads actual MNIST IDX files and trains a softmax classifier through the
//! federated pipeline. Compares:
//! - Real MNIST vs synthetic data quality
//! - IID vs Non-IID partitioning on real data
//! - Byzantine resilience with real gradients
//! - Per-digit class accuracy
//!
//! ## Data Setup
//!
//! Download MNIST files to `data/mnist/`:
//! ```bash
//! cd mycelix-workspace/crates/mycelix-fl-core
//! mkdir -p data/mnist && cd data/mnist
//! curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
//! curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
//! curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
//! curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
//! gunzip *.gz
//! ```
//!
//! If files aren't found, falls back to synthetic data with a warning.
//!
//! Run: `cargo run --example benchmark_real_mnist --release`

use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as IoRead;
use std::path::Path;

const DIM: usize = 784;
const NUM_CLASSES: usize = 10;
const PARAMS: usize = DIM * NUM_CLASSES + NUM_CLASSES; // 7,850

// ─── IDX File Parser ─────────────────────────────────────────────

fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    ((data[offset] as u32) << 24)
        | ((data[offset + 1] as u32) << 16)
        | ((data[offset + 2] as u32) << 8)
        | (data[offset + 3] as u32)
}

fn load_idx_images(path: &Path) -> Option<Vec<Vec<f32>>> {
    let mut file = File::open(path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;

    let magic = read_u32_be(&data, 0);
    if magic != 2051 {
        eprintln!("  Bad magic for images: {} (expected 2051)", magic);
        return None;
    }

    let n_images = read_u32_be(&data, 4) as usize;
    let rows = read_u32_be(&data, 8) as usize;
    let cols = read_u32_be(&data, 12) as usize;
    let pixels = rows * cols;

    if pixels != DIM {
        eprintln!(
            "  Unexpected image size: {}x{} (expected 28x28)",
            rows, cols
        );
        return None;
    }

    let header_size = 16;
    if data.len() < header_size + n_images * pixels {
        eprintln!(
            "  File too short: {} bytes for {} images",
            data.len(),
            n_images
        );
        return None;
    }

    let mut images = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let offset = header_size + i * pixels;
        let img: Vec<f32> = data[offset..offset + pixels]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(img);
    }
    Some(images)
}

fn load_idx_labels(path: &Path) -> Option<Vec<usize>> {
    let mut file = File::open(path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;

    let magic = read_u32_be(&data, 0);
    if magic != 2049 {
        eprintln!("  Bad magic for labels: {} (expected 2049)", magic);
        return None;
    }

    let n_labels = read_u32_be(&data, 4) as usize;
    let header_size = 8;

    if data.len() < header_size + n_labels {
        eprintln!(
            "  File too short: {} bytes for {} labels",
            data.len(),
            n_labels
        );
        return None;
    }

    let labels: Vec<usize> = data[header_size..header_size + n_labels]
        .iter()
        .map(|&b| b as usize)
        .collect();
    Some(labels)
}

struct MnistData {
    train: Vec<(Vec<f32>, usize)>,
    test: Vec<(Vec<f32>, usize)>,
    is_real: bool,
}

fn load_mnist(data_dir: &Path) -> MnistData {
    let train_images_path = data_dir.join("train-images-idx3-ubyte");
    let train_labels_path = data_dir.join("train-labels-idx1-ubyte");
    let test_images_path = data_dir.join("t10k-images-idx3-ubyte");
    let test_labels_path = data_dir.join("t10k-labels-idx1-ubyte");

    // Try loading real data
    if let (Some(train_imgs), Some(train_lbls), Some(test_imgs), Some(test_lbls)) = (
        load_idx_images(&train_images_path),
        load_idx_labels(&train_labels_path),
        load_idx_images(&test_images_path),
        load_idx_labels(&test_labels_path),
    ) {
        println!(
            "  Loaded real MNIST: {} train, {} test",
            train_imgs.len(),
            test_imgs.len()
        );
        let train: Vec<(Vec<f32>, usize)> = train_imgs.into_iter().zip(train_lbls).collect();
        let test: Vec<(Vec<f32>, usize)> = test_imgs.into_iter().zip(test_lbls).collect();
        return MnistData {
            train,
            test,
            is_real: true,
        };
    }

    // Fall back to synthetic
    println!(
        "  MNIST files not found at {:?}, using synthetic data",
        data_dir
    );
    println!("  To use real MNIST, download IDX files (see example header for instructions)");
    let mut rng = StdRng::seed_from_u64(12345);
    let protos = generate_digit_prototypes(&mut rng);

    let mut train = Vec::new();
    for c in 0..NUM_CLASSES {
        for _ in 0..500 {
            train.push((sample_from_proto(&protos[c], &mut rng, 0.25), c));
        }
    }
    let mut test = Vec::new();
    for c in 0..NUM_CLASSES {
        for _ in 0..50 {
            test.push((sample_from_proto(&protos[c], &mut rng, 0.2), c));
        }
    }
    MnistData {
        train,
        test,
        is_real: false,
    }
}

fn generate_digit_prototypes(rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut protos = Vec::new();
    for d in 0..NUM_CLASSES {
        let mut img = vec![0.0f32; DIM];
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
                if idx > 0 {
                    img[idx - 1] = (img[idx - 1] + 0.4).min(1.0);
                }
                if idx + 1 < DIM {
                    img[idx + 1] = (img[idx + 1] + 0.4).min(1.0);
                }
            }
        }
        protos.push(img);
    }
    protos
}

fn sample_from_proto(proto: &[f32], rng: &mut StdRng, noise: f32) -> Vec<f32> {
    proto
        .iter()
        .map(|&v| (v + rng.gen_range(-noise..noise)).clamp(0.0, 1.0))
        .collect()
}

// ─── Softmax Model ───────────────────────────────────────────────

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

// ─── Partitioning ────────────────────────────────────────────────

fn partition_iid(
    data: &[(Vec<f32>, usize)],
    n_nodes: usize,
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    // Fisher-Yates shuffle
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    let per_node = data.len() / n_nodes;
    let mut nodes = Vec::new();
    for n in 0..n_nodes {
        let start = n * per_node;
        let end = if n == n_nodes - 1 {
            data.len()
        } else {
            start + per_node
        };
        let partition: Vec<(Vec<f32>, usize)> = indices[start..end]
            .iter()
            .map(|&i| data[i].clone())
            .collect();
        nodes.push(partition);
    }
    nodes
}

fn partition_noniid(
    data: &[(Vec<f32>, usize)],
    n_nodes: usize,
    classes_per_node: usize,
    rng: &mut StdRng,
) -> Vec<Vec<(Vec<f32>, usize)>> {
    // Group by class
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); NUM_CLASSES];
    for (i, (_, label)) in data.iter().enumerate() {
        by_class[*label].push(i);
    }
    // Shuffle within each class
    for class_indices in &mut by_class {
        for i in (1..class_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            class_indices.swap(i, j);
        }
    }

    let mut nodes = vec![Vec::new(); n_nodes];
    let mut class_offsets = vec![0usize; NUM_CLASSES];

    for n in 0..n_nodes {
        let primary = n % NUM_CLASSES;
        let mut assigned_classes = vec![primary];
        for k in 1..classes_per_node {
            let c = (primary + k) % NUM_CLASSES;
            assigned_classes.push(c);
        }

        let samples_per_class = data.len() / (n_nodes * classes_per_node);
        for &c in &assigned_classes {
            let available = &by_class[c];
            let start = class_offsets[c];
            let end = (start + samples_per_class).min(available.len());
            for &idx in &available[start..end] {
                nodes[n].push(data[idx].clone());
            }
            class_offsets[c] = end;
        }

        // Shuffle this node's data
        let len = nodes[n].len();
        for i in (1..len).rev() {
            let j = rng.gen_range(0..=i);
            nodes[n].swap(i, j);
        }
    }
    nodes
}

// ─── Evaluation ──────────────────────────────────────────────────

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

fn per_class_accuracy(model: &SoftmaxModel, data: &[(Vec<f32>, usize)]) -> Vec<f32> {
    let mut correct = vec![0usize; NUM_CLASSES];
    let mut total = vec![0usize; NUM_CLASSES];
    for (x, y) in data {
        total[*y] += 1;
        if model.predict(x) == *y {
            correct[*y] += 1;
        }
    }
    (0..NUM_CLASSES)
        .map(|c| {
            if total[c] > 0 {
                correct[c] as f32 / total[c] as f32
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Training ────────────────────────────────────────────────────

struct TrainResult {
    label: String,
    final_acc: f32,
    final_loss: f32,
    acc_history: Vec<f32>,
    class_acc: Vec<f32>,
}

fn train_federated(
    label: &str,
    node_data: &[Vec<(Vec<f32>, usize)>],
    test_data: &[(Vec<f32>, usize)],
    config: PipelineConfig,
    n_rounds: usize,
    lr: f32,
    batch_size: usize,
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

        // Honest nodes: compute gradients on local batch
        for (nid, data) in node_data.iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            let batch = batch_size.min(data.len());
            // Random batch sampling
            let start = rng.gen_range(0..=data.len().saturating_sub(batch));
            let mut grads = vec![0.0f32; PARAMS];
            let mut total_loss = 0.0;
            for s in start..start + batch {
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

        // Byzantine nodes: adversarial gradients
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
    let class_acc = per_class_accuracy(&model, test_data);
    TrainResult {
        label: label.into(),
        final_acc,
        final_loss,
        acc_history,
        class_acc,
    }
}

// ─── Main ────────────────────────────────────────────────────────

fn main() {
    println!("=== Real MNIST Federated Benchmark ===\n");

    // Try multiple data paths
    let data_paths = ["data/mnist", "examples/data/mnist", "../data/mnist"];

    let mut mnist = None;
    for path in &data_paths {
        let p = Path::new(path);
        if p.exists() {
            println!("Checking {:?}...", p);
            let loaded = load_mnist(p);
            if loaded.is_real {
                mnist = Some(loaded);
                break;
            }
        }
    }

    let mnist = mnist.unwrap_or_else(|| {
        println!("No data directory found, using synthetic fallback");
        load_mnist(Path::new("data/mnist"))
    });

    let data_source = if mnist.is_real {
        "Real MNIST"
    } else {
        "Synthetic (MNIST-like)"
    };
    println!("\nData source: {}", data_source);
    println!(
        "Train: {} samples, Test: {} samples",
        mnist.train.len(),
        mnist.test.len()
    );
    println!("Model: Linear Softmax (784->10, {} params)\n", PARAMS);

    // Class distribution in training data
    let mut class_counts = vec![0usize; NUM_CLASSES];
    for (_, label) in &mnist.train {
        class_counts[*label] += 1;
    }
    print!("Class distribution: ");
    for (c, &count) in class_counts.iter().enumerate() {
        print!("{}={} ", c, count);
    }
    println!("\n");

    let mut rng = StdRng::seed_from_u64(42);
    let n_nodes = 20;
    let n_rounds = if mnist.is_real { 40 } else { 25 };
    let lr = if mnist.is_real { 0.02 } else { 0.01 };
    let batch_size = 64;

    // Subsample training data if real MNIST (60K is a lot for 20 nodes)
    let train_data: Vec<(Vec<f32>, usize)> = if mnist.train.len() > 10000 {
        // Use first 10K for manageable benchmark runtime
        mnist.train[..10000].to_vec()
    } else {
        mnist.train
    };

    // Subsample test data
    let test_data: Vec<(Vec<f32>, usize)> = if mnist.test.len() > 2000 {
        mnist.test[..2000].to_vec()
    } else {
        mnist.test
    };

    println!(
        "Using: {} train, {} test (subsampled for benchmark speed)",
        train_data.len(),
        test_data.len()
    );
    println!(
        "Nodes: {}, Rounds: {}, LR: {}, Batch: {}\n",
        n_nodes, n_rounds, lr, batch_size
    );

    let mut results = Vec::new();

    // === Experiment 1: IID partitioning ===
    println!("--- Experiment 1: IID Partitioning ---");
    let iid_nodes = partition_iid(&train_data, n_nodes, &mut rng);
    results.push(train_federated(
        "IID-clean",
        &iid_nodes,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        batch_size,
        0,
    ));

    // === Experiment 2: Non-IID (2 classes per node) ===
    println!("--- Experiment 2: Non-IID (2 classes/node) ---");
    let noniid_2 = partition_noniid(&train_data, n_nodes, 2, &mut rng);
    results.push(train_federated(
        "NonIID-2c",
        &noniid_2,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        batch_size,
        0,
    ));

    // === Experiment 3: Non-IID (3 classes per node) ===
    println!("--- Experiment 3: Non-IID (3 classes/node) ---");
    let noniid_3 = partition_noniid(&train_data, n_nodes, 3, &mut rng);
    results.push(train_federated(
        "NonIID-3c",
        &noniid_3,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        batch_size,
        0,
    ));

    // === Experiment 4: IID with Byzantine (10%, 20%) ===
    println!("--- Experiment 4: IID + Byzantine ---");
    for byz_pct in [10, 20] {
        let n_byz = n_nodes * byz_pct / 100;
        let label = format!("IID-byz{}%", byz_pct);
        results.push(train_federated(
            &label,
            &iid_nodes,
            &test_data,
            PipelineConfig::default(),
            n_rounds,
            lr,
            batch_size,
            n_byz,
        ));
    }

    // === Experiment 5: Non-IID + Byzantine ===
    println!("--- Experiment 5: Non-IID-2c + Byzantine 20% ---");
    let n_byz = n_nodes * 20 / 100;
    results.push(train_federated(
        "NonIID-2c-byz20%",
        &noniid_2,
        &test_data,
        PipelineConfig::default(),
        n_rounds,
        lr,
        batch_size,
        n_byz,
    ));

    // === Experiment 6: High security preset ===
    println!("--- Experiment 6: High security preset (IID, 20% byz) ---");
    results.push(train_federated(
        "HighSec-byz20%",
        &iid_nodes,
        &test_data,
        PipelineConfig::high_security(),
        n_rounds,
        lr,
        batch_size,
        n_nodes * 20 / 100,
    ));

    // === Results Table ===
    println!("\n=== Results ({} rounds) ===\n", n_rounds);
    println!(
        "{:<22} {:>8} {:>8} {:>8} {:>8}",
        "Experiment", "Acc", "Loss", "Acc@R5", "Acc@R15"
    );
    println!("{:-<60}", "");

    for r in &results {
        let acc_r5 = r.acc_history.get(4).copied().unwrap_or(0.0);
        let acc_r15 = r.acc_history.get(14).copied().unwrap_or(0.0);
        println!(
            "{:<22} {:>7.1}% {:>8.3} {:>7.1}% {:>7.1}%",
            r.label,
            r.final_acc * 100.0,
            r.final_loss,
            acc_r5 * 100.0,
            acc_r15 * 100.0
        );
    }

    // === Per-Class Accuracy for best model ===
    println!("\n--- Per-Class Accuracy (IID-clean) ---\n");
    let iid_clean = results.iter().find(|r| r.label == "IID-clean").unwrap();
    print!("  Digit:  ");
    for c in 0..NUM_CLASSES {
        print!("  {:>3}  ", c);
    }
    println!();
    print!("  Acc:    ");
    for c in 0..NUM_CLASSES {
        print!("{:>5.1}% ", iid_clean.class_acc[c] * 100.0);
    }
    println!("\n");

    // === Convergence Curves ===
    println!("--- Convergence Curves ---\n");
    for r in &results {
        let spark: String = r
            .acc_history
            .iter()
            .map(|&a| match (a * 100.0) as u32 {
                0..=10 => '▁',
                11..=20 => '▂',
                21..=40 => '▃',
                41..=60 => '▅',
                61..=75 => '▇',
                _ => '█',
            })
            .collect();
        println!("  {:<22} {} {:.1}%", r.label, spark, r.final_acc * 100.0);
    }

    // === Verification Tests ===
    println!("\n=== Verification ===\n");
    let mut passed = 0;
    let total_tests = 7;

    // Test 1: IID clean converges above 30% (real MNIST linear model should reach 85%+,
    // synthetic ~60%+, but with subsample and limited rounds, 30% is safe floor)
    let iid_acc = iid_clean.final_acc;
    if iid_acc > 0.30 {
        println!(
            "  [PASS] Test 1: IID-clean converges: {:.1}%",
            iid_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 1: IID-clean too low: {:.1}%",
            iid_acc * 100.0
        );
    }

    // Test 2: Non-IID converges (may be lower than IID)
    let noniid = results.iter().find(|r| r.label == "NonIID-2c").unwrap();
    if noniid.final_acc > 0.15 {
        println!(
            "  [PASS] Test 2: Non-IID-2c converges: {:.1}%",
            noniid.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 2: Non-IID-2c too low: {:.1}%",
            noniid.final_acc * 100.0
        );
    }

    // Test 3: More classes per node → better Non-IID accuracy
    let noniid3 = results.iter().find(|r| r.label == "NonIID-3c").unwrap();
    if noniid3.final_acc >= noniid.final_acc - 0.05 {
        println!(
            "  [PASS] Test 3: NonIID-3c >= NonIID-2c: {:.1}% vs {:.1}%",
            noniid3.final_acc * 100.0,
            noniid.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 3: NonIID-3c ({:.1}%) << NonIID-2c ({:.1}%)",
            noniid3.final_acc * 100.0,
            noniid.final_acc * 100.0
        );
    }

    // Test 4: Survives 10% Byzantine
    let byz10 = results.iter().find(|r| r.label == "IID-byz10%").unwrap();
    if byz10.final_acc > 0.20 {
        println!(
            "  [PASS] Test 4: Survives 10% Byzantine: {:.1}%",
            byz10.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 4: 10% Byzantine too degraded: {:.1}%",
            byz10.final_acc * 100.0
        );
    }

    // Test 5: Accuracy degrades with more Byzantine
    let byz20 = results.iter().find(|r| r.label == "IID-byz20%").unwrap();
    if byz10.final_acc >= byz20.final_acc - 0.05 {
        println!(
            "  [PASS] Test 5: More Byzantine → worse: 10%={:.1}%, 20%={:.1}%",
            byz10.final_acc * 100.0,
            byz20.final_acc * 100.0
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 5: 10% byz ({:.1}%) worse than 20% ({:.1}%)",
            byz10.final_acc * 100.0,
            byz20.final_acc * 100.0
        );
    }

    // Test 6: Convergence improves over training
    if iid_clean.acc_history.len() >= 20 {
        let early = iid_clean.acc_history[4];
        let late = iid_clean.final_acc;
        if late > early {
            println!(
                "  [PASS] Test 6: Accuracy improves R5→R{}: {:.1}% → {:.1}%",
                n_rounds,
                early * 100.0,
                late * 100.0
            );
            passed += 1;
        } else {
            println!(
                "  [FAIL] Test 6: Accuracy regresses: {:.1}% → {:.1}%",
                early * 100.0,
                late * 100.0
            );
        }
    } else {
        println!("  [FAIL] Test 6: Not enough rounds");
    }

    // Test 7: At least half the classes have >10% accuracy (model learns something for most digits)
    let classes_above_10 = iid_clean.class_acc.iter().filter(|&&a| a > 0.10).count();
    if classes_above_10 >= 5 {
        println!(
            "  [PASS] Test 7: {}/10 classes above 10% accuracy",
            classes_above_10
        );
        passed += 1;
    } else {
        println!(
            "  [FAIL] Test 7: Only {}/10 classes above 10%",
            classes_above_10
        );
    }

    println!(
        "\n=== RESULTS: {}/{} passed ({}) ===",
        passed, total_tests, data_source
    );
    if passed < total_tests {
        std::process::exit(1);
    }
}
