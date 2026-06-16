// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MNIST HDC Dimension Ablation Study
//!
//! Proves that HDC dimensionality is a meaningful hyperparameter by sweeping
//! across 5 dimension sizes. Validates the 16,384D choice used in Symthaea.
//!
//! ## Ablation Conditions
//! | Dimension | Levels | Retrain Iters | Expected Accuracy |
//! |-----------|--------|---------------|-------------------|
//! | 256       | 32     | 5             | ~50-65%           |
//! | 1,024     | 32     | 5             | ~70-78%           |
//! | 4,096     | 32     | 5             | ~82-87%           |
//! | 8,192     | 32     | 5             | ~85-90%           |
//! | 16,384    | 32     | 5             | ~87-92%           |
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_ablation_mnist --release
//! ```

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;

const DATA_DIR: &str = "data/benchmarks/mnist";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       MNIST HDC Dimension Ablation Study                   ║");
    println!("║       Validating Dimensionality as Key Hyperparameter      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: MNIST data not found at {}", DATA_DIR);
        eprintln!("Download MNIST first (benchmark_mnist_hdc has instructions).");
        return;
    }

    println!("Loading MNIST data...");
    let train_images = read_idx_images(&data_path.join("train-images-idx3-ubyte"));
    let train_labels = read_idx_labels(&data_path.join("train-labels-idx1-ubyte"));
    let test_images = read_idx_images(&data_path.join("t10k-images-idx3-ubyte"));
    let test_labels = read_idx_labels(&data_path.join("t10k-labels-idx1-ubyte"));
    println!();

    // Ablation conditions: (dim, levels, retrain_iters, label)
    let conditions: Vec<(usize, usize, usize, &str)> = vec![
        (256, 32, 5, "256D"),
        (1024, 32, 5, "1,024D"),
        (4096, 32, 5, "4,096D"),
        (8192, 32, 5, "8,192D"),
        (16384, 32, 5, "16,384D"),
    ];

    #[allow(clippy::type_complexity)]
    let mut results: Vec<(usize, &str, f64, f64, Vec<f64>, f64)> = Vec::new();

    for (dim, levels, retrain_iters, label) in &conditions {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Dimension: {} (levels={}, retrain={})",
            label, levels, retrain_iters
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let total_start = Instant::now();

        let mut classifier = HdcClassifier::new(*dim, *levels);

        println!("  Training...");
        classifier.train(&train_images, &train_labels);

        println!("  Baseline test...");
        let baseline = classifier.test(&test_images, &test_labels);
        println!(
            "  Baseline accuracy: {:.2}% ({}/{})",
            baseline.accuracy * 100.0,
            baseline.correct,
            baseline.total
        );

        println!("  Retraining (lr=0.1, {} iters)...", retrain_iters);
        classifier.retrain(&train_images, &train_labels, 0.1, *retrain_iters);

        println!("  Final test...");
        let final_result = classifier.test(&test_images, &test_labels);
        let total_time = total_start.elapsed().as_secs_f64();

        println!(
            "  Final accuracy: {:.2}% ({}/{})",
            final_result.accuracy * 100.0,
            final_result.correct,
            final_result.total
        );
        println!("  Total time: {:.1}s\n", total_time);

        results.push((
            *dim,
            label,
            baseline.accuracy,
            final_result.accuracy,
            final_result.per_class_accuracy,
            total_time,
        ));
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                   DIMENSION ABLATION RESULTS                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:>8} │ {:>10} │ {:>10} │ {:>8} │ {:>8} ║",
        "Dim", "Baseline", "Retrained", "Gain", "Time"
    );
    println!("╟──────────┼────────────┼────────────┼──────────┼──────────╢");

    for (_dim, label, baseline, retrained, _per_class, time) in &results {
        let gain = (*retrained - *baseline) * 100.0;
        println!(
            "║ {:>8} │ {:>9.2}% │ {:>9.2}% │ {:>+7.2}pp │ {:>7.1}s ║",
            label,
            baseline * 100.0,
            retrained * 100.0,
            gain,
            time,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Validation: monotonic accuracy increase with dimension
    println!("ABLATION VALIDATION");
    println!("═══════════════════════════════════════════════════════════════");

    let retrained_accs: Vec<f64> = results.iter().map(|(_, _, _, r, _, _)| *r).collect();

    let mut monotonic = true;
    for i in 1..retrained_accs.len() {
        let pass = retrained_accs[i] >= retrained_accs[i - 1];
        if !pass {
            monotonic = false;
        }
        println!(
            "  {} >= {}: {:.2}% >= {:.2}%  {}",
            results[i].1,
            results[i - 1].1,
            retrained_accs[i] * 100.0,
            retrained_accs[i - 1] * 100.0,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    let smallest = retrained_accs[0];
    let largest = retrained_accs[retrained_accs.len() - 1];
    let spread = (largest - smallest) * 100.0;

    println!(
        "\n  Monotonic increase:      {}",
        if monotonic { "PASS" } else { "FAIL" }
    );
    println!(
        "  Spread ({}D → {}D): {:.1}pp",
        results[0].0,
        results[results.len() - 1].0,
        spread
    );
    println!(
        "  16,384D > 80%:           {:.2}%  {}",
        largest * 100.0,
        if largest > 0.80 { "PASS" } else { "FAIL" }
    );

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "MNIST HDC Dimension Ablation",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "conditions": results.iter().map(|(dim, label, baseline, retrained, per_class, time)| {
            serde_json::json!({
                "dimension": dim,
                "label": label,
                "baseline_accuracy": baseline,
                "retrained_accuracy": retrained,
                "per_class_accuracy": per_class,
                "time_s": time,
            })
        }).collect::<Vec<_>>(),
        "validation": {
            "monotonic_increase": monotonic,
            "spread_pp": spread,
            "best_above_80": largest > 0.80,
        },
    });

    std::fs::create_dir_all("data/benchmarks/ablation-mnist").ok();
    if let Ok(f) = File::create("data/benchmarks/ablation-mnist/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to data/benchmarks/ablation-mnist/results.json");
    }
}

/// Minimal HDC classifier for ablation (no spatial/adaptive features).
struct HdcClassifier {
    dim: usize,
    n_levels: usize,
    level_hvs: Vec<ContinuousHV>,
    position_hvs: Vec<ContinuousHV>,
    class_prototypes: Vec<Option<ContinuousHV>>,
    class_counts: Vec<usize>,
}

impl HdcClassifier {
    fn new(dim: usize, n_levels: usize) -> Self {
        let t = Instant::now();

        // Level HVs with progressive random-flip encoding
        let base_hv = ContinuousHV::random(dim, 1000);
        let flips_per_level = dim / n_levels.max(1);
        let mut level_hvs: Vec<ContinuousHV> = Vec::with_capacity(n_levels);
        level_hvs.push(base_hv);

        let mut flip_seed: u64 = 3000;
        for l in 1..n_levels {
            let prev = &level_hvs[l - 1];
            let mut new_values = prev.values.clone();
            for _ in 0..flips_per_level {
                flip_seed = flip_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx = (flip_seed >> 33) as usize % dim;
                new_values[idx] = -new_values[idx];
            }
            level_hvs.push(ContinuousHV::from_vec(new_values));
        }

        let position_hvs: Vec<ContinuousHV> = (0..784)
            .map(|p| ContinuousHV::random(dim, 10000 + p as u64))
            .collect();

        println!("  Init (dim={}): {:.0}ms", dim, t.elapsed().as_millis());

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
            class_prototypes: (0..10).map(|_| None).collect(),
            class_counts: vec![0; 10],
        }
    }

    fn encode(&self, pixels: &[u8]) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];
        let level_size = 256.0 / self.n_levels as f32;

        for (pos, &pixel) in pixels.iter().enumerate() {
            let level = ((pixel as f32 / level_size) as usize).min(self.n_levels - 1);
            let bound = self.position_hvs[pos].bind(&self.level_hvs[level]);
            for (acc, &val) in accumulator.iter_mut().zip(bound.values.iter()) {
                *acc += val;
            }
        }

        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }
        ContinuousHV::from_vec(accumulator)
    }

    fn train(&mut self, images: &[Vec<u8>], labels: &[u8]) {
        let t = Instant::now();
        let mut accumulators: Vec<Vec<f32>> = (0..10).map(|_| vec![0.0f32; self.dim]).collect();

        for (img, &label) in images.iter().zip(labels.iter()) {
            let encoded = self.encode(img);
            let class = label as usize;
            for (acc, &val) in accumulators[class].iter_mut().zip(encoded.values.iter()) {
                *acc += val;
            }
            self.class_counts[class] += 1;
        }

        for (class, accumulator) in accumulators.iter_mut().enumerate().take(10) {
            if self.class_counts[class] > 0 {
                let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in accumulator.iter_mut() {
                        *v /= norm;
                    }
                }
                self.class_prototypes[class] = Some(ContinuousHV::from_vec(accumulator.clone()));
            }
        }

        println!(
            "  Trained {} samples in {:.1}s",
            images.len(),
            t.elapsed().as_secs_f64()
        );
    }

    fn retrain(&mut self, images: &[Vec<u8>], labels: &[u8], lr: f32, iterations: usize) {
        for iter in 0..iterations {
            let t = Instant::now();
            let mut corrections = 0;

            for (img, &label) in images.iter().zip(labels.iter()) {
                let encoded = self.encode(img);
                let actual = label as usize;

                let mut best_class = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (class, proto) in self.class_prototypes.iter().enumerate() {
                    if let Some(ref p) = proto {
                        let sim = encoded.similarity(p);
                        if sim > best_sim {
                            best_sim = sim;
                            best_class = class;
                        }
                    }
                }

                if best_class != actual {
                    if let Some(ref mut proto) = self.class_prototypes[best_class] {
                        for (p, &e) in proto.values.iter_mut().zip(encoded.values.iter()) {
                            *p -= lr * e;
                        }
                    }
                    if let Some(ref mut proto) = self.class_prototypes[actual] {
                        for (p, &e) in proto.values.iter_mut().zip(encoded.values.iter()) {
                            *p += lr * e;
                        }
                    }
                    corrections += 1;
                }
            }

            let accuracy = 1.0 - corrections as f64 / images.len() as f64;
            println!(
                "    Iter {}/{}: {} corrections, train acc = {:.2}% ({:.1}s)",
                iter + 1,
                iterations,
                corrections,
                accuracy * 100.0,
                t.elapsed().as_secs_f64()
            );

            if corrections < images.len() / 200 {
                println!("    Early stopping");
                break;
            }
        }

        // Normalize prototypes
        for ref mut p in self.class_prototypes.iter_mut().flatten() {
            let norm: f32 = p.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut p.values {
                    *v /= norm;
                }
            }
        }
    }

    fn test(&self, images: &[Vec<u8>], labels: &[u8]) -> TestResult {
        let t = Instant::now();
        let mut correct = 0;
        let mut per_class_correct = [0usize; 10];
        let mut per_class_total = [0usize; 10];

        for (img, &label) in images.iter().zip(labels.iter()) {
            let encoded = self.encode(img);
            let actual = label as usize;

            let mut best_class = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (class, proto) in self.class_prototypes.iter().enumerate() {
                if let Some(ref p) = proto {
                    let sim = encoded.similarity(p);
                    if sim > best_sim {
                        best_sim = sim;
                        best_class = class;
                    }
                }
            }

            per_class_total[actual] += 1;
            if best_class == actual {
                correct += 1;
                per_class_correct[actual] += 1;
            }
        }

        let n = images.len();
        TestResult {
            accuracy: correct as f64 / n as f64,
            correct,
            total: n,
            per_class_accuracy: per_class_correct
                .iter()
                .zip(per_class_total.iter())
                .map(|(&c, &t)| if t > 0 { c as f64 / t as f64 } else { 0.0 })
                .collect(),
            time_s: t.elapsed().as_secs_f64(),
        }
    }
}

struct TestResult {
    accuracy: f64,
    correct: usize,
    total: usize,
    per_class_accuracy: Vec<f64>,
    #[allow(dead_code)]
    time_s: f64,
}

/// Parse IDX image file format
fn read_idx_images(path: &Path) -> Vec<Vec<u8>> {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "Invalid image file magic number");

    let n_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let n_rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let n_cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels = n_rows * n_cols;

    println!("  Images: {}, Size: {}x{}", n_images, n_rows, n_cols);

    let data = &buf[16..];
    (0..n_images)
        .map(|i| data[i * pixels..(i + 1) * pixels].to_vec())
        .collect()
}

/// Parse IDX label file format
fn read_idx_labels(path: &Path) -> Vec<u8> {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "Invalid label file magic number");

    let n_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    println!("  Labels: {}", n_labels);

    buf[8..8 + n_labels].to_vec()
}