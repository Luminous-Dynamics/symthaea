// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MNIST Cognitive Loop Benchmark
//!
//! Compares HDC-only classification vs CfC-integrated classification on MNIST.
//!
//! ## Method
//! 1. **HDC-only baseline**: Encode pixels → position-level binding → class prototypes → cosine classify
//! 2. **CfC-integrated**: Same HDC encoding → feed through cognitive loop's CfC temporal network →
//!    build CfC output prototypes from hidden states → cosine classify on CfC space
//!
//! The CfC layer adds temporal dynamics and prediction error tracking. The question is whether
//! the CfC's learned temporal representation improves or degrades classification accuracy
//! compared to raw HDC prototypes.
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_mnist_cognitive --release
//! ```

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::hdc::unified_hv::ContinuousHV;

const DATA_DIR: &str = "data/benchmarks/mnist";

// ── MNIST IDX file parsing ──────────────────────────────────────────────────

fn read_idx_images(path: &Path) -> Vec<Vec<u8>> {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "Invalid image file magic number");

    let n_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let n_rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let n_cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels_per_image = n_rows * n_cols;

    println!("  Images: {n_images}, Size: {n_rows}x{n_cols}");

    let data = &buf[16..];
    (0..n_images)
        .map(|i| {
            let start = i * pixels_per_image;
            data[start..start + pixels_per_image].to_vec()
        })
        .collect()
}

fn read_idx_labels(path: &Path) -> Vec<u8> {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "Invalid label file magic number");

    let n_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    println!("  Labels: {n_labels}");

    buf[8..8 + n_labels].to_vec()
}

// ── HDC MNIST Encoder ───────────────────────────────────────────────────────

struct HdcMnistEncoder {
    dim: usize,
    n_levels: usize,
    level_hvs: Vec<ContinuousHV>,
    position_hvs: Vec<ContinuousHV>,
}

impl HdcMnistEncoder {
    fn new(dim: usize, n_levels: usize) -> Self {
        let base_hv = ContinuousHV::random(dim, 1000);
        let flips_per_level = dim / n_levels.max(1);

        let mut level_hvs = Vec::with_capacity(n_levels);
        level_hvs.push(base_hv);

        let mut flip_seed: u64 = 3000;
        for l in 1..n_levels {
            let prev = &level_hvs[l - 1];
            let mut new_values = prev.values.clone();
            for _f in 0..flips_per_level {
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

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
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
}

// ── Cosine similarity helper ────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(b[..n].iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║          MNIST: HDC-only vs CfC-Integrated Classification           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // ── Load data ────────────────────────────────────────────────────────────
    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: MNIST data not found at {DATA_DIR}");
        eprintln!("Run the data download script first.");
        return;
    }

    println!("Loading MNIST data...");
    let train_images = read_idx_images(&data_path.join("train-images-idx3-ubyte"));
    let train_labels = read_idx_labels(&data_path.join("train-labels-idx1-ubyte"));
    let test_images = read_idx_images(&data_path.join("t10k-images-idx3-ubyte"));
    let test_labels = read_idx_labels(&data_path.join("t10k-labels-idx1-ubyte"));
    println!();

    let dim = 4096;
    let n_levels = 32;

    // ── Step 1: HDC Encoding ────────────────────────────────────────────────
    println!("Initializing HDC encoder (dim={dim}, levels={n_levels})...");
    let encoder = HdcMnistEncoder::new(dim, n_levels);

    // ── Step 2: Build HDC-only class prototypes ─────────────────────────────
    println!("\n━━━ Phase 1: HDC-Only Baseline ━━━");
    let hdc_start = Instant::now();

    let mut hdc_prototypes: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; 10];
    let mut class_counts = [0usize; 10];

    for (img, &label) in train_images.iter().zip(train_labels.iter()) {
        let encoded = encoder.encode(img);
        let c = label as usize;
        for (p, &e) in hdc_prototypes[c].iter_mut().zip(encoded.values.iter()) {
            *p += e;
        }
        class_counts[c] += 1;
    }

    // Normalize
    for proto in &mut hdc_prototypes {
        let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in proto.iter_mut() {
                *v /= norm;
            }
        }
    }

    println!("  HDC training: {:.1}s", hdc_start.elapsed().as_secs_f64());

    // Retrain (5 iterations)
    let retrain_start = Instant::now();
    for iter in 0..5 {
        let mut corrections = 0;
        for (img, &label) in train_images.iter().zip(train_labels.iter()) {
            let encoded = encoder.encode(img);
            let actual = label as usize;

            let mut best_class = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, proto) in hdc_prototypes.iter().enumerate() {
                let sim = cosine_similarity(&encoded.values, proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_class = c;
                }
            }

            if best_class != actual {
                for (p, &e) in hdc_prototypes[best_class]
                    .iter_mut()
                    .zip(encoded.values.iter())
                {
                    *p -= 0.1 * e;
                }
                for (p, &e) in hdc_prototypes[actual].iter_mut().zip(encoded.values.iter()) {
                    *p += 0.1 * e;
                }
                corrections += 1;
            }
        }
        let acc = 1.0 - corrections as f64 / train_images.len() as f64;
        println!(
            "  Retrain {}/5: {corrections} corrections, train acc = {:.2}%",
            iter + 1,
            acc * 100.0
        );
    }
    // Normalize after retrain
    for proto in &mut hdc_prototypes {
        let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in proto.iter_mut() {
                *v /= norm;
            }
        }
    }
    println!("  Retrain: {:.1}s", retrain_start.elapsed().as_secs_f64());

    // Test HDC-only
    let test_start = Instant::now();
    let mut hdc_correct = 0;
    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let encoded = encoder.encode(img);
        let mut best_class = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (c, proto) in hdc_prototypes.iter().enumerate() {
            let sim = cosine_similarity(&encoded.values, proto);
            if sim > best_sim {
                best_sim = sim;
                best_class = c;
            }
        }
        if best_class == label as usize {
            hdc_correct += 1;
        }
    }
    let hdc_accuracy = hdc_correct as f64 / test_images.len() as f64;
    let hdc_test_time = test_start.elapsed().as_secs_f64();
    println!(
        "\n  HDC-only accuracy: {:.2}% ({hdc_correct}/{}) in {:.1}s",
        hdc_accuracy * 100.0,
        test_images.len(),
        hdc_test_time
    );

    // ── Step 3: CfC-Integrated ──────────────────────────────────────────────
    println!("\n━━━ Phase 2: CfC-Integrated (HDC → Cognitive Loop) ━━━");

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0, // always learn
        ..Default::default()
    })
    .unwrap();

    let cfc_output_dim = service.stats().total_cycles; // just to get service running
    let _ = cfc_output_dim;

    // Warm up with a few samples
    println!("  Warming up cognitive loop (100 samples)...");
    let warmup_start = Instant::now();
    for img in train_images.iter().take(100) {
        let hdv = encoder.encode(img);
        let _ = service.cycle_with_hv(&hdv);
    }
    println!(
        "  Warmup: {:.1}s ({:.1}ms/cycle)",
        warmup_start.elapsed().as_secs_f64(),
        warmup_start.elapsed().as_secs_f64() * 10.0
    );

    // Build CfC output prototypes from training set
    // Use a subset (5000 samples) for speed
    let train_subset = 5000;
    println!("  Building CfC output prototypes ({train_subset} training samples)...");
    let cfc_train_start = Instant::now();

    let mut cfc_prototypes: Vec<Vec<f32>> = Vec::new();
    let mut cfc_counts = [0usize; 10];
    let mut cfc_dim = 0;

    for (img, &label) in train_images
        .iter()
        .zip(train_labels.iter())
        .take(train_subset)
    {
        let hdv = encoder.encode(img);
        let result = service.cycle_with_hv(&hdv);
        let c = label as usize;

        if cfc_prototypes.is_empty() {
            cfc_dim = result.output.len();
            cfc_prototypes = vec![vec![0.0f32; cfc_dim]; 10];
        }

        for (p, &o) in cfc_prototypes[c].iter_mut().zip(result.output.iter()) {
            *p += o;
        }
        cfc_counts[c] += 1;
    }

    // Normalize CfC prototypes
    for proto in &mut cfc_prototypes {
        let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in proto.iter_mut() {
                *v /= norm;
            }
        }
    }

    println!(
        "  CfC training: {:.1}s (output dim={cfc_dim})",
        cfc_train_start.elapsed().as_secs_f64()
    );

    // Test CfC-integrated
    println!("  Testing CfC-integrated classification...");
    let cfc_test_start = Instant::now();
    let mut cfc_correct = 0;
    let mut total_prediction_error = 0.0f64;

    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let hdv = encoder.encode(img);
        let result = service.cycle_with_hv(&hdv);
        total_prediction_error += result.prediction_error as f64;

        let mut best_class = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (c, proto) in cfc_prototypes.iter().enumerate() {
            let sim = cosine_similarity(&result.output, proto);
            if sim > best_sim {
                best_sim = sim;
                best_class = c;
            }
        }
        if best_class == label as usize {
            cfc_correct += 1;
        }
    }
    let cfc_accuracy = cfc_correct as f64 / test_images.len() as f64;
    let cfc_test_time = cfc_test_start.elapsed().as_secs_f64();
    let avg_prediction_error = total_prediction_error / test_images.len() as f64;

    println!(
        "\n  CfC-integrated accuracy: {:.2}% ({cfc_correct}/{}) in {:.1}s",
        cfc_accuracy * 100.0,
        test_images.len(),
        cfc_test_time
    );
    println!("  Avg prediction error: {:.6}", avg_prediction_error);
    println!(
        "  Avg CfC inference: {:.3}ms/sample",
        cfc_test_time / test_images.len() as f64 * 1000.0
    );

    // ── Summary ─────────────────────────────────────────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                         COMPARISON SUMMARY                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:35} │ {:>8} │ {:>10} ║",
        "Method", "Accuracy", "ms/sample"
    );
    println!("╟─────────────────────────────────────┼──────────┼────────────╢");
    println!(
        "║ {:35} │ {:>7.2}% │ {:>9.3} ║",
        "HDC-only (4K, 32L, 5 retrain)",
        hdc_accuracy * 100.0,
        hdc_test_time / test_images.len() as f64 * 1000.0
    );
    println!(
        "║ {:35} │ {:>7.2}% │ {:>9.3} ║",
        "CfC-integrated (HDC→CfC→classify)",
        cfc_accuracy * 100.0,
        cfc_test_time / test_images.len() as f64 * 1000.0
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    let delta = cfc_accuracy - hdc_accuracy;
    if delta > 0.0 {
        println!("\n  CfC advantage: +{:.2}pp", delta * 100.0);
    } else if delta < 0.0 {
        println!("\n  CfC disadvantage: {:.2}pp", delta * 100.0);
    } else {
        println!("\n  Identical accuracy");
    }

    println!(
        "  CfC avg prediction error: {:.6} (lower = better temporal model)",
        avg_prediction_error
    );
    println!("  Total cognitive cycles: {}", service.stats().total_cycles);
}