// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CIFAR-10 HDC Classification Benchmark
//!
//! Tests Symthaea's HDC encoding on CIFAR-10 (Krizhevsky 2009), a standard
//! 10-class color image benchmark that is significantly harder than MNIST.
//!
//! ## Method
//! 1. Level encoding: Quantize pixel intensities into Q levels (per channel)
//! 2. Position encoding: One random HV per spatial position (1024 = 32x32)
//! 3. Channel encoding: One random HV per color channel (R, G, B)
//! 4. Encode: bundle(bind(position_i, bind(channel_j, level[pixel_ij])))
//! 5. Train: Average all encoded HVs per class → 10 class prototypes
//! 6. Classify: argmax cosine_similarity(encoded_test, class_prototype)
//!
//! ## Honest Assessment
//! CIFAR-10 is designed for CNNs (state-of-art >96%). Pixel-level HDC encoding
//! cannot capture the spatial hierarchies that convolutional architectures learn.
//! Expected accuracy: 25-35% (above 10% random baseline, below CNN performance).
//! The value is: (a) validating color+spatial HDC encoding, (b) establishing a
//! baseline for future multi-scale HDC approaches.
//!
//! ## Dataset
//! CIFAR-10 (Krizhevsky 2009): 50K training + 10K test, 32x32 RGB, 10 classes
//! Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
//! Format: Custom binary — 10000 images per batch file
//!
//! ## Download
//! ```bash
//! cd data/benchmarks/cifar10
//! curl -sL https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xzf -
//! ```
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_cifar10_hdc --release
//! ```

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

const CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

const IMAGE_SIZE: usize = 32 * 32; // 1024 pixels per channel
const CHANNELS: usize = 3; // RGB
const PIXELS_PER_IMAGE: usize = IMAGE_SIZE * CHANNELS; // 3072
const _IMAGES_PER_BATCH: usize = 10000;

fn data_dir() -> PathBuf {
    env::var("SYMTHAEA_CIFAR10_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/cifar10/cifar-10-batches-bin"))
}

// ─────────────────────────────────────────────────────────────────────
// Data loading
// ─────────────────────────────────────────────────────────────────────

/// CIFAR-10 binary format: each record = 1 label byte + 3072 pixel bytes
/// Pixels are stored as: 1024 red, 1024 green, 1024 blue (planar)
fn load_batch(path: &Path) -> Option<(Vec<Vec<u8>>, Vec<u8>)> {
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;

    let record_size = 1 + PIXELS_PER_IMAGE; // label + pixels
    let n_images = buf.len() / record_size;

    let mut images = Vec::with_capacity(n_images);
    let mut labels = Vec::with_capacity(n_images);

    for i in 0..n_images {
        let offset = i * record_size;
        labels.push(buf[offset]);
        images.push(buf[offset + 1..offset + record_size].to_vec());
    }

    Some((images, labels))
}

fn load_training_data(dir: &Path) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    for batch_num in 1..=5 {
        let batch_path = dir.join(format!("data_batch_{}.bin", batch_num));
        if let Some((images, labels)) = load_batch(&batch_path) {
            println!("  Loaded batch {}: {} images", batch_num, images.len());
            all_images.extend(images);
            all_labels.extend(labels);
        } else {
            eprintln!("  Warning: could not load {}", batch_path.display());
        }
    }

    (all_images, all_labels)
}

fn load_test_data(dir: &Path) -> (Vec<Vec<u8>>, Vec<u8>) {
    let test_path = dir.join("test_batch.bin");
    load_batch(&test_path).unwrap_or_else(|| {
        eprintln!("Cannot load test batch: {}", test_path.display());
        (vec![], vec![])
    })
}

// ─────────────────────────────────────────────────────────────────────
// HDC CIFAR-10 Classifier
// ─────────────────────────────────────────────────────────────────────

struct HdcCifar10Classifier {
    dim: usize,
    n_levels: usize,
    /// Level HVs (one per quantization level)
    level_hvs: Vec<ContinuousHV>,
    /// Position HVs (one per spatial position, 1024 = 32x32)
    position_hvs: Vec<ContinuousHV>,
    /// Channel HVs (R, G, B)
    channel_hvs: [ContinuousHV; 3],
    /// Class prototypes (one per class 0-9)
    class_prototypes: Vec<Option<ContinuousHV>>,
    /// Number of samples per class
    class_counts: Vec<usize>,
}

impl HdcCifar10Classifier {
    fn new(dim: usize, n_levels: usize) -> Self {
        let t = Instant::now();

        // Generate level HVs with progressive flip encoding (ordinal similarity)
        let base_hv = ContinuousHV::random(dim, 5000);
        let flips_per_level = dim / n_levels.max(1);

        let mut level_hvs = Vec::with_capacity(n_levels);
        level_hvs.push(base_hv);

        let mut flip_seed: u64 = 7000;
        for l in 1..n_levels {
            let prev = &level_hvs[l - 1];
            let mut new_values = prev.values.clone();

            for _f in 0..flips_per_level {
                flip_seed = flip_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (flip_seed >> 33) as usize % dim;
                new_values[idx] = -new_values[idx];
            }

            level_hvs.push(ContinuousHV::from_values(new_values));
            let _ = l; // suppress warning
        }

        // Generate position HVs (one per spatial position)
        let position_hvs: Vec<ContinuousHV> = (0..IMAGE_SIZE)
            .map(|i| ContinuousHV::random(dim, 10000 + i as u64))
            .collect();

        // Channel HVs (R=0, G=1, B=2)
        let channel_hvs = [
            ContinuousHV::random(dim, 20000),
            ContinuousHV::random(dim, 20001),
            ContinuousHV::random(dim, 20002),
        ];

        println!(
            "  HDC CIFAR-10: dim={}, levels={}, init={:.1}ms",
            dim,
            n_levels,
            t.elapsed().as_millis()
        );

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
            channel_hvs,
            class_prototypes: vec![None; 10],
            class_counts: vec![0; 10],
        }
    }

    /// Encode a single CIFAR-10 image (3072 bytes: 1024 R + 1024 G + 1024 B)
    fn encode(&self, pixels: &[u8]) -> ContinuousHV {
        let mut acc = ContinuousHV::zero(self.dim);

        for ch in 0..CHANNELS {
            let ch_offset = ch * IMAGE_SIZE;
            for pos in 0..IMAGE_SIZE {
                let pixel_val = pixels[ch_offset + pos] as usize;
                let level = (pixel_val * self.n_levels) / 256;
                let level = level.min(self.n_levels - 1);

                // bind(position, bind(channel, level))
                let bound = self.channel_hvs[ch]
                    .bind(&self.level_hvs[level])
                    .bind(&self.position_hvs[pos]);

                acc.add_in_place(&bound);
            }
        }

        acc.normalize();
        acc
    }

    /// Train on a batch of images
    fn train(&mut self, images: &[Vec<u8>], labels: &[u8]) {
        for (img, &label) in images.iter().zip(labels.iter()) {
            let class = label as usize;
            if class >= 10 {
                continue;
            }

            let encoded = self.encode(img);

            match &mut self.class_prototypes[class] {
                None => {
                    self.class_prototypes[class] = Some(encoded);
                    self.class_counts[class] = 1;
                }
                Some(proto) => {
                    proto.add_in_place(&encoded);
                    self.class_counts[class] += 1;
                }
            }
        }
    }

    /// Normalize prototypes after training
    fn finalize(&mut self) {
        for (proto, &count) in self
            .class_prototypes
            .iter_mut()
            .zip(self.class_counts.iter())
        {
            if let Some(p) = proto {
                if count > 1 {
                    p.scale_in_place(1.0 / count as f32);
                }
                p.normalize();
            }
        }
    }

    /// Classify a single image
    fn classify(&self, pixels: &[u8]) -> usize {
        let encoded = self.encode(pixels);
        let mut best_class = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, proto) in self.class_prototypes.iter().enumerate() {
            if let Some(p) = proto {
                let sim = encoded.similarity(p);
                if sim > best_sim {
                    best_sim = sim;
                    best_class = i;
                }
            }
        }

        best_class
    }

    /// Retrain: remove misclassified samples from prototypes, re-add correctly
    fn retrain(&mut self, images: &[Vec<u8>], labels: &[u8]) {
        let mut new_prototypes: Vec<Option<ContinuousHV>> = vec![None; 10];
        let mut new_counts = vec![0usize; 10];

        for (img, &label) in images.iter().zip(labels.iter()) {
            let class = label as usize;
            if class >= 10 {
                continue;
            }

            let encoded = self.encode(img);
            let predicted = self.classify(img);

            if predicted == class {
                // Correctly classified: add to prototype
                match &mut new_prototypes[class] {
                    None => {
                        new_prototypes[class] = Some(encoded);
                        new_counts[class] = 1;
                    }
                    Some(proto) => {
                        proto.add_in_place(&encoded);
                        new_counts[class] += 1;
                    }
                }
            } else {
                // Misclassified: add to correct class, subtract from wrong class
                match &mut new_prototypes[class] {
                    None => {
                        new_prototypes[class] = Some(encoded.clone());
                        new_counts[class] = 1;
                    }
                    Some(proto) => {
                        proto.add_in_place(&encoded);
                        new_counts[class] += 1;
                    }
                }
                if let Some(wrong_proto) = &mut new_prototypes[predicted] {
                    let mut neg = encoded;
                    neg.scale_in_place(-1.0);
                    wrong_proto.add_in_place(&neg);
                }
            }
        }

        self.class_prototypes = new_prototypes;
        self.class_counts = new_counts;
        self.finalize();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       CIFAR-10 HDC Classification Benchmark                ║");
    println!("║       Color Image Recognition via Hyperdimensional Computing║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dir = data_dir();
    if !dir.exists() {
        eprintln!("CIFAR-10 data not found at: {}", dir.display());
        eprintln!(
            "Download: cd data/benchmarks/cifar10 && curl -sL https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xzf -"
        );
        eprintln!("Set SYMTHAEA_CIFAR10_DATA_DIR to override.");
        return;
    }

    // ── Load data ────────────────────────────────────────────────────
    println!("Loading training data...");
    let (train_images, train_labels) = load_training_data(&dir);
    println!("  Total training: {} images\n", train_images.len());

    println!("Loading test data...");
    let (test_images, test_labels) = load_test_data(&dir);
    println!("  Total test: {} images\n", test_images.len());

    if train_images.is_empty() || test_images.is_empty() {
        eprintln!("No data loaded. Check file paths.");
        return;
    }

    // ── Train + evaluate ────────────────────────────────────────────
    let dim = 4096;
    let n_levels = 32;
    let n_retrains = 3;
    // Use a subset for faster training (full 50K is slow at 4096D)
    let train_subset = 10000;

    let mut classifier = HdcCifar10Classifier::new(dim, n_levels);

    println!(
        "Training on {} images (subset of {})...",
        train_subset,
        train_images.len()
    );
    let t = Instant::now();
    classifier.train(
        &train_images[..train_subset.min(train_images.len())],
        &train_labels[..train_subset.min(train_labels.len())],
    );
    classifier.finalize();
    println!("  Initial training: {:.1}s\n", t.elapsed().as_secs_f64());

    // Retrain passes
    for pass in 0..n_retrains {
        let t = Instant::now();
        classifier.retrain(
            &train_images[..train_subset.min(train_images.len())],
            &train_labels[..train_subset.min(train_labels.len())],
        );
        println!(
            "  Retrain pass {}: {:.1}s",
            pass + 1,
            t.elapsed().as_secs_f64()
        );
    }
    println!();

    // ── Evaluate on test set ────────────────────────────────────────
    println!("Evaluating on {} test images...", test_images.len());
    let t = Instant::now();
    let mut correct = 0;
    let mut per_class_correct = [0u32; 10];
    let mut per_class_total = [0u32; 10];
    let mut confusion = [[0u32; 10]; 10]; // [true][predicted]

    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let class = label as usize;
        if class >= 10 {
            continue;
        }
        let predicted = classifier.classify(img);
        per_class_total[class] += 1;
        confusion[class][predicted] += 1;
        if predicted == class {
            correct += 1;
            per_class_correct[class] += 1;
        }
    }

    let total = test_images.len();
    let accuracy = correct as f64 / total as f64;
    let eval_time = t.elapsed();

    println!(
        "  Time: {:.1}s ({:.0} images/sec)\n",
        eval_time.as_secs_f64(),
        total as f64 / eval_time.as_secs_f64()
    );

    // ── Results ─────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Overall accuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        total
    );
    println!("  Random baseline:  10.0%");
    println!("  CNN SOTA:         96.5%+ (EfficientNet)");
    println!();

    println!("  Per-class accuracy:");
    for i in 0..10 {
        let class_acc = if per_class_total[i] > 0 {
            per_class_correct[i] as f64 / per_class_total[i] as f64
        } else {
            0.0
        };
        println!(
            "    {:12} {:.1}% ({}/{})",
            CLASSES[i],
            class_acc * 100.0,
            per_class_correct[i],
            per_class_total[i]
        );
    }

    println!();
    println!(
        "  Configuration: dim={}, levels={}, retrains={}, train_subset={}",
        dim, n_levels, n_retrains, train_subset
    );

    // ── Save results ────────────────────────────────────────────────
    let results = serde_json::json!({
        "benchmark": "CIFAR-10",
        "method": "HDC pixel-level encoding",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "dim": dim,
        "n_levels": n_levels,
        "n_retrains": n_retrains,
        "train_subset": train_subset,
        "per_class_accuracy": (0..10).map(|i| {
            serde_json::json!({
                "class": CLASSES[i],
                "accuracy": if per_class_total[i] > 0 {
                    per_class_correct[i] as f64 / per_class_total[i] as f64
                } else { 0.0 },
                "correct": per_class_correct[i],
                "total": per_class_total[i],
            })
        }).collect::<Vec<_>>(),
        "eval_time_secs": eval_time.as_secs_f64(),
        "note": "Pixel-level HDC baseline. Expected 25-35%. CNN SOTA >96%.",
    });

    let results_path = PathBuf::from("data/benchmarks/cifar10/results.json");
    if let Ok(file) = File::create(&results_path) {
        let _ = serde_json::to_writer_pretty(file, &results);
        println!("\n  Results saved to {}", results_path.display());
    }
}