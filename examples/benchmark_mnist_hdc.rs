//! # MNIST HDC Classification Benchmark
//!
//! Standard MNIST handwritten digit recognition using Hyperdimensional Computing.
//! This validates Symthaea's HDC encoding and classification pipeline against
//! the most widely used ML benchmark dataset.
//!
//! ## Method
//! 1. Level encoding: Quantize pixel intensities into Q levels, each mapped to a random HV
//! 2. Position encoding: One random HV per pixel position (784 positions)
//! 3. Encode: bundle(bind(position_i, level[pixel_i]) for each pixel)
//! 4. Train: Average (bundle) all encoded HVs per digit class → 10 class prototypes
//! 5. Classify: argmax cosine_similarity(encoded_test, class_prototype)
//!
//! ## Expected Results
//! - HDC at dim=4096, Q=16: ~88-90% accuracy
//! - HDC at dim=8192, Q=32: ~90-92% accuracy
//! - HDC at dim=16384, Q=64: ~92-94% accuracy
//!
//! ## Literature Reference
//! Imani et al. "HDC: Hyperdimensional Computing for Efficient Classification" (2019)
//! Kanerva "Hyperdimensional Computing: An Introduction to Computing in Distributed
//! Representation with High-Dimensional Random Vectors" (2009)
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_mnist_hdc --release
//! ```

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;

const DATA_DIR: &str = "data/benchmarks/mnist";

/// Parse IDX file format (MNIST binary format)
fn read_idx_images(path: &Path) -> Vec<Vec<u8>> {
    let mut file = File::open(path).expect(&format!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "Invalid image file magic number");

    let n_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let n_rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let n_cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels_per_image = n_rows * n_cols;

    println!("  Images: {}, Size: {}x{}", n_images, n_rows, n_cols);

    let data = &buf[16..];
    (0..n_images)
        .map(|i| {
            let start = i * pixels_per_image;
            data[start..start + pixels_per_image].to_vec()
        })
        .collect()
}

fn read_idx_labels(path: &Path) -> Vec<u8> {
    let mut file = File::open(path).expect(&format!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "Invalid label file magic number");

    let n_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    println!("  Labels: {}", n_labels);

    buf[8..8 + n_labels].to_vec()
}

/// HDC MNIST classifier
struct HdcMnistClassifier {
    dim: usize,
    n_levels: usize,
    /// Level hypervectors (one per quantization level)
    level_hvs: Vec<ContinuousHV>,
    /// Position hypervectors (one per pixel position, 784 for MNIST)
    position_hvs: Vec<ContinuousHV>,
    /// Class prototypes (one per digit 0-9)
    class_prototypes: Vec<Option<ContinuousHV>>,
    /// Number of samples per class (for averaging)
    class_counts: Vec<usize>,
}

impl HdcMnistClassifier {
    fn new(dim: usize, n_levels: usize) -> Self {
        println!("  Initializing HDC classifier: dim={}, levels={}", dim, n_levels);
        let t = Instant::now();

        // Generate level HVs using thermometer-like encoding
        // Each level is a smooth interpolation between two random endpoints
        let base_hv = ContinuousHV::random(dim, 1000);
        let end_hv = ContinuousHV::random(dim, 2000);

        let level_hvs: Vec<ContinuousHV> = (0..n_levels)
            .map(|l| {
                let alpha = l as f32 / (n_levels - 1).max(1) as f32;
                let values: Vec<f32> = base_hv
                    .values
                    .iter()
                    .zip(end_hv.values.iter())
                    .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
                    .collect();
                ContinuousHV::from_vec(values)
            })
            .collect();

        // Generate position HVs (one per pixel)
        let position_hvs: Vec<ContinuousHV> = (0..784)
            .map(|p| ContinuousHV::random(dim, 10000 + p as u64))
            .collect();

        println!("  Init time: {:.1}ms", t.elapsed().as_millis());

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
            class_prototypes: (0..10).map(|_| None).collect(),
            class_counts: vec![0; 10],
        }
    }

    /// Encode a single image into an HDC vector
    fn encode(&self, pixels: &[u8]) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];
        let level_size = 256.0 / self.n_levels as f32;

        for (pos, &pixel) in pixels.iter().enumerate() {
            let level = ((pixel as f32 / level_size) as usize).min(self.n_levels - 1);

            // bind(position, level) then accumulate
            let bound = self.position_hvs[pos].bind(&self.level_hvs[level]);
            for (acc, &val) in accumulator.iter_mut().zip(bound.values.iter()) {
                *acc += val;
            }
        }

        // Normalize
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Train on a set of images and labels
    fn train(&mut self, images: &[Vec<u8>], labels: &[u8]) {
        let t = Instant::now();
        let n = images.len();

        // Accumulators for class prototypes
        let mut accumulators: Vec<Vec<f32>> = (0..10).map(|_| vec![0.0f32; self.dim]).collect();

        for (i, (img, &label)) in images.iter().zip(labels.iter()).enumerate() {
            let encoded = self.encode(img);
            let class = label as usize;

            for (acc, &val) in accumulators[class].iter_mut().zip(encoded.values.iter()) {
                *acc += val;
            }
            self.class_counts[class] += 1;

            if (i + 1) % 10000 == 0 {
                let elapsed = t.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                println!(
                    "  Training: {}/{} ({:.0} samples/sec)",
                    i + 1,
                    n,
                    rate
                );
            }
        }

        // Normalize accumulators to create class prototypes
        for class in 0..10 {
            if self.class_counts[class] > 0 {
                let norm: f32 = accumulators[class]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                if norm > 0.0 {
                    for v in &mut accumulators[class] {
                        *v /= norm;
                    }
                }
                self.class_prototypes[class] = Some(ContinuousHV::from_vec(accumulators[class].clone()));
            }
        }

        let elapsed = t.elapsed().as_secs_f64();
        println!(
            "  Training complete: {} samples in {:.1}s ({:.0}/s)",
            n,
            elapsed,
            n as f64 / elapsed
        );
        println!(
            "  Class distribution: {:?}",
            self.class_counts
        );
    }

    /// Retrain prototypes using ISOLET-proven direct prototype modification.
    /// For misclassified samples: subtract (scaled) from wrong class prototype,
    /// add (scaled) to correct class prototype. Normalize only at the end.
    fn retrain(&mut self, images: &[Vec<u8>], labels: &[u8], lr: f32, iterations: usize) {
        for iter in 0..iterations {
            let t = Instant::now();
            let mut corrections = 0;

            for (img, &label) in images.iter().zip(labels.iter()) {
                let encoded = self.encode(img);
                let actual = label as usize;

                // Classify using pre-encoded HV to avoid double-encoding
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
                    // Subtract (scaled) from wrong class prototype
                    if let Some(ref mut proto) = self.class_prototypes[best_class] {
                        for (p, &e) in proto.values.iter_mut().zip(encoded.values.iter()) {
                            *p -= lr * e;
                        }
                    }
                    // Add (scaled) to correct class prototype
                    if let Some(ref mut proto) = self.class_prototypes[actual] {
                        for (p, &e) in proto.values.iter_mut().zip(encoded.values.iter()) {
                            *p += lr * e;
                        }
                    }
                    corrections += 1;
                }
            }

            let accuracy = 1.0 - corrections as f64 / images.len() as f64;
            let elapsed = t.elapsed().as_secs_f64();
            println!(
                "  Retrain iter {}/{}: {} corrections, train acc = {:.2}% ({:.1}s)",
                iter + 1,
                iterations,
                corrections,
                accuracy * 100.0,
                elapsed
            );

            // Early stop if very few corrections
            if corrections < images.len() / 200 {
                println!("  Early stopping: corrections < 0.5% of training set");
                break;
            }
        }

        // Normalize prototypes once at end (preserves relative magnitudes during training)
        for proto in &mut self.class_prototypes {
            if let Some(ref mut p) = proto {
                let norm: f32 = p.values.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut p.values {
                        *v /= norm;
                    }
                }
            }
        }
    }

    /// Classify a single image
    fn classify(&self, pixels: &[u8]) -> (usize, f32) {
        let encoded = self.encode(pixels);

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

        (best_class, best_sim)
    }

    /// Test accuracy on a dataset
    fn test(&self, images: &[Vec<u8>], labels: &[u8]) -> TestResult {
        let t = Instant::now();
        let n = images.len();

        let mut correct = 0;
        let mut per_class_correct = vec![0usize; 10];
        let mut per_class_total = vec![0usize; 10];
        let mut confusion = vec![vec![0usize; 10]; 10];

        for (img, &label) in images.iter().zip(labels.iter()) {
            let (predicted, _sim) = self.classify(img);
            let actual = label as usize;

            per_class_total[actual] += 1;
            confusion[actual][predicted] += 1;

            if predicted == actual {
                correct += 1;
                per_class_correct[actual] += 1;
            }
        }

        let accuracy = correct as f64 / n as f64;
        let elapsed = t.elapsed().as_secs_f64();

        TestResult {
            accuracy,
            correct,
            total: n,
            per_class_accuracy: per_class_correct
                .iter()
                .zip(per_class_total.iter())
                .map(|(&c, &t)| if t > 0 { c as f64 / t as f64 } else { 0.0 })
                .collect(),
            confusion,
            inference_time_ms: elapsed * 1000.0 / n as f64,
        }
    }
}

struct TestResult {
    accuracy: f64,
    correct: usize,
    total: usize,
    per_class_accuracy: Vec<f64>,
    #[allow(dead_code)]
    confusion: Vec<Vec<usize>>,
    inference_time_ms: f64,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       MNIST HDC Classification Benchmark                   ║");
    println!("║       Symthaea Hyperdimensional Computing Validation       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check data exists
    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: MNIST data not found at {}", DATA_DIR);
        eprintln!("Run the data download script first.");
        return;
    }

    // Load data
    println!("Loading MNIST data...");
    let train_images = read_idx_images(&data_path.join("train-images-idx3-ubyte"));
    let train_labels = read_idx_labels(&data_path.join("train-labels-idx1-ubyte"));
    let test_images = read_idx_images(&data_path.join("t10k-images-idx3-ubyte"));
    let test_labels = read_idx_labels(&data_path.join("t10k-labels-idx1-ubyte"));
    println!();

    // Run benchmark at multiple dimensions
    // (dim, levels, retrain_iters, label)
    let configs: Vec<(usize, usize, usize, &str)> = vec![
        (4096, 32, 0, "Standard (4K dim, 32 levels)"),
        (4096, 32, 5, "Standard+Retrain (4K, 5 iters)"),
        (8192, 32, 0, "Extended (8K dim, 32 levels)"),
        (8192, 32, 5, "Extended+Retrain (8K, 5 iters)"),
    ];

    let mut results = Vec::new();

    for (dim, levels, retrain_iters, label) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {}", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let total_start = Instant::now();

        let mut classifier = HdcMnistClassifier::new(*dim, *levels);

        // Train
        println!("\nTraining...");
        classifier.train(&train_images, &train_labels);

        // Retrain if configured
        if *retrain_iters > 0 {
            println!("\nRetraining (lr=0.1, {} iterations)...", retrain_iters);
            classifier.retrain(&train_images, &train_labels, 0.1, *retrain_iters);
        }

        // Test
        println!("\nTesting...");
        let result = classifier.test(&test_images, &test_labels);

        let total_time = total_start.elapsed().as_secs_f64();

        println!("\n  Overall Accuracy: {:.2}% ({}/{})",
            result.accuracy * 100.0, result.correct, result.total);
        println!("  Avg inference time: {:.3}ms per sample", result.inference_time_ms);
        println!("  Total time: {:.1}s", total_time);

        println!("\n  Per-class accuracy:");
        for (digit, acc) in result.per_class_accuracy.iter().enumerate() {
            println!("    Digit {}: {:.1}%", digit, acc * 100.0);
        }

        results.push((*dim, *levels, *retrain_iters, label, result));
        println!();
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ {:30} │ {:>8} │ {:>10} ║", "Configuration", "Accuracy", "Infer/ms");
    println!("╟────────────────────────────────┼──────────┼────────────╢");

    for (_dim, _levels, _retrain, label, result) in results.iter() {
        println!(
            "║ {:30} │ {:>7.2}% │ {:>9.3} ║",
            label, result.accuracy * 100.0, result.inference_time_ms
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Validation thresholds
    println!("VALIDATION THRESHOLDS");
    println!("═══════════════════════════════════════════════════════════════");

    let best_accuracy = results
        .iter()
        .map(|(_, _, _, _, r)| r.accuracy)
        .fold(f64::NEG_INFINITY, f64::max);

    let checks = vec![
        ("Accuracy > 80% (minimum HDC)", best_accuracy > 0.80),
        ("Accuracy > 85% (good HDC)", best_accuracy > 0.85),
        ("Accuracy > 90% (strong HDC)", best_accuracy > 0.90),
    ];

    for (name, pass) in &checks {
        println!(
            "  {}: {}",
            name,
            if *pass { "PASS" } else { "FAIL" }
        );
    }

    println!("\nBest accuracy: {:.2}%", best_accuracy * 100.0);

    // Save results to JSON
    let result_json = serde_json::json!({
        "benchmark": "MNIST HDC Classification",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "results": results.iter().map(|(dim, levels, retrain, label, r)| {
            serde_json::json!({
                "config": *label,
                "dim": *dim,
                "levels": *levels,
                "retrain_iterations": *retrain,
                "accuracy": r.accuracy,
                "correct": r.correct,
                "total": r.total,
                "inference_time_ms": r.inference_time_ms,
                "per_class_accuracy": r.per_class_accuracy,
            })
        }).collect::<Vec<_>>(),
        "best_accuracy": best_accuracy,
        "validation_passed": best_accuracy > 0.85,
    });

    let result_path = "data/benchmarks/mnist/results.json";
    if let Ok(f) = File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to {}", result_path);
    }
}
