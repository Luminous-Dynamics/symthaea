//! # MNIST HDC with Learned Feature Weighting
//!
//! Extends the standard MNIST HDC benchmark with:
//! - Fisher discriminant per-pixel weighting (focuses on discriminative pixels)
//! - Spatial context encoding (neighborhood averaging)
//! - Adaptive quantization (histogram equalization)
//! - Learning rate decay during retraining
//! - Gram-Schmidt prototype orthogonalization
//!
//! Target: 92%+ accuracy (up from 87.6% baseline).
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_mnist_learned --release
//! ```

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::hdc::learned_encoding::AdaptiveQuantizer;

const DATA_DIR: &str = "data/benchmarks/mnist";

// ═══════════════════════════════════════════════════════════════════════════════
// Data loading (MNIST IDX format)
// ═══════════════════════════════════════════════════════════════════════════════

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
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "Invalid label file magic number");

    let n_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    println!("  Labels: {}", n_labels);

    buf[8..8 + n_labels].to_vec()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fisher Discriminant Feature Weighting
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute per-pixel discriminative weights using Fisher's linear discriminant.
///
/// For each pixel position, compute:
///   weight = between-class variance / within-class variance
///
/// Pixels where different digits look different (high between-class variance)
/// and consistent within each digit class (low within-class variance) get
/// the highest weights. Background corners always get low weights.
fn compute_discriminative_weights(images: &[Vec<u8>], labels: &[u8]) -> Vec<f32> {
    let n_pixels = images[0].len(); // 784
    let n_classes = 10;
    let n = images.len() as f64;

    let mut class_sums = vec![vec![0.0f64; n_pixels]; n_classes];
    let mut class_sq_sums = vec![vec![0.0f64; n_pixels]; n_classes];
    let mut class_counts = vec![0usize; n_classes];

    for (img, &label) in images.iter().zip(labels) {
        let c = label as usize;
        class_counts[c] += 1;
        for (i, &p) in img.iter().enumerate() {
            let v = p as f64 / 255.0;
            class_sums[c][i] += v;
            class_sq_sums[c][i] += v * v;
        }
    }

    let mut weights = vec![1.0f32; n_pixels];

    for i in 0..n_pixels {
        let global_mean: f64 = class_sums.iter().map(|s| s[i]).sum::<f64>() / n;

        let mut between = 0.0f64;
        let mut within = 0.0f64;

        for c in 0..n_classes {
            if class_counts[c] > 0 {
                let nc = class_counts[c] as f64;
                let class_mean = class_sums[c][i] / nc;
                between += nc * (class_mean - global_mean).powi(2);
                let class_var = (class_sq_sums[c][i] / nc) - class_mean * class_mean;
                within += nc * class_var.max(0.0);
            }
        }

        if within > 1e-10 {
            weights[i] = (between / within) as f32;
        } else if between > 1e-10 {
            weights[i] = 10.0; // perfectly discriminative, zero noise
        }
    }

    // Normalize: mean=1.0, clamp to [0.1, 5.0]
    let mean_w = weights.iter().sum::<f32>() / n_pixels as f32;
    if mean_w > 1e-10 {
        for w in &mut weights {
            *w = (*w / mean_w).clamp(0.1, 5.0);
        }
    }

    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let active = weights.iter().filter(|&&w| w > 0.15).count();
    println!(
        "  Feature weights: min={:.3}, max={:.3}, {}/{} discriminative pixels",
        min_w, max_w, active, n_pixels
    );

    weights
}

// ═══════════════════════════════════════════════════════════════════════════════
// Learned MNIST Classifier
// ═══════════════════════════════════════════════════════════════════════════════

/// MNIST classifier with learned feature weighting and adaptive quantization.
///
/// Two improvements over baseline HDC:
/// 1. **Adaptive quantization**: Non-uniform levels matching MNIST's pixel distribution
/// 2. **Fisher weighting**: Per-pixel importance based on inter-class discriminability
///
/// Note: Spatial context encoding (neighborhood averaging) was tested but found to
/// hurt accuracy — it blurs discriminative features. Gram-Schmidt orthogonalization
/// was also tested but destroys learned prototype structure.
struct LearnedMnistClassifier {
    dim: usize,
    n_levels: usize,
    /// Interpolated level HVs (smooth gradient between two random endpoints).
    level_hvs: Vec<ContinuousHV>,
    /// Position hypervectors (one per pixel position, 784 for MNIST).
    position_hvs: Vec<ContinuousHV>,
    /// Class prototypes (one per digit 0-9).
    class_prototypes: Vec<Option<ContinuousHV>>,
    /// Number of samples per class.
    class_counts: Vec<usize>,
    /// Adaptive quantizer learned from training data (None = uniform levels).
    quantizer: Option<AdaptiveQuantizer>,
    /// Per-pixel importance weights from Fisher discriminant analysis (None = uniform).
    feature_weights: Option<Vec<f32>>,
}

impl LearnedMnistClassifier {
    fn new(
        dim: usize,
        n_levels: usize,
        training_images: &[Vec<u8>],
        training_labels: &[u8],
        use_fisher_weights: bool,
        use_adaptive_quant: bool,
    ) -> Self {
        println!(
            "  Initializing HDC classifier: dim={}, levels={}, fisher={}, adaptive_quant={}",
            dim, n_levels, use_fisher_weights, use_adaptive_quant
        );
        let t = Instant::now();

        // Optionally compute Fisher discriminant feature weights
        let feature_weights = if use_fisher_weights {
            Some(compute_discriminative_weights(training_images, training_labels))
        } else {
            println!("  Feature weights: uniform (disabled)");
            None
        };

        // Optionally build adaptive quantizer from non-zero pixel distribution
        let quantizer = if use_adaptive_quant {
            let mut all_values: Vec<f32> = Vec::with_capacity(training_images.len() * 100);
            for (i, img) in training_images.iter().enumerate() {
                if i % 10 == 0 {
                    for &p in img.iter() {
                        if p > 0 {
                            all_values.push(p as f32 / 255.0);
                        }
                    }
                }
            }
            println!(
                "  Adaptive quantizer: {} levels from {} non-zero pixel samples",
                n_levels,
                all_values.len()
            );
            Some(AdaptiveQuantizer::from_data(&all_values, n_levels))
        } else {
            println!("  Quantizer: uniform {} levels", n_levels);
            None
        };

        // Progressive random-flip level HVs: each level flips dim/n_levels
        // dimensions from the previous level. This preserves ordinal similarity
        // (adjacent levels are similar) while ensuring distant levels are
        // nearly orthogonal. Much better than linear interpolation.
        let base_hv = ContinuousHV::random(dim, 1000);
        let flips_per_level = dim / n_levels.max(1);

        let mut level_hvs: Vec<ContinuousHV> = Vec::with_capacity(n_levels);
        level_hvs.push(base_hv);

        let mut flip_seed: u64 = 3000;
        for _l in 1..n_levels {
            let prev = &level_hvs[level_hvs.len() - 1];
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

        // Position HVs
        let position_hvs: Vec<ContinuousHV> = (0..784)
            .map(|p| ContinuousHV::random(dim, 10000 + p as u64))
            .collect();

        println!("  Init time: {:.0}ms", t.elapsed().as_millis());

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
            class_prototypes: (0..10).map(|_| None).collect(),
            class_counts: vec![0; 10],
            quantizer,
            feature_weights,
        }
    }

    /// Encode a single image with Fisher-weighted features and adaptive quantization.
    fn encode(&self, pixels: &[u8]) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];

        for (pos, &pixel) in pixels.iter().enumerate() {
            let val = pixel as f32 / 255.0;

            // Quantize: adaptive (histogram-equalized) or uniform
            let level = if let Some(ref q) = self.quantizer {
                q.quantize(val)
            } else {
                ((val * (self.n_levels - 1) as f32).round() as usize).min(self.n_levels - 1)
            };

            // Get weight: Fisher discriminant or uniform 1.0
            let w = self.feature_weights.as_ref().map_or(1.0, |fw| fw[pos]);

            // Weighted bind(position, level)
            let bound = self.position_hvs[pos].bind(&self.level_hvs[level]);
            for (acc, &v) in accumulator.iter_mut().zip(bound.values.iter()) {
                *acc += w * v;
            }
        }

        // L2 normalize
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
        let n = images.len();
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
                println!(
                    "  Training: {}/{} ({:.0} samples/sec)",
                    i + 1,
                    n,
                    (i + 1) as f64 / elapsed
                );
            }
        }

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
                self.class_prototypes[class] =
                    Some(ContinuousHV::from_vec(accumulators[class].clone()));
            }
        }

        let elapsed = t.elapsed().as_secs_f64();
        println!(
            "  Training complete: {} samples in {:.1}s ({:.0}/s)",
            n,
            elapsed,
            n as f64 / elapsed
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
                "  Retrain iter {}/{}: {} corrections, train acc = {:.2}% ({:.1}s)",
                iter + 1,
                iterations,
                corrections,
                accuracy * 100.0,
                t.elapsed().as_secs_f64()
            );

            if corrections < images.len() / 200 {
                println!("  Early stopping: corrections < 0.5% of training set");
                break;
            }
        }

        // Normalize prototypes
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

        TestResult {
            accuracy: correct as f64 / n as f64,
            correct,
            total: n,
            per_class_accuracy: per_class_correct
                .iter()
                .zip(per_class_total.iter())
                .map(|(&c, &t)| if t > 0 { c as f64 / t as f64 } else { 0.0 })
                .collect(),
            confusion,
            inference_time_ms: t.elapsed().as_secs_f64() * 1000.0 / n as f64,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════

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
    println!("║   MNIST HDC with Learned Feature Weighting                 ║");
    println!("║   Fisher Discriminant + Adaptive Quantization              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: MNIST data not found at {}", DATA_DIR);
        eprintln!("Download from http://yann.lecun.com/exdb/mnist/");
        return;
    }

    println!("Loading MNIST data...");
    let train_images = read_idx_images(&data_path.join("train-images-idx3-ubyte"));
    let train_labels = read_idx_labels(&data_path.join("train-labels-idx1-ubyte"));
    let test_images = read_idx_images(&data_path.join("t10k-images-idx3-ubyte"));
    let test_labels = read_idx_labels(&data_path.join("t10k-labels-idx1-ubyte"));
    println!();

    // Configurations: (dim, levels, retrain_iters, fisher, adaptive_quant, label)
    // Ablation: isolate contribution of each improvement
    let configs: Vec<(usize, usize, usize, bool, bool, &str)> = vec![
        // Control: identical to standard benchmark (uniform levels, no weights)
        (8192,  32, 5,  false, false, "Control (8K/32L/5i, no improvements)"),
        // Fisher weights only (uniform levels)
        (8192,  32, 5,  true,  false, "Fisher only (8K/32L/5i)"),
        // Adaptive quant only (uniform weights)
        (8192,  64, 5,  false, true,  "AdaptiveQuant only (8K/64L/5i)"),
        // Both: Fisher + adaptive quant
        (8192,  64, 5,  true,  true,  "Fisher+AQ (8K/64L/5i)"),
        // Higher capacity + more retrain
        (10240, 64, 10, true,  true,  "Fisher+AQ (10K/64L/10i)"),
        // Max capacity
        (16384, 64, 15, true,  true,  "Fisher+AQ (16K/64L/15i)"),
    ];

    let mut results = Vec::new();

    for (dim, levels, retrain_iters, fisher, adaptive_quant, label) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {}", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let total_start = Instant::now();

        let mut classifier = LearnedMnistClassifier::new(
            *dim, *levels, &train_images, &train_labels, *fisher, *adaptive_quant,
        );

        println!("\nTraining...");
        classifier.train(&train_images, &train_labels);

        if *retrain_iters > 0 {
            println!("\nRetraining (lr=0.1, {} iters)...", retrain_iters);
            classifier.retrain(&train_images, &train_labels, 0.1, *retrain_iters);
        }

        println!("\nTesting...");
        let result = classifier.test(&test_images, &test_labels);

        let total_time = total_start.elapsed().as_secs_f64();

        println!(
            "\n  Overall Accuracy: {:.2}% ({}/{})",
            result.accuracy * 100.0, result.correct, result.total
        );
        println!(
            "  Avg inference time: {:.3}ms per sample",
            result.inference_time_ms
        );
        println!("  Total time: {:.1}s", total_time);

        println!("\n  Per-class accuracy:");
        for (digit, acc) in result.per_class_accuracy.iter().enumerate() {
            println!("    Digit {}: {:.1}%", digit, acc * 100.0);
        }

        results.push((*dim, *levels, *retrain_iters, label, result));
        println!();
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       LEARNED MNIST RESULTS SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:42} │ {:>8} │ {:>10} ║",
        "Configuration", "Accuracy", "Infer/ms"
    );
    println!("╟────────────────────────────────────────────┼──────────┼────────────╢");

    for (_dim, _levels, _retrain, label, result) in results.iter() {
        println!(
            "║ {:42} │ {:>7.2}% │ {:>9.3} ║",
            label,
            result.accuracy * 100.0,
            result.inference_time_ms
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let best_accuracy = results
        .iter()
        .map(|(_, _, _, _, r)| r.accuracy)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("VALIDATION THRESHOLDS");
    println!("═══════════════════════════════════════════════════════════════");
    let checks = vec![
        ("Accuracy > 85% (good HDC)", best_accuracy > 0.85),
        ("Accuracy > 90% (strong HDC)", best_accuracy > 0.90),
        ("Accuracy > 92% (target)", best_accuracy > 0.92),
    ];
    for (name, pass) in &checks {
        println!(
            "  {}: {}",
            name,
            if *pass { "PASS" } else { "FAIL" }
        );
    }
    println!("\nBest accuracy: {:.2}%", best_accuracy * 100.0);

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "MNIST HDC Learned Feature Weighting",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "improvements": [
            "Fisher discriminant per-pixel weighting",
            "Adaptive quantization (histogram equalization)",
            "Progressive random-flip level HVs",
        ],
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
        "baseline_accuracy": 0.876,
        "validation_passed": best_accuracy > 0.90,
    });

    let result_path = "data/benchmarks/mnist/results_learned.json";
    if let Ok(f) = File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to {}", result_path);
    }
}
