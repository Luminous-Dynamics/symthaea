// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! ## Validated Results (Feb 2026)
//! - HDC at dim=4096, Q=32, 5 retrain: 88.49% accuracy
//! - HDC at dim=8192, Q=32, 5 retrain: 88.01% accuracy
//! - Higher dimensions show diminishing returns with position-level binding
//! - 92%+ requires convolutional/multi-scale encoding (not pixel-level)
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

use symthaea::hdc::learned_encoding::{AdaptiveQuantizer, SpatialContextEncoder};
use symthaea::hdc::unified_hv::ContinuousHV;

const DATA_DIR: &str = "data/benchmarks/mnist";

/// Parse IDX file format (MNIST binary format)
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
        println!(
            "  Initializing HDC classifier: dim={}, levels={}",
            dim, n_levels
        );
        let t = Instant::now();

        // Generate level HVs using progressive random-flip encoding
        // Level 0 starts as a random HV. Each subsequent level flips
        // dim/n_levels random dimensions from the previous level.
        // This preserves ordinal similarity (adjacent levels are similar)
        // while ensuring distant levels are nearly orthogonal.
        let base_hv = ContinuousHV::random(dim, 1000);
        let flips_per_level = dim / n_levels.max(1);

        let mut level_hvs: Vec<ContinuousHV> = Vec::with_capacity(n_levels);
        level_hvs.push(base_hv);

        // Simple deterministic PRNG for flip index selection
        let mut flip_seed: u64 = 3000;
        for l in 1..n_levels {
            let prev = &level_hvs[l - 1];
            let mut new_values = prev.values.clone();

            // Flip flips_per_level dimensions by negating them
            for _f in 0..flips_per_level {
                flip_seed = flip_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx = (flip_seed >> 33) as usize % dim;
                new_values[idx] = -new_values[idx];
            }

            level_hvs.push(ContinuousHV::from_vec(new_values));
        }

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
                println!("  Training: {}/{} ({:.0} samples/sec)", i + 1, n, rate);
            }
        }

        // Normalize accumulators to create class prototypes
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

        let elapsed = t.elapsed().as_secs_f64();
        println!(
            "  Training complete: {} samples in {:.1}s ({:.0}/s)",
            n,
            elapsed,
            n as f64 / elapsed
        );
        println!("  Class distribution: {:?}", self.class_counts);
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
        for ref mut p in self.class_prototypes.iter_mut().flatten() {
            let norm: f32 = p.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut p.values {
                    *v /= norm;
                }
            }
        }
    }

    /// Apply Gram-Schmidt orthogonalization to class prototypes.
    /// This reduces inter-class interference by making prototypes more orthogonal.
    /// Uses a soft version: each prototype has a fraction of its projection onto
    /// prior prototypes removed, controlled by `strength` (0.0=none, 1.0=full).
    fn orthogonalize_prototypes(&mut self, strength: f32) {
        let n_classes = self.class_prototypes.len();

        // Collect all prototype vectors
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        for (i, proto) in self.class_prototypes.iter().enumerate() {
            if let Some(ref p) = proto {
                vectors.push(p.values.clone());
                indices.push(i);
            }
        }

        if vectors.len() < 2 {
            return;
        }

        // Modified Gram-Schmidt: subtract projection onto previous vectors
        for i in 1..vectors.len() {
            for j in 0..i {
                // Compute projection coefficient: <v_i, v_j> / <v_j, v_j>
                let dot: f32 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let norm_sq: f32 = vectors[j].iter().map(|x| x * x).sum();
                if norm_sq > 1e-10 {
                    let coeff = dot / norm_sq * strength;
                    let vj_copy: Vec<f32> = vectors[j].clone();
                    for (vi, &vj) in vectors[i].iter_mut().zip(vj_copy.iter()) {
                        *vi -= coeff * vj;
                    }
                }
            }
        }

        // Normalize and write back
        for (idx, vec) in indices.iter().zip(vectors.iter()) {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let normalized: Vec<f32> = vec.iter().map(|v| v / norm).collect();
                self.class_prototypes[*idx] = Some(ContinuousHV::from_vec(normalized));
            }
        }

        // Report inter-class similarity after orthogonalization
        let mut max_sim = f32::NEG_INFINITY;
        let mut avg_sim = 0.0f32;
        let mut count = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                if let (Some(ref a), Some(ref b)) =
                    (&self.class_prototypes[i], &self.class_prototypes[j])
                {
                    let sim = a.similarity(b);
                    avg_sim += sim;
                    count += 1;
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
            }
        }
        if count > 0 {
            avg_sim /= count as f32;
        }
        println!(
            "  Orthogonalization (strength={:.1}): avg inter-class sim={:.4}, max={:.4}",
            strength, avg_sim, max_sim
        );
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
        let mut per_class_correct = [0usize; 10];
        let mut per_class_total = [0usize; 10];
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

/// Enhanced MNIST classifier using spatial context + interpolated levels + adaptive quantization.
struct EnhancedMnistClassifier {
    dim: usize,
    #[allow(dead_code)]
    n_levels: usize,
    /// Interpolated level HVs (smooth gradient between two random endpoints).
    level_hvs: Vec<ContinuousHV>,
    /// Position hypervectors (one per pixel position, 784 for MNIST).
    position_hvs: Vec<ContinuousHV>,
    /// Class prototypes (one per digit 0-9).
    class_prototypes: Vec<Option<ContinuousHV>>,
    /// Number of samples per class.
    class_counts: Vec<usize>,
    /// Spatial context encoder for neighborhood features.
    spatial: SpatialContextEncoder,
    /// Adaptive quantizer learned from training data.
    quantizer: AdaptiveQuantizer,
}

impl EnhancedMnistClassifier {
    fn new(dim: usize, n_levels: usize, training_images: &[Vec<u8>]) -> Self {
        println!(
            "  Initializing Enhanced HDC classifier: dim={}, levels={}",
            dim, n_levels
        );
        let t = Instant::now();

        // Collect all pixel values normalized to [0,1] for adaptive quantization
        let mut all_values: Vec<f32> = Vec::with_capacity(training_images.len() * 100);
        // Sample every 10th image to keep this fast
        for (i, img) in training_images.iter().enumerate() {
            if i % 10 == 0 {
                for &p in img.iter() {
                    if p > 0 {
                        // Skip zero pixels (background)
                        all_values.push(p as f32 / 255.0);
                    }
                }
            }
        }
        let quantizer = AdaptiveQuantizer::from_data(&all_values, n_levels);
        println!(
            "  Adaptive quantizer: {} levels from {} non-zero pixel samples",
            n_levels,
            all_values.len()
        );

        // Interpolated level HVs: smooth gradient between two random endpoints
        let base = ContinuousHV::random(dim, 42);
        let end = ContinuousHV::random(dim, 43);
        let level_hvs: Vec<ContinuousHV> = (0..n_levels)
            .map(|l| {
                let alpha = if n_levels <= 1 {
                    0.0
                } else {
                    l as f32 / (n_levels - 1) as f32
                };
                let values: Vec<f32> = base
                    .values
                    .iter()
                    .zip(end.values.iter())
                    .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
                    .collect();
                ContinuousHV::from_vec(values)
            })
            .collect();

        // Position HVs
        let position_hvs: Vec<ContinuousHV> = (0..784)
            .map(|p| ContinuousHV::random(dim, 10000 + p as u64))
            .collect();

        let spatial = SpatialContextEncoder::mnist();

        println!("  Init time: {:.1}ms", t.elapsed().as_millis());

        Self {
            dim,
            n_levels,
            level_hvs,
            position_hvs,
            class_prototypes: (0..10).map(|_| None).collect(),
            class_counts: vec![0; 10],
            spatial,
            quantizer,
        }
    }

    /// Encode a single image with spatial context preprocessing.
    fn encode(&self, pixels: &[u8]) -> ContinuousHV {
        // Normalize to [0, 1] and apply spatial context
        let normalized: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
        let augmented = self.spatial.augment(&normalized);

        let mut accumulator = vec![0.0f32; self.dim];

        for (pos, &val) in augmented.iter().enumerate() {
            let level = self.quantizer.quantize(val);

            // bind(position, level) then accumulate
            let bound = self.position_hvs[pos].bind(&self.level_hvs[level]);
            for (acc, &v) in accumulator.iter_mut().zip(bound.values.iter()) {
                *acc += v;
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
        for ref mut p in self.class_prototypes.iter_mut().flatten() {
            let norm: f32 = p.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut p.values {
                    *v /= norm;
                }
            }
        }
    }

    fn orthogonalize_prototypes(&mut self, strength: f32) {
        let n_classes = self.class_prototypes.len();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        for (i, proto) in self.class_prototypes.iter().enumerate() {
            if let Some(ref p) = proto {
                vectors.push(p.values.clone());
                indices.push(i);
            }
        }
        if vectors.len() < 2 {
            return;
        }

        for i in 1..vectors.len() {
            for j in 0..i {
                let dot: f32 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let norm_sq: f32 = vectors[j].iter().map(|x| x * x).sum();
                if norm_sq > 1e-10 {
                    let coeff = dot / norm_sq * strength;
                    let vj_copy: Vec<f32> = vectors[j].clone();
                    for (vi, &vj) in vectors[i].iter_mut().zip(vj_copy.iter()) {
                        *vi -= coeff * vj;
                    }
                }
            }
        }

        for (idx, vec) in indices.iter().zip(vectors.iter()) {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let normalized: Vec<f32> = vec.iter().map(|v| v / norm).collect();
                self.class_prototypes[*idx] = Some(ContinuousHV::from_vec(normalized));
            }
        }

        let mut max_sim = f32::NEG_INFINITY;
        let mut avg_sim = 0.0f32;
        let mut count = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                if let (Some(ref a), Some(ref b)) =
                    (&self.class_prototypes[i], &self.class_prototypes[j])
                {
                    let sim = a.similarity(b);
                    avg_sim += sim;
                    count += 1;
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
            }
        }
        if count > 0 {
            avg_sim /= count as f32;
        }
        println!(
            "  Orthogonalization (strength={:.1}): avg inter-class sim={:.4}, max={:.4}",
            strength, avg_sim, max_sim
        );
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
        let mut per_class_correct = [0usize; 10];
        let mut per_class_total = [0usize; 10];
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

    // ── Part 1: Standard HDC configs ───────────────────────────────────────
    // (dim, levels, retrain_iters, orthogonalize_strength, label)
    let configs: Vec<(usize, usize, usize, f32, &str)> = vec![
        (4096, 32, 0, 0.0, "Standard (4K dim, 32 levels)"),
        (4096, 32, 5, 0.0, "Standard+Retrain (4K, 5 iters)"),
        (8192, 32, 0, 0.0, "Extended (8K dim, 32 levels)"),
        (8192, 32, 5, 0.0, "Extended+Retrain (8K, 5 iters)"),
    ];

    let mut results = Vec::new();

    for (dim, levels, retrain_iters, gs_strength, label) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {}", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let total_start = Instant::now();

        let mut classifier = HdcMnistClassifier::new(*dim, *levels);

        println!("\nTraining...");
        classifier.train(&train_images, &train_labels);

        if *retrain_iters > 0 {
            println!("\nRetraining (lr=0.1, {} iterations)...", retrain_iters);
            classifier.retrain(&train_images, &train_labels, 0.1, *retrain_iters);
        }

        if *gs_strength > 0.0 {
            println!("\nOrthogonalizing prototypes...");
            classifier.orthogonalize_prototypes(*gs_strength);
        }

        println!("\nTesting...");
        let result = classifier.test(&test_images, &test_labels);

        let total_time = total_start.elapsed().as_secs_f64();

        println!(
            "\n  Overall Accuracy: {:.2}% ({}/{})",
            result.accuracy * 100.0,
            result.correct,
            result.total
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

        results.push((*dim, *levels, *retrain_iters, *gs_strength, label, result));
        println!();
    }

    // ── Part 2: Enhanced HDC with spatial context + interpolated levels ──
    let enhanced_configs: Vec<(usize, usize, usize, f32, &str)> = vec![
        (8192, 64, 5, 0.5, "Enhanced (8K, 64L, spatial, 5i)"),
        (10240, 64, 10, 0.5, "Enhanced (10K, 64L, spatial, 10i)"),
        (16384, 64, 10, 0.5, "Enhanced (16K, 64L, spatial, 10i)"),
    ];

    for (dim, levels, retrain_iters, gs_strength, label) in &enhanced_configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {} [ENHANCED]", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let total_start = Instant::now();

        let mut classifier = EnhancedMnistClassifier::new(*dim, *levels, &train_images);

        println!("\nTraining...");
        classifier.train(&train_images, &train_labels);

        if *retrain_iters > 0 {
            println!("\nRetraining (lr=0.1, {} iterations)...", retrain_iters);
            classifier.retrain(&train_images, &train_labels, 0.1, *retrain_iters);
        }

        if *gs_strength > 0.0 {
            println!("\nOrthogonalizing prototypes...");
            classifier.orthogonalize_prototypes(*gs_strength);
        }

        println!("\nTesting...");
        let result = classifier.test(&test_images, &test_labels);

        let total_time = total_start.elapsed().as_secs_f64();

        println!(
            "\n  Overall Accuracy: {:.2}% ({}/{})",
            result.accuracy * 100.0,
            result.correct,
            result.total
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

        results.push((*dim, *levels, *retrain_iters, *gs_strength, label, result));
        println!();
    }

    // Summary table
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS SUMMARY                             ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:35} │ {:>8} │ {:>10} ║",
        "Configuration", "Accuracy", "Infer/ms"
    );
    println!("╟─────────────────────────────────────┼──────────┼────────────╢");

    for (_dim, _levels, _retrain, _gs, label, result) in results.iter() {
        println!(
            "║ {:35} │ {:>7.2}% │ {:>9.3} ║",
            label,
            result.accuracy * 100.0,
            result.inference_time_ms
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Validation thresholds
    println!("VALIDATION THRESHOLDS");
    println!("═══════════════════════════════════════════════════════════════");

    let best_accuracy = results
        .iter()
        .map(|(_, _, _, _, _, r)| r.accuracy)
        .fold(f64::NEG_INFINITY, f64::max);

    let checks = vec![
        ("Accuracy > 80% (minimum HDC)", best_accuracy > 0.80),
        ("Accuracy > 85% (good HDC)", best_accuracy > 0.85),
        ("Accuracy > 90% (strong HDC)", best_accuracy > 0.90),
    ];

    for (name, pass) in &checks {
        println!("  {}: {}", name, if *pass { "PASS" } else { "FAIL" });
    }

    println!("\nBest accuracy: {:.2}%", best_accuracy * 100.0);

    // Save results to JSON
    let result_json = serde_json::json!({
        "benchmark": "MNIST HDC Classification",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "results": results.iter().map(|(dim, levels, retrain, gs, label, r)| {
            serde_json::json!({
                "config": *label,
                "dim": *dim,
                "levels": *levels,
                "retrain_iterations": *retrain,
                "gram_schmidt_strength": *gs,
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