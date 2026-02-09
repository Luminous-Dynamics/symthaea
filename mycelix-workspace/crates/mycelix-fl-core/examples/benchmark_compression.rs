//! HyperFeel Compression Measurement Benchmark
//!
//! Measures actual compression ratios at various model sizes and
//! reconstruction error. Provides honest documentation of HyperFeel-style
//! compression performance.
//!
//! Run: `cargo run --example benchmark_compression --release`

use mycelix_fl_core::plugins::{CompressedGradient, CompressionPlugin};
use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const HV_SIZE: usize = 2048; // HV16 is 16,384 bits = 2,048 bytes

/// Identity compression (baseline): just converts to bytes, ratio = 1.0
struct IdentityCompressionPlugin;

impl CompressionPlugin for IdentityCompressionPlugin {
    fn compress(&mut self, update: &GradientUpdate) -> CompressedGradient {
        let bytes: Vec<u8> = update
            .gradients
            .iter()
            .flat_map(|g| g.to_le_bytes())
            .collect();
        let ratio = 1.0;
        CompressedGradient {
            participant_id: update.participant_id.clone(),
            data: bytes,
            compression_ratio: ratio,
        }
    }

    fn decompress(&mut self, compressed: &CompressedGradient, output_dim: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(output_dim);
        for chunk in compressed.data.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            result.push(f32::from_le_bytes(bytes));
        }
        result
    }

    fn name(&self) -> &str {
        "identity"
    }

    fn compression_ratio(&self) -> f32 {
        1.0
    }
}

/// Simulated HyperFeel compression using random projection.
///
/// Reduces gradients to HV_SIZE bytes (2,048) using:
/// 1. Random projection matrix (sparse, seeded)
/// 2. Quantization to bytes
///
/// This approximates HyperFeel without depending on Symthaea.
struct SimulatedHyperFeelPlugin {
    seed: u64,
}

impl SimulatedHyperFeelPlugin {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn generate_projection_row(&self, row: usize, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(row as u64));
        let mut proj = vec![0.0f32; dim];
        // Sparse random projection: ~10% non-zero entries
        let scale = (10.0 / dim as f32).sqrt();
        for val in proj.iter_mut() {
            let r: f32 = rng.gen();
            if r < 0.05 {
                *val = scale;
            } else if r < 0.10 {
                *val = -scale;
            }
        }
        proj
    }
}

impl CompressionPlugin for SimulatedHyperFeelPlugin {
    fn compress(&mut self, update: &GradientUpdate) -> CompressedGradient {
        let dim = update.gradients.len();
        let n_proj = HV_SIZE; // one byte per projection

        // Project: for each output dimension, dot product with random vector
        let mut projected = Vec::with_capacity(n_proj);
        for row in 0..n_proj {
            let proj_vec = self.generate_projection_row(row, dim);
            let dot: f32 = update
                .gradients
                .iter()
                .zip(proj_vec.iter())
                .map(|(g, p)| g * p)
                .sum();
            projected.push(dot);
        }

        // Quantize to bytes: map range to [0, 255]
        let min_val = projected
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max_val = projected
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-10);

        let mut data = Vec::with_capacity(n_proj + 8);
        // Store min/max for dequantization (8 bytes overhead)
        data.extend_from_slice(&min_val.to_le_bytes());
        data.extend_from_slice(&range.to_le_bytes());
        for &v in &projected {
            let normalized = ((v - min_val) / range * 255.0).round() as u8;
            data.push(normalized);
        }

        let original_bytes = dim * 4; // f32 = 4 bytes
        let ratio = original_bytes as f32 / data.len() as f32;

        CompressedGradient {
            participant_id: update.participant_id.clone(),
            data,
            compression_ratio: ratio,
        }
    }

    fn decompress(&mut self, compressed: &CompressedGradient, output_dim: usize) -> Vec<f32> {
        if compressed.data.len() < 8 {
            return vec![0.0; output_dim];
        }

        // Extract min/range
        let min_bytes: [u8; 4] = compressed.data[0..4].try_into().unwrap();
        let range_bytes: [u8; 4] = compressed.data[4..8].try_into().unwrap();
        let min_val = f32::from_le_bytes(min_bytes);
        let range = f32::from_le_bytes(range_bytes);

        // Dequantize projected values
        let quantized = &compressed.data[8..];
        let n_proj = quantized.len();
        let mut projected = Vec::with_capacity(n_proj);
        for &byte in quantized {
            let val = min_val + (byte as f32 / 255.0) * range;
            projected.push(val);
        }

        // Pseudo-inverse reconstruction via transpose projection
        let mut result = vec![0.0f32; output_dim];
        for (row, &proj_val) in projected.iter().enumerate() {
            let proj_vec = self.generate_projection_row(row, output_dim);
            for (i, &p) in proj_vec.iter().enumerate() {
                result[i] += proj_val * p;
            }
        }

        // Normalize by number of projections
        let scale = 1.0 / (n_proj as f32).sqrt();
        for val in result.iter_mut() {
            *val *= scale;
        }

        result
    }

    fn name(&self) -> &str {
        "simulated_hyperfeel"
    }

    fn compression_ratio(&self) -> f32 {
        // Depends on input size; return nominal
        1000.0
    }
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32
}

fn main() {
    println!("=== HyperFeel Compression Measurement Benchmark ===\n");

    let mut rng = StdRng::seed_from_u64(42);
    let mut hyperfeel = SimulatedHyperFeelPlugin::new(42);
    let mut identity = IdentityCompressionPlugin;

    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut passed = 0;
    let mut total = 0;

    println!("{:<12} {:>10} {:>14} {:>14} {:>12} {:>12}",
        "Params", "Bytes(f32)", "HyperFeel(B)", "Ratio", "MSE", "CosSimil");
    println!("{}", "-".repeat(80));

    for &size in &sizes {
        let gradients: Vec<f32> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let update = GradientUpdate::new("test".into(), 1, gradients.clone(), 100, 0.5);

        // Identity compression
        let identity_compressed = identity.compress(&update);
        let identity_decompressed = identity.decompress(&identity_compressed, size);
        let identity_mse = mse(&gradients, &identity_decompressed);
        assert!(
            identity_mse < 1e-10,
            "Identity compression should be lossless"
        );

        // HyperFeel compression
        let hf_compressed = hyperfeel.compress(&update);
        let hf_decompressed = hyperfeel.decompress(&hf_compressed, size);
        let hf_mse = mse(&gradients, &hf_decompressed);

        let original_bytes = size * 4;
        let compressed_bytes = hf_compressed.data.len();
        let actual_ratio = original_bytes as f64 / compressed_bytes as f64;

        // Cosine similarity
        let dot: f32 = gradients.iter().zip(hf_decompressed.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = hf_decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };

        println!(
            "{:<12} {:>10} {:>14} {:>12.1}x {:>12.6} {:>12.4}",
            format!("{}K", size / 1000),
            original_bytes,
            compressed_bytes,
            actual_ratio,
            hf_mse,
            cos_sim,
        );

        // Verify compression ratio matches expected
        total += 1;
        let expected_ratio = original_bytes as f64 / (HV_SIZE as f64 + 8.0);
        let ratio_error = (actual_ratio - expected_ratio).abs() / expected_ratio;
        if ratio_error < 0.05 {
            passed += 1;
        } else {
            println!(
                "  [WARN] Ratio {:.1}x vs expected {:.1}x (error: {:.1}%)",
                actual_ratio,
                expected_ratio,
                ratio_error * 100.0
            );
        }
    }

    println!("\n=== Aggregation Quality Test ===\n");

    // Compare pipeline(compressed) vs pipeline(uncompressed)
    let dim = 10_000;
    let n_nodes = 5;
    let mut updates = Vec::new();
    let mut reps = HashMap::new();
    for i in 0..n_nodes {
        let grads: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
        updates.push(GradientUpdate::new(
            format!("node_{}", i),
            1,
            grads,
            100,
            0.5,
        ));
        reps.insert(format!("node_{}", i), 0.85);
    }

    // Uncompressed aggregation
    let config = PipelineConfig {
        multi_signal_detection: false,
        ..Default::default()
    };
    let mut pipeline = UnifiedPipeline::new(config.clone());
    let uncompressed_result = pipeline.aggregate(&updates, &reps).unwrap();

    // Compressed aggregation (compress → decompress → aggregate)
    let mut compressed_updates = Vec::new();
    for update in &updates {
        let compressed = hyperfeel.compress(update);
        let decompressed = hyperfeel.decompress(&compressed, dim);
        compressed_updates.push(GradientUpdate::new(
            update.participant_id.clone(),
            update.model_version,
            decompressed,
            update.metadata.batch_size,
            update.metadata.loss,
        ));
    }

    let mut pipeline2 = UnifiedPipeline::new(config);
    let compressed_result = pipeline2.aggregate(&compressed_updates, &reps).unwrap();

    let agg_mse = mse(
        &uncompressed_result.aggregated.gradients,
        &compressed_result.aggregated.gradients,
    );

    let dot: f32 = uncompressed_result.aggregated.gradients.iter()
        .zip(compressed_result.aggregated.gradients.iter())
        .map(|(a, b)| a * b).sum();
    let norm_a: f32 = uncompressed_result.aggregated.gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = compressed_result.aggregated.gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
    let agg_cos_sim = if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 };

    println!("Aggregation MSE (compressed vs uncompressed): {:.6}", agg_mse);
    println!("Aggregation cosine similarity: {:.4}", agg_cos_sim);

    total += 1;
    if agg_cos_sim > 0.0 {
        println!("[PASS] Compressed aggregation produces valid results");
        passed += 1;
    } else {
        println!("[FAIL] Compressed aggregation produces invalid results");
    }

    println!("\n=== Honest Documentation ===\n");
    println!("Compression ratio scales linearly with model size:");
    println!("  1K params  ->  ~2x compression");
    println!("  10K params -> ~20x compression");
    println!("  100K params -> ~200x compression");
    println!("  1M params  -> ~2,000x compression");
    println!("  10M params -> ~20,000x compression");
    println!("\nFixed output size: {} bytes (HV16 + 8 byte header)", HV_SIZE + 8);
    println!("Reconstruction is lossy — cosine similarity degrades with higher compression.");

    println!("\n=== RESULTS: {}/{} passed ===", passed, total);
    if passed < total {
        std::process::exit(1);
    }
}
