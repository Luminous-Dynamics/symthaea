// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TurboQuant HDC vector compression benchmark.
//!
//! Measures compression ratio, cosine fidelity, and throughput at various bit widths.
//!
//! Run: cargo run --example benchmark_hv_compression --release --features turbo-quant

#[cfg(feature = "turbo-quant")]
fn main() {
    use std::time::Instant;
    use symthaea::hdc::hv_compression::HvCompressor;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    println!("=== TurboQuant HDC Compression Benchmark ===");
    println!("Dimension: {HDC_DIMENSION}");
    println!(
        "Original size: {} bytes ({:.1} KB)",
        HDC_DIMENSION * 4,
        HDC_DIMENSION as f64 * 4.0 / 1024.0
    );
    println!();

    // Generate test vectors
    let n_vectors = 1000;
    let vectors: Vec<ContinuousHV> = (0..n_vectors)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    // Test at different bit widths
    for bits in [2, 3, 4, 6, 8] {
        let compressor = match HvCompressor::new(bits) {
            Ok(c) => c,
            Err(e) => {
                println!("{bits}-bit: FAILED to create compressor: {e}");
                continue;
            }
        };

        // Compression throughput
        let start = Instant::now();
        let compressed: Vec<_> = vectors.iter().map(|v| compressor.compress(v)).collect();
        let compress_elapsed = start.elapsed();

        // Decompression throughput
        let start = Instant::now();
        let decompressed: Vec<ContinuousHV> = compressed
            .iter()
            .map(|c| compressor.decompress(c))
            .collect();
        let decompress_elapsed = start.elapsed();

        // Fidelity: cosine similarity between original and reconstructed
        let mut sim_sum = 0.0f64;
        let mut sim_min = 1.0f64;
        for (orig, recon) in vectors.iter().zip(decompressed.iter()) {
            let sim = orig.cosine_similarity(recon) as f64;
            sim_sum += sim;
            if sim < sim_min {
                sim_min = sim;
            }
        }
        let sim_mean = sim_sum / n_vectors as f64;

        // Size
        let compressed_bytes: usize = compressed.iter().map(|c| c.data.len()).sum();
        let original_bytes = n_vectors * HDC_DIMENSION * 4;
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        let compress_rate = n_vectors as f64 / compress_elapsed.as_secs_f64();
        let decompress_rate = n_vectors as f64 / decompress_elapsed.as_secs_f64();

        println!("{bits}-bit PolarQuant:");
        println!(
            "  Ratio:        {ratio:.1}x ({:.1} KB → {:.1} KB per vector)",
            HDC_DIMENSION as f64 * 4.0 / 1024.0,
            compressed_bytes as f64 / n_vectors as f64 / 1024.0
        );
        println!("  Cosine sim:   mean={sim_mean:.6}  min={sim_min:.6}");
        println!(
            "  Compress:     {compress_rate:.0} vectors/sec ({:.1} us/vector)",
            compress_elapsed.as_micros() as f64 / n_vectors as f64
        );
        println!(
            "  Decompress:   {decompress_rate:.0} vectors/sec ({:.1} us/vector)",
            decompress_elapsed.as_micros() as f64 / n_vectors as f64
        );
        println!();
    }

    // Episodic memory simulation: how much memory saved?
    let compressor_4bit = HvCompressor::new(4).unwrap();
    let n_memories = 10_000;
    let uncompressed_mb = n_memories as f64 * HDC_DIMENSION as f64 * 4.0 / 1_048_576.0;
    let sample = compressor_4bit.compress(&vectors[0]);
    let compressed_mb = n_memories as f64 * sample.data.len() as f64 / 1_048_576.0;
    println!("=== Episodic Memory Impact (4-bit, {n_memories} memories) ===");
    println!("  Uncompressed: {uncompressed_mb:.1} MB");
    println!("  Compressed:   {compressed_mb:.1} MB");
    println!(
        "  Savings:      {:.1} MB ({:.0}%)",
        uncompressed_mb - compressed_mb,
        (1.0 - compressed_mb / uncompressed_mb) * 100.0
    );
}

#[cfg(not(feature = "turbo-quant"))]
fn main() {
    eprintln!("This benchmark requires the turbo-quant feature.");
    eprintln!("Run: cargo run --example benchmark_hv_compression --release --features turbo-quant");
    std::process::exit(1);
}