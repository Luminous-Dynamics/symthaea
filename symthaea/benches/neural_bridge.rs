// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge v2 Benchmarks
//!
//! Measures encoding latency for cached and uncached scenarios.

#![cfg(feature = "neural-bridge")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::path::Path;

/// Benchmark uncached encoding (full BGE-M3 + probe pipeline).
fn bench_encode_uncached(c: &mut Criterion) {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping benchmark: probe weights not found");
        return;
    }

    // Load model outside of benchmark loop
    let mut bridge = match NeuralBridgeV2::load_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    let texts = [
        "justice",
        "What is consciousness?",
        "The neural correlates of phenomenal experience in biological systems",
    ];

    let mut group = c.benchmark_group("neural_bridge_encode");
    group.sample_size(10); // Reduce samples due to slow encoding

    for text in &texts {
        // Clear cache before each benchmark to measure uncached performance
        bridge.clear_cache();

        group.bench_with_input(BenchmarkId::new("uncached", text.len()), text, |b, text| {
            b.iter(|| {
                bridge.clear_cache(); // Ensure uncached
                bridge.encode_to_hdc(black_box(*text)).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark cached encoding (cache hit).
fn bench_encode_cached(c: &mut Criterion) {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping benchmark: probe weights not found");
        return;
    }

    let mut bridge = match NeuralBridgeV2::load_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    let text = "What is the nature of consciousness?";

    // Pre-populate cache
    bridge.encode_to_hdc(text).unwrap();

    c.bench_function("neural_bridge_cached", |b| {
        b.iter(|| bridge.encode_to_hdc(black_box(text)).unwrap())
    });

    // Report cache stats
    let stats = bridge.stats();
    println!("\nCache stats after benchmark:");
    println!("  Total calls: {}", stats.total_calls);
    println!("  HDC hits: {}", stats.hdc_cache_hits);
    println!("  Hit rate: {:.1}%", stats.hdc_cache_hit_rate() * 100.0);
}

/// Benchmark probe projection only (no encoder).
fn bench_probe_only(c: &mut Criterion) {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping benchmark: probe weights not found");
        return;
    }

    let probe = match NeuralBridge::load(weights_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load probe: {}", e);
            return;
        }
    };

    // Simulate a 1024-dim embedding
    let embedding: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();

    c.bench_function("neural_bridge_probe_only", |b| {
        b.iter(|| probe.project_to_packed(black_box(&embedding)).unwrap())
    });
}

/// Benchmark batch encoding.
fn bench_batch_encode(c: &mut Criterion) {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping benchmark: probe weights not found");
        return;
    }

    let mut bridge = match NeuralBridgeV2::load_default() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    let texts: Vec<&str> = vec![
        "consciousness",
        "awareness",
        "sentience",
        "qualia",
        "phenomenal experience",
    ];

    let mut group = c.benchmark_group("neural_bridge_batch");
    group.sample_size(10);

    group.bench_function("batch_5_texts", |b| {
        b.iter(|| {
            bridge.clear_cache();
            bridge.encode_batch_to_hdc(black_box(&texts)).unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_cached,
    bench_probe_only,
    bench_encode_uncached,
    bench_batch_encode,
);

criterion_main!(benches);
