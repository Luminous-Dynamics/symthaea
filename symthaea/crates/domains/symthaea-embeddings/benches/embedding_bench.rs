// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Criterion benchmarks for symthaea-embeddings.
//!
//! Run simulation benchmarks (always available):
//!   cargo bench -p symthaea-embeddings
//!
//! Run burn benchmarks (requires `burn` feature):
//!   cargo bench -p symthaea-embeddings --features burn

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea_embeddings::{Qwen3Config, Qwen3Embedder};

// ── Simulation benchmarks (always available) ─────────────────────────

fn simulation_single_embed(c: &mut Criterion) {
    let config = Qwen3Config::simulated();
    let mut embedder = Qwen3Embedder::new(config).unwrap();

    c.bench_function("simulation_single_embed", |b| {
        b.iter(|| {
            embedder.clear_cache();
            embedder
                .embed(black_box("The quick brown fox jumps over the lazy dog"))
                .unwrap()
        })
    });
}

fn simulation_cached_embed(c: &mut Criterion) {
    let config = Qwen3Config::simulated();
    let mut embedder = Qwen3Embedder::new(config).unwrap();
    // Prime the cache
    embedder
        .embed("The quick brown fox jumps over the lazy dog")
        .unwrap();

    c.bench_function("simulation_cached_embed", |b| {
        b.iter(|| {
            embedder
                .embed(black_box("The quick brown fox jumps over the lazy dog"))
                .unwrap()
        })
    });
}

fn simulation_batch_8(c: &mut Criterion) {
    let config = Qwen3Config::simulated();
    let mut embedder = Qwen3Embedder::new(config).unwrap();

    let texts: Vec<&str> = vec![
        "The quick brown fox",
        "jumps over the lazy dog",
        "NixOS is a Linux distribution",
        "Holographic liquid brain architecture",
        "HDC hypervector encoding",
        "Semantic similarity search",
        "Quantum computing fundamentals",
        "Rust programming language",
    ];

    c.bench_function("simulation_batch_8", |b| {
        b.iter(|| {
            embedder.clear_cache();
            embedder.embed_batch(black_box(&texts)).unwrap()
        })
    });
}

fn bridge_project_single(c: &mut Criterion) {
    use symthaea_embeddings::HdcBridge;

    let bridge = HdcBridge::new();
    let embedding: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

    c.bench_function("bridge_project_single", |b| {
        b.iter(|| bridge.project(black_box(&embedding)))
    });
}

fn bridge_project_batch(c: &mut Criterion) {
    use symthaea_embeddings::HdcBridge;

    let bridge = HdcBridge::new();
    let mut group = c.benchmark_group("bridge_project_batch");

    for &size in &[4, 16, 64] {
        let embeddings: Vec<Vec<f32>> = (0..size)
            .map(|i| {
                (0..1024)
                    .map(|j| ((i * 1000 + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        group.bench_function(format!("{size}"), |b| {
            b.iter(|| bridge.project_batch(black_box(&embeddings)))
        });
    }

    group.finish();
}

fn simulation_dimensions(c: &mut Criterion) {
    let dims = [64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("simulation_mrl_dimensions");

    for &dim in &dims {
        let config = Qwen3Config::simulated().with_matryoshka_dim(dim);
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        group.bench_function(format!("{dim}D"), |b| {
            b.iter(|| {
                embedder.clear_cache();
                embedder
                    .embed(black_box("Matryoshka dimension reduction benchmark"))
                    .unwrap()
            })
        });
    }

    group.finish();
}

/// Full pipeline benchmark: embed text → HdcBridge → BinaryHV.
fn bridge_project_roundtrip(c: &mut Criterion) {
    use symthaea_embeddings::HdcBridge;

    let config = Qwen3Config::simulated();
    let mut embedder = Qwen3Embedder::new(config).unwrap();
    let bridge = HdcBridge::for_qwen3();

    c.bench_function("bridge_project_roundtrip", |b| {
        b.iter(|| {
            embedder.clear_cache();
            let result = embedder
                .embed(black_box("The quick brown fox jumps over the lazy dog"))
                .unwrap();
            bridge.project(black_box(&result.embedding))
        })
    });
}

criterion_group!(
    simulation_benches,
    simulation_single_embed,
    simulation_cached_embed,
    simulation_batch_8,
    simulation_dimensions,
    bridge_project_single,
    bridge_project_batch,
    bridge_project_roundtrip,
);

// ── Burn benchmarks (feature-gated) ─────────────────────────────────

#[cfg(feature = "burn")]
mod burn_benches {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::*;
    use symthaea_embeddings::qwen3::model::Qwen3ModelConfig;

    type B = NdArray;

    pub fn burn_tiny_single_forward(c: &mut Criterion) {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        c.bench_function("burn_tiny_single_forward", |b| {
            b.iter(|| {
                let ids = Tensor::<B, 2, Int>::zeros([1, 8], &device);
                model.forward(black_box(ids), None)
            })
        });
    }

    pub fn burn_tiny_batch_8(c: &mut Criterion) {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        c.bench_function("burn_tiny_batch_8_forward", |b| {
            b.iter(|| {
                let ids = Tensor::<B, 2, Int>::zeros([8, 16], &device);
                model.forward(black_box(ids), None)
            })
        });
    }
}

// ── Hub benchmarks (feature-gated) ──────────────────────────────────

#[cfg(feature = "burn-hub")]
mod hub_benches {
    use super::*;

    pub fn burn_hub_single_embed(c: &mut Criterion) {
        let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
        let mut embedder = match Qwen3Embedder::new(config) {
            Ok(e) if e.is_model_loaded() => e,
            _ => {
                eprintln!("Skipping burn_hub_single_embed: model not available");
                return;
            }
        };

        c.bench_function("burn_hub_single_embed", |b| {
            b.iter(|| {
                embedder.clear_cache();
                embedder
                    .embed(black_box("The quick brown fox jumps over the lazy dog"))
                    .unwrap()
            })
        });
    }

    pub fn burn_hub_batch_8(c: &mut Criterion) {
        let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
        let mut embedder = match Qwen3Embedder::new(config) {
            Ok(e) if e.is_model_loaded() => e,
            _ => {
                eprintln!("Skipping burn_hub_batch_8: model not available");
                return;
            }
        };

        let texts: Vec<&str> = vec![
            "The quick brown fox",
            "jumps over the lazy dog",
            "NixOS is a Linux distribution",
            "Holographic liquid brain architecture",
            "HDC hypervector encoding",
            "Semantic similarity search",
            "Quantum computing fundamentals",
            "Rust programming language",
        ];

        c.bench_function("burn_hub_batch_8", |b| {
            b.iter(|| {
                embedder.clear_cache();
                embedder.embed_batch(black_box(&texts)).unwrap()
            })
        });
    }
}

#[cfg(feature = "burn")]
criterion_group!(
    burn_bench_group,
    burn_benches::burn_tiny_single_forward,
    burn_benches::burn_tiny_batch_8,
);

#[cfg(feature = "burn-hub")]
criterion_group!(
    hub_bench_group,
    hub_benches::burn_hub_single_embed,
    hub_benches::burn_hub_batch_8,
);

// Conditional criterion_main! based on features
#[cfg(all(feature = "burn-hub", feature = "burn"))]
criterion_main!(simulation_benches, burn_bench_group, hub_bench_group);

#[cfg(all(feature = "burn", not(feature = "burn-hub")))]
criterion_main!(simulation_benches, burn_bench_group);

#[cfg(not(feature = "burn"))]
criterion_main!(simulation_benches);
