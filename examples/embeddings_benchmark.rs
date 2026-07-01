// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Embeddings throughput benchmark.
//!
//! Measures sequential and batch inference performance for the Qwen3 embedder.
//!
//! # Usage
//!
//! ```bash
//! # Simulation mode (no model needed)
//! EMBEDDINGS_SIMULATED=1 cargo run --release --features embeddings --example embeddings_benchmark
//!
//! # With a local model directory
//! QWEN3_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B cargo run --release --features embeddings --example embeddings_benchmark
//!
//! # Default: auto-downloads from HF Hub (requires network)
//! cargo run --release --features embeddings --example embeddings_benchmark
//! ```

#[cfg(not(feature = "embeddings"))]
fn main() {
    eprintln!("This example requires the `embeddings` feature:");
    eprintln!("  cargo run --release --features embeddings --example embeddings_benchmark");
    std::process::exit(1);
}

#[cfg(feature = "embeddings")]
fn main() {
    embeddings_benchmark::run();
}

#[cfg(feature = "embeddings")]
mod embeddings_benchmark {
    use std::time::Instant;
    use symthaea::embeddings::qwen3::{Qwen3Config, Qwen3Embedder};

    const NUM_TEXTS: usize = 100;
    const WARMUP_COUNT: usize = 5;

    /// Generate diverse benchmark texts of varying lengths.
    fn generate_texts(count: usize) -> Vec<String> {
        let templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process natural language efficiently.",
            "Quantum computing explores the boundaries of classical physics and information theory.",
            "Rust provides memory safety without garbage collection through its ownership system.",
            "Neural networks with attention mechanisms have revolutionized natural language processing tasks.",
            "Hyperdimensional computing represents information using high-dimensional random vectors for robust distributed representations.",
            "The Johnson-Lindenstrauss lemma guarantees that high-dimensional points can be projected to lower dimensions while approximately preserving pairwise distances between all points.",
            "Consciousness remains one of the hardest problems in philosophy and neuroscience, with theories ranging from integrated information theory to global workspace theory attempting to explain subjective experience.",
        ];
        (0..count)
            .map(|i| {
                let base = &templates[i % templates.len()];
                if i < templates.len() {
                    base.to_string()
                } else {
                    format!("{base} (variant {i})")
                }
            })
            .collect()
    }

    pub fn run() {
        println!("=== Embeddings Throughput Benchmark ===\n");

        // Resolve config from environment
        let config = if std::env::var("EMBEDDINGS_SIMULATED").is_ok() {
            println!("Mode: SIMULATED (no model loaded)\n");
            Qwen3Config::simulated()
        } else if let Ok(path) = std::env::var("QWEN3_MODEL_PATH") {
            println!("Mode: LOCAL model at {path}\n");
            Qwen3Config::qwen3_06b(&path)
        } else {
            println!("Mode: HF Hub download (Qwen/Qwen3-Embedding-0.6B)\n");
            Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B")
        };

        let dim = config.embedding_dim;
        let mut embedder = match Qwen3Embedder::new(config) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to create embedder: {e}");
                std::process::exit(1);
            }
        };

        println!(
            "Model loaded: {} (dim={dim})\n",
            if embedder.is_model_loaded() {
                "Burn inference"
            } else {
                "simulation"
            }
        );

        let texts = generate_texts(NUM_TEXTS);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // ── Warmup ──
        println!("Warming up ({WARMUP_COUNT} texts)...");
        for t in &text_refs[..WARMUP_COUNT] {
            let _ = embedder.embed(t);
        }
        embedder.clear_cache();

        // ── Sequential benchmark ──
        println!("Running sequential benchmark ({NUM_TEXTS} texts)...");
        let mut latencies = Vec::with_capacity(NUM_TEXTS);
        let seq_start = Instant::now();
        for t in &text_refs {
            embedder.clear_cache(); // Ensure no cache hits
            let t0 = Instant::now();
            let _ = embedder.embed(t);
            latencies.push(t0.elapsed().as_secs_f64() * 1000.0); // ms
        }
        let seq_elapsed = seq_start.elapsed().as_secs_f64();
        let seq_throughput = NUM_TEXTS as f64 / seq_elapsed;

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg_latency: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        println!("\n--- Sequential Results ---");
        println!("  Throughput: {seq_throughput:.1} texts/sec");
        println!("  Avg latency: {avg_latency:.3} ms");
        println!("  P50 latency: {p50:.3} ms");
        println!("  P99 latency: {p99:.3} ms");

        // ── Batch benchmark ──
        println!("\nRunning batch benchmark ({NUM_TEXTS} texts in one call)...");
        embedder.clear_cache();
        let batch_start = Instant::now();
        let _ = embedder.embed_batch(&text_refs);
        let batch_elapsed = batch_start.elapsed().as_secs_f64();
        let batch_throughput = NUM_TEXTS as f64 / batch_elapsed;

        println!("\n--- Batch Results ---");
        println!("  Throughput: {batch_throughput:.1} texts/sec");
        println!("  Total time: {:.1} ms", batch_elapsed * 1000.0);

        // ── Summary ──
        let speedup = batch_throughput / seq_throughput;
        println!("\n--- Summary ---");
        println!("  Batch speedup: {speedup:.2}x over sequential");
        println!("  Embedding dim: {dim}");
        println!("  Texts benchmarked: {NUM_TEXTS}");
    }
}
