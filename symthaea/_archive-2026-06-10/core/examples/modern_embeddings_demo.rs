// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Modern Embeddings Demo
//!
//! Demonstrates the unified embedding interface for modern transformer models,
//! replacing the problematic BERT layer extraction.
//!
//! Run with:
//!   cargo run --example modern_embeddings_demo --features neural-bridge

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::modern_embeddings::{
    EmbeddingConfig, ModelBackend, PhenomenalLayerAnalyzer, UnifiedEmbedder, print_model_summary,
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("========================================================================");
    println!("            MODERN EMBEDDINGS DEMO - Unified Interface                  ");
    println!("========================================================================\n");

    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This demo requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example modern_embeddings_demo --features neural-bridge");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_demo()
}

#[cfg(feature = "neural-bridge")]
fn run_demo() -> Result<()> {
    // 1. Show model status
    println!("1. AVAILABLE MODELS\n");
    print_model_summary();
    println!();

    // 2. Create unified embedder
    println!("2. CREATING UNIFIED EMBEDDER\n");

    let config = EmbeddingConfig {
        backend: ModelBackend::BgeM3,
        use_gpu: true,
        enable_cache: true,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Backend: {:?}", config.backend);
    println!("  Use GPU: {}", config.use_gpu);
    println!("  Cache: {}", config.enable_cache);
    println!();

    println!("Loading model...");
    let start = std::time::Instant::now();
    let mut embedder = UnifiedEmbedder::new(config)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!();

    println!("Model Info:");
    println!("  Name: {}", embedder.model_name());
    println!("  Dimension: {}", embedder.dimension());
    println!("  Layers: {}", embedder.num_layers());
    println!(
        "  Phenomenal Corridor: Layer {}",
        embedder.phenomenal_corridor_layer()
    );
    println!("  Using GPU: {}", embedder.is_gpu());
    println!();

    // 3. Basic embedding
    println!("3. BASIC EMBEDDING\n");

    let texts = [
        "The vivid redness of the sunset filled my awareness",
        "The recursive algorithm terminates in O(n log n) time",
        "What is it like to experience consciousness?",
        "Matrix multiplication computes dot products efficiently",
    ];

    for text in &texts {
        let start = std::time::Instant::now();
        let embedding = embedder.embed(text)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("Text: \"{}\"", &text[..text.len().min(50)]);
        println!(
            "  Embedding: [{:.4}, {:.4}, {:.4}, ... ] (dim={})",
            embedding[0],
            embedding[1],
            embedding[2],
            embedding.len()
        );
        println!("  Time: {:.2}ms", elapsed_ms);
        println!();
    }

    // 4. Layer-wise extraction
    println!("4. LAYER-WISE EXTRACTION (H2 Hypothesis Testing)\n");

    let phenomenal_text = "The subjective experience of seeing red";
    let layers_to_extract = [0, 6, 12, 18, 22, 23];

    println!("Text: \"{}\"", phenomenal_text);
    println!();
    println!("Layer | Depth | First 3 values               | Time");
    println!("------|-------|------------------------------|-------");

    for layer_idx in layers_to_extract {
        let output = embedder.extract_layer(phenomenal_text, layer_idx)?;
        let depth = output.layer_idx as f64 / (embedder.num_layers() - 1) as f64;

        println!(
            "{:5} | {:5.1}% | [{:7.4}, {:7.4}, {:7.4}] | {:4}us",
            output.layer_idx,
            depth * 100.0,
            output.activation[0],
            output.activation[1],
            output.activation[2],
            output.processing_time_us
        );
    }
    println!();

    // 5. Full H2 analysis
    println!("5. PHENOMENAL LAYER ANALYSIS (H2 Hypothesis)\n");

    let mut analyzer = PhenomenalLayerAnalyzer::new()?;

    let test_texts = [
        ("The vivid qualia of pain", "Phenomenal"),
        ("The algorithm computes the result", "Functional"),
    ];

    for (text, category) in test_texts {
        println!("Text: \"{}\" ({})", text, category);

        let result = analyzer.analyze(text)?;

        println!("  Analysis time: {}us", result.total_time_us);
        println!("  Phenomenal corridor:");
        println!(
            "    Peak layer: {} (depth: {:.1}%)",
            result.phenomenal_corridor.peak_layer,
            result.phenomenal_corridor.peak_depth * 100.0
        );
        println!(
            "    Peak score: {:.4}",
            result.phenomenal_corridor.peak_score
        );
        println!(
            "    Corridor width: {} layers",
            result.phenomenal_corridor.width
        );
        println!(
            "    Matches H2 prediction (~92%): {}",
            if result.phenomenal_corridor.matches_h2_prediction {
                "YES"
            } else {
                "NO"
            }
        );

        // Show layer scores
        println!("  Layer scores (top 5 by phenomenal score):");
        let mut sorted_scores = result.layer_scores.clone();
        sorted_scores.sort_by(|a, b| b.phenomenal_score.partial_cmp(&a.phenomenal_score).unwrap());

        for score in sorted_scores.iter().take(5) {
            println!(
                "    Layer {:2} ({:5.1}%): phen={:.4}, unity={:.4}, phi={:.4}",
                score.layer_idx,
                score.depth_fraction * 100.0,
                score.phenomenal_score,
                score.unity,
                score.phi_loading
            );
        }
        println!();
    }

    // 6. Statistics
    println!("6. EMBEDDER STATISTICS\n");
    let stats = embedder.stats();
    println!("  Total calls: {}", stats.total_calls);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Layer extractions: {}", stats.layer_extractions);
    println!("  Total time: {}us", stats.total_time_us);

    println!();
    println!("========================================================================");
    println!("                          DEMO COMPLETE                                 ");
    println!("========================================================================");

    Ok(())
}
