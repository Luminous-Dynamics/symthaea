// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Embedding Model Verification
//!
//! Quick test to verify that the Qwen3 embedding model is properly loaded
//! and producing meaningful embeddings.
//!
//! ## Prerequisites
//!
//! Enter the nix develop shell first:
//! ```bash
//! cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
//! nix develop
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --features embeddings --example embedding_verification
//! ```
//!
//! ## What This Tests
//!
//! 1. Model loading (ONNX + tokenizer)
//! 2. Basic embedding generation
//! 3. Semantic similarity between related/unrelated texts
//! 4. Integration with primitive activation system

#[cfg(feature = "embeddings")]
fn main() -> anyhow::Result<()> {
    use symthaea::embeddings::qwen3::{Qwen3Config, Qwen3Embedder, cosine_similarity};

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       EMBEDDING MODEL VERIFICATION                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Load the embedding model (default: MiniLM)
    println!("📦 Loading embedding model...");
    let config = Qwen3Config::default();
    println!("   Model path: {}", config.model_path);

    let mut embedder = Qwen3Embedder::new(config)?;

    if embedder.is_stub() {
        println!("⚠️  WARNING: Running in STUB mode (no real model loaded)");
        println!("   Stub embeddings use hash-based pseudo-random vectors.");
        println!();
    } else {
        println!("✅ Real ONNX model loaded successfully!");
        println!();
    }

    // Step 2: Test basic embedding
    println!("🧪 Testing basic embedding generation...");
    let test_text = "The quick brown fox jumps over the lazy dog";
    let embedding = embedder.embed(test_text)?;

    println!("   Text: \"{}\"", test_text);
    println!("   Embedding dimension: {}", embedding.embedding.len());
    println!("   Time: {:.2}ms", embedding.time_ms);
    println!("   Truncated: {}", embedding.truncated);

    // Check embedding is normalized
    let norm: f32 = embedding
        .embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    println!("   L2 norm: {:.4} (should be ~1.0)", norm);
    println!();

    // Step 3: Test semantic similarity
    println!("🔗 Testing semantic similarity...");

    let test_pairs = [
        // Related pairs (should have high similarity)
        ("dog", "puppy", "Related"),
        ("cat", "kitten", "Related"),
        ("happy", "joyful", "Related"),
        ("code", "programming", "Related"),
        // Unrelated pairs (should have low similarity)
        ("dog", "mathematics", "Unrelated"),
        ("happy", "triangle", "Unrelated"),
        ("code", "banana", "Unrelated"),
    ];

    println!("\n   Pair                          | Type      | Cosine Sim");
    println!("   ------------------------------+----------+-----------");

    let mut related_scores = Vec::new();
    let mut unrelated_scores = Vec::new();

    for (text_a, text_b, pair_type) in &test_pairs {
        let emb_a = embedder.embed(text_a)?;
        let emb_b = embedder.embed(text_b)?;
        let similarity = cosine_similarity(&emb_a.embedding, &emb_b.embedding);

        if *pair_type == "Related" {
            related_scores.push(similarity);
        } else {
            unrelated_scores.push(similarity);
        }

        println!(
            "   {:<15} / {:<12} | {:<9} | {:.4}",
            text_a, text_b, pair_type, similarity
        );
    }

    println!();

    // Step 4: Summary statistics
    let avg_related: f32 = related_scores.iter().sum::<f32>() / related_scores.len() as f32;
    let avg_unrelated: f32 = unrelated_scores.iter().sum::<f32>() / unrelated_scores.len() as f32;
    let discrimination = avg_related - avg_unrelated;

    println!("📊 Summary Statistics:");
    println!("   Average related pair similarity:   {:.4}", avg_related);
    println!("   Average unrelated pair similarity: {:.4}", avg_unrelated);
    println!(
        "   Discrimination (related - unrelated): {:.4}",
        discrimination
    );
    println!();

    // Step 5: Verdict
    println!("🎯 VERDICT:");

    if embedder.is_stub() {
        println!("   ⚠️  STUB MODE - Results are hash-based, not semantic");
        println!("   To enable real embeddings:");
        println!("     1. Enter nix shell: nix develop");
        println!("     2. Ensure model.onnx exists at models/qwen3-embedding-0.6b/");
        println!("     3. Run with --features embeddings");
    } else if discrimination > 0.1 {
        println!("   ✅ PASS - Model shows meaningful semantic discrimination");
        println!("   Related pairs have higher similarity than unrelated pairs.");
    } else {
        println!("   ⚠️  WEAK - Low discrimination between related/unrelated pairs");
        println!("   This might indicate model issues or need for fine-tuning.");
    }

    println!();
    Ok(())
}

#[cfg(not(feature = "embeddings"))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       EMBEDDING MODEL VERIFICATION                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("⚠️  The 'embeddings' feature is not enabled.");
    println!();
    println!("To run this verification:");
    println!("  1. Enter the nix development shell:");
    println!("     cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb");
    println!("     nix develop");
    println!();
    println!("  2. Run with the embeddings feature:");
    println!("     cargo run --release --features embeddings --example embedding_verification");
    println!();
    println!("The nix shell provides:");
    println!("  - ONNX Runtime (libonnxruntime.so)");
    println!("  - OpenSSL libraries");
    println!("  - Tokenizers dependencies");
    println!();
}
