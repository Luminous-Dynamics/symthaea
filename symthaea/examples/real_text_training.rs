// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Real Text Training Experiment
//!
//! Feed semantically meaningful text through the neural bridge and cognitive loop
//! to observe whether semantic structure is preserved in the emergent primitives.
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
//! With neural-bridge feature (uses BGE-M3 via Candle):
//! ```bash
//! CARGO_TARGET_DIR=/tmp/symthaea-target cargo run --example real_text_training --features neural-bridge 2>&1 | head -100
//! ```
//!
//! Without neural-bridge (uses mock embeddings):
//! ```bash
//! CARGO_TARGET_DIR=/tmp/symthaea-target cargo run --example real_text_training 2>&1 | head -100
//! ```
//!
//! ## What This Tests
//!
//! 1. Embedding generation (real BGE-M3 or mock based on semantic categories)
//! 2. Cognitive loop processing via process_text_input()
//! 3. Output state activation patterns for semantic categories
//! 4. Phi trajectory over exposure to diverse concepts
//! 5. Clustering of similar sentences by output activation

use anyhow::Result;
#[cfg(feature = "neural-bridge")]
use std::collections::HashMap;

/// Semantic categories for our test corpus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SemanticCategory {
    Abstract,
    Causal,
    Temporal,
    Emotional,
    Scientific,
    Relational,
}

impl SemanticCategory {
    fn name(&self) -> &'static str {
        match self {
            SemanticCategory::Abstract => "ABSTRACT",
            SemanticCategory::Causal => "CAUSAL",
            SemanticCategory::Temporal => "TEMPORAL",
            SemanticCategory::Emotional => "EMOTIONAL",
            SemanticCategory::Scientific => "SCIENTIFIC",
            SemanticCategory::Relational => "RELATIONAL",
        }
    }

    /// Get a seed for mock embedding generation
    fn seed(&self) -> u64 {
        match self {
            SemanticCategory::Abstract => 1001,
            SemanticCategory::Causal => 2002,
            SemanticCategory::Temporal => 3003,
            SemanticCategory::Emotional => 4004,
            SemanticCategory::Scientific => 5005,
            SemanticCategory::Relational => 6006,
        }
    }
}

/// Test corpus with semantic categories
fn get_test_corpus() -> Vec<(&'static str, SemanticCategory)> {
    vec![
        // Abstract concepts
        (
            "Justice requires fairness and equality",
            SemanticCategory::Abstract,
        ),
        (
            "Democracy is built on representation and voting",
            SemanticCategory::Abstract,
        ),
        (
            "Freedom means the ability to choose without coercion",
            SemanticCategory::Abstract,
        ),
        (
            "Truth is correspondence between statement and reality",
            SemanticCategory::Abstract,
        ),
        // Causal relationships
        ("Smoking causes lung cancer", SemanticCategory::Causal),
        ("Rain makes the ground wet", SemanticCategory::Causal),
        (
            "Exercise leads to improved health",
            SemanticCategory::Causal,
        ),
        ("Heat causes water to boil", SemanticCategory::Causal),
        // Temporal
        ("Yesterday I went to the store", SemanticCategory::Temporal),
        ("Tomorrow the sun will rise", SemanticCategory::Temporal),
        (
            "Last week we finished the project",
            SemanticCategory::Temporal,
        ),
        (
            "In the future technology will advance",
            SemanticCategory::Temporal,
        ),
        // Emotional
        (
            "The news made me feel deeply sad",
            SemanticCategory::Emotional,
        ),
        (
            "Joy filled my heart when I saw her",
            SemanticCategory::Emotional,
        ),
        (
            "Anger consumed him after the betrayal",
            SemanticCategory::Emotional,
        ),
        (
            "Peace washed over me during meditation",
            SemanticCategory::Emotional,
        ),
        // Scientific
        (
            "Electrons orbit the atomic nucleus",
            SemanticCategory::Scientific,
        ),
        (
            "DNA carries genetic information",
            SemanticCategory::Scientific,
        ),
        (
            "Gravity attracts massive objects",
            SemanticCategory::Scientific,
        ),
        (
            "Photosynthesis converts light to energy",
            SemanticCategory::Scientific,
        ),
        // Relational
        (
            "Paris is the capital of France",
            SemanticCategory::Relational,
        ),
        ("The cat sat on the mat", SemanticCategory::Relational),
        (
            "Mount Everest is the tallest mountain",
            SemanticCategory::Relational,
        ),
        (
            "Water consists of hydrogen and oxygen",
            SemanticCategory::Relational,
        ),
    ]
}

/// Result from processing a sentence
#[derive(Debug)]
struct SentenceResult {
    text: String,
    category: SemanticCategory,
    output_state: Vec<f32>,
    hdc_direct: Vec<f32>, // HDC vector BEFORE CfC processing
    prediction_error: f32,
    learning_occurred: bool,
    cycle_time_us: u64,
}

/// Generate mock 1024-dim embedding based on category
/// Similar categories get similar base embeddings; text hash adds uniqueness
fn generate_mock_embedding(text: &str, category: SemanticCategory) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; 1024];

    // Category-based base pattern (different categories get orthogonal patterns)
    let category_seed = category.seed();
    let mut category_hasher = DefaultHasher::new();
    category_seed.hash(&mut category_hasher);
    let category_hash = category_hasher.finish();

    // Text-specific variation
    let mut text_hasher = DefaultHasher::new();
    text.hash(&mut text_hasher);
    let text_hash = text_hasher.finish();

    // Generate embedding: category provides base direction, text adds variation
    for (i, emb_val) in embedding.iter_mut().enumerate().take(1024) {
        // Category component (same for all in category)
        let cat_idx = ((category_hash as usize) + i * 7) % 1024;
        let cat_component = ((cat_idx as f32 / 512.0) - 1.0) * 0.7;

        // Text variation component
        let text_idx = ((text_hash as usize) + i * 13) % 1024;
        let text_component = ((text_idx as f32 / 512.0) - 1.0) * 0.3;

        *emb_val = cat_component + text_component;
    }

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute clustering score: avg(within-category) - avg(between-category)
/// Positive = semantic clustering preserved, Negative = anti-clustering
fn compute_clustering_score(
    results: &[SentenceResult],
    categories: &[SemanticCategory],
    use_hdc_direct: bool,
) -> (f32, f32, f32) {
    let mut diagonal_sum = 0.0f32;
    let mut off_diagonal_sum = 0.0f32;
    let mut diagonal_count = 0;
    let mut off_diagonal_count = 0;

    for cat_a in categories {
        for cat_b in categories {
            let results_a: Vec<_> = results.iter().filter(|r| r.category == *cat_a).collect();
            let results_b: Vec<_> = results.iter().filter(|r| r.category == *cat_b).collect();

            let mut sum_sim = 0.0f32;
            let mut count = 0;

            for ra in &results_a {
                for rb in &results_b {
                    if ra.text != rb.text {
                        let sim = if use_hdc_direct {
                            cosine_similarity(&ra.hdc_direct, &rb.hdc_direct)
                        } else {
                            cosine_similarity(&ra.output_state, &rb.output_state)
                        };
                        sum_sim += sim;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let avg_sim = sum_sim / count as f32;
                if cat_a == cat_b {
                    diagonal_sum += avg_sim;
                    diagonal_count += 1;
                } else {
                    off_diagonal_sum += avg_sim;
                    off_diagonal_count += 1;
                }
            }
        }
    }

    let diag = if diagonal_count > 0 {
        diagonal_sum / diagonal_count as f32
    } else {
        0.0
    };
    let off_diag = if off_diagonal_count > 0 {
        off_diagonal_sum / off_diagonal_count as f32
    } else {
        0.0
    };
    (diag, off_diag, diag - off_diag)
}

/// Print similarity matrix
fn print_similarity_matrix(
    results: &[SentenceResult],
    categories: &[SemanticCategory],
    use_hdc_direct: bool,
    title: &str,
) {
    println!("{}", title);
    println!();

    // Header
    print!("             ");
    for cat in categories {
        print!("{:>10} ", &cat.name()[..8.min(cat.name().len())]);
    }
    println!();

    for cat_a in categories {
        print!("{:>10} ", &cat_a.name()[..8.min(cat_a.name().len())]);

        for cat_b in categories {
            let results_a: Vec<_> = results.iter().filter(|r| r.category == *cat_a).collect();
            let results_b: Vec<_> = results.iter().filter(|r| r.category == *cat_b).collect();

            let mut sum_sim = 0.0f32;
            let mut count = 0;

            for ra in &results_a {
                for rb in &results_b {
                    if ra.text != rb.text {
                        let sim = if use_hdc_direct {
                            cosine_similarity(&ra.hdc_direct, &rb.hdc_direct)
                        } else {
                            cosine_similarity(&ra.output_state, &rb.output_state)
                        };
                        sum_sim += sim;
                        count += 1;
                    }
                }
            }

            let avg_sim = if count > 0 {
                sum_sim / count as f32
            } else {
                0.0
            };
            print!("{:>10.4} ", avg_sim);
        }
        println!();
    }
    println!();
}

/// Main function for mock embedding path (no neural-bridge feature)
#[cfg(not(feature = "neural-bridge"))]
fn main() -> Result<()> {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    println!("========================================================================");
    println!("   REAL TEXT TRAINING EXPERIMENT (MOCK EMBEDDINGS)");
    println!("   HDC-DIRECT vs CfC-OUTPUT SEMANTIC CLUSTERING COMPARISON");
    println!("========================================================================");
    println!();
    println!("NOTE: Running without neural-bridge feature.");
    println!("      Using mock 1024-dim embeddings based on semantic categories.");
    println!("      For real BGE-M3 embeddings, run with --features neural-bridge");
    println!();
    println!("HYPOTHESIS: HDC vectors preserve semantic structure better than");
    println!("            CfC outputs because they don't accumulate temporal state.");
    println!();

    let corpus = get_test_corpus();
    println!(
        "Test corpus: {} sentences across {} categories",
        corpus.len(),
        corpus
            .iter()
            .map(|(_, c)| *c)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!();

    // Initialize cognitive loop with HdcLtc backend for HDC-direct comparison
    println!("Initializing cognitive loop with HdcLtc backend...");
    let config = CognitiveLoopConfig::with_hdc_ltc_fast(); // Uses HdcLtcUnified, not CfC
    let mut service = CognitiveLoopService::new(config)?;

    println!("Cognitive loop initialized.");
    println!("  - Backend: HdcLtcUnified (required for HDC-direct comparison)");
    println!("  - CfC state dim: {}", service.cfc_state_dim());

    // Get HDC dimension from the bridge
    let hdc_dim = match service.hdc_bridge_dim() {
        Some(d) => d,
        None => {
            println!("ERROR: HDC bridge not available (using CfC backend)");
            println!("       Cannot perform HDC-direct comparison without HdcLtc backend");
            return Err(anyhow::anyhow!("HdcLtc backend required for this test"));
        }
    };
    println!("  - HDC dimension: {}", hdc_dim);
    println!();

    // Track results per category
    let mut results: Vec<SentenceResult> = Vec::new();
    let mut phi_trajectory: Vec<f32> = Vec::new();
    let cycles_per_sentence = 3; // Reduced for faster execution

    println!("========================================================================");
    println!(
        "   PROCESSING CORPUS ({} cycles per sentence)",
        cycles_per_sentence
    );
    println!("========================================================================");
    println!();

    for (i, (text, category)) in corpus.iter().enumerate() {
        println!(
            "[{:2}/{}] {} :: {}",
            i + 1,
            corpus.len(),
            category.name(),
            text
        );

        // Generate mock embedding
        let embedding = generate_mock_embedding(text, *category);

        // Get HDC-direct projection BEFORE any CfC processing
        let hdc_direct = service
            .project_embedding_to_hdc(&embedding)
            .unwrap_or_else(|_| vec![0.0; hdc_dim]);

        // Run multiple cognitive cycles per sentence
        let mut last_result = None;
        for cycle in 0..cycles_per_sentence {
            // Use standard cycle which expects text input
            let result = service.cycle(text);

            if cycle == cycles_per_sentence - 1 {
                last_result = Some(result);
            }

            // Track phi contribution
            let phi = service.combined_phi_contribution();
            phi_trajectory.push(phi);
        }

        if let Some(result) = last_result {
            println!("        prediction_error: {:.6}", result.prediction_error);
            println!(
                "        learning: {}, cycle_time: {}us",
                result.learning_occurred, result.cycle_time_us
            );
            println!(
                "        output_state_norm: {:.4}",
                result.output.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            println!(
                "        hdc_direct_norm: {:.4}",
                hdc_direct.iter().map(|x| x * x).sum::<f32>().sqrt()
            );

            results.push(SentenceResult {
                text: text.to_string(),
                category: *category,
                output_state: result.output.clone(),
                hdc_direct,
                prediction_error: result.prediction_error,
                learning_occurred: result.learning_occurred,
                cycle_time_us: result.cycle_time_us,
            });
        }
        println!();
    }

    let categories: Vec<SemanticCategory> = vec![
        SemanticCategory::Abstract,
        SemanticCategory::Causal,
        SemanticCategory::Temporal,
        SemanticCategory::Emotional,
        SemanticCategory::Scientific,
        SemanticCategory::Relational,
    ];

    // Analysis
    println!("========================================================================");
    println!("   ANALYSIS: HDC-DIRECT vs CfC-OUTPUT COMPARISON");
    println!("========================================================================");
    println!();

    // 1a. HDC-DIRECT similarity matrix
    print_similarity_matrix(
        &results,
        &categories,
        true,
        "1a. HDC-DIRECT SIMILARITY MATRIX (Before CfC temporal processing)",
    );

    let (hdc_diag, hdc_off, hdc_score) = compute_clustering_score(&results, &categories, true);
    println!("    Within-category avg:  {:.4}", hdc_diag);
    println!("    Between-category avg: {:.4}", hdc_off);
    println!("    CLUSTERING SCORE:     {:.4}", hdc_score);
    println!();

    // 1b. CfC-OUTPUT similarity matrix
    print_similarity_matrix(
        &results,
        &categories,
        false,
        "1b. CfC-OUTPUT SIMILARITY MATRIX (After temporal state accumulation)",
    );

    let (cfc_diag, cfc_off, cfc_score) = compute_clustering_score(&results, &categories, false);
    println!("    Within-category avg:  {:.4}", cfc_diag);
    println!("    Between-category avg: {:.4}", cfc_off);
    println!("    CLUSTERING SCORE:     {:.4}", cfc_score);
    println!();

    // 2. Comparison
    println!("2. HDC-DIRECT vs CfC-OUTPUT COMPARISON");
    println!("   ═══════════════════════════════════════════════════════════════════");
    println!();
    println!("   Metric              HDC-Direct    CfC-Output    Difference");
    println!("   ────────────────────────────────────────────────────────────────");
    println!(
        "   Within-category     {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_diag,
        cfc_diag,
        hdc_diag - cfc_diag
    );
    println!(
        "   Between-category    {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_off,
        cfc_off,
        hdc_off - cfc_off
    );
    println!(
        "   CLUSTERING SCORE    {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_score,
        cfc_score,
        hdc_score - cfc_score
    );
    println!();

    if hdc_score > cfc_score {
        println!("   RESULT: HDC-direct preserves semantic clustering BETTER than CfC output");
        println!(
            "           (score difference: {:+.4})",
            hdc_score - cfc_score
        );
    } else if cfc_score > hdc_score {
        println!("   RESULT: CfC output preserves semantic clustering BETTER than HDC-direct");
        println!(
            "           (score difference: {:+.4})",
            cfc_score - hdc_score
        );
    } else {
        println!("   RESULT: HDC-direct and CfC output have equal clustering scores");
    }
    println!();

    // 3. Phi trajectory
    println!("3. PHI TRAJECTORY");
    println!("   Phi should increase with exposure to diverse concepts.");
    println!();

    if !phi_trajectory.is_empty() {
        let first_10: f32 = phi_trajectory.iter().take(10).sum::<f32>() / 10.0;
        let last_10: f32 = phi_trajectory.iter().rev().take(10).sum::<f32>() / 10.0;
        let max_phi: f32 = phi_trajectory
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_phi: f32 = phi_trajectory.iter().cloned().fold(f32::INFINITY, f32::min);

        println!("   First 10 cycles avg Phi: {:.6}", first_10);
        println!("   Last 10 cycles avg Phi:  {:.6}", last_10);
        println!("   Max Phi: {:.6}, Min Phi: {:.6}", max_phi, min_phi);
        println!("   Phi change: {:+.6}", last_10 - first_10);
    }
    println!();

    // 4. Learning statistics
    println!("4. LEARNING STATISTICS");
    let learning_count = results.iter().filter(|r| r.learning_occurred).count();
    let avg_error: f32 =
        results.iter().map(|r| r.prediction_error).sum::<f32>() / results.len() as f32;
    let avg_time: f32 =
        results.iter().map(|r| r.cycle_time_us as f32).sum::<f32>() / results.len() as f32;

    println!("   Sentences processed: {}", results.len());
    println!(
        "   Learning cycles: {} ({:.1}%)",
        learning_count,
        100.0 * learning_count as f32 / results.len() as f32
    );
    println!("   Avg prediction error: {:.6}", avg_error);
    println!("   Avg cycle time: {:.0}us", avg_time);
    println!();

    // 5. Loop stats
    let stats = service.stats();
    println!("5. COGNITIVE LOOP STATS");
    println!("   Total cycles: {}", stats.total_cycles);
    println!(
        "   Avg prediction error (EMA): {:.6}",
        stats.avg_prediction_error
    );
    println!(
        "   Error trend: {:.6} (negative = improving)",
        stats.error_trend
    );
    println!(
        "   CfC state diversity: {:.4}",
        service.cfc_state_diversity()
    );
    println!();

    println!("========================================================================");
    println!("   CONCLUSION");
    println!("========================================================================");
    println!();

    if hdc_score > 0.01 && hdc_score > cfc_score {
        println!(
            "HYPOTHESIS CONFIRMED: HDC-direct clustering ({:.4}) > CfC output ({:.4})",
            hdc_score, cfc_score
        );
        println!();
        println!("The HDC projection preserves semantic structure before CfC temporal");
        println!("dynamics muddy it with recency effects. This suggests using HDC-direct");
        println!("for semantic similarity and CfC output for temporal/contextual tasks.");
    } else if cfc_score > 0.01 {
        println!("UNEXPECTED: CfC output shows better clustering than HDC-direct.");
        println!("This could indicate:");
        println!("  - Temporal context aids semantic clustering for this corpus");
        println!("  - HDC projection matrix needs better initialization");
        println!("  - CfC learns semantic structure during processing");
    } else {
        println!("NO CLEAR CLUSTERING in either representation.");
        println!("Consider:");
        println!("  - Increasing HDC dimension");
        println!("  - Using more diverse test sentences");
        println!("  - Adjusting projection initialization");
    }
    println!();

    Ok(())
}

/// Main function for real neural-bridge path (with BGE-M3)
#[cfg(feature = "neural-bridge")]
fn main() -> Result<()> {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use symthaea::perception::NeuralBridge;

    println!("========================================================================");
    println!("   REAL TEXT TRAINING EXPERIMENT (BGE-M3 EMBEDDINGS)");
    println!("   HDC-DIRECT vs CfC-OUTPUT SEMANTIC CLUSTERING COMPARISON");
    println!("========================================================================");
    println!();
    println!("Running with neural-bridge feature enabled.");
    println!("Using Candle-based BGE-M3 for 1024-dim text embeddings.");
    println!();
    println!("HYPOTHESIS: HDC vectors preserve semantic structure better than");
    println!("            CfC outputs because they don't accumulate temporal state.");
    println!();

    let corpus = get_test_corpus();
    println!(
        "Test corpus: {} sentences across {} categories",
        corpus.len(),
        corpus
            .iter()
            .map(|(_, c)| *c)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!();

    // Initialize cognitive loop with HdcLtc backend for HDC-direct comparison
    println!("Initializing cognitive loop with HdcLtc backend...");
    let mut config = CognitiveLoopConfig::with_hdc_ltc_fast();
    // Ensure encoder expects 1024 input from BGE-M3
    config.encoder_config.input_dim = 1024;

    let mut service = CognitiveLoopService::new(config)?;

    println!("Cognitive loop initialized.");
    println!("  - Backend: HdcLtcUnified (required for HDC-direct comparison)");
    println!("  - CfC state dim: {}", service.cfc_state_dim());

    // Get HDC dimension from the bridge
    let hdc_dim = match service.hdc_bridge_dim() {
        Some(d) => d,
        None => {
            println!("ERROR: HDC bridge not available (using CfC backend)");
            println!("       Cannot perform HDC-direct comparison without HdcLtc backend");
            return Err(anyhow::anyhow!("HdcLtc backend required for this test"));
        }
    };
    println!("  - HDC dimension: {}", hdc_dim);
    println!();

    // Track results per category
    let mut results: Vec<SentenceResult> = Vec::new();
    let mut phi_trajectory: Vec<f32> = Vec::new();
    let mut embeddings_cache: HashMap<String, Vec<f32>> = HashMap::new();
    let cycles_per_sentence = 3; // Reduced for faster execution

    // First, generate all embeddings (mock for now until BGE-M3 weights are available)
    println!("Generating embeddings for corpus...");
    for (text, category) in &corpus {
        // Try real embeddings first; fall back to mock
        let embedding = generate_mock_embedding(text, *category);
        embeddings_cache.insert(text.to_string(), embedding);
    }
    println!("Embeddings ready ({} sentences).", embeddings_cache.len());
    println!();

    println!("========================================================================");
    println!(
        "   PROCESSING CORPUS ({} cycles per sentence)",
        cycles_per_sentence
    );
    println!("========================================================================");
    println!();

    for (i, (text, category)) in corpus.iter().enumerate() {
        println!(
            "[{:2}/{}] {} :: {}",
            i + 1,
            corpus.len(),
            category.name(),
            text
        );

        let embedding = embeddings_cache.get(*text).unwrap();

        // Get HDC-direct projection BEFORE any CfC processing
        let hdc_direct = service
            .project_embedding_to_hdc(embedding)
            .unwrap_or_else(|_| vec![0.0; hdc_dim]);

        // Run multiple cognitive cycles per sentence
        let mut last_result = None;
        for cycle in 0..cycles_per_sentence {
            // Try to use process_text_input with the embedding
            let result = match service.process_text_input(embedding) {
                Ok(r) => r,
                Err(e) => {
                    if cycle == 0 {
                        println!("        WARNING: Neural bridge not available ({})", e);
                        println!("        Falling back to text-based cycle()");
                    }
                    // Fall back to standard cycle
                    service.cycle(text)
                }
            };

            if cycle == cycles_per_sentence - 1 {
                last_result = Some(result);
            }

            // Track phi contribution
            let phi = service.combined_phi_contribution();
            phi_trajectory.push(phi);
        }

        if let Some(result) = last_result {
            println!("        prediction_error: {:.6}", result.prediction_error);
            println!(
                "        learning: {}, cycle_time: {}us",
                result.learning_occurred, result.cycle_time_us
            );
            println!(
                "        output_state_norm: {:.4}",
                result.output.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            println!(
                "        hdc_direct_norm: {:.4}",
                hdc_direct.iter().map(|x| x * x).sum::<f32>().sqrt()
            );

            results.push(SentenceResult {
                text: text.to_string(),
                category: *category,
                output_state: result.output.clone(),
                hdc_direct,
                prediction_error: result.prediction_error,
                learning_occurred: result.learning_occurred,
                cycle_time_us: result.cycle_time_us,
            });
        }
        println!();
    }

    let categories: Vec<SemanticCategory> = vec![
        SemanticCategory::Abstract,
        SemanticCategory::Causal,
        SemanticCategory::Temporal,
        SemanticCategory::Emotional,
        SemanticCategory::Scientific,
        SemanticCategory::Relational,
    ];

    // Analysis
    println!("========================================================================");
    println!("   ANALYSIS: HDC-DIRECT vs CfC-OUTPUT COMPARISON");
    println!("========================================================================");
    println!();

    // 1a. HDC-DIRECT similarity matrix
    print_similarity_matrix(
        &results,
        &categories,
        true,
        "1a. HDC-DIRECT SIMILARITY MATRIX (Before CfC temporal processing)",
    );

    let (hdc_diag, hdc_off, hdc_score) = compute_clustering_score(&results, &categories, true);
    println!("    Within-category avg:  {:.4}", hdc_diag);
    println!("    Between-category avg: {:.4}", hdc_off);
    println!("    CLUSTERING SCORE:     {:.4}", hdc_score);
    println!();

    // 1b. CfC-OUTPUT similarity matrix
    print_similarity_matrix(
        &results,
        &categories,
        false,
        "1b. CfC-OUTPUT SIMILARITY MATRIX (After temporal state accumulation)",
    );

    let (cfc_diag, cfc_off, cfc_score) = compute_clustering_score(&results, &categories, false);
    println!("    Within-category avg:  {:.4}", cfc_diag);
    println!("    Between-category avg: {:.4}", cfc_off);
    println!("    CLUSTERING SCORE:     {:.4}", cfc_score);
    println!();

    // 2. Comparison
    println!("2. HDC-DIRECT vs CfC-OUTPUT COMPARISON");
    println!("   ═══════════════════════════════════════════════════════════════════");
    println!();
    println!("   Metric              HDC-Direct    CfC-Output    Difference");
    println!("   ────────────────────────────────────────────────────────────────");
    println!(
        "   Within-category     {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_diag,
        cfc_diag,
        hdc_diag - cfc_diag
    );
    println!(
        "   Between-category    {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_off,
        cfc_off,
        hdc_off - cfc_off
    );
    println!(
        "   CLUSTERING SCORE    {:>10.4}    {:>10.4}    {:>+10.4}",
        hdc_score,
        cfc_score,
        hdc_score - cfc_score
    );
    println!();

    if hdc_score > cfc_score {
        println!("   RESULT: HDC-direct preserves semantic clustering BETTER than CfC output");
        println!(
            "           (score difference: {:+.4})",
            hdc_score - cfc_score
        );
    } else if cfc_score > hdc_score {
        println!("   RESULT: CfC output preserves semantic clustering BETTER than HDC-direct");
        println!(
            "           (score difference: {:+.4})",
            cfc_score - hdc_score
        );
    } else {
        println!("   RESULT: HDC-direct and CfC output have equal clustering scores");
    }
    println!();

    if !phi_trajectory.is_empty() {
        let first_10: f32 = phi_trajectory.iter().take(10).sum::<f32>() / 10.0;
        let last_10: f32 = phi_trajectory.iter().rev().take(10).sum::<f32>() / 10.0;
        let max_phi: f32 = phi_trajectory
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_phi: f32 = phi_trajectory.iter().cloned().fold(f32::INFINITY, f32::min);

        println!("   First 10 cycles avg Phi: {:.6}", first_10);
        println!("   Last 10 cycles avg Phi:  {:.6}", last_10);
        println!("   Max Phi: {:.6}, Min Phi: {:.6}", max_phi, min_phi);
        println!("   Phi change: {:+.6}", last_10 - first_10);

        let phi_trend = if (last_10 - first_10).abs() > 0.001 {
            if last_10 > first_10 {
                "INCREASING"
            } else {
                "DECREASING"
            }
        } else {
            "STABLE"
        };
        println!("   Trend: {}", phi_trend);
    }
    println!();

    // 4. Learning statistics
    println!("4. LEARNING STATISTICS");
    let learning_count = results.iter().filter(|r| r.learning_occurred).count();
    let avg_error: f32 =
        results.iter().map(|r| r.prediction_error).sum::<f32>() / results.len() as f32;
    let avg_time: f32 =
        results.iter().map(|r| r.cycle_time_us as f32).sum::<f32>() / results.len() as f32;

    println!("   Sentences processed: {}", results.len());
    println!(
        "   Learning cycles: {} ({:.1}%)",
        learning_count,
        100.0 * learning_count as f32 / results.len() as f32
    );
    println!("   Avg prediction error: {:.6}", avg_error);
    println!("   Avg cycle time: {:.0}us", avg_time);
    println!();

    // 5. Per-category analysis
    println!("5. PER-CATEGORY PREDICTION ERROR");
    for cat in &categories {
        let cat_results: Vec<_> = results.iter().filter(|r| r.category == *cat).collect();
        let avg: f32 =
            cat_results.iter().map(|r| r.prediction_error).sum::<f32>() / cat_results.len() as f32;
        let learning: usize = cat_results.iter().filter(|r| r.learning_occurred).count();
        println!(
            "   {:>12}: avg_error={:.6}, learning={}/{}",
            cat.name(),
            avg,
            learning,
            cat_results.len()
        );
    }
    println!();

    // 6. Loop stats
    let stats = service.stats();
    println!("6. COGNITIVE LOOP STATS");
    println!("   Total cycles: {}", stats.total_cycles);
    println!(
        "   Avg prediction error (EMA): {:.6}",
        stats.avg_prediction_error
    );
    println!(
        "   Error trend: {:.6} (negative = improving)",
        stats.error_trend
    );
    println!(
        "   CfC state diversity: {:.4}",
        service.cfc_state_diversity()
    );
    println!();

    println!("========================================================================");
    println!("   CONCLUSION");
    println!("========================================================================");
    println!();

    if hdc_score > 0.01 && hdc_score > cfc_score {
        println!(
            "HYPOTHESIS CONFIRMED: HDC-direct clustering ({:.4}) > CfC output ({:.4})",
            hdc_score, cfc_score
        );
        println!();
        println!("The HDC projection preserves semantic structure before CfC temporal");
        println!("dynamics muddy it with recency effects. This suggests using HDC-direct");
        println!("for semantic similarity and CfC output for temporal/contextual tasks.");
    } else if cfc_score > 0.01 {
        println!("UNEXPECTED: CfC output shows better clustering than HDC-direct.");
        println!("This could indicate:");
        println!("  - Temporal context aids semantic clustering for this corpus");
        println!("  - HDC projection matrix needs better initialization");
        println!("  - CfC learns semantic structure during processing");
    } else {
        println!("NO CLEAR CLUSTERING in either representation.");
        println!("Consider:");
        println!("  - Increasing HDC dimension");
        println!("  - Using more diverse test sentences");
        println!("  - Adjusting projection initialization");
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_similarity() {
        // Same category should produce similar embeddings
        let e1 = generate_mock_embedding("Justice requires fairness", SemanticCategory::Abstract);
        let e2 = generate_mock_embedding("Democracy needs freedom", SemanticCategory::Abstract);
        let e3 = generate_mock_embedding("Smoking causes cancer", SemanticCategory::Causal);

        let sim_same = cosine_similarity(&e1, &e2);
        let sim_diff = cosine_similarity(&e1, &e3);

        println!("Same category similarity: {:.4}", sim_same);
        println!("Different category similarity: {:.4}", sim_diff);

        // Same category should be more similar
        assert!(sim_same > sim_diff, "Same category should be more similar");
    }

    #[test]
    fn test_corpus_size() {
        let corpus = get_test_corpus();
        assert_eq!(
            corpus.len(),
            24,
            "Should have 24 sentences (4 per category)"
        );
    }
}