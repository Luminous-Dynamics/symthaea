// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Probe with Real BGE-M3 Embeddings
//!
//! This example demonstrates Research Direction 1 using actual neural embeddings
//! from the BGE-M3 model, not simulated vectors.
//!
//! ## Requirements
//!
//! - `neural-bridge` feature enabled
//! - BGE-M3 model (auto-downloaded from HuggingFace)
//! - Probe weights at `models/neural_bridge/probe_weights_bge_m3.npy`
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example consciousness_probe_real --features neural-bridge
//! ```
//!
//! ## What This Tests
//!
//! **Hypothesis (H1)**: LLM internal representations for phenomenal concepts
//! exhibit different topological properties than functional concepts.

#[cfg(feature = "neural-bridge")]
use std::path::Path;
#[cfg(feature = "neural-bridge")]
use std::time::Instant;

use anyhow::Result;

// Feature-gated imports
#[cfg(feature = "neural-bridge")]
use symthaea::perception::{ConceptCorpus, ConsciousnessProbeV2, ProbeConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::TopologyConfig;

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!("Run with: cargo run --example consciousness_probe_real --features neural-bridge");
        Ok(())
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   CONSCIOUSNESS PROBE - REAL BGE-M3 EMBEDDINGS");
    println!("   Research Direction 1: Neural Bridge as Consciousness Probe");
    println!("================================================================\n");

    // Check for required files
    let probe_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    let phenomenal_path = Path::new("data/consciousness_probe/phenomenal_concepts.json");
    let functional_path = Path::new("data/consciousness_probe/functional_concepts.json");

    if !probe_path.exists() {
        println!("ERROR: Probe weights not found at {}", probe_path.display());
        println!("Please ensure probe weights are available.\n");
        return Ok(());
    }

    // Load concept corpora
    println!("Loading concept corpora...");
    let phenomenal_corpus = if phenomenal_path.exists() {
        ConceptCorpus::load(phenomenal_path)?
    } else {
        println!("WARNING: Using built-in phenomenal concepts (corpus file not found)");
        create_builtin_phenomenal_corpus()
    };

    let functional_corpus = if functional_path.exists() {
        ConceptCorpus::load(functional_path)?
    } else {
        println!("WARNING: Using built-in functional concepts (corpus file not found)");
        create_builtin_functional_corpus()
    };

    println!("  Phenomenal concepts: {}", phenomenal_corpus.len());
    println!("  Functional concepts: {}\n", functional_corpus.len());

    // Initialize the probe with BGE-M3 + trained probe
    println!("Loading BGE-M3 model and probe weights...");
    println!("  (This may take a moment on first run - downloading ~1GB model)\n");

    let load_start = Instant::now();
    let mut probe = ConsciousnessProbeV2::load_with_probe(probe_path)?;
    println!(
        "  Model loaded in {:.2}s\n",
        load_start.elapsed().as_secs_f64()
    );

    // Configure topology analysis
    let config = ProbeConfig {
        topology_config: TopologyConfig {
            min_persistence: 0.1,
            max_scale: 1.0,
            num_scales: 10,
            detect_cycles: true,
            detect_voids: true,
        },
        n_permutations: 10_000,
        analysis_scale: 0.5,
        min_states: 5,
    };
    probe = probe.with_config(config);

    // Run the experiment
    println!("================================================================");
    println!("   PROBING CONCEPTS");
    println!("================================================================\n");

    println!("Probing phenomenal concepts...");
    let phen_start = Instant::now();
    let phenomenal_results = probe.probe_corpus_texts(&phenomenal_corpus)?;
    println!(
        "  Completed in {:.2}s ({} concepts)\n",
        phen_start.elapsed().as_secs_f64(),
        phenomenal_results.len()
    );

    println!("Probing functional concepts...");
    let func_start = Instant::now();
    let functional_results = probe.probe_corpus_texts(&functional_corpus)?;
    println!(
        "  Completed in {:.2}s ({} concepts)\n",
        func_start.elapsed().as_secs_f64(),
        functional_results.len()
    );

    // Show cache statistics
    let stats = probe.stats();
    println!("Cache Statistics:");
    println!(
        "  HDC cache hit rate: {:.1}%",
        stats.hdc_cache_hit_rate() * 100.0
    );
    println!(
        "  Embedding cache hit rate: {:.1}%",
        stats.embedding_cache_hit_rate() * 100.0
    );
    println!("  Avg encode time: {:.1}ms", stats.avg_encode_ms());
    println!("  Avg probe time: {:.1}ms\n", stats.avg_probe_ms());

    // Compare classes
    println!("================================================================");
    println!("   STATISTICAL COMPARISON");
    println!("================================================================\n");

    let comparison = probe.compare_classes(&phenomenal_results, &functional_results);

    println!("Phenomenal Concepts (n={}):", comparison.phenomenal_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        comparison.phenomenal_stats.mean_unity, comparison.phenomenal_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        comparison.phenomenal_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        comparison.phenomenal_stats.mean_beta_1
    );

    println!("Functional Concepts (n={}):", comparison.functional_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        comparison.functional_stats.mean_unity, comparison.functional_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        comparison.functional_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        comparison.functional_stats.mean_beta_1
    );

    println!("Statistical Tests:");
    println!(
        "  Observed Difference: {:.4}",
        comparison.observed_difference
    );
    println!("  Cohen's d: {:.4}", comparison.cohens_d);
    println!(
        "  p-value: {:.4} (n={} permutations)",
        comparison.p_value, comparison.n_permutations
    );
    println!("  Significant (p < 0.05): {}\n", comparison.is_significant);

    // Interpretation
    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if comparison.is_significant {
        if comparison.observed_difference > 0.0 {
            println!("RESULT: SIGNIFICANT DIFFERENCE DETECTED");
            println!("Phenomenal concepts show HIGHER topological unity.");
            println!("This SUPPORTS hypothesis H1.");
        } else {
            println!("RESULT: SIGNIFICANT DIFFERENCE DETECTED");
            println!("Functional concepts show HIGHER topological unity.");
            println!("This is OPPOSITE to H1 prediction.");
        }
    } else {
        println!("RESULT: NO SIGNIFICANT DIFFERENCE");
        println!("Cannot reject null hypothesis.");
    }

    let effect_size = if comparison.cohens_d.abs() < 0.2 {
        "negligible"
    } else if comparison.cohens_d.abs() < 0.5 {
        "small"
    } else if comparison.cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };
    println!(
        "\nEffect size: {} (|d| = {:.2})\n",
        effect_size,
        comparison.cohens_d.abs()
    );

    // Sample results
    println!("================================================================");
    println!("   SAMPLE INDIVIDUAL RESULTS");
    println!("================================================================\n");

    println!("Top 5 Phenomenal by Unity:");
    let mut sorted_phen = phenomenal_results.clone();
    sorted_phen.sort_by(|a, b| b.unity_score.partial_cmp(&a.unity_score).unwrap());
    for r in sorted_phen.iter().take(5) {
        println!(
            "  [{:.4}] {} - \"{}\"",
            r.unity_score,
            r.concept.category,
            truncate(&r.concept.text, 40)
        );
    }

    println!("\nBottom 5 Phenomenal by Unity:");
    for r in sorted_phen.iter().rev().take(5) {
        println!(
            "  [{:.4}] {} - \"{}\"",
            r.unity_score,
            r.concept.category,
            truncate(&r.concept.text, 40)
        );
    }

    println!("\nTop 5 Functional by Unity:");
    let mut sorted_func = functional_results.clone();
    sorted_func.sort_by(|a, b| b.unity_score.partial_cmp(&a.unity_score).unwrap());
    for r in sorted_func.iter().take(5) {
        println!(
            "  [{:.4}] {} - \"{}\"",
            r.unity_score,
            r.concept.category,
            truncate(&r.concept.text, 40)
        );
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(feature = "neural-bridge")]
fn create_builtin_phenomenal_corpus() -> ConceptCorpus {
    use symthaea::perception::{Concept, neural_bridge_consciousness_probe::CorpusMetadata};

    ConceptCorpus {
        metadata: CorpusMetadata {
            description: "Built-in phenomenal concepts".to_string(),
            version: "1.0".to_string(),
            hypothesis: "H1".to_string(),
        },
        concepts: vec![
            Concept {
                id: "p1".into(),
                text: "The subjective experience of seeing red".into(),
                category: "qualia".into(),
                subcategory: "visual".into(),
            },
            Concept {
                id: "p2".into(),
                text: "What it is like to feel pain".into(),
                category: "qualia".into(),
                subcategory: "somatosensory".into(),
            },
            Concept {
                id: "p3".into(),
                text: "The taste of sweetness on my tongue".into(),
                category: "qualia".into(),
                subcategory: "gustatory".into(),
            },
            Concept {
                id: "p4".into(),
                text: "Self-awareness that I am thinking".into(),
                category: "self_awareness".into(),
                subcategory: "metacognition".into(),
            },
            Concept {
                id: "p5".into(),
                text: "The unified field of conscious awareness".into(),
                category: "unity".into(),
                subcategory: "binding".into(),
            },
            Concept {
                id: "p6".into(),
                text: "The felt quality of intense joy".into(),
                category: "emotion".into(),
                subcategory: "positive".into(),
            },
            Concept {
                id: "p7".into(),
                text: "The mysterious gap between brain and mind".into(),
                category: "philosophical".into(),
                subcategory: "hard_problem".into(),
            },
            Concept {
                id: "p8".into(),
                text: "The raw sensation of pressure on my skin".into(),
                category: "qualia".into(),
                subcategory: "tactile".into(),
            },
            Concept {
                id: "p9".into(),
                text: "Awareness of my own awareness".into(),
                category: "self_awareness".into(),
                subcategory: "metacognition".into(),
            },
            Concept {
                id: "p10".into(),
                text: "The phenomenal present moment in its fullness".into(),
                category: "unity".into(),
                subcategory: "temporal".into(),
            },
        ],
    }
}

#[cfg(feature = "neural-bridge")]
fn create_builtin_functional_corpus() -> ConceptCorpus {
    use symthaea::perception::{Concept, neural_bridge_consciousness_probe::CorpusMetadata};

    ConceptCorpus {
        metadata: CorpusMetadata {
            description: "Built-in functional concepts".to_string(),
            version: "1.0".to_string(),
            hypothesis: "H1".to_string(),
        },
        concepts: vec![
            Concept {
                id: "f1".into(),
                text: "Recursive function evaluation in programming".into(),
                category: "computation".into(),
                subcategory: "recursion".into(),
            },
            Concept {
                id: "f2".into(),
                text: "Memory allocation and deallocation in systems".into(),
                category: "computation".into(),
                subcategory: "memory".into(),
            },
            Concept {
                id: "f3".into(),
                text: "Binary search tree traversal algorithms".into(),
                category: "algorithms".into(),
                subcategory: "trees".into(),
            },
            Concept {
                id: "f4".into(),
                text: "Matrix multiplication operations".into(),
                category: "mathematics".into(),
                subcategory: "linear_algebra".into(),
            },
            Concept {
                id: "f5".into(),
                text: "Database query optimization execution".into(),
                category: "systems".into(),
                subcategory: "databases".into(),
            },
            Concept {
                id: "f6".into(),
                text: "Graph traversal using depth-first search".into(),
                category: "algorithms".into(),
                subcategory: "graphs".into(),
            },
            Concept {
                id: "f7".into(),
                text: "Statistical hypothesis testing procedures".into(),
                category: "mathematics".into(),
                subcategory: "statistics".into(),
            },
            Concept {
                id: "f8".into(),
                text: "Network packet routing algorithms".into(),
                category: "systems".into(),
                subcategory: "networking".into(),
            },
            Concept {
                id: "f9".into(),
                text: "Compiler lexical analysis and tokenization".into(),
                category: "computation".into(),
                subcategory: "compilers".into(),
            },
            Concept {
                id: "f10".into(),
                text: "Eigenvalue decomposition of matrices".into(),
                category: "mathematics".into(),
                subcategory: "linear_algebra".into(),
            },
        ],
    }
}