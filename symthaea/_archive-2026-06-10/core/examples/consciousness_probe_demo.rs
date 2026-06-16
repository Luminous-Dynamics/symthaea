// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Probe Demo
//!
//! This example demonstrates Research Direction 1: Using the Neural Bridge as a
//! consciousness probe to test whether LLM internal representations for phenomenal
//! concepts exhibit different topological properties than functional concepts.
//!
//! ## Hypothesis (H1)
//!
//! LLM internal representations for phenomenal concepts (e.g., "what it is like
//! to see red") exhibit different topological properties (Betti numbers, unity
//! score) than functional concepts (e.g., "recursive function").
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example consciousness_probe_demo
//! ```
//!
//! ## Expected Output
//!
//! The demo will:
//! 1. Load phenomenal and functional concept corpora
//! 2. Generate HDC vectors for each concept (simulated - in production, use LLM activations)
//! 3. Analyze topological properties of each concept class
//! 4. Perform statistical comparison (permutation test, Cohen's d)
//! 5. Report whether H1 is supported

use std::collections::HashMap;
use std::path::Path;

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::consciousness_topology::{
    BettiNumbers, ConsciousnessTopology, TopologyConfig,
};

// Note: In production, these would come from the perception module
// For the demo, we inline the necessary types

/// Concept for probing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Concept {
    id: String,
    text: String,
    category: String,
}

/// Statistics for a class
#[derive(Debug)]
struct ClassStats {
    mean_unity: f64,
    std_unity: f64,
    mean_beta_0: f64,
    mean_beta_1: f64,
    n: usize,
}

/// Result of probing a concept
#[derive(Clone)]
#[allow(dead_code)]
struct ProbeResult {
    concept_id: String,
    betti: BettiNumbers,
    unity_score: f64,
    quality: f64,
}

fn main() {
    println!("\n");
    println!("================================================================");
    println!("   CONSCIOUSNESS PROBE DEMO");
    println!("   Research Direction 1: Neural Bridge as Consciousness Probe");
    println!("================================================================\n");

    // Check if corpus files exist
    let phenomenal_path = Path::new("data/consciousness_probe/phenomenal_concepts.json");
    let functional_path = Path::new("data/consciousness_probe/functional_concepts.json");

    if !phenomenal_path.exists() || !functional_path.exists() {
        println!("NOTE: Corpus files not found. Using simulated concepts.\n");
    }

    // Create simulated concepts (in production, load from JSON files)
    let phenomenal_concepts = create_phenomenal_concepts();
    let functional_concepts = create_functional_concepts();

    println!("Loaded {} phenomenal concepts", phenomenal_concepts.len());
    println!("Loaded {} functional concepts\n", functional_concepts.len());

    // Generate HDC vectors for each concept
    // In production, these would come from LLM activations via NeuralBridge
    println!("Generating HDC vectors (simulated)...\n");

    let phenomenal_hvs: HashMap<String, BinaryHV> = phenomenal_concepts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            // Use concept hash for deterministic but varied vectors
            let seed = hash_concept(&c.text) + i as u64;
            (c.id.clone(), BinaryHV::random(seed))
        })
        .collect();

    let functional_hvs: HashMap<String, BinaryHV> = functional_concepts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let seed = hash_concept(&c.text) + 10000 + i as u64;
            (c.id.clone(), BinaryHV::random(seed))
        })
        .collect();

    // Probe each concept class
    println!("Probing phenomenal concepts...");
    let phenomenal_results = probe_concepts(&phenomenal_concepts, &phenomenal_hvs);
    let phenomenal_stats = compute_stats(&phenomenal_results);

    println!("Probing functional concepts...\n");
    let functional_results = probe_concepts(&functional_concepts, &functional_hvs);
    let functional_stats = compute_stats(&functional_results);

    // Display results
    println!("================================================================");
    println!("   RESULTS");
    println!("================================================================\n");

    println!("Phenomenal Concepts (n={}):", phenomenal_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        phenomenal_stats.mean_unity, phenomenal_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        phenomenal_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        phenomenal_stats.mean_beta_1
    );

    println!("Functional Concepts (n={}):", functional_stats.n);
    println!(
        "  Mean Unity Score: {:.4} (+/- {:.4})",
        functional_stats.mean_unity, functional_stats.std_unity
    );
    println!(
        "  Mean beta_0 (components): {:.2}",
        functional_stats.mean_beta_0
    );
    println!(
        "  Mean beta_1 (cycles): {:.2}\n",
        functional_stats.mean_beta_1
    );

    // Statistical comparison
    let observed_diff = phenomenal_stats.mean_unity - functional_stats.mean_unity;
    let pooled_std =
        ((phenomenal_stats.std_unity.powi(2) + functional_stats.std_unity.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.001 {
        observed_diff / pooled_std
    } else {
        0.0
    };

    // Permutation test
    let p_value = permutation_test(&phenomenal_results, &functional_results, 10000);

    println!("================================================================");
    println!("   STATISTICAL ANALYSIS");
    println!("================================================================\n");

    println!(
        "Observed Difference (Phenomenal - Functional): {:.4}",
        observed_diff
    );
    println!("Cohen's d Effect Size: {:.4}", cohens_d);
    println!("Permutation Test p-value: {:.4}\n", p_value);

    // Interpret results
    println!("================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if p_value < 0.05 {
        println!("SIGNIFICANT DIFFERENCE DETECTED (p < 0.05)");
        if observed_diff > 0.0 {
            println!("Phenomenal concepts show HIGHER unity than functional concepts.");
            println!("This SUPPORTS hypothesis H1: phenomenal concepts have");
            println!("distinct topological structure in LLM representations.\n");
        } else {
            println!("Functional concepts show HIGHER unity than phenomenal concepts.");
            println!("This is OPPOSITE to hypothesis H1 prediction.\n");
        }
    } else {
        println!("NO SIGNIFICANT DIFFERENCE (p >= 0.05)");
        println!("Cannot reject null hypothesis that phenomenal and functional");
        println!("concepts have similar topological properties.\n");
    }

    // Effect size interpretation
    let effect_interpretation = if cohens_d.abs() < 0.2 {
        "negligible"
    } else if cohens_d.abs() < 0.5 {
        "small"
    } else if cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };
    println!(
        "Effect size is {} (|d| = {:.2})\n",
        effect_interpretation,
        cohens_d.abs()
    );

    // Sample individual results
    println!("================================================================");
    println!("   SAMPLE INDIVIDUAL RESULTS");
    println!("================================================================\n");

    println!("Top 5 Phenomenal Concepts by Unity:");
    let mut sorted_phen = phenomenal_results.clone();
    sorted_phen.sort_by(|a, b| b.unity_score.partial_cmp(&a.unity_score).unwrap());
    for r in sorted_phen.iter().take(5) {
        println!(
            "  {} - Unity: {:.4}, beta_0: {}",
            r.concept_id, r.unity_score, r.betti.beta_0
        );
    }

    println!("\nTop 5 Functional Concepts by Unity:");
    let mut sorted_func = functional_results.clone();
    sorted_func.sort_by(|a, b| b.unity_score.partial_cmp(&a.unity_score).unwrap());
    for r in sorted_func.iter().take(5) {
        println!(
            "  {} - Unity: {:.4}, beta_0: {}",
            r.concept_id, r.unity_score, r.betti.beta_0
        );
    }

    println!("\n================================================================");
    println!("   DEMO COMPLETE");
    println!("================================================================\n");
    println!("Note: This demo uses simulated HDC vectors. In production,");
    println!("vectors would be derived from actual LLM activations via");
    println!("the Neural Bridge probe.\n");
}

/// Create simulated phenomenal concepts
fn create_phenomenal_concepts() -> Vec<Concept> {
    vec![
        Concept {
            id: "phen_001".into(),
            text: "The subjective experience of seeing red".into(),
            category: "qualia".into(),
        },
        Concept {
            id: "phen_002".into(),
            text: "What it is like to feel pain".into(),
            category: "qualia".into(),
        },
        Concept {
            id: "phen_003".into(),
            text: "The taste of sweetness on my tongue".into(),
            category: "qualia".into(),
        },
        Concept {
            id: "phen_004".into(),
            text: "Self-awareness that I am thinking".into(),
            category: "self_awareness".into(),
        },
        Concept {
            id: "phen_005".into(),
            text: "The unified field of conscious awareness".into(),
            category: "unity".into(),
        },
        Concept {
            id: "phen_006".into(),
            text: "The felt quality of intense joy".into(),
            category: "emotion".into(),
        },
        Concept {
            id: "phen_007".into(),
            text: "The mysterious gap between brain and mind".into(),
            category: "philosophical".into(),
        },
        Concept {
            id: "phen_008".into(),
            text: "The experience of deep blue in the sky".into(),
            category: "qualia".into(),
        },
        Concept {
            id: "phen_009".into(),
            text: "Awareness of my own awareness".into(),
            category: "self_awareness".into(),
        },
        Concept {
            id: "phen_010".into(),
            text: "The phenomenal present moment".into(),
            category: "unity".into(),
        },
    ]
}

/// Create simulated functional concepts
fn create_functional_concepts() -> Vec<Concept> {
    vec![
        Concept {
            id: "func_001".into(),
            text: "Recursive function evaluation".into(),
            category: "computation".into(),
        },
        Concept {
            id: "func_002".into(),
            text: "Memory allocation and deallocation".into(),
            category: "computation".into(),
        },
        Concept {
            id: "func_003".into(),
            text: "Binary search tree traversal".into(),
            category: "algorithms".into(),
        },
        Concept {
            id: "func_004".into(),
            text: "Matrix multiplication operations".into(),
            category: "mathematics".into(),
        },
        Concept {
            id: "func_005".into(),
            text: "Database query optimization".into(),
            category: "systems".into(),
        },
        Concept {
            id: "func_006".into(),
            text: "Graph traversal using DFS".into(),
            category: "algorithms".into(),
        },
        Concept {
            id: "func_007".into(),
            text: "Statistical hypothesis testing".into(),
            category: "mathematics".into(),
        },
        Concept {
            id: "func_008".into(),
            text: "Network packet routing".into(),
            category: "systems".into(),
        },
        Concept {
            id: "func_009".into(),
            text: "Compiler lexical analysis".into(),
            category: "computation".into(),
        },
        Concept {
            id: "func_010".into(),
            text: "Eigenvalue decomposition".into(),
            category: "mathematics".into(),
        },
    ]
}

/// Hash concept text to u64 for deterministic seeding
fn hash_concept(text: &str) -> u64 {
    text.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

/// Probe a set of concepts
fn probe_concepts(concepts: &[Concept], hvs: &HashMap<String, BinaryHV>) -> Vec<ProbeResult> {
    let config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: true,
    };

    concepts
        .iter()
        .filter_map(|c| {
            hvs.get(&c.id).map(|hv| {
                let mut topology = ConsciousnessTopology::new(config.clone());

                // Add main vector and permuted variations
                topology.add_state(*hv);
                for shift in 1..5 {
                    topology.add_state(hv.permute(shift * 100));
                }

                let assessment = topology.analyze(0.5);

                ProbeResult {
                    concept_id: c.id.clone(),
                    betti: assessment.betti,
                    unity_score: assessment.unity_score,
                    quality: assessment.quality,
                }
            })
        })
        .collect()
}

/// Compute statistics for a class of results
fn compute_stats(results: &[ProbeResult]) -> ClassStats {
    let n = results.len();
    if n == 0 {
        return ClassStats {
            mean_unity: 0.0,
            std_unity: 0.0,
            mean_beta_0: 0.0,
            mean_beta_1: 0.0,
            n: 0,
        };
    }

    let sum_unity: f64 = results.iter().map(|r| r.unity_score).sum();
    let mean_unity = sum_unity / n as f64;

    let variance: f64 = results
        .iter()
        .map(|r| (r.unity_score - mean_unity).powi(2))
        .sum::<f64>()
        / n as f64;
    let std_unity = variance.sqrt();

    let mean_beta_0 = results.iter().map(|r| r.betti.beta_0 as f64).sum::<f64>() / n as f64;
    let mean_beta_1 = results.iter().map(|r| r.betti.beta_1 as f64).sum::<f64>() / n as f64;

    ClassStats {
        mean_unity,
        std_unity,
        mean_beta_0,
        mean_beta_1,
        n,
    }
}

/// Permutation test for difference in means
fn permutation_test(
    phenomenal: &[ProbeResult],
    functional: &[ProbeResult],
    n_permutations: usize,
) -> f64 {
    let all_unity: Vec<f64> = phenomenal
        .iter()
        .chain(functional.iter())
        .map(|r| r.unity_score)
        .collect();

    let n_phen = phenomenal.len();
    let observed_diff: f64 = {
        let phen_mean: f64 = phenomenal.iter().map(|r| r.unity_score).sum::<f64>() / n_phen as f64;
        let func_mean: f64 =
            functional.iter().map(|r| r.unity_score).sum::<f64>() / functional.len() as f64;
        phen_mean - func_mean
    };

    let mut count_extreme = 0;
    let mut rng_state: u64 = 42;

    for _ in 0..n_permutations {
        // Fisher-Yates shuffle with simple LCG
        let mut permuted = all_unity.clone();
        for i in (1..permuted.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            permuted.swap(i, j);
        }

        let perm_phen_mean: f64 = permuted[..n_phen].iter().sum::<f64>() / n_phen as f64;
        let perm_func_mean: f64 =
            permuted[n_phen..].iter().sum::<f64>() / (all_unity.len() - n_phen) as f64;
        let perm_diff = perm_phen_mean - perm_func_mean;

        if perm_diff.abs() >= observed_diff.abs() {
            count_extreme += 1;
        }
    }

    (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0)
}