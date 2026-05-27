// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding Associative Memory Experiment
//!
//! Tests whether HDC binding helps distinguish associated concepts.
//!
//! Hypothesis: bind(red, apple) is more similar to bind(red, apple) than to
//! bind(green, apple), enabling associative recall.
//!
//! Key question: Can binding solve "which apple is red?"
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example binding_associative_memory --features neural-bridge --release
//! ```

use std::time::Instant;

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::bge_m3::BgeM3;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example binding_associative_memory --features neural-bridge"
        );
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   BINDING ASSOCIATIVE MEMORY EXPERIMENT");
    println!("   Can binding distinguish 'red apple' from 'green apple'?");
    println!("================================================================\n");

    // Test scenarios: Object + Attribute combinations
    let test_scenarios = vec![
        // (object, correct_attribute, distractor_attributes)
        ("apple", "red", vec!["green", "yellow", "blue"]),
        ("apple", "green", vec!["red", "yellow", "blue"]),
        ("sky", "blue", vec!["red", "green", "gray"]),
        ("grass", "green", vec!["brown", "yellow", "blue"]),
        ("sun", "bright", vec!["dim", "dark", "faint"]),
        ("fire", "hot", vec!["cold", "warm", "cool"]),
        ("ice", "cold", vec!["hot", "warm", "lukewarm"]),
        ("night", "dark", vec!["bright", "light", "sunny"]),
        ("ocean", "deep", vec!["shallow", "thin", "narrow"]),
        ("mountain", "tall", vec!["short", "small", "low"]),
        ("whisper", "quiet", vec!["loud", "noisy", "booming"]),
        ("thunder", "loud", vec!["quiet", "soft", "silent"]),
        ("honey", "sweet", vec!["bitter", "sour", "salty"]),
        ("lemon", "sour", vec!["sweet", "bitter", "bland"]),
        ("stone", "hard", vec!["soft", "squishy", "fluffy"]),
        ("pillow", "soft", vec!["hard", "rigid", "firm"]),
        ("cheetah", "fast", vec!["slow", "sluggish", "lazy"]),
        ("snail", "slow", vec!["fast", "quick", "rapid"]),
        ("elephant", "large", vec!["small", "tiny", "miniature"]),
        ("ant", "small", vec!["large", "huge", "massive"]),
    ];

    // Load BGE-M3 encoder
    println!("Loading BGE-M3 encoder...");
    let load_start = Instant::now();
    let encoder = BgeM3::load_from_hub("BAAI/bge-m3")?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    println!("================================================================");
    println!("   TEST 1: ASSOCIATIVE RETRIEVAL");
    println!("   Given bind(object, ?), which attribute matches best?");
    println!("================================================================\n");

    let mut bind_correct = 0;
    let mut bundle_correct = 0;
    let mut total = 0;

    let mut bind_margins = Vec::new();
    let mut bundle_margins = Vec::new();

    for (object, correct_attr, distractors) in &test_scenarios {
        let obj_emb = encoder.encode(object)?;
        let correct_emb = encoder.encode(correct_attr)?;

        let obj_hv = embedding_to_hv16(&obj_emb);
        let correct_hv = embedding_to_hv16(&correct_emb);

        // Create the correct binding
        let correct_bind = obj_hv.bind(&correct_hv);
        let correct_bundle = BinaryHV::bundle(&[obj_hv, correct_hv]);

        // Test retrieval: which attribute is most similar when bound to object?
        let mut bind_sims = Vec::new();
        let mut bundle_sims = Vec::new();

        // Add correct attribute
        bind_sims.push((
            correct_attr.to_string(),
            hv_similarity(&correct_bind, &correct_bind),
        ));
        bundle_sims.push((
            correct_attr.to_string(),
            hv_similarity(&correct_bundle, &correct_bundle),
        ));

        // Add distractors
        for distractor in distractors {
            let dist_emb = encoder.encode(distractor)?;
            let dist_hv = embedding_to_hv16(&dist_emb);

            let dist_bind = obj_hv.bind(&dist_hv);
            let dist_bundle = BinaryHV::bundle(&[obj_hv, dist_hv]);

            // Similarity to the QUERY (correct binding/bundle)
            bind_sims.push((
                distractor.to_string(),
                hv_similarity(&correct_bind, &dist_bind),
            ));
            bundle_sims.push((
                distractor.to_string(),
                hv_similarity(&correct_bundle, &dist_bundle),
            ));
        }

        // Find best match for binding
        let bind_best = bind_sims
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let bundle_best = bundle_sims
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        // Compute margin (difference between correct and best distractor)
        let bind_correct_sim = bind_sims[0].1;
        let bind_best_distractor = bind_sims[1..]
            .iter()
            .map(|x| x.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let bind_margin = bind_correct_sim - bind_best_distractor;
        bind_margins.push(bind_margin);

        let bundle_correct_sim = bundle_sims[0].1;
        let bundle_best_distractor = bundle_sims[1..]
            .iter()
            .map(|x| x.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let bundle_margin = bundle_correct_sim - bundle_best_distractor;
        bundle_margins.push(bundle_margin);

        if bind_best.0 == *correct_attr {
            bind_correct += 1;
        }
        if bundle_best.0 == *correct_attr {
            bundle_correct += 1;
        }
        total += 1;
    }

    println!("Retrieval Accuracy:");
    println!(
        "  Binding: {}/{} ({:.1}%)",
        bind_correct,
        total,
        100.0 * bind_correct as f64 / total as f64
    );
    println!(
        "  Bundling: {}/{} ({:.1}%)\n",
        bundle_correct,
        total,
        100.0 * bundle_correct as f64 / total as f64
    );

    println!("Margin (correct - best distractor):");
    println!(
        "  Binding: {:.4} (std: {:.4})",
        mean(&bind_margins),
        std_dev(&bind_margins)
    );
    println!(
        "  Bundling: {:.4} (std: {:.4})\n",
        mean(&bundle_margins),
        std_dev(&bundle_margins)
    );

    println!("================================================================");
    println!("   TEST 2: DISAMBIGUATION");
    println!("   Given 'red' query, distinguish bind(red,apple) from bind(green,apple)");
    println!("================================================================\n");

    // Test: Can we retrieve the correct object-attribute pair given just the attribute?
    let disambiguation_tests = vec![
        // (query_attribute, object, paired_attribute, competing_attribute)
        ("red", "apple", "red", "green"),
        ("green", "apple", "green", "red"),
        ("blue", "sky", "blue", "gray"),
        ("hot", "fire", "hot", "cold"),
        ("cold", "ice", "cold", "hot"),
        ("loud", "thunder", "loud", "quiet"),
        ("quiet", "whisper", "quiet", "loud"),
        ("sweet", "honey", "sweet", "bitter"),
        ("fast", "cheetah", "fast", "slow"),
        ("slow", "snail", "slow", "fast"),
    ];

    let mut bind_disambig_correct = 0;
    let mut bundle_disambig_correct = 0;
    let mut disambig_total = 0;

    println!("Query     | Object  | Correct | Competing | Bind→ | Bundle→");
    println!("----------|---------|---------|-----------|-------|--------");

    for (query, object, correct, competing) in &disambiguation_tests {
        let query_emb = encoder.encode(query)?;
        let obj_emb = encoder.encode(object)?;
        let correct_emb = encoder.encode(correct)?;
        let competing_emb = encoder.encode(competing)?;

        let query_hv = embedding_to_hv16(&query_emb);
        let obj_hv = embedding_to_hv16(&obj_emb);
        let correct_hv = embedding_to_hv16(&correct_emb);
        let competing_hv = embedding_to_hv16(&competing_emb);

        // Create bindings: object+correct vs object+competing
        let correct_bind = obj_hv.bind(&correct_hv);
        let competing_bind = obj_hv.bind(&competing_hv);

        let correct_bundle = BinaryHV::bundle(&[obj_hv, correct_hv]);
        let competing_bundle = BinaryHV::bundle(&[obj_hv, competing_hv]);

        // For binding: unbind query from each to recover object, see which is closer
        // bind(obj, attr) ⊗ attr = obj (approximately, due to self-inverse property)
        let recovered_from_correct = correct_bind.bind(&query_hv);
        let recovered_from_competing = competing_bind.bind(&query_hv);

        let bind_sim_correct = hv_similarity(&recovered_from_correct, &obj_hv);
        let bind_sim_competing = hv_similarity(&recovered_from_competing, &obj_hv);

        // For bundling: just measure similarity to query
        let bundle_sim_correct = hv_similarity(&correct_bundle, &query_hv);
        let bundle_sim_competing = hv_similarity(&competing_bundle, &query_hv);

        let bind_winner = if bind_sim_correct > bind_sim_competing {
            "✓"
        } else {
            "✗"
        };
        let bundle_winner = if bundle_sim_correct > bundle_sim_competing {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:9} | {:7} | {:7} | {:9} | {:5} | {:6}",
            query, object, correct, competing, bind_winner, bundle_winner
        );

        if bind_sim_correct > bind_sim_competing {
            bind_disambig_correct += 1;
        }
        if bundle_sim_correct > bundle_sim_competing {
            bundle_disambig_correct += 1;
        }
        disambig_total += 1;
    }

    println!("\nDisambiguation Accuracy:");
    println!(
        "  Binding: {}/{} ({:.1}%)",
        bind_disambig_correct,
        disambig_total,
        100.0 * bind_disambig_correct as f64 / disambig_total as f64
    );
    println!(
        "  Bundling: {}/{} ({:.1}%)\n",
        bundle_disambig_correct,
        disambig_total,
        100.0 * bundle_disambig_correct as f64 / disambig_total as f64
    );

    println!("================================================================");
    println!("   TEST 3: ANALOGY COMPLETION");
    println!("   red:apple :: green:?  (using binding algebra)");
    println!("================================================================\n");

    let analogies = vec![
        // (a, b, c, expected_d) where a:b :: c:d
        ("red", "apple", "yellow", "banana"),
        ("hot", "fire", "cold", "ice"),
        ("fast", "cheetah", "slow", "snail"),
        ("large", "elephant", "small", "ant"),
        ("sweet", "honey", "sour", "lemon"),
        ("bright", "sun", "dark", "night"),
        ("loud", "thunder", "quiet", "whisper"),
        ("hard", "stone", "soft", "pillow"),
        ("deep", "ocean", "tall", "mountain"),
        ("blue", "sky", "green", "grass"),
    ];

    let candidates = vec![
        "banana", "ice", "snail", "ant", "lemon", "night", "whisper", "pillow", "mountain",
        "grass", "apple", "fire", "cheetah", "elephant", "honey", "sun", "thunder", "stone",
        "ocean", "sky",
    ];

    let mut bind_analogy_correct = 0;
    let mut bundle_analogy_correct = 0;
    let mut analogy_total = 0;

    println!("Analogy                    | Expected | Bind→    | Bundle→");
    println!("---------------------------|----------|----------|----------");

    for (a, b, c, expected_d) in &analogies {
        let a_emb = encoder.encode(a)?;
        let b_emb = encoder.encode(b)?;
        let c_emb = encoder.encode(c)?;

        let a_hv = embedding_to_hv16(&a_emb);
        let b_hv = embedding_to_hv16(&b_emb);
        let c_hv = embedding_to_hv16(&c_emb);

        // Binding analogy: d = bind(unbind(b, a), c) = bind(bind(b, a), c)
        // Since bind is self-inverse: a ⊗ a = identity (approximately)
        // So: a:b relationship = bind(a, b)
        // Apply to c: bind(bind(a, b), c) ≈ d if a:b :: c:d
        let relationship = a_hv.bind(&b_hv);
        let bind_predicted = relationship.bind(&c_hv);

        // Bundling analogy: average of relationship components
        // Not a good fit for analogies, but let's test
        let bundle_relationship = BinaryHV::bundle(&[a_hv, b_hv]);
        let bundle_predicted = BinaryHV::bundle(&[bundle_relationship, c_hv]);

        // Find best matching candidate
        let mut bind_best = ("", f64::NEG_INFINITY);
        let mut bundle_best = ("", f64::NEG_INFINITY);

        for candidate in &candidates {
            let cand_emb = encoder.encode(candidate)?;
            let cand_hv = embedding_to_hv16(&cand_emb);

            let bind_sim = hv_similarity(&bind_predicted, &cand_hv);
            let bundle_sim = hv_similarity(&bundle_predicted, &cand_hv);

            if bind_sim > bind_best.1 {
                bind_best = (candidate, bind_sim);
            }
            if bundle_sim > bundle_best.1 {
                bundle_best = (candidate, bundle_sim);
            }
        }

        let bind_mark = if bind_best.0 == *expected_d {
            "✓"
        } else {
            ""
        };
        let bundle_mark = if bundle_best.0 == *expected_d {
            "✓"
        } else {
            ""
        };

        println!(
            "{}:{} :: {}:?         | {:8} | {:8}{} | {:8}{}",
            a, b, c, expected_d, bind_best.0, bind_mark, bundle_best.0, bundle_mark
        );

        if bind_best.0 == *expected_d {
            bind_analogy_correct += 1;
        }
        if bundle_best.0 == *expected_d {
            bundle_analogy_correct += 1;
        }
        analogy_total += 1;
    }

    println!("\nAnalogy Completion Accuracy:");
    println!(
        "  Binding: {}/{} ({:.1}%)",
        bind_analogy_correct,
        analogy_total,
        100.0 * bind_analogy_correct as f64 / analogy_total as f64
    );
    println!(
        "  Bundling: {}/{} ({:.1}%)\n",
        bundle_analogy_correct,
        analogy_total,
        100.0 * bundle_analogy_correct as f64 / analogy_total as f64
    );

    println!("================================================================");
    println!("   SUMMARY");
    println!("================================================================\n");

    println!("                          | Binding | Bundling | Winner");
    println!("--------------------------|---------|----------|--------");
    println!(
        "Associative Retrieval     | {:5.1}%  | {:6.1}%  | {}",
        100.0 * bind_correct as f64 / total as f64,
        100.0 * bundle_correct as f64 / total as f64,
        if bind_correct > bundle_correct {
            "Binding"
        } else if bundle_correct > bind_correct {
            "Bundling"
        } else {
            "Tie"
        }
    );
    println!(
        "Disambiguation            | {:5.1}%  | {:6.1}%  | {}",
        100.0 * bind_disambig_correct as f64 / disambig_total as f64,
        100.0 * bundle_disambig_correct as f64 / disambig_total as f64,
        if bind_disambig_correct > bundle_disambig_correct {
            "Binding"
        } else if bundle_disambig_correct > bind_disambig_correct {
            "Bundling"
        } else {
            "Tie"
        }
    );
    println!(
        "Analogy Completion        | {:5.1}%  | {:6.1}%  | {}",
        100.0 * bind_analogy_correct as f64 / analogy_total as f64,
        100.0 * bundle_analogy_correct as f64 / analogy_total as f64,
        if bind_analogy_correct > bundle_analogy_correct {
            "Binding"
        } else if bundle_analogy_correct > bind_analogy_correct {
            "Bundling"
        } else {
            "Tie"
        }
    );

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn embedding_to_hv16(embedding: &[f32]) -> BinaryHV {
    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / embedding.len();

    for tile in 0..tiles {
        for (i, &val) in embedding.iter().enumerate() {
            let perturbation = ((tile * embedding.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    BinaryHV::from_bipolar(&expanded)
}

#[cfg(feature = "neural-bridge")]
fn hv_similarity(a: &BinaryHV, b: &BinaryHV) -> f64 {
    let hamming = a.hamming_distance(b);
    1.0 - (hamming as f64 / HDC_DIMENSION as f64)
}

#[cfg(feature = "neural-bridge")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}
