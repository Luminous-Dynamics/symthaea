// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical Scorer Demo - Solving Bundling Saturation
//!
//! Demonstrates how hierarchical clustering solves the bundling saturation problem
//! when training on large vocabularies (CMU Dict has 135K words).

use std::path::Path;
use symthaea_stt::hierarchical_scorer::HierarchicalScorer;
use symthaea_stt::lexicon::CmuDictionary;

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn main() {
    separator('=', 70);
    println!("  HIERARCHICAL HDC SCORER");
    println!("  Solving Bundling Saturation via Linguistic Clustering");
    separator('=', 70);
    println!();

    // Load CMU dictionary
    println!("Loading CMU Dictionary...");
    let dict_path = Path::new("data/cmudict-0.7b");
    let dict = if dict_path.exists() {
        CmuDictionary::load(dict_path).unwrap_or_else(|_| {
            println!("Failed to load dictionary, using empty.");
            CmuDictionary::new()
        })
    } else {
        println!("Dictionary not found at {}", dict_path.display());
        println!("Using built-in word samples.");
        CmuDictionary::new()
    };

    let word_count = dict.len();
    println!("  Loaded {} words\n", word_count);

    // Create hierarchical scorer
    let mut scorer = HierarchicalScorer::new();

    // Train on CMU Dict patterns
    separator('-', 70);
    println!("  Phase 1: Training on English Phoneme Patterns");
    separator('-', 70);
    println!();

    // Sample words from different categories
    let stop_words = [
        "CAT", "DOG", "BIT", "TOP", "CUP", "BAD", "KID", "PIG", "TAP", "DIP",
    ];
    let fricative_words = ["SEE", "ZOO", "FISH", "THE", "SHOE", "VISION", "SAFE", "VAN"];
    let nasal_words = ["MAN", "NO", "SING", "NAME", "MOON", "NINE", "AMONG"];
    let vowel_words = ["EAT", "OAT", "ATE", "OUT", "OWL", "EYE", "AWE"];

    // Function to train on word category
    fn train_category(
        scorer: &mut HierarchicalScorer,
        dict: &CmuDictionary,
        words: &[&str],
        cat_name: &str,
    ) -> usize {
        let mut trained = 0;
        for word in words {
            if let Some(entry) = dict.get(&word.to_uppercase()) {
                let refs: Vec<&str> = entry.primary.iter().map(|s| s.as_str()).collect();
                for _ in 0..5 {
                    scorer.train_sequence(&refs);
                }
                trained += 1;
            }
        }
        println!("  Trained {} {} words", trained, cat_name);
        trained
    }

    // If dictionary is loaded, use it
    let total_trained = if word_count > 0 {
        let mut t = 0;
        t += train_category(&mut scorer, &dict, &stop_words, "stop-initial");
        t += train_category(&mut scorer, &dict, &fricative_words, "fricative-heavy");
        t += train_category(&mut scorer, &dict, &nasal_words, "nasal-heavy");
        t += train_category(&mut scorer, &dict, &vowel_words, "vowel-only");
        t
    } else {
        // Use hardcoded phoneme sequences if no dict
        let patterns: Vec<Vec<&str>> = vec![
            vec!["K", "AE", "T"], // cat
            vec!["D", "AO", "G"], // dog
            vec!["S", "IY"],      // see
            vec!["M", "AE", "N"], // man
            vec!["N", "OW"],      // no
            vec!["IY", "T"],      // eat
        ];

        for pattern in &patterns {
            for _ in 0..10 {
                scorer.train_sequence(pattern);
            }
        }
        println!("  Trained {} built-in patterns", patterns.len());
        patterns.len()
    };

    // Show stats
    let stats = scorer.stats();
    println!();
    println!("  Training Statistics:");
    println!("    Total patterns:           {}", stats.total_patterns);
    println!(
        "    Active Manner clusters:   {} / 7",
        stats.active_manner_clusters
    );
    println!(
        "    Active M×P clusters:      {} / 70",
        stats.active_mp_clusters
    );
    println!(
        "    Avg patterns/Manner:      {:.1}",
        stats.avg_patterns_per_manner
    );
    println!(
        "    Avg patterns/M×P:         {:.1}",
        stats.avg_patterns_per_mp
    );
    println!(
        "    Avg Manner memory density: {:.3}",
        stats.avg_manner_density
    );
    println!();

    // Test discrimination at different levels
    separator('-', 70);
    println!("  Phase 2: Discrimination Testing");
    separator('-', 70);
    println!();

    // Test patterns
    let trained_cvc = vec!["K", "AE", "T"]; // cat - trained
    let similar_cvc = vec!["G", "AH", "T"]; // gut - similar structure
    let different = vec!["S", "IY", "Z"]; // seize - different manner

    // Level 2 (Manner×Place - finest granularity)
    scorer.set_level(2);
    println!("  Level 2 (Manner×Place - 70 clusters):");
    let l2_trained = scorer.score_sequence(&trained_cvc);
    let l2_similar = scorer.score_sequence(&similar_cvc);
    let l2_different = scorer.score_sequence(&different);
    println!("    'CAT' (trained):  {:.4}", l2_trained);
    println!("    'GUT' (similar):  {:.4}", l2_similar);
    println!("    'SEIZE' (diff):   {:.4}", l2_different);
    println!(
        "    Discrimination (CAT vs SEIZE): {:+.4}",
        l2_trained - l2_different
    );
    println!();

    // Level 1 (Manner only - coarser)
    scorer.set_level(1);
    println!("  Level 1 (Manner - 7 clusters):");
    let l1_trained = scorer.score_sequence(&trained_cvc);
    let l1_similar = scorer.score_sequence(&similar_cvc);
    let l1_different = scorer.score_sequence(&different);
    println!("    'CAT' (trained):  {:.4}", l1_trained);
    println!("    'GUT' (similar):  {:.4}", l1_similar);
    println!("    'SEIZE' (diff):   {:.4}", l1_different);
    println!(
        "    Discrimination (CAT vs SEIZE): {:+.4}",
        l1_trained - l1_different
    );
    println!();

    // Cross-manner discrimination test
    separator('-', 70);
    println!("  Phase 3: Cross-Manner Isolation Test");
    separator('-', 70);
    println!();

    scorer.set_level(2);

    // Train heavily on stop patterns
    let stop_patterns: Vec<Vec<&str>> = vec![
        vec!["P", "IH", "T"], // pit
        vec!["B", "AE", "T"], // bat
        vec!["T", "AO", "P"], // top
        vec!["K", "AE", "P"], // cap
    ];

    for pattern in &stop_patterns {
        for _ in 0..50 {
            scorer.train_sequence(pattern);
        }
    }
    println!("  Heavy training on Stop-initial patterns (50 reps each)");
    println!();

    // Test: stop pattern vs fricative pattern
    let test_stop = vec!["D", "IH", "G"]; // dig (stop-vowel-stop)
    let test_fric = vec!["S", "AE", "D"]; // sad (fricative-vowel-stop)

    let stop_score = scorer.score_sequence(&test_stop);
    let fric_score = scorer.score_sequence(&test_fric);

    println!("  Test Results:");
    println!("    'DIG' (Stop-V-Stop):     {:.4}", stop_score);
    println!("    'SAD' (Fric-V-Stop):     {:.4}", fric_score);
    println!();
    println!("    Discrimination: {:+.4}", stop_score - fric_score);
    println!();

    if stop_score > fric_score + 0.01 {
        println!("  [SUCCESS] Hierarchical clustering isolates manner classes!");
        println!("  Stop patterns score higher because they route to trained clusters.");
    } else if stop_score > fric_score {
        println!("  [PARTIAL] Some manner isolation detected.");
    } else {
        println!("  [INFO] Scores similar - may need more training data.");
    }

    // Compare to saturation baseline
    separator('=', 70);
    println!("  SATURATION ANALYSIS");
    separator('=', 70);
    println!();

    let patterns_per_cluster = stats.avg_patterns_per_mp;
    let theoretical_single_memory = stats.total_patterns as f32;

    println!("  With Hierarchical Clustering:");
    println!("    Patterns per cluster:     {:.1}", patterns_per_cluster);
    println!(
        "    Memory density:           {:.3}",
        stats.avg_manner_density
    );
    println!();
    println!("  Without Clustering (single memory):");
    println!(
        "    All patterns bundled:     {:.0}",
        theoretical_single_memory
    );
    println!("    Expected density:         ~0.500 (saturated)");
    println!();

    if stats.avg_manner_density < 0.48 {
        println!("  [SUCCESS] Hierarchical bundling prevents saturation!");
        println!("  Each cluster stays well below the ~0.5 saturation threshold.");
    } else {
        println!("  [INFO] Memory density approaching saturation.");
        println!("  Consider adding more clusters (e.g., by first phoneme).");
    }

    separator('=', 70);
    println!("  DEMO COMPLETE");
    separator('=', 70);
}
