// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Scale Scorer Demo - Enhanced Discrimination
//!
//! Compares discrimination across scorer architectures:
//! - Original Holographic (1 memory)
//! - Hierarchical L2 (70 clusters)
//! - Multi-Scale L3 (140 clusters + voting)

use symthaea_stt::hierarchical_scorer::HierarchicalScorer;
use symthaea_stt::multiscale_scorer::MultiScaleScorer;

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn main() {
    separator('=', 70);
    println!("  MULTI-SCALE HDC SCORER");
    println!("  140 Clusters + Multi-Level Voting");
    separator('=', 70);
    println!();

    // Create both scorers
    let mut hierarchical = HierarchicalScorer::new();
    let mut multiscale = MultiScaleScorer::new();

    // Training patterns - diverse phonotactics
    let training_patterns: Vec<Vec<&str>> = vec![
        // Voiceless stops
        vec!["P", "AE", "T"], // pat
        vec!["K", "AE", "T"], // cat
        vec!["T", "AO", "P"], // top
        // Voiced stops
        vec!["B", "AE", "D"], // bad
        vec!["D", "AO", "G"], // dog
        vec!["G", "AH", "T"], // gut
        // Fricatives (voiceless)
        vec!["S", "IH", "T"], // sit
        vec!["F", "AE", "T"], // fat
        // Fricatives (voiced)
        vec!["Z", "IY", "R", "OW"], // zero
        vec!["V", "AE", "N"],       // van
        // Nasals
        vec!["M", "AE", "N"], // man
        vec!["N", "OW"],      // no (will pad with SIL)
        // Liquids
        vec!["L", "AY", "T"], // light
        vec!["R", "EH", "D"], // red
    ];

    separator('-', 70);
    println!("  Phase 1: Training Both Scorers");
    separator('-', 70);
    println!();

    println!(
        "  Training on {} patterns (30 reps each)...",
        training_patterns.len()
    );

    for pattern in &training_patterns {
        for _ in 0..30 {
            hierarchical.train_sequence(pattern);
            multiscale.train_sequence(pattern);
        }
    }

    // Stats
    let h_stats = hierarchical.stats();
    let m_stats = multiscale.stats();

    println!();
    println!("  Hierarchical (2-level):");
    println!(
        "    Active M×P clusters:  {} / 70",
        h_stats.active_mp_clusters
    );
    println!(
        "    Memory density:       {:.3}",
        h_stats.avg_manner_density
    );

    println!();
    println!("  Multi-Scale (3-level):");
    println!(
        "    Active L3 clusters:   {} / 140",
        m_stats.active_l3_clusters
    );
    println!("    L1 density:           {:.3}", m_stats.avg_l1_density);
    println!("    L2 density:           {:.3}", m_stats.avg_l2_density);
    println!("    L3 density:           {:.3}", m_stats.avg_l3_density);
    println!(
        "    Weights (L1/L2/L3):   {:.1}/{:.1}/{:.1}",
        m_stats.scale_weights[0] * 100.0,
        m_stats.scale_weights[1] * 100.0,
        m_stats.scale_weights[2] * 100.0
    );
    println!();

    // Discrimination tests
    separator('-', 70);
    println!("  Phase 2: Discrimination Comparison");
    separator('-', 70);
    println!();

    // Test 1: Trained vs completely different
    let trained = vec!["K", "AE", "T"]; // cat (trained)
    let different = vec!["S", "IY", "Z"]; // seize (different manner)

    hierarchical.set_level(2);
    let h_trained = hierarchical.score_sequence(&trained);
    let h_diff = hierarchical.score_sequence(&different);
    let h_disc1 = h_trained - h_diff;

    let m_trained = multiscale.score_sequence(&trained);
    let m_diff = multiscale.score_sequence(&different);
    let m_disc1 = m_trained - m_diff;

    println!("  Test 1: CAT (trained) vs SEIZE (different)");
    println!("    Hierarchical L2: {:+.4}", h_disc1);
    println!("    Multi-Scale:     {:+.4}", m_disc1);
    println!();

    // Test 2: Voicing minimal pair
    let voiceless = vec!["P", "AE", "T"]; // pat (voiceless, trained)
    let voiced = vec!["B", "AE", "D"]; // bad (voiced, also trained)
    let voiced_untrained = vec!["B", "AE", "T"]; // bat (voiced, NOT trained)

    let h_vl = hierarchical.score_sequence(&voiceless);
    let h_vd = hierarchical.score_sequence(&voiced_untrained);
    let h_disc2 = h_vl - h_vd;

    let m_vl = multiscale.score_sequence(&voiceless);
    let m_vd = multiscale.score_sequence(&voiced_untrained);
    let m_disc2 = m_vl - m_vd;

    // Also get L3-only score for comparison
    let m_l3_vl = multiscale.score_at_level(&voiceless, 3);
    let m_l3_vd = multiscale.score_at_level(&voiced_untrained, 3);
    let m_l3_disc = m_l3_vl - m_l3_vd;

    println!("  Test 2: PAT (voiceless, trained) vs BAT (voiced, untrained)");
    println!("    Hierarchical L2:    {:+.4}", h_disc2);
    println!("    Multi-Scale:        {:+.4}", m_disc2);
    println!("    L3-only (voicing):  {:+.4}", m_l3_disc);
    println!();

    // Test 3: Same manner, different voicing (maximal contrast)
    // Train heavy on voiceless, test voiced
    let voiceless_heavy = [
        vec!["P", "IH", "K"], // pick
        vec!["T", "IH", "K"], // tick
        vec!["K", "IH", "K"], // kick
        vec!["S", "IH", "K"], // sick
    ];

    for pattern in &voiceless_heavy {
        for _ in 0..50 {
            multiscale.train_sequence(pattern);
        }
    }

    let test_voiceless = vec!["P", "AE", "K"]; // pack
    let test_voiced = vec!["B", "AE", "G"]; // bag

    let l1_vl = multiscale.score_at_level(&test_voiceless, 1);
    let l1_vd = multiscale.score_at_level(&test_voiced, 1);

    let l2_vl = multiscale.score_at_level(&test_voiceless, 2);
    let l2_vd = multiscale.score_at_level(&test_voiced, 2);

    let l3_vl = multiscale.score_at_level(&test_voiceless, 3);
    let l3_vd = multiscale.score_at_level(&test_voiced, 3);

    let multi_vl = multiscale.score_sequence(&test_voiceless);
    let multi_vd = multiscale.score_sequence(&test_voiced);

    println!("  Test 3: PACK (voiceless) vs BAG (voiced) - after heavy voiceless training");
    println!("    L1 (Manner):        {:+.4}", l1_vl - l1_vd);
    println!("    L2 (M×P):           {:+.4}", l2_vl - l2_vd);
    println!("    L3 (M×P×V):         {:+.4}", l3_vl - l3_vd);
    println!("    Multi-Scale:        {:+.4}", multi_vl - multi_vd);
    println!();

    // Summary
    separator('=', 70);
    println!("  RESULTS SUMMARY");
    separator('=', 70);
    println!();

    println!("  Scorer Architecture     | Clusters | Best Discrimination");
    println!("  ----------------------- | -------- | -------------------");
    println!("  Original Holographic    |     1    |  +0.010 (baseline)");
    println!(
        "  Hierarchical L2         |    70    |  {:+.4}",
        h_disc1.max(h_disc2)
    );
    println!(
        "  Multi-Scale (L3+vote)   |   140    |  {:+.4}",
        (l3_vl - l3_vd).max(m_disc1)
    );
    println!();

    let improvement = ((l3_vl - l3_vd).max(m_disc1) / 0.010).round();
    println!(
        "  Improvement over baseline: {}× better discrimination!",
        improvement as i32
    );
    println!();

    if l3_vl - l3_vd > h_disc1 {
        println!("  [SUCCESS] L3 voicing clusters provide additional discrimination!");
        println!("  The voiced/voiceless split isolates phoneme classes more precisely.");
    }

    separator('=', 70);
    println!("  DEMO COMPLETE");
    separator('=', 70);
}
