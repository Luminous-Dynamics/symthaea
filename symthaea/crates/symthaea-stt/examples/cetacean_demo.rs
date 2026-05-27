// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cetacean Scorer Demo
//!
//! Demonstrates HDC grammar learning on whale communication patterns.
//! This shows why HDC excels at structural relations with small vocabularies.

use symthaea_stt::cetacean_scorer::{
    ALL_CAUS, CetaceanManner, CetaceanPlace, CetaceanScorer, CetaceanUnit, MANNER_NAMES,
    PLACE_NAMES,
};

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn main() {
    separator('=', 70);
    println!("  CETACEAN SCORER - HDC Grammar Learning Demo");
    separator('=', 70);
    println!();

    // Create scorer
    let mut scorer = CetaceanScorer::new();
    println!(
        "Initialized Cetacean Scorer with {} articulatory units",
        ALL_CAUS.len()
    );
    println!();

    // Show the articulatory unit inventory
    println!("Cetacean Articulatory Unit Inventory:");
    separator('-', 50);
    println!(
        "{:15} {:>10} {:>10} {:>10}",
        "", "Upsweep", "Downsweep", "Flat"
    );
    for (i, manner) in MANNER_NAMES.iter().enumerate() {
        print!("{:15}", manner);
        for (j, _place) in PLACE_NAMES.iter().enumerate() {
            let cau = CetaceanUnit::from_indices(i, j).unwrap();
            print!(" {:>10}", cau.short_name());
        }
        println!();
    }
    println!();

    // Define characteristic patterns for different species
    // Sperm whale: Repeated clicks (codas)
    let sperm_coda = vec![
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
    ];

    // Humpback whale: Complex whistles with sweeps
    let humpback_song = vec![
        CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
        CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Downsweep),
        CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
    ];

    // Killer whale (orca): Mixed calls
    let orca_call = vec![
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Upsweep),
        CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Downsweep),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Flat),
    ];

    // Print patterns
    println!("Training Patterns:");
    separator('-', 50);
    print!("Sperm Whale Coda:    ");
    for u in &sperm_coda {
        print!("{} ", u);
    }
    println!();
    print!("Humpback Song:       ");
    for u in &humpback_song {
        print!("{} ", u);
    }
    println!();
    print!("Orca Call:           ");
    for u in &orca_call {
        print!("{} ", u);
    }
    println!();
    println!();

    // Phase 1: Score before training (baseline)
    println!("Phase 1: Baseline Scores (before training)");
    separator('-', 50);
    let baseline_sperm = scorer.score_sequence(&sperm_coda);
    let baseline_humpback = scorer.score_sequence(&humpback_song);
    let baseline_orca = scorer.score_sequence(&orca_call);
    println!("Sperm Whale Coda:    {:.4}", baseline_sperm);
    println!("Humpback Song:       {:.4}", baseline_humpback);
    println!("Orca Call:           {:.4}", baseline_orca);
    println!();

    // Phase 2: Train on Sperm Whale patterns only
    println!("Phase 2: Training on Sperm Whale Codas (50 sequences)");
    separator('-', 50);
    for _ in 0..50 {
        scorer.train_sequence(&sperm_coda);
    }
    let stats = scorer.stats();
    println!("Training count: {}", stats.training_count);
    println!("Grammar density: {:.3}", stats.grammar_density);
    println!();

    // Phase 3: Score after training
    println!("Phase 3: Scores After Training");
    separator('-', 50);
    let trained_sperm = scorer.score_sequence(&sperm_coda);
    let trained_humpback = scorer.score_sequence(&humpback_song);
    let trained_orca = scorer.score_sequence(&orca_call);
    println!(
        "Sperm Whale Coda:    {:.4} (delta = {:+.4})",
        trained_sperm,
        trained_sperm - baseline_sperm
    );
    println!(
        "Humpback Song:       {:.4} (delta = {:+.4})",
        trained_humpback,
        trained_humpback - baseline_humpback
    );
    println!(
        "Orca Call:           {:.4} (delta = {:+.4})",
        trained_orca,
        trained_orca - baseline_orca
    );
    println!();

    // Phase 4: Discrimination test
    println!("Phase 4: Discrimination Analysis");
    separator('-', 50);
    let disc_sperm_humpback = scorer.discriminate(&sperm_coda, &humpback_song);
    let disc_sperm_orca = scorer.discriminate(&sperm_coda, &orca_call);
    println!(
        "Sperm vs Humpback: {:+.4} (positive = Sperm favored)",
        disc_sperm_humpback
    );
    println!(
        "Sperm vs Orca:     {:+.4} (positive = Sperm favored)",
        disc_sperm_orca
    );
    println!();

    // Phase 5: Test generalization to novel patterns
    println!("Phase 5: Generalization to Novel Patterns");
    separator('-', 50);

    // Novel sperm-like pattern (same manner, slight variation)
    let novel_sperm = vec![
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Downsweep),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
    ];

    // Completely different pattern
    let noise_pattern = vec![
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Downsweep),
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Upsweep),
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Downsweep),
        CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Upsweep),
    ];

    let novel_sperm_score = scorer.score_sequence(&novel_sperm);
    let noise_score = scorer.score_sequence(&noise_pattern);

    print!("Novel Sperm-like:    ");
    for u in &novel_sperm {
        print!("{} ", u);
    }
    println!(" -> {:.4}", novel_sperm_score);

    print!("Noise Pattern:       ");
    for u in &noise_pattern {
        print!("{} ", u);
    }
    println!(" -> {:.4}", noise_score);
    println!();

    // Analysis
    println!("Analysis");
    separator('=', 70);
    println!();
    println!("The Cetacean Scorer demonstrates HDC's strength: structural relations.");
    println!();
    println!("Key observations:");
    println!("  1. Small vocabulary (9 units) avoids bundling saturation");
    println!("  2. Training on one pattern increases its score while");
    println!("     relatively decreasing scores for dissimilar patterns");
    println!("  3. Novel patterns similar to training data still resonate");
    println!("     (soft matching, unlike traditional n-grams)");
    println!();
    println!("This is the 'Whale Bridge' strategy: use HDC where it excels,");
    println!("not for bulk vocabulary storage but for grammar/structure learning.");
    println!();

    // Final verification
    if disc_sperm_humpback > 0.0 && trained_sperm > trained_humpback {
        println!("[OK] Discrimination successful: Trained pattern recognized.");
    } else {
        println!("[WARN] Unexpected: discrimination may need tuning.");
    }
}
