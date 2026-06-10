// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inspect trained phoneme prototypes
//!
//! Analyzes the topology of learned HDC prototypes:
//! - Similarity matrix between phonemes
//! - Checks for orthogonality (the "0.99 similarity bug")
//! - Finds phoneme clusters

use std::collections::HashMap;
use std::env;
use symthaea_stt::bootstrap::TrainedPrototypes;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("data/models/dev-clean_phoneme_prototypes.bin");

    println!("Loading prototypes from: {}", model_path);

    let prototypes = match TrainedPrototypes::load(model_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading prototypes: {}", e);
            std::process::exit(1);
        }
    };

    let pairs = prototypes.as_pairs();
    let n = pairs.len();

    println!("\n=== PROTOTYPE TOPOLOGY ANALYSIS ===");
    println!("Total phonemes: {}\n", n);

    // Compute all pairwise similarities
    let mut similarities: Vec<(String, String, f32)> = Vec::new();
    let mut all_sims: Vec<f32> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = pairs[i].1.similarity(&pairs[j].1);
            similarities.push((pairs[i].0.clone(), pairs[j].0.clone(), sim));
            all_sims.push(sim);
        }
    }

    // Statistics
    all_sims.sort_by(|a, b| a.total_cmp(b));
    let min_sim = all_sims.first().copied().unwrap_or(0.0);
    let max_sim = all_sims.last().copied().unwrap_or(0.0);
    let mean_sim: f32 = all_sims.iter().sum::<f32>() / all_sims.len() as f32;
    let median_sim = all_sims[all_sims.len() / 2];

    println!("=== SIMILARITY STATISTICS ===");
    println!("Min similarity:    {:.4}", min_sim);
    println!("Max similarity:    {:.4}", max_sim);
    println!("Mean similarity:   {:.4}", mean_sim);
    println!("Median similarity: {:.4}", median_sim);

    // Check for the 0.99 similarity bug
    let high_sim_threshold = 0.9;
    let high_sim_count = all_sims.iter().filter(|&&s| s > high_sim_threshold).count();
    let total_pairs = all_sims.len();

    println!("\n=== ORTHOGONALITY CHECK ===");
    if max_sim > 0.95 {
        println!(
            "WARNING: Max similarity {:.4} is dangerously high!",
            max_sim
        );
        println!("         This could indicate prototype collapse.");
    } else if max_sim > 0.8 {
        println!("CAUTION: Max similarity {:.4} is somewhat high.", max_sim);
    } else {
        println!(
            "GOOD: Max similarity {:.4} indicates healthy orthogonality.",
            max_sim
        );
    }

    println!(
        "Pairs with sim > {}: {} / {} ({:.1}%)",
        high_sim_threshold,
        high_sim_count,
        total_pairs,
        100.0 * high_sim_count as f32 / total_pairs as f32
    );

    // Most similar pairs (potential issues)
    similarities.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!("\n=== MOST SIMILAR PHONEME PAIRS ===");
    println!("(Acoustically similar phonemes SHOULD be close)");
    for (p1, p2, sim) in similarities.iter().take(15) {
        let status = if *sim > 0.8 {
            "HIGH"
        } else if *sim > 0.5 {
            "mid"
        } else {
            "ok"
        };
        println!("  {:>6} <-> {:<6}  {:.4}  [{}]", p1, p2, sim, status);
    }

    // Least similar pairs (good orthogonality)
    println!("\n=== LEAST SIMILAR PHONEME PAIRS ===");
    println!("(Different phonemes SHOULD be far apart)");
    for (p1, p2, sim) in similarities.iter().rev().take(10) {
        println!("  {:>6} <-> {:<6}  {:.4}", p1, p2, sim);
    }

    // Group by phoneme type and check within-group similarity
    println!("\n=== PHONEME TYPE ANALYSIS ===");

    // Vowels (should cluster together somewhat)
    let vowels = [
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    ];
    let vowels_with_stress: Vec<String> = vowels
        .iter()
        .flat_map(|v| {
            vec![
                format!("{}", v),
                format!("{}0", v),
                format!("{}1", v),
                format!("{}2", v),
            ]
        })
        .collect();

    // Stops
    let stops = ["P", "B", "T", "D", "K", "G"];

    // Fricatives
    let fricatives = ["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"];

    fn avg_within_group(pairs: &[(String, HV16)], group: &[&str]) -> Option<f32> {
        let group_lower: Vec<String> = group
            .iter()
            .flat_map(|g| {
                vec![
                    g.to_string(),
                    format!("{}0", g),
                    format!("{}1", g),
                    format!("{}2", g),
                ]
            })
            .collect();
        let members: Vec<_> = pairs
            .iter()
            .filter(|(name, _)| {
                let base = name.trim_end_matches(|c: char| c.is_ascii_digit());
                group_lower
                    .iter()
                    .any(|g| g.trim_end_matches(|c: char| c.is_ascii_digit()) == base)
            })
            .collect();

        if members.len() < 2 {
            return None;
        }

        let mut total = 0.0f32;
        let mut count = 0;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                total += members[i].1.similarity(&members[j].1);
                count += 1;
            }
        }

        if count > 0 {
            Some(total / count as f32)
        } else {
            None
        }
    }

    if let Some(v) = avg_within_group(&pairs, &vowels) {
        println!("Vowels avg similarity:     {:.4}", v);
    }
    if let Some(v) = avg_within_group(&pairs, &stops) {
        println!("Stops avg similarity:      {:.4}", v);
    }
    if let Some(v) = avg_within_group(&pairs, &fricatives) {
        println!("Fricatives avg similarity: {:.4}", v);
    }

    // Voiced vs voiceless pairs
    println!("\n=== VOICED/VOICELESS PAIRS ===");
    let voiced_pairs = [
        ("P", "B"),
        ("T", "D"),
        ("K", "G"),
        ("F", "V"),
        ("S", "Z"),
        ("SH", "ZH"),
        ("TH", "DH"),
    ];

    for (voiceless, voiced) in &voiced_pairs {
        let v1 = pairs.iter().find(|(n, _)| n == *voiceless);
        let v2 = pairs.iter().find(|(n, _)| n == *voiced);
        if let (Some((_, hv1)), Some((_, hv2))) = (v1, v2) {
            let sim = hv1.similarity(hv2);
            println!(
                "  {} <-> {}  {:.4}  (should be similar)",
                voiceless, voiced, sim
            );
        }
    }

    // Instance counts
    println!("\n=== PHONEME INSTANCE COUNTS ===");
    let mut counts: Vec<_> = prototypes.counts.iter().collect();
    counts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    println!("Top 10:");
    for (name, count) in counts.iter().take(10) {
        println!("  {:>6}: {:>6}", name, count);
    }

    println!("\nBottom 10:");
    for (name, count) in counts.iter().rev().take(10) {
        println!("  {:>6}: {:>6}", name, count);
    }

    // Overall health assessment
    println!("\n=== OVERALL HEALTH ASSESSMENT ===");

    let health_score = if max_sim > 0.95 {
        "CRITICAL: Prototype collapse detected"
    } else if max_sim > 0.8 {
        "WARNING: Some prototypes may be collapsing"
    } else if mean_sim > 0.3 {
        "CAUTION: Mean similarity is higher than expected"
    } else if mean_sim < -0.1 {
        "GOOD: Strong orthogonality between phonemes"
    } else {
        "OK: Reasonable prototype separation"
    };

    println!("{}", health_score);
    println!(
        "\nConclusion: {} phonemes trained with mean separation {:.4}",
        n, mean_sim
    );
}

use symthaea_stt::hdc::HV16;
