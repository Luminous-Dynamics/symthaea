// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Whale Analysis Pipeline
//!
//! Complete pipeline: Whale Audio → CAU Classification → HDC Grammar Scoring
//!
//! This demonstrates the full Cetacean Scorer system on real whale recordings.

use std::path::Path;
use symthaea_stt::cetacean_classifier::{CetaceanClassifier, CetaceanConfig, collapse_to_segments};
use symthaea_stt::cetacean_scorer::{CetaceanManner, CetaceanPlace, CetaceanScorer, CetaceanUnit};

fn main() -> std::io::Result<()> {
    println!("======================================================================");
    println!("  WHALE ANALYSIS PIPELINE");
    println!("  Audio → CAU Classification → HDC Grammar Scoring");
    println!("======================================================================");
    println!();

    // Find whale audio files
    let whale_dir = Path::new("data/whales/sperm");
    let files: Vec<_> = std::fs::read_dir(whale_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "wav")
                .unwrap_or(false)
        })
        .collect();

    if files.is_empty() {
        println!("No whale audio files found in {}", whale_dir.display());
        println!("Run: bash scripts/download_whales.sh");
        return Ok(());
    }

    println!("Found {} whale audio files\n", files.len());

    // Create classifier with cetacean-optimized config
    let config = CetaceanConfig {
        sample_rate: 44100,
        frame_duration: 0.02,
        hop_duration: 0.01,
        n_fft: 2048,
        f_min: 100.0,
        f_max: 20000.0,
        transient_threshold: 0.3,
        harmonicity_threshold: 0.35,
        fm_slope_threshold: 400.0,
    };

    let mut classifier = CetaceanClassifier::new(config);
    let mut scorer = CetaceanScorer::new();

    // Process each file
    let mut all_sequences: Vec<Vec<CetaceanUnit>> = Vec::new();

    for entry in &files {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();

        println!("Processing: {}", filename);
        println!("----------------------------------------------------------------------");

        // Classify audio into CAU sequence
        match classifier.classify_file(&path) {
            Ok(classifications) => {
                println!("  Frames analyzed: {}", classifications.len());

                // Collapse to segments
                let segments = collapse_to_segments(&classifications);
                println!("  Segments found:  {}", segments.len());

                // Count CAU types
                let mut manner_counts = [0usize; 3];
                let mut place_counts = [0usize; 3];

                for (_, _, unit) in &segments {
                    match unit.manner {
                        CetaceanManner::Click => manner_counts[0] += 1,
                        CetaceanManner::Whistle => manner_counts[1] += 1,
                        CetaceanManner::Burst => manner_counts[2] += 1,
                    }
                    match unit.place {
                        CetaceanPlace::Upsweep => place_counts[0] += 1,
                        CetaceanPlace::Downsweep => place_counts[1] += 1,
                        CetaceanPlace::Flat => place_counts[2] += 1,
                    }
                }

                println!("\n  Manner distribution:");
                println!("    Clicks:   {:4}", manner_counts[0]);
                println!("    Whistles: {:4}", manner_counts[1]);
                println!("    Bursts:   {:4}", manner_counts[2]);

                println!("\n  Place distribution:");
                println!("    Upsweep:   {:4}", place_counts[0]);
                println!("    Downsweep: {:4}", place_counts[1]);
                println!("    Flat:      {:4}", place_counts[2]);

                // Extract CAU sequence for scoring
                let sequence: Vec<CetaceanUnit> = segments.iter().map(|(_, _, u)| *u).collect();

                if !sequence.is_empty() {
                    // Score the sequence
                    let score = scorer.score_sequence(&sequence);
                    println!("\n  HDC Grammar Score: {:.4}", score);

                    // Show first few CAUs
                    print!("\n  Sequence preview: ");
                    for (i, u) in sequence.iter().take(10).enumerate() {
                        if i > 0 {
                            print!(" → ");
                        }
                        print!("{}", u);
                    }
                    if sequence.len() > 10 {
                        print!(" ... ({} more)", sequence.len() - 10);
                    }
                    println!();

                    all_sequences.push(sequence);
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }

        println!();
    }

    // Train scorer on discovered patterns
    if !all_sequences.is_empty() {
        println!("======================================================================");
        println!("  GRAMMAR LEARNING");
        println!("======================================================================");
        println!();

        println!(
            "Training HDC grammar on {} sequences...",
            all_sequences.len()
        );

        for seq in &all_sequences {
            scorer.train_sequence(seq);
        }

        let stats = scorer.stats();
        println!("  Training count:  {}", stats.training_count);
        println!("  Grammar density: {:.3}", stats.grammar_density);
        println!();

        // Re-score after training
        println!("Scores after training:");
        println!("----------------------------------------------------------------------");

        for (i, seq) in all_sequences.iter().enumerate() {
            let score = scorer.score_sequence(seq);
            println!("  Sequence {}: {:.4} ({} units)", i + 1, score, seq.len());
        }

        // Test discrimination
        println!();
        println!("======================================================================");
        println!("  DISCRIMINATION TEST");
        println!("======================================================================");
        println!();

        // Compare first sequence against synthetic patterns
        if let Some(first_seq) = all_sequences.first() {
            // Pure click pattern (likely sperm whale coda)
            let click_pattern = vec![
                CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
                CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
                CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            ];

            // Pure whistle pattern (likely humpback)
            let whistle_pattern = vec![
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Downsweep),
                CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Flat),
            ];

            let real_score = scorer.score_sequence(first_seq);
            let click_score = scorer.score_sequence(&click_pattern);
            let whistle_score = scorer.score_sequence(&whistle_pattern);

            println!("Real whale recording: {:.4}", real_score);
            println!("Synthetic click-only: {:.4}", click_score);
            println!("Synthetic whistle-only: {:.4}", whistle_score);
            println!();

            let disc = scorer.discriminate(first_seq, &click_pattern);
            println!("Discrimination (Real vs Click): {:+.4}", disc);

            let disc2 = scorer.discriminate(first_seq, &whistle_pattern);
            println!("Discrimination (Real vs Whistle): {:+.4}", disc2);
        }
    }

    println!();
    println!("======================================================================");
    println!("  PIPELINE COMPLETE");
    println!("======================================================================");

    Ok(())
}
