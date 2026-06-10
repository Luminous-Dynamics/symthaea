// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Grammar Trainer
//!
//! Train the Holographic Scorer's grammar memory on CMU Dict phoneme sequences.
//!
//! # Usage
//!
//! ```bash
//! # Train grammar memory from CMU Dict
//! symthaea-grammar train --dict data/dict/cmudict.dict --output models/grammar_memory.bin
//!
//! # Train with phoneme basis from train_omni.py
//! symthaea-grammar train --dict data/dict/cmudict.dict --basis models/phoneme_hdc_basis.bin
//! ```

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use symthaea_stt::holographic_scorer::HolographicScorer;
use symthaea_stt::lexicon::CmuDictionary;

// ============================================================================
// CLI STRUCTURE
// ============================================================================

#[derive(Parser)]
#[command(name = "symthaea-grammar")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Luminous Dynamics")]
#[command(about = "Train Holographic Grammar Memory from CMU Dict")]
#[command(long_about = r#"
╔═══════════════════════════════════════════════════════════════════════╗
║                  HOLOGRAPHIC GRAMMAR TRAINER                          ║
║                                                                       ║
║   "Teaching language through hyperdimensional resonance"             ║
║                                                                       ║
║   Trains grammar memory using HDC sequence encoding:                 ║
║   H_t = Π(H_{t-1}) ⊕ Φ_current                                       ║
║                                                                       ║
║   This creates an "infinite n-gram" model in a fixed-size vector.    ║
╚═══════════════════════════════════════════════════════════════════════╝
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train grammar memory from phoneme sequences
    Train {
        /// Path to CMU Dict file
        #[arg(long, default_value = "data/dict/cmudict.dict")]
        dict: PathBuf,

        /// Path to phoneme HDC basis (from train_omni.py)
        #[arg(long)]
        basis: Option<PathBuf>,

        /// Output path for trained grammar memory
        #[arg(long, default_value = "models/grammar_memory.bin")]
        output: PathBuf,

        /// Also build and save trigram constraints
        #[arg(long)]
        trigrams: bool,

        /// Limit number of sequences (for testing)
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Test grammar resonance on sample sequences
    Test {
        /// Path to grammar memory
        #[arg(long, default_value = "models/grammar_memory.bin")]
        grammar: PathBuf,

        /// Path to phoneme HDC basis
        #[arg(long)]
        basis: Option<PathBuf>,
    },
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            dict,
            basis,
            output,
            trigrams,
            limit,
        } => {
            train_grammar(
                &dict,
                basis.as_deref(),
                &output,
                trigrams,
                limit,
                cli.verbose,
            )?;
        }
        Commands::Test { grammar, basis } => {
            test_grammar(&grammar, basis.as_deref(), cli.verbose)?;
        }
    }

    Ok(())
}

// ============================================================================
// TRAINING
// ============================================================================

/// Strip stress markers from ARPAbet phonemes (e.g., "AH0" -> "AH")
fn strip_stress(phoneme: &str) -> String {
    phoneme
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string()
}

fn train_grammar(
    dict_path: &Path,
    basis_path: Option<&Path>,
    output_path: &Path,
    build_trigrams: bool,
    limit: Option<usize>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║            HOLOGRAPHIC GRAMMAR MEMORY TRAINING                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load CMU Dict
    println!("Loading CMU Dict from {:?}...", dict_path);
    let dict = CmuDictionary::load(dict_path)?;
    println!("  Loaded {} entries", dict.len());

    // Create scorer (with basis if provided)
    let mut scorer = HolographicScorer::new();

    if let Some(bp) = basis_path {
        println!("Loading phoneme basis from {:?}...", bp);
        let basis = HolographicScorer::load_basis(bp)?;
        println!("  Loaded {} phoneme vectors", basis.len());
        scorer.set_phoneme_basis(basis);
    }

    // Extract phoneme sequences
    println!("\nExtracting phoneme sequences...");
    let mut sequences: Vec<Vec<String>> = Vec::new();

    for word in dict.words() {
        if let Some(entry) = dict.get(word) {
            // Primary pronunciation
            let seq: Vec<String> = entry.primary.iter().map(|p| strip_stress(p)).collect();
            sequences.push(seq);

            // Alternates
            for alt in &entry.alternates {
                let seq: Vec<String> = alt.iter().map(|p| strip_stress(p)).collect();
                sequences.push(seq);
            }
        }

        if let Some(lim) = limit {
            if sequences.len() >= lim {
                break;
            }
        }
    }

    println!("  Extracted {} phoneme sequences", sequences.len());

    if verbose {
        // Show some statistics
        let total_phonemes: usize = sequences.iter().map(|s| s.len()).sum();
        let avg_len = total_phonemes as f32 / sequences.len() as f32;
        println!("  Total phonemes: {}", total_phonemes);
        println!("  Average sequence length: {:.2}", avg_len);

        // Phoneme frequency
        let mut freq: HashMap<String, usize> = HashMap::new();
        for seq in &sequences {
            for p in seq {
                *freq.entry(p.clone()).or_default() += 1;
            }
        }
        println!("  Unique phonemes: {}", freq.len());
    }

    // Train grammar memory
    println!("\nTraining grammar memory...");
    scorer.train_grammar(&sequences);

    // Optionally build trigram constraints
    if build_trigrams {
        println!("\nBuilding trigram constraints...");
        scorer.build_trigram_constraints(&sequences);
    }

    // Save grammar memory
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    println!("\nSaving grammar memory to {:?}...", output_path);
    scorer.save_grammar(output_path)?;

    // Save trigrams if built
    if build_trigrams {
        let trigram_path = output_path.with_extension("trigrams.json");
        println!("Saving trigram constraints to {:?}...", trigram_path);
        // Note: trigram saving would need to be implemented
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    GRAMMAR TRAINING COMPLETE                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Grammar Memory: {:?}", output_path);
    println!("  Trained on {} sequences", sequences.len());
    println!();
    println!("  Next: Use grammar memory in articulatory decoder");
    println!();

    Ok(())
}

// ============================================================================
// TESTING
// ============================================================================

fn test_grammar(
    grammar_path: &Path,
    basis_path: Option<&Path>,
    _verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║               HOLOGRAPHIC GRAMMAR RESONANCE TEST                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create scorer
    let mut scorer = HolographicScorer::new();

    if let Some(bp) = basis_path {
        println!("Loading phoneme basis from {:?}...", bp);
        let basis = HolographicScorer::load_basis(bp)?;
        scorer.set_phoneme_basis(basis);
    }

    // Load grammar
    println!("Loading grammar memory from {:?}...", grammar_path);
    scorer.load_grammar(grammar_path)?;

    // Test some sequences
    println!("\nTesting resonance scores:\n");

    let test_cases = vec![
        // Common English sequences
        (vec!["HH", "EH", "L", "OW"], "hello"),
        (vec!["W", "ER", "L", "D"], "world"),
        (vec!["T", "IY", "M"], "team"),
        (vec!["S", "P", "IY", "CH"], "speech"),
        (
            vec!["R", "EH", "K", "AH", "G", "N", "IH", "SH", "AH", "N"],
            "recognition",
        ),
        // Phonotactically valid but rare
        (vec!["S", "T", "R", "EH", "NG", "TH"], "strength"),
        // Invalid sequences (should have lower resonance)
        (vec!["NG", "K", "P", "F"], "invalid-ngkpf"),
        (vec!["ZH", "TH", "DH"], "invalid-zhthdh"),
    ];

    for (sequence, label) in test_cases {
        scorer.reset();

        let mut scores: Vec<f32> = Vec::new();
        for phoneme in &sequence {
            let s = scorer.score(&phoneme.to_string());
            scores.push(s);
        }

        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let final_score = *scores.last().unwrap_or(&0.0);

        println!(
            "  {:20} {:40} avg={:.3} final={:.3}",
            label,
            sequence.join(" "),
            avg_score,
            final_score
        );
    }

    println!("\n(Higher scores indicate better resonance with trained grammar)");

    Ok(())
}
