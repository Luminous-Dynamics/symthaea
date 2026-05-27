// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contrastive Prototype Refinement Tool
//!
//! Applies the Wisdom Engine's Priority 3 fix: push similar prototypes apart
//! to prevent mode collapse in phoneme recognition.
//!
//! Usage:
//! ```bash
//! cargo run --release --bin refine-prototypes -- \
//!     --input data/models/dev-clean_adaptive_prototypes.bin \
//!     --output data/models/dev-clean_refined_prototypes.bin \
//!     --min-separation 0.3
//! ```

use clap::Parser;
use console::style;
use std::path::PathBuf;

use symthaea_stt::{HV16, TrainedPrototypes};

#[derive(Parser)]
#[command(
    name = "refine-prototypes",
    about = "Apply contrastive refinement to push similar prototypes apart",
    version,
    author
)]
struct Cli {
    /// Input prototypes file
    #[arg(short, long)]
    input: PathBuf,

    /// Output prototypes file
    #[arg(short, long)]
    output: PathBuf,

    /// Minimum separation between any two prototypes (cosine similarity)
    /// Lower = more separated. Try 0.2-0.4 for best results.
    #[arg(long, default_value = "0.3")]
    min_separation: f32,

    /// Maximum iterations for separation algorithm
    #[arg(long, default_value = "200")]
    max_iterations: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("     WISDOM ENGINE: CONTRASTIVE PROTOTYPE REFINEMENT       ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!("  Priority 3: Push similar prototypes apart to prevent mode collapse");
    println!();

    // Load prototypes
    println!("📂 Loading prototypes from {:?}...", cli.input);
    let mut prototypes = match TrainedPrototypes::load(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "{} Failed to load prototypes: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    println!("  ✓ Loaded {} phonemes", prototypes.len());
    println!();

    // Analyze initial separation
    println!("📊 Analyzing initial prototype separation...");
    let (initial_stats, confusion_pairs) = analyze_separation(&prototypes);
    println!("  Min similarity: {:.4}", initial_stats.min_sim);
    println!("  Max similarity: {:.4}", initial_stats.max_sim);
    println!("  Mean similarity: {:.4}", initial_stats.mean_sim);
    println!(
        "  Pairs above {}: {}",
        cli.min_separation, initial_stats.pairs_above_threshold
    );
    println!();

    if cli.verbose && !confusion_pairs.is_empty() {
        println!("  Most similar pairs (potential confusion):");
        for (i, (a, b, sim)) in confusion_pairs.iter().take(10).enumerate() {
            println!("    {}. {} <-> {} (sim={:.4})", i + 1, a, b, sim);
        }
        println!();
    }

    // Apply contrastive refinement
    println!(
        "🔧 Applying contrastive separation (target: <{})...",
        cli.min_separation
    );
    apply_aggressive_separation(&mut prototypes, cli.min_separation, cli.max_iterations);
    println!();

    // Analyze final separation
    println!("📊 Analyzing final prototype separation...");
    let (final_stats, _) = analyze_separation(&prototypes);
    println!("  Min similarity: {:.4}", final_stats.min_sim);
    println!("  Max similarity: {:.4}", final_stats.max_sim);
    println!("  Mean similarity: {:.4}", final_stats.mean_sim);
    println!(
        "  Pairs above {}: {}",
        cli.min_separation, final_stats.pairs_above_threshold
    );
    println!();

    // Report improvement
    let improvement = initial_stats.max_sim - final_stats.max_sim;
    if improvement > 0.0 {
        println!(
            "  {} Max similarity reduced by {:.4}",
            style("✓").green(),
            improvement
        );
    }

    // Save refined prototypes
    println!("💾 Saving refined prototypes to {:?}...", cli.output);
    if let Err(e) = prototypes.save(&cli.output) {
        eprintln!(
            "{} Failed to save prototypes: {}",
            style("ERROR:").red().bold(),
            e
        );
        std::process::exit(1);
    }
    println!("  ✓ Saved {} phonemes", prototypes.len());
    println!();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                   REFINEMENT COMPLETE                      ")
            .bold()
            .green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").cyan()
    );
    println!();
    println!(
        "  Next step: Evaluate with stt-eval --prototypes {:?}",
        cli.output
    );
}

struct SeparationStats {
    min_sim: f32,
    max_sim: f32,
    mean_sim: f32,
    pairs_above_threshold: usize,
}

fn analyze_separation(
    prototypes: &TrainedPrototypes,
) -> (SeparationStats, Vec<(String, String, f32)>) {
    let pairs = prototypes.as_pairs();
    let mut min_sim = f32::INFINITY;
    let mut max_sim = f32::NEG_INFINITY;
    let mut sum_sim = 0.0;
    let mut count = 0;
    let mut confusion_pairs = Vec::new();

    for i in 0..pairs.len() {
        for j in (i + 1)..pairs.len() {
            let sim = pairs[i].1.similarity(&pairs[j].1);
            min_sim = min_sim.min(sim);
            max_sim = max_sim.max(sim);
            sum_sim += sim;
            count += 1;

            confusion_pairs.push((pairs[i].0.clone(), pairs[j].0.clone(), sim));
        }
    }

    // Sort by similarity descending
    confusion_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let stats = SeparationStats {
        min_sim,
        max_sim,
        mean_sim: if count > 0 {
            sum_sim / count as f32
        } else {
            0.0
        },
        pairs_above_threshold: confusion_pairs.iter().filter(|(_, _, s)| *s > 0.3).count(),
    };

    (stats, confusion_pairs)
}

/// Apply aggressive contrastive separation until all pairs are below threshold
fn apply_aggressive_separation(
    prototypes: &mut TrainedPrototypes,
    min_separation: f32,
    max_iterations: usize,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    for iteration in 0..max_iterations {
        let labels: Vec<String> = prototypes.prototypes.keys().cloned().collect();
        let mut max_sim_seen: f32 = 0.0;
        let mut pairs_fixed = 0;

        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                let label_i = &labels[i];
                let label_j = &labels[j];

                let proto_i = match prototypes.prototypes.get(label_i) {
                    Some(hv) => *hv,
                    None => continue,
                };
                let proto_j = match prototypes.prototypes.get(label_j) {
                    Some(hv) => *hv,
                    None => continue,
                };

                let similarity = proto_i.similarity(&proto_j);
                max_sim_seen = max_sim_seen.max(similarity);

                // If too similar, push apart
                if similarity > min_separation {
                    pairs_fixed += 1;

                    // Adaptive repulsion strength - push harder when closer
                    let repulsion_strength =
                        ((similarity - min_separation) / (1.0 - min_separation)).powf(0.5) * 0.5;

                    // Generate different seeds for each prototype
                    let mut hasher_i = DefaultHasher::new();
                    label_i.hash(&mut hasher_i);
                    iteration.hash(&mut hasher_i);
                    let seed_i = hasher_i.finish();

                    let mut hasher_j = DefaultHasher::new();
                    label_j.hash(&mut hasher_j);
                    iteration.hash(&mut hasher_j);
                    (iteration + 1000).hash(&mut hasher_j); // Different seed
                    let seed_j = hasher_j.finish();

                    // Apply repulsion
                    let new_i =
                        contrastive_repel_seeded(&proto_i, &proto_j, repulsion_strength, seed_i);
                    let new_j =
                        contrastive_repel_seeded(&proto_j, &proto_i, repulsion_strength, seed_j);

                    prototypes.prototypes.insert(label_i.clone(), new_i);
                    prototypes.prototypes.insert(label_j.clone(), new_j);
                }
            }
        }

        // Progress reporting
        if iteration == 0 || iteration % 20 == 0 || pairs_fixed == 0 {
            println!(
                "    Iter {:3}: {} pairs above {:.2}, max_sim={:.4}",
                iteration, pairs_fixed, min_separation, max_sim_seen
            );
        }

        // Stop if converged
        if max_sim_seen <= min_separation {
            println!("    ✓ Converged after {} iterations", iteration + 1);
            break;
        }
    }
}

/// Contrastive repulsion with seeded randomness
fn contrastive_repel_seeded(proto: &HV16, negative: &HV16, strength: f32, seed: u64) -> HV16 {
    const HDC_WORDS: usize = 16;

    let mut result_words = proto.words;
    let bits_to_flip = (2048.0 * strength * 0.8) as usize;
    let mut flipped = 0;

    // Use blake3 hash of seed for deterministic randomness
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    let hash_result = hasher.finalize();
    let hash_bytes = hash_result.as_bytes();

    for word_idx in 0..HDC_WORDS {
        let proto_word = proto.words[word_idx];
        let neg_word = negative.words[word_idx];
        let agreement = !(proto_word ^ neg_word); // 1 where both same

        let hash_byte = hash_bytes[word_idx % 32] as usize;

        for bit in 0usize..128 {
            if flipped >= bits_to_flip {
                break;
            }
            let mask = 1u128 << bit;
            if (agreement & mask) != 0 {
                // Use hash to select which agreeing bits to flip
                if (bit.wrapping_add(word_idx * 7).wrapping_add(hash_byte)) % 3 == 0 {
                    result_words[word_idx] ^= mask;
                    flipped += 1;
                }
            }
        }
        if flipped >= bits_to_flip {
            break;
        }
    }

    HV16 {
        words: result_words,
    }
}
