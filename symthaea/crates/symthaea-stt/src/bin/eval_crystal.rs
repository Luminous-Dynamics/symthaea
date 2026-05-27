// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluate Crystal Reservoir Classifier
//!
//! Loads a trained Crystal classifier and evaluates on test data using
//! Viterbi decoding with bigram language model.

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::PathBuf;

use symthaea_stt::{
    AudioConfig, AudioFrontend, CrystalReservoir, OnlinePrototypeClassifier,
    eval::phoneme_error_rate_detailed, id_to_audio_path, load_alignments,
};

#[derive(Parser)]
#[command(
    name = "eval-crystal",
    about = "Evaluate Crystal Reservoir classifier",
    version,
    author
)]
struct Cli {
    /// Trained Crystal classifier file
    #[arg(short, long)]
    classifier: PathBuf,

    /// Parquet file with test alignments
    #[arg(short, long)]
    alignments: PathBuf,

    /// LibriSpeech audio directory
    #[arg(short = 'd', long)]
    audio_dir: PathBuf,

    /// Maximum utterances to evaluate (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Frame hop duration (seconds)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Use delta features
    #[arg(long, default_value = "true")]
    use_deltas: bool,

    /// Per-utterance cepstral mean normalization
    #[arg(long)]
    cmn: bool,

    /// Per-utterance CMVN
    #[arg(long)]
    cmvn: bool,

    /// Viterbi self-loop penalty
    #[arg(long, default_value = "1.0")]
    penalty: f32,

    /// Bigram LM weight
    #[arg(long, default_value = "0.1")]
    lm_weight: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Stack multiple frames together for temporal context
fn stack_frames(features: &[Vec<f32>], center: usize, context_frames: usize) -> Vec<f32> {
    let n_dims = if features.is_empty() {
        0
    } else {
        features[0].len()
    };
    let half = context_frames / 2;
    let mut stacked = Vec::with_capacity(n_dims * context_frames);

    for offset in 0..context_frames {
        let idx = (center + offset).saturating_sub(half);
        let idx = idx.min(features.len().saturating_sub(1));
        stacked.extend_from_slice(&features[idx]);
    }

    stacked
}

/// Simple Viterbi decoder with bigram LM
fn viterbi_decode(
    logits: &[Vec<f32>],
    phonemes: &[String],
    bigram_counts: &HashMap<(usize, usize), usize>,
    unigram_counts: &[usize],
    penalty: f32,
    lm_weight: f32,
) -> Vec<String> {
    if logits.is_empty() {
        return vec![];
    }

    let n_states = phonemes.len();
    let n_frames = logits.len();
    let uniform_log = -(n_states as f32).ln();
    let _total_unigrams: f32 = unigram_counts.iter().sum::<usize>() as f32;

    // Viterbi DP
    let mut dp = vec![vec![f32::NEG_INFINITY; n_states]; n_frames];
    let mut backptr = vec![vec![0usize; n_states]; n_frames];

    // Initialize first frame
    for s in 0..n_states {
        dp[0][s] = logits[0][s];
    }

    // Forward pass
    for t in 1..n_frames {
        for s in 0..n_states {
            let emission = logits[t][s];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0;

            for prev in 0..n_states {
                let trans = if prev == s {
                    // Self-loop
                    -penalty
                } else {
                    // Transition with bigram LM
                    let bigram_count = bigram_counts.get(&(prev, s)).copied().unwrap_or(0) as f32;
                    let prev_count = unigram_counts[prev] as f32;
                    let bigram_log = if prev_count > 0.0 && bigram_count > 0.0 {
                        (bigram_count / prev_count).ln()
                    } else {
                        -10.0 // Smoothing
                    };
                    -penalty + lm_weight * (bigram_log - uniform_log)
                };

                let score = dp[t - 1][prev] + trans + emission;
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }

            dp[t][s] = best_score;
            backptr[t][s] = best_prev;
        }
    }

    // Find best final state
    let mut best_final = 0;
    let mut best_score = f32::NEG_INFINITY;
    for s in 0..n_states {
        if dp[n_frames - 1][s] > best_score {
            best_score = dp[n_frames - 1][s];
            best_final = s;
        }
    }

    // Backtrack
    let mut path = vec![best_final];
    for t in (1..n_frames).rev() {
        let prev = backptr[t][*path.last().unwrap()];
        path.push(prev);
    }
    path.reverse();

    // Collapse consecutive same phonemes
    let mut result = Vec::new();
    let mut prev = usize::MAX;
    for &s in &path {
        if s != prev {
            result.push(phonemes[s].clone());
            prev = s;
        }
    }

    result
}

/// Serializable state for the Crystal classifier
#[derive(serde::Serialize, serde::Deserialize)]
struct CrystalClassifierState {
    phonemes: Vec<String>,
    prototypes: Vec<Vec<f32>>,
    counts: Vec<usize>,
    n_filters: usize,
    context_frames: usize,
    feature_dim: usize,
    n_mels: usize,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!(
        "{}",
        style("    CRYSTAL RESERVOIR EVALUATION                           ")
            .bold()
            .magenta()
    );
    println!(
        "{}",
        style("    Gabor STRFs + Prototype Matching + Viterbi             ").magenta()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!();

    // Load classifier
    println!("Loading Crystal classifier from {:?}...", cli.classifier);
    let data = std::fs::read(&cli.classifier).expect("Failed to read classifier file");
    let state: CrystalClassifierState = bincode::deserialize(&data).expect("Failed to deserialize");

    let phonemes = state.phonemes;
    let n_filters = state.n_filters;
    let context_frames = state.context_frames;
    let feature_dim = state.feature_dim;
    let n_mels = state.n_mels;

    // Recreate Crystal Reservoir (deterministic from params)
    let crystal = CrystalReservoir::new(n_filters, context_frames, feature_dim, n_mels);

    // Recreate classifier with loaded prototypes
    let mut classifier = OnlinePrototypeClassifier::new(phonemes.clone(), n_filters);
    classifier.prototypes = state.prototypes;
    classifier.counts = state.counts;

    println!("  ✓ Loaded {} phoneme prototypes", phonemes.len());
    println!("  ✓ Crystal Reservoir: {} Gabor filters", n_filters);

    // Load test alignments
    println!("\nLoading test alignments from {:?}...", cli.alignments);
    let alignments = match load_alignments(&cli.alignments) {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{} Failed to load alignments: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };
    println!("  ✓ Loaded {} utterance alignments", alignments.len());

    // Build bigram counts from training data
    println!("\n  Building bigram LM from alignments...");
    let mut bigram_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut unigram_counts = vec![0usize; phonemes.len()];

    for alignment in alignments.values() {
        let mut prev_idx: Option<usize> = None;
        for segment in &alignment.phonemes {
            if let Some(idx) = phonemes.iter().position(|p| p == &segment.phoneme) {
                unigram_counts[idx] += 1;
                if let Some(prev) = prev_idx {
                    *bigram_counts.entry((prev, idx)).or_insert(0) += 1;
                }
                prev_idx = Some(idx);
            }
        }
    }
    println!("  ✓ Built bigram LM ({} transitions)", bigram_counts.len());

    // Audio frontend
    let audio_config = AudioConfig::default();
    let mut frontend = AudioFrontend::new(audio_config);

    // Evaluate
    let total = if cli.max_utterances > 0 {
        alignments.len().min(cli.max_utterances)
    } else {
        alignments.len()
    };

    println!("\n  Evaluating {} utterances...", total);
    println!("    Penalty: {}, LM weight: {}", cli.penalty, cli.lm_weight);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.magenta} [{bar:40.magenta/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("◆◇-"),
    );

    let mut total_per = 0.0;
    let mut evaluated = 0;
    let mut total_insertions = 0;
    let mut total_deletions = 0;
    let mut total_substitutions = 0;
    let mut total_ref_len = 0;

    for (i, (id, alignment)) in alignments.iter().take(total).enumerate() {
        pb.set_position(i as u64);

        let audio_path = id_to_audio_path(id, &cli.audio_dir);
        let audio_path = match audio_path {
            Some(p) if p.exists() => p,
            _ => continue,
        };

        let (samples, _) = match AudioFrontend::load_audio(&audio_path) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let mut features = if cli.use_deltas {
            frontend.extract_features_with_deltas(&samples)
        } else {
            frontend.extract_features(&samples)
        };

        if features.is_empty() {
            continue;
        }

        // CMVN
        if (cli.cmn || cli.cmvn) && !features.is_empty() {
            let n_dims = features[0].len();
            let n_frames = features.len() as f32;
            let mut mean = vec![0.0f32; n_dims];
            for frame in &features {
                for (j, &v) in frame.iter().enumerate() {
                    mean[j] += v;
                }
            }
            for m in &mut mean {
                *m /= n_frames;
            }
            for frame in &mut features {
                for (j, v) in frame.iter_mut().enumerate() {
                    *v -= mean[j];
                }
            }
            if cli.cmvn {
                let mut var = vec![0.0f32; n_dims];
                for frame in &features {
                    for (j, &v) in frame.iter().enumerate() {
                        var[j] += v * v;
                    }
                }
                for v in &mut var {
                    *v = (*v / n_frames).sqrt().max(1e-6);
                }
                for frame in &mut features {
                    for (j, v) in frame.iter_mut().enumerate() {
                        *v /= var[j];
                    }
                }
            }
        }

        // Get logits for each frame
        let mut all_logits = Vec::with_capacity(features.len());
        for frame_idx in 0..features.len() {
            let stacked = stack_frames(&features, frame_idx, context_frames);
            let crystal_features = crystal.forward(&stacked);
            let logits = classifier.get_logits(&crystal_features);
            all_logits.push(logits);
        }

        // Viterbi decode
        let predicted = viterbi_decode(
            &all_logits,
            &phonemes,
            &bigram_counts,
            &unigram_counts,
            cli.penalty,
            cli.lm_weight,
        );

        // Get reference (strip stress for comparison)
        let reference: Vec<String> = alignment
            .phonemes
            .iter()
            .map(|s| {
                let p = &s.phoneme;
                if p.ends_with('0') || p.ends_with('1') || p.ends_with('2') {
                    p[..p.len() - 1].to_string()
                } else {
                    p.clone()
                }
            })
            .collect();

        let predicted_stripped: Vec<String> = predicted
            .iter()
            .map(|p| {
                if p.ends_with('0') || p.ends_with('1') || p.ends_with('2') {
                    p[..p.len() - 1].to_string()
                } else {
                    p.clone()
                }
            })
            .collect();

        // Convert to &[&str] for PER computation
        let ref_strs: Vec<&str> = reference.iter().map(|s| s.as_str()).collect();
        let hyp_strs: Vec<&str> = predicted_stripped.iter().map(|s| s.as_str()).collect();

        // Compute PER
        let (per, result, _) = phoneme_error_rate_detailed(&ref_strs, &hyp_strs);
        total_per += per;
        total_insertions += result.insertions;
        total_deletions += result.deletions;
        total_substitutions += result.substitutions;
        total_ref_len += reference.len();
        evaluated += 1;
    }

    pb.finish_with_message("done");

    // Results
    println!(
        "\n{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
    println!(
        "{}",
        style("                    EVALUATION RESULTS                       ")
            .bold()
            .white()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );

    if evaluated > 0 {
        let _avg_per = total_per / evaluated as f32;
        let overall_per = (total_insertions + total_deletions + total_substitutions) as f32
            / total_ref_len as f32
            * 100.0;

        println!("\n  Evaluated: {} utterances", evaluated);
        println!("  Insertions:         {}", total_insertions);
        println!("  Deletions:          {}", total_deletions);
        println!("  Substitutions:      {}", total_substitutions);
        println!("  Reference phonemes: {}", total_ref_len);
        println!();

        if overall_per < 40.0 {
            println!(
                "  {}",
                style(format!("PER: {:.1}%", overall_per)).bold().green()
            );
        } else if overall_per < 55.0 {
            println!(
                "  {}",
                style(format!("PER: {:.1}%", overall_per)).bold().yellow()
            );
        } else {
            println!(
                "  {}",
                style(format!("PER: {:.1}%", overall_per)).bold().red()
            );
        }
    } else {
        println!("\n  No utterances evaluated!");
    }

    println!();
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").magenta()
    );
}
