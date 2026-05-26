// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluate Direct Linear Classifier
//!
//! Computes Phoneme Error Rate (PER) using the trained direct classifier.

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea_stt::{
    load_alignments, AudioConfig, AudioFrontend, CmuDictionary, DirectClassifier, EvalResult,
    LtcCell, LtcConfig, TextToPhonemes,
};

#[derive(Parser)]
#[command(
    name = "eval-direct",
    about = "Evaluate direct linear classifier on LibriSpeech",
    version,
    author
)]
struct Cli {
    /// Path to trained classifier file
    #[arg(short = 'c', long, default_value = "data/models/direct_classifier.bin")]
    classifier: PathBuf,

    /// Path to test dataset directory (LibriSpeech format)
    #[arg(short, long, default_value = "data/librispeech/LibriSpeech/dev-clean")]
    test_dir: PathBuf,

    /// Path to CMU dictionary
    #[arg(short, long, default_value = "data/dict/cmudict.dict")]
    dictionary: PathBuf,

    /// Maximum number of utterances to evaluate (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Minimum confidence for emission
    #[arg(long, default_value = "0.05")]
    min_confidence: f32,

    /// Minimum consecutive frames to emit a phoneme
    #[arg(long, default_value = "3")]
    min_duration: usize,

    /// Use delta features (must match training)
    #[arg(long, default_value = "true")]
    use_deltas: bool,

    /// Override prior correction scale (-1 = use stored value)
    #[arg(long, default_value = "-1.0")]
    prior_scale: f32,

    /// Normalize weight vectors to unit norm (remove class frequency bias)
    #[arg(long)]
    normalize_weights: bool,

    /// Skip reservoir (use raw stacked features - must match training)
    #[arg(long)]
    skip_reservoir: bool,

    /// Smooth logits over N frames before decoding (0 = no smoothing)
    #[arg(long, default_value = "0")]
    smooth_window: usize,

    /// Per-utterance cepstral mean normalization (must match training)
    #[arg(long)]
    cmn: bool,

    /// Per-utterance cepstral mean + variance normalization (overrides --cmn)
    #[arg(long)]
    cmvn: bool,

    /// Viterbi self-loop penalty (log-scale penalty for phoneme transitions, 0=disabled)
    #[arg(long, default_value = "0.0")]
    viterbi_penalty: f32,

    /// Path to alignment parquet for building bigram LM (enables bigram Viterbi)
    #[arg(long)]
    bigram_alignments: Option<PathBuf>,

    /// Language model weight for bigram Viterbi (scales bigram log-probs)
    #[arg(long, default_value = "1.0")]
    lm_weight: f32,

    /// Acoustic model scale (multiplies logits before decoding; >1 = sharper, <1 = softer)
    #[arg(long, default_value = "1.0")]
    am_scale: f32,

    /// Insertion penalty (extra cost per phoneme transition, controls output length)
    #[arg(long, default_value = "0.0")]
    insertion_penalty: f32,

    /// Collapse stress variants before decoding (AH0/AH1/AH2 → AH)
    /// Reduces state space from ~69 to ~44 for cleaner Viterbi paths
    #[arg(long)]
    collapse_stress: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// LibriSpeech utterance
struct Utterance {
    audio_path: PathBuf,
    transcript: String,
    #[allow(dead_code)]
    speaker_id: String,
    #[allow(dead_code)]
    chapter_id: String,
    #[allow(dead_code)]
    utterance_id: String,
}

/// Strip stress markers from phoneme (e.g., "IH1" -> "IH", "AH0" -> "AH")
fn strip_stress(phoneme: &str) -> String {
    let bytes = phoneme.as_bytes();
    if !bytes.is_empty() && bytes[bytes.len() - 1].is_ascii_digit() {
        phoneme[..phoneme.len() - 1].to_string()
    } else {
        phoneme.to_string()
    }
}

/// Decode phoneme sequence from frame-level predictions
fn decode_sequence(
    predictions: &[(String, f32)],
    min_confidence: f32,
    min_duration: usize,
) -> Vec<String> {
    if predictions.is_empty() {
        return Vec::new();
    }

    // Segment into runs of the same phoneme
    let mut result: Vec<String> = Vec::new();
    let mut run_start = 0;

    while run_start < predictions.len() {
        let current = &predictions[run_start].0;
        let mut run_end = run_start + 1;

        while run_end < predictions.len() && &predictions[run_end].0 == current {
            run_end += 1;
        }

        let run_len = run_end - run_start;

        // Only emit if: long enough run AND sufficient average confidence
        if run_len >= min_duration {
            let avg_conf: f32 = predictions[run_start..run_end]
                .iter()
                .map(|(_, c)| c)
                .sum::<f32>()
                / run_len as f32;

            if avg_conf >= min_confidence && result.last() != Some(current) {
                result.push(current.clone());
            }
        }

        run_start = run_end;
    }

    result
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

/// Viterbi decoder: find optimal phoneme path given logits and transition penalty
/// penalty > 0 means switching to a different phoneme costs `penalty` log-score
fn viterbi_decode(all_logits: &[Vec<f32>], penalty: f32, phonemes: &[String]) -> Vec<String> {
    let n_frames = all_logits.len();
    let n_states = phonemes.len();
    if n_frames == 0 || n_states == 0 {
        return Vec::new();
    }

    // Viterbi tables: score[t][s] = best log-score ending in state s at frame t
    // backptr[t][s] = previous state on best path
    let mut score = vec![vec![f32::NEG_INFINITY; n_states]; n_frames];
    let mut backptr = vec![vec![0usize; n_states]; n_frames];

    // Initialize first frame
    for s in 0..n_states {
        score[0][s] = all_logits[0][s];
    }

    // Forward pass
    for t in 1..n_frames {
        for s in 0..n_states {
            let emission = all_logits[t][s];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0;

            for prev_s in 0..n_states {
                let transition = if prev_s == s { 0.0 } else { -penalty };
                let candidate = score[t - 1][prev_s] + transition;
                if candidate > best_score {
                    best_score = candidate;
                    best_prev = prev_s;
                }
            }

            score[t][s] = best_score + emission;
            backptr[t][s] = best_prev;
        }
    }

    // Backtrace
    let mut best_final = 0;
    let mut best_final_score = f32::NEG_INFINITY;
    for s in 0..n_states {
        if score[n_frames - 1][s] > best_final_score {
            best_final_score = score[n_frames - 1][s];
            best_final = s;
        }
    }

    let mut path = vec![0usize; n_frames];
    path[n_frames - 1] = best_final;
    for t in (1..n_frames).rev() {
        path[t - 1] = backptr[t][path[t]];
    }

    // Deduplicate consecutive same phonemes
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

/// Bigram language model: log P(next_phone | prev_phone) matrix
struct BigramLM {
    /// `log_probs[prev_idx][next_idx] = log P(phonemes[next_idx] | phonemes[prev_idx])`
    log_probs: Vec<Vec<f32>>,
    /// Unigram log probs for first-frame initialization
    unigram_log_probs: Vec<f32>,
}

/// Build bigram LM from alignment data
/// If phonemes are stress-collapsed, alignment phonemes are stripped before lookup
fn build_bigram_lm(alignments_path: &Path, phonemes: &[String]) -> Result<BigramLM, String> {
    let alignments = load_alignments(alignments_path)?;

    let n_states = phonemes.len();
    let phone_to_idx: HashMap<String, usize> = phonemes
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), i))
        .collect();

    // Detect if phonemes are stress-collapsed (no digits in any phoneme name)
    let is_collapsed = phonemes
        .iter()
        .all(|p| !p.bytes().last().is_some_and(|b| b.is_ascii_digit()));

    // Count bigrams and unigrams
    let mut bigram_counts = vec![vec![0u64; n_states]; n_states];
    let mut unigram_counts = vec![0u64; n_states];
    let mut total_bigrams = 0u64;
    let mut total_unigrams = 0u64;

    for alignment in alignments.values() {
        let phone_indices: Vec<usize> = alignment
            .phonemes
            .iter()
            .filter_map(|seg| {
                let key = if is_collapsed {
                    strip_stress(&seg.phoneme)
                } else {
                    seg.phoneme.clone()
                };
                phone_to_idx.get(&key).copied()
            })
            .collect();

        for &idx in &phone_indices {
            unigram_counts[idx] += 1;
            total_unigrams += 1;
        }

        for pair in phone_indices.windows(2) {
            bigram_counts[pair[0]][pair[1]] += 1;
            total_bigrams += 1;
        }
    }

    // Compute log probabilities with add-1 (Laplace) smoothing
    let mut log_probs = vec![vec![0.0f32; n_states]; n_states];
    for prev in 0..n_states {
        let row_total: u64 = bigram_counts[prev].iter().sum();
        let denominator = (row_total + n_states as u64) as f32; // Laplace smoothing
        for next in 0..n_states {
            let count = bigram_counts[prev][next] as f32 + 1.0; // add-1
            log_probs[prev][next] = (count / denominator).ln();
        }
    }

    // Unigram log probs
    let mut unigram_log_probs = vec![0.0f32; n_states];
    let uni_denom = (total_unigrams + n_states as u64) as f32;
    for s in 0..n_states {
        unigram_log_probs[s] = ((unigram_counts[s] as f32 + 1.0) / uni_denom).ln();
    }

    println!(
        "  ✓ Built bigram LM from {} alignments ({} bigrams, {} unigrams)",
        alignments.len(),
        total_bigrams,
        total_unigrams
    );

    // Show some statistics
    let mut top_bigrams: Vec<(usize, usize, u64)> = Vec::new();
    for prev in 0..n_states {
        for next in 0..n_states {
            if bigram_counts[prev][next] > 0 {
                top_bigrams.push((prev, next, bigram_counts[prev][next]));
            }
        }
    }
    top_bigrams.sort_by(|a, b| b.2.cmp(&a.2));
    println!("    Top 5 bigrams:");
    for &(prev, next, count) in top_bigrams.iter().take(5) {
        let pct = count as f32 / total_bigrams as f32 * 100.0;
        println!(
            "      {} -> {}: {} ({:.1}%)",
            phonemes[prev], phonemes[next], count, pct
        );
    }

    Ok(BigramLM {
        log_probs,
        unigram_log_probs,
    })
}

/// Viterbi decoder with bigram language model
/// Combines base transition penalty with bigram LM modulation:
///   transition = -penalty + lm_weight * (bigram_log_prob - uniform_log_prob)
/// At lm_weight=0, this is equivalent to uniform Viterbi with the given penalty.
/// At lm_weight>0, common transitions get less penalty, rare get more.
fn viterbi_decode_bigram(
    all_logits: &[Vec<f32>],
    bigram_lm: &BigramLM,
    lm_weight: f32,
    penalty: f32,
    insertion_penalty: f32,
    phonemes: &[String],
) -> Vec<String> {
    let n_frames = all_logits.len();
    let n_states = phonemes.len();
    if n_frames == 0 || n_states == 0 {
        return Vec::new();
    }

    // Uniform log-prob for transitions (excluding self-loop)
    let uniform_log_prob = (1.0 / (n_states - 1) as f32).ln();

    let mut score = vec![vec![f32::NEG_INFINITY; n_states]; n_frames];
    let mut backptr = vec![vec![0usize; n_states]; n_frames];

    // Initialize first frame
    for s in 0..n_states {
        score[0][s] = all_logits[0][s];
    }

    // Forward pass
    for t in 1..n_frames {
        for s in 0..n_states {
            let emission = all_logits[t][s];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0;

            for prev_s in 0..n_states {
                let transition = if prev_s == s {
                    0.0 // self-loop: free
                } else {
                    // Base penalty + LM modulation + insertion penalty
                    -penalty - insertion_penalty
                        + lm_weight * (bigram_lm.log_probs[prev_s][s] - uniform_log_prob)
                };
                let candidate = score[t - 1][prev_s] + transition;
                if candidate > best_score {
                    best_score = candidate;
                    best_prev = prev_s;
                }
            }

            score[t][s] = best_score + emission;
            backptr[t][s] = best_prev;
        }
    }

    // Backtrace
    let mut best_final = 0;
    let mut best_final_score = f32::NEG_INFINITY;
    for s in 0..n_states {
        if score[n_frames - 1][s] > best_final_score {
            best_final_score = score[n_frames - 1][s];
            best_final = s;
        }
    }

    let mut path = vec![0usize; n_frames];
    path[n_frames - 1] = best_final;
    for t in (1..n_frames).rev() {
        path[t - 1] = backptr[t][path[t]];
    }

    // Deduplicate consecutive same phonemes
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

/// Collapse stress variants: maps each phoneme to its stress-stripped base
/// Returns (collapsed_phonemes, mapping) where `mapping[original_idx]` = collapsed_idx
fn build_stress_collapse_map(phonemes: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut collapsed_phonemes: Vec<String> = Vec::new();
    let mut base_to_idx: HashMap<String, usize> = HashMap::new();
    let mut mapping = Vec::with_capacity(phonemes.len());

    for ph in phonemes {
        let base = strip_stress(ph);
        let collapsed_idx = if let Some(&idx) = base_to_idx.get(&base) {
            idx
        } else {
            let idx = collapsed_phonemes.len();
            collapsed_phonemes.push(base.clone());
            base_to_idx.insert(base, idx);
            idx
        };
        mapping.push(collapsed_idx);
    }

    (collapsed_phonemes, mapping)
}

/// Collapse logits by taking max over stress variants (preserves logit scale)
fn collapse_logits(
    all_logits: &[Vec<f32>],
    mapping: &[usize],
    n_collapsed: usize,
) -> Vec<Vec<f32>> {
    let mut collapsed = Vec::with_capacity(all_logits.len());
    for logits in all_logits {
        let mut c = vec![f32::NEG_INFINITY; n_collapsed];
        for (orig_idx, &logit) in logits.iter().enumerate() {
            let cidx = mapping[orig_idx];
            if logit > c[cidx] {
                c[cidx] = logit;
            }
        }
        collapsed.push(c);
    }
    collapsed
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("         DIRECT CLASSIFIER EVALUATION                            ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    // Load classifier
    println!("Loading classifier from {:?}...", cli.classifier);
    let mut classifier = match DirectClassifier::load(&cli.classifier) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "{} Failed to load classifier: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    // Override prior_scale if specified
    if cli.prior_scale >= 0.0 {
        classifier.prior_scale = cli.prior_scale;
    }

    // Normalize weights if requested
    if cli.normalize_weights {
        classifier.normalize_weights();
        println!("  ✓ Weight vectors normalized to unit norm");
    }

    println!(
        "  ✓ Loaded classifier with {} phonemes, reservoir={}, prior_scale={}",
        classifier.phonemes.len(),
        classifier.config.reservoir_size,
        classifier.prior_scale
    );

    // Build stress collapse map if requested
    let (collapsed_phonemes, collapse_mapping) = if cli.collapse_stress {
        let (cp, cm) = build_stress_collapse_map(&classifier.phonemes);
        println!(
            "  ✓ Stress collapsing: {} → {} phoneme classes",
            classifier.phonemes.len(),
            cp.len()
        );
        (Some(cp), Some(cm))
    } else {
        (None, None)
    };

    // Determine decode phonemes (collapsed or original)
    let decode_phonemes: &[String] = if let Some(ref cp) = collapsed_phonemes {
        cp
    } else {
        &classifier.phonemes
    };

    // Build bigram LM if alignment path provided (using decode phonemes)
    let bigram_lm = if let Some(ref bigram_path) = cli.bigram_alignments {
        println!("Building bigram LM from {:?}...", bigram_path);
        match build_bigram_lm(bigram_path, decode_phonemes) {
            Ok(lm) => Some(lm),
            Err(e) => {
                eprintln!(
                    "{} Failed to build bigram LM: {}",
                    style("ERROR:").red().bold(),
                    e
                );
                None
            }
        }
    } else {
        None
    };

    // Load dictionary
    let dictionary = if cli.dictionary.exists() {
        println!("Loading dictionary from {:?}...", cli.dictionary);
        match CmuDictionary::load(&cli.dictionary) {
            Ok(d) => {
                println!("  ✓ Loaded {} words", d.len());
                Some(d)
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load dictionary: {}", e);
                None
            }
        }
    } else {
        eprintln!(
            "{} Dictionary not found: {:?}",
            style("WARNING:").yellow().bold(),
            cli.dictionary
        );
        None
    };

    // Load test utterances
    println!("Scanning test directory {:?}...", cli.test_dir);
    let utterances = match scan_librispeech_dir(&cli.test_dir) {
        Ok(u) => u,
        Err(e) => {
            eprintln!(
                "{} Failed to scan test directory: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    let total_utterances = if cli.max_utterances > 0 {
        utterances.len().min(cli.max_utterances)
    } else {
        utterances.len()
    };

    println!(
        "  ✓ Found {} utterances, evaluating {}",
        utterances.len(),
        total_utterances
    );

    // Create audio frontend
    let audio_config = AudioConfig::default();
    let mut frontend = AudioFrontend::new(audio_config);

    // Create reservoir matching classifier config (unless skip_reservoir)
    let context_frames = classifier.config.context_frames;
    let n_mels = 40;
    let feature_dim = if cli.use_deltas { n_mels * 3 } else { n_mels };
    let input_size = feature_dim * context_frames;

    let mut reservoir = if !cli.skip_reservoir {
        let mut ltc_config = LtcConfig::default();
        ltc_config.hidden_size = classifier.config.reservoir_size;
        Some(LtcCell::new_reservoir(input_size, ltc_config, 0.9, 1.0))
    } else {
        None
    };

    let frame_hop = classifier.config.frame_duration;

    println!("\n  Eval config:");
    println!(
        "    Input: {} dims ({}mel × {}Δ × {}ctx)",
        input_size,
        n_mels,
        if cli.use_deltas { 3 } else { 1 },
        context_frames
    );
    println!("    Min duration: {} frames", cli.min_duration);
    println!("    Min confidence: {}", cli.min_confidence);
    if bigram_lm.is_some() {
        let penalty = if cli.viterbi_penalty > 0.0 {
            cli.viterbi_penalty
        } else {
            0.95
        };
        println!(
            "    Decoder: Viterbi + bigram LM (penalty={}, lm_weight={})",
            penalty, cli.lm_weight
        );
    } else if cli.viterbi_penalty > 0.0 {
        println!("    Decoder: Viterbi (penalty={})", cli.viterbi_penalty);
    } else {
        println!(
            "    Decoder: Greedy (min_dur={}, min_conf={})",
            cli.min_duration, cli.min_confidence
        );
    }

    // Run evaluation
    println!();
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                     RUNNING EVALUATION                           ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    let pb = ProgressBar::new(total_utterances as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut total_phoneme_result = EvalResult::default();
    let mut total_audio_seconds = 0.0f32;
    let mut total_inference_seconds = 0.0f32;
    let mut phoneme_counts: HashMap<String, usize> = HashMap::new();

    let text_to_phonemes = dictionary.as_ref().map(|d| TextToPhonemes::new(d.clone()));

    for (i, utterance) in utterances.iter().take(total_utterances).enumerate() {
        pb.set_message(
            utterance
                .audio_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );

        // Load audio
        let (audio, sample_rate) = match AudioFrontend::load_audio(&utterance.audio_path) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let audio_duration = audio.len() as f32 / sample_rate as f32;
        total_audio_seconds += audio_duration;

        let start = Instant::now();

        // Extract features
        let mut features = if cli.use_deltas {
            frontend.extract_features_with_deltas(&audio)
        } else {
            frontend.extract_features(&audio)
        };

        // Per-utterance cepstral mean (and optionally variance) normalization
        if (cli.cmn || cli.cmvn) && !features.is_empty() {
            let n_dims = features[0].len();
            let n_frames = features.len() as f32;
            let mut mean = vec![0.0f32; n_dims];
            for frame in &features {
                for (i, &v) in frame.iter().enumerate() {
                    mean[i] += v;
                }
            }
            for m in &mut mean {
                *m /= n_frames;
            }
            for frame in &mut features {
                for (i, v) in frame.iter_mut().enumerate() {
                    *v -= mean[i];
                }
            }
            if cli.cmvn {
                let mut var = vec![0.0f32; n_dims];
                for frame in &features {
                    for (i, &v) in frame.iter().enumerate() {
                        var[i] += v * v;
                    }
                }
                for v in &mut var {
                    *v = (*v / n_frames).sqrt().max(1e-6);
                }
                for frame in &mut features {
                    for (i, v) in frame.iter_mut().enumerate() {
                        *v /= var[i];
                    }
                }
            }
        }

        if let Some(ref mut res) = reservoir {
            res.reset();
        }

        // Compute raw logits for all frames
        let n_phonemes = classifier.phonemes.len();
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(features.len());

        for frame_idx in 0..features.len() {
            let input = if context_frames > 1 {
                stack_frames(&features, frame_idx, context_frames)
            } else {
                features[frame_idx].clone()
            };

            let logits = if let Some(ref mut res) = reservoir {
                res.forward(&input, frame_hop);
                classifier.compute_logits(res.state())
            } else {
                classifier.compute_logits(&input)
            };
            all_logits.push(logits);
        }

        // Smooth logits if requested
        if cli.smooth_window > 1 {
            let half = cli.smooth_window / 2;
            let mut smoothed = Vec::with_capacity(all_logits.len());
            for t in 0..all_logits.len() {
                let start_t = t.saturating_sub(half);
                let end_t = (t + half + 1).min(all_logits.len());
                let count = (end_t - start_t) as f32;
                let mut avg = vec![0.0f32; n_phonemes];
                for frame in &all_logits[start_t..end_t] {
                    for (i, v) in frame.iter().enumerate() {
                        avg[i] += v;
                    }
                }
                for v in &mut avg {
                    *v /= count;
                }
                smoothed.push(avg);
            }
            all_logits = smoothed;
        }

        // Apply prior correction (subtract log-priors to remove frequency bias)
        if cli.prior_scale > 0.0 && !classifier.log_priors.is_empty() {
            for logits in &mut all_logits {
                for (i, v) in logits.iter_mut().enumerate() {
                    if i < classifier.log_priors.len() {
                        *v -= cli.prior_scale * classifier.log_priors[i];
                    }
                }
            }
        }

        // Apply acoustic model scaling
        if cli.am_scale != 1.0 {
            for logits in &mut all_logits {
                for v in logits.iter_mut() {
                    *v *= cli.am_scale;
                }
            }
        }

        // Collapse stress variants if requested
        let decode_logits: Vec<Vec<f32>>;
        let logits_ref =
            if let (Some(cm), Some(cp)) = (&collapse_mapping, &collapsed_phonemes) {
                decode_logits = collapse_logits(&all_logits, cm, cp.len());
                &decode_logits
            } else {
                &all_logits
            };

        // Decode phoneme sequence
        let hyp_phonemes = if let Some(ref lm) = bigram_lm {
            // Viterbi + bigram LM (penalty defaults to 0.95 if not specified)
            let penalty = if cli.viterbi_penalty > 0.0 {
                cli.viterbi_penalty
            } else {
                0.95
            };
            viterbi_decode_bigram(
                logits_ref,
                lm,
                cli.lm_weight,
                penalty,
                cli.insertion_penalty,
                decode_phonemes,
            )
        } else if cli.viterbi_penalty > 0.0 {
            // Viterbi decoding with uniform transition penalty
            viterbi_decode(logits_ref, cli.viterbi_penalty, decode_phonemes)
        } else {
            // Greedy decoding with min_duration filtering
            let mut frame_predictions: Vec<(String, f32)> = Vec::with_capacity(logits_ref.len());
            for logits in logits_ref {
                let mut best_idx = 0;
                let mut best_logit = f32::NEG_INFINITY;
                for (i, &l) in logits.iter().enumerate() {
                    if l > best_logit {
                        best_logit = l;
                        best_idx = i;
                    }
                }
                let max_l = best_logit;
                let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
                let confidence = 1.0 / exp_sum;
                frame_predictions.push((decode_phonemes[best_idx].clone(), confidence));
            }
            decode_sequence(&frame_predictions, cli.min_confidence, cli.min_duration)
        };

        let inference_time = start.elapsed().as_secs_f32();
        total_inference_seconds += inference_time;

        // Count predicted phonemes
        for ph in &hyp_phonemes {
            *phoneme_counts.entry(ph.clone()).or_insert(0) += 1;
        }

        // Get reference phonemes from transcript (strip SIL tokens)
        let ref_phonemes: Vec<String> = if let Some(ref t2p) = text_to_phonemes {
            t2p.convert(&utterance.transcript)
                .into_iter()
                .filter(|p| p != "SIL" && p != "SPN" && p != "NSN")
                .collect()
        } else {
            Vec::new()
        };

        // Compute PER (strip stress markers for fairer comparison)
        if !ref_phonemes.is_empty() {
            let ref_stripped: Vec<String> = ref_phonemes.iter().map(|s| strip_stress(s)).collect();
            let hyp_stripped: Vec<String> = hyp_phonemes.iter().map(|s| strip_stress(s)).collect();
            let ref_strs: Vec<&str> = ref_stripped.iter().map(|s| s.as_str()).collect();
            let hyp_strs: Vec<&str> = hyp_stripped.iter().map(|s| s.as_str()).collect();

            let alignment = symthaea_stt::eval::levenshtein_align(&ref_strs, &hyp_strs);
            let result = symthaea_stt::eval::eval_from_alignment(&alignment);
            total_phoneme_result.merge(&result);
        }

        if cli.verbose && i < 5 {
            println!("\n  [{}] REF: {}", i, utterance.transcript);
            println!("       HYP: {}", hyp_phonemes.join(" "));
            println!("       REF PH: {}", ref_phonemes.join(" "));
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate metrics
    let per = total_phoneme_result.error_rate() * 100.0;
    let rtf = if total_audio_seconds > 0.0 {
        total_inference_seconds / total_audio_seconds
    } else {
        0.0
    };

    // Print results
    println!();
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                     EVALUATION RESULTS                           ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    println!("{}:", style("Summary").bold());
    println!("  Utterances evaluated:  {}", total_utterances);
    println!("  Total audio:           {:.1}s", total_audio_seconds);
    println!("  Total inference:       {:.1}s", total_inference_seconds);
    println!("  Real-time factor:      {:.2}x", rtf);
    println!();

    println!("{}:", style("Phoneme Error Rate (PER)").bold());
    println!(
        "  Reference phonemes: {}",
        total_phoneme_result.reference_length
    );
    println!(
        "  Hypothesis phonemes:{}",
        total_phoneme_result.hypothesis_length
    );
    println!("  Correct:            {}", total_phoneme_result.correct);
    println!(
        "  Substitutions:      {}",
        total_phoneme_result.substitutions
    );
    println!("  Insertions:         {}", total_phoneme_result.insertions);
    println!("  Deletions:          {}", total_phoneme_result.deletions);
    println!("  {}:               {:.1}%", style("PER").bold(), per);
    println!();

    // Show top predicted phonemes
    if cli.verbose {
        let mut counts: Vec<_> = phoneme_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        println!("  Top predicted phonemes:");
        for (ph, count) in counts.iter().take(10) {
            println!("    {}: {}", ph, count);
        }
        println!();
    }

    // Assessment
    let status = if per < 30.0 {
        (
            style("EXCELLENT").green().bold(),
            "System is learning phonemes!",
        )
    } else if per < 50.0 {
        (
            style("GOOD").green().bold(),
            "Meaningful phoneme discrimination",
        )
    } else if per < 70.0 {
        (
            style("PROGRESSING").yellow().bold(),
            "Better than random, improving",
        )
    } else {
        (style("NEEDS WORK").red().bold(), "Still needs improvement")
    };

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!(
        "║           DIRECT CLASSIFIER: {}                   ║",
        status.0
    );
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  PER: {:.1}% - {}║", per, status.1);
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

/// Scan LibriSpeech directory structure
fn scan_librispeech_dir(base_dir: &Path) -> Result<Vec<Utterance>, String> {
    let mut utterances = Vec::new();

    let entries = fs::read_dir(base_dir).map_err(|e| format!("Failed to read directory: {}", e))?;

    for speaker_entry in entries.flatten() {
        if !speaker_entry.path().is_dir() {
            continue;
        }

        let speaker_id = speaker_entry.file_name().to_string_lossy().to_string();

        let chapter_entries = fs::read_dir(speaker_entry.path())
            .map_err(|e| format!("Failed to read speaker dir: {}", e))?;

        for chapter_entry in chapter_entries.flatten() {
            if !chapter_entry.path().is_dir() {
                continue;
            }

            let chapter_id = chapter_entry.file_name().to_string_lossy().to_string();
            let chapter_path = chapter_entry.path();

            let trans_file = chapter_path.join(format!("{}-{}.trans.txt", speaker_id, chapter_id));
            if !trans_file.exists() {
                continue;
            }

            let file =
                File::open(&trans_file).map_err(|e| format!("Failed to open transcript: {}", e))?;
            let reader = BufReader::new(file);

            for line in reader.lines().flatten() {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() != 2 {
                    continue;
                }

                let utterance_id = parts[0].to_string();
                let transcript = parts[1].to_uppercase();

                let audio_path = chapter_path.join(format!("{}.flac", utterance_id));
                if !audio_path.exists() {
                    continue;
                }

                utterances.push(Utterance {
                    audio_path,
                    transcript,
                    speaker_id: speaker_id.clone(),
                    chapter_id: chapter_id.clone(),
                    utterance_id,
                });
            }
        }
    }

    Ok(utterances)
}
