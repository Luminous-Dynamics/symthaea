// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluate Liquid Projection Encoder
//!
//! Computes Phoneme Error Rate (PER) using the trained encoder.

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea_stt::{
    AudioConfig, AudioFrontend, CmuDictionary, EvalResult, LiquidProjection, LtcCell, LtcConfig,
    TextToPhonemes,
};

#[derive(Parser)]
#[command(
    name = "eval-reservoir",
    about = "Evaluate Liquid Projection encoder on LibriSpeech",
    version,
    author
)]
struct Cli {
    /// Path to trained encoder file
    #[arg(short, long, default_value = "data/models/liquid_projection.bin")]
    encoder: PathBuf,

    /// Path to test dataset directory (LibriSpeech format)
    #[arg(short, long, default_value = "data/librispeech/LibriSpeech/dev-clean")]
    test_dir: PathBuf,

    /// Path to CMU dictionary
    #[arg(short, long, default_value = "data/dict/cmudict.dict")]
    dictionary: PathBuf,

    /// Maximum number of utterances to evaluate (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Frame hop duration (seconds)
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Voting window size (frames)
    #[arg(long, default_value = "8")]
    vote_window: usize,

    /// Minimum similarity threshold for decoding
    #[arg(long, default_value = "0.3")]
    min_similarity: f32,

    /// Number of context frames to stack (1 = no stacking)
    #[arg(long, default_value = "1")]
    context_frames: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Stack multiple frames together for temporal context
fn stack_frames(features: &[Vec<f32>], center: usize, context_frames: usize) -> Vec<f32> {
    let n_mels = if features.is_empty() {
        40
    } else {
        features[0].len()
    };
    let half = context_frames / 2;
    let mut stacked = Vec::with_capacity(n_mels * context_frames);

    for offset in 0..context_frames {
        let idx = (center + offset).saturating_sub(half);
        let idx = idx.min(features.len().saturating_sub(1));
        stacked.extend_from_slice(&features[idx]);
    }

    stacked
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

/// Decode phoneme sequence using voting window + minimum duration for temporal smoothing
fn decode_with_voting(
    frame_predictions: &[(String, f32)],
    window_size: usize,
    min_similarity: f32,
) -> Vec<String> {
    use std::collections::HashMap;

    if frame_predictions.is_empty() {
        return Vec::new();
    }

    let half_window = window_size / 2;
    let min_duration = window_size; // Require phoneme to dominate for this many frames

    // First pass: compute smoothed predictions for each frame
    let mut smoothed: Vec<String> = Vec::with_capacity(frame_predictions.len());

    for center in 0..frame_predictions.len() {
        // Define window bounds
        let start = center.saturating_sub(half_window);
        let end = (center + half_window + 1).min(frame_predictions.len());

        // Count votes in window (weighted by similarity)
        let mut votes: HashMap<&str, f32> = HashMap::new();
        for (phoneme, sim) in &frame_predictions[start..end] {
            if *sim >= min_similarity {
                *votes.entry(phoneme.as_str()).or_insert(0.0) += sim;
            }
        }

        // Find winner
        let winner = votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, _)| p.to_string())
            .unwrap_or_default();

        smoothed.push(winner);
    }

    // Second pass: segment into runs and filter by minimum duration
    let mut result: Vec<String> = Vec::new();
    let mut run_start = 0;

    while run_start < smoothed.len() {
        let current = &smoothed[run_start];
        let mut run_end = run_start + 1;

        // Find end of this run (consecutive same phoneme)
        while run_end < smoothed.len() && &smoothed[run_end] == current {
            run_end += 1;
        }

        let run_len = run_end - run_start;

        // Only emit if run is long enough and different from last emission
        if run_len >= min_duration && !current.is_empty() && result.last() != Some(current) {
            result.push(current.clone());
        }

        run_start = run_end;
    }

    result
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("         LIQUID PROJECTION EVALUATION                             ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    // Load encoder
    println!("Loading encoder from {:?}...", cli.encoder);
    let encoder = match LiquidProjection::load(&cli.encoder) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "{} Failed to load encoder: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };
    println!("  ✓ Loaded encoder with {} phonemes", encoder.targets.len());

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

    // Create reservoir matching the encoder's config
    let mut ltc_config = LtcConfig::default();
    ltc_config.hidden_size = encoder.config.reservoir_size;
    let n_mels = 40;
    let input_size = n_mels * cli.context_frames;
    let mut reservoir = LtcCell::new(input_size, ltc_config);

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

        // Extract features and run through reservoir
        let features = frontend.extract_features(&audio);
        reservoir.reset();

        // Collect all frame predictions first
        let mut frame_predictions: Vec<(String, f32)> = Vec::with_capacity(features.len());

        for i in 0..features.len() {
            // Stack frames for temporal context
            let input = if cli.context_frames > 1 {
                stack_frames(&features, i, cli.context_frames)
            } else {
                features[i].clone()
            };
            reservoir.forward(&input, cli.frame_hop);
            let state = reservoir.state();

            // Decode phoneme from reservoir state
            if let Some((phoneme, sim)) = encoder.decode(state) {
                frame_predictions.push((phoneme, sim));
            }
        }

        // Apply voting window to smooth predictions
        let hyp_phonemes =
            decode_with_voting(&frame_predictions, cli.vote_window, cli.min_similarity);

        let inference_time = start.elapsed().as_secs_f32();
        total_inference_seconds += inference_time;

        // Get reference phonemes from transcript
        let ref_phonemes = if let Some(ref t2p) = text_to_phonemes {
            t2p.convert(&utterance.transcript)
        } else {
            Vec::new()
        };

        // Compute phoneme error rate
        if !ref_phonemes.is_empty() {
            let ref_strs: Vec<&str> = ref_phonemes.iter().map(|s| s.as_str()).collect();
            let hyp_strs: Vec<&str> = hyp_phonemes.iter().map(|s| s.as_str()).collect();

            let alignment = symthaea_stt::eval::levenshtein_align(&ref_strs, &hyp_strs);
            let result = symthaea_stt::eval::eval_from_alignment(&alignment);
            total_phoneme_result.merge(&result);
        }

        if cli.verbose && i < 5 {
            println!("\n  [{}] REF: {}", i, utterance.transcript);
            println!("       HYP PH: {}", hyp_phonemes.join(" "));
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate metrics
    let per = total_phoneme_result.error_rate() * 100.0;
    let rtf = total_inference_seconds / total_audio_seconds;

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

    // Assessment
    let status = if per < 30.0 {
        (style("PASS").green().bold(), "System is learning phonemes!")
    } else if per < 60.0 {
        (
            style("PROGRESSING").yellow().bold(),
            "Better than random, needs improvement",
        )
    } else {
        (
            style("NEEDS WORK").red().bold(),
            "Still struggling with phoneme discrimination",
        )
    };

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!(
        "║           LIQUID PROJECTION: {}                    ║",
        status.0
    );
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  PER: {:.1}% - {}║", per, status.1);
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

/// Scan LibriSpeech directory structure
fn scan_librispeech_dir(base_dir: &Path) -> Result<Vec<Utterance>, String> {
    let mut utterances = Vec::new();

    // LibriSpeech structure: base/speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
    // Transcript: base/speaker_id/chapter_id/speaker_id-chapter_id.trans.txt

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

            // Find transcript file
            let trans_file = chapter_path.join(format!("{}-{}.trans.txt", speaker_id, chapter_id));
            if !trans_file.exists() {
                continue;
            }

            // Read transcripts
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
