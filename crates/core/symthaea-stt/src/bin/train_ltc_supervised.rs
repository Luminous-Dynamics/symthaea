// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supervised LTC Training — HdcLtcUnifiedNeuron Architecture
//!
//! Trains a 16,384D unified HDC-LTC neuron for phoneme recognition using
//! the same architecture as Broca's language generation pipeline.
//!
//! Key difference from basic LTC: the neuron STATE is a 16,384D hypervector
//! that evolves through HDC bind/bundle operations with SIMD acceleration.
//!
//! Usage:
//!   cargo run --release -p symthaea-stt --bin train-ltc-supervised -- \
//!     --alignments data/alignments/data/dev_clean-00000-of-00001.parquet \
//!     --audio-dir data/librispeech/LibriSpeech/dev-clean \
//!     --output data/models/v2/unified_ltc.bin \
//!     --epochs 10 --lr 0.01

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

use symthaea_core::genesis::GenesisSeed;
use symthaea_stt::{
    AudioConfig, AudioFrontend, audio_hdc_encoder::AudioHdcEncoder, id_to_audio_path,
    load_alignments, phoneme_hdc_ltc::PhonemeHdcLtc,
};

#[derive(Parser)]
#[command(
    name = "train-ltc-supervised",
    about = "Train unified HDC-LTC neuron for phoneme recognition (16,384D)",
    version,
    author
)]
struct Cli {
    /// Parquet file with phoneme alignments
    #[arg(short, long)]
    alignments: PathBuf,

    /// LibriSpeech audio directory
    #[arg(short = 'd', long)]
    audio_dir: PathBuf,

    /// Output model file
    #[arg(short, long, default_value = "data/models/v2/unified_ltc.bin")]
    output: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value = "0.01")]
    lr: f32,

    /// Base time constant (seconds). 20ms matches phoneme duration.
    #[arg(long, default_value = "0.020")]
    tau_base: f32,

    /// Maximum utterances (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Frame hop in seconds
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Genesis seed phrase
    #[arg(long, default_value = "unified-stt-v1")]
    genesis: String,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("╔══════════════════════════════════════════════════════════╗").cyan()
    );
    println!(
        "{}",
        style("║   UNIFIED HDC-LTC TRAINING — 16,384D Phoneme Neuron     ║").cyan()
    );
    println!(
        "{}",
        style("╚══════════════════════════════════════════════════════════╝").cyan()
    );
    println!();

    let genesis = GenesisSeed::from_phrase(&cli.genesis);

    // Load alignments
    println!("  Loading alignments from {:?}...", cli.alignments);
    let alignments = match load_alignments(&cli.alignments) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load alignments: {e}");
            std::process::exit(1);
        }
    };
    println!("    {} utterances with alignments", alignments.len());

    // Collect all phoneme labels
    let mut phoneme_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for alignment in alignments.values() {
        for seg in &alignment.phonemes {
            phoneme_set.insert(seg.phoneme.clone());
        }
    }
    let phoneme_labels: Vec<String> = phoneme_set.into_iter().collect();
    println!("    {} unique phonemes", phoneme_labels.len());

    // Create audio HDC encoder (mel → 16,384D)
    let mel_dim = 40;
    let encoder = AudioHdcEncoder::new(mel_dim, &genesis);

    // Create unified phoneme HDC-LTC neuron (16,384D state)
    let mut phoneme_ltc = PhonemeHdcLtc::new(&genesis, &phoneme_labels, cli.tau_base);

    // Audio frontend for mel extraction
    let mut frontend = AudioFrontend::new(AudioConfig::default());

    // Prepare utterance IDs
    let mut utterance_ids: Vec<String> = alignments.keys().cloned().collect();
    utterance_ids.sort();
    if cli.max_utterances > 0 && utterance_ids.len() > cli.max_utterances {
        utterance_ids.truncate(cli.max_utterances);
    }

    println!();
    println!("  Architecture: HdcLtcUnifiedNetwork (3×8 = 24 neurons, 16,384D)");
    println!("  Phonemes:     {}", phoneme_labels.len());
    println!("  Neurons:      {}", phoneme_ltc.num_neurons());
    println!("  Epochs:       {}", cli.epochs);
    println!("  LR:           {}", cli.lr);
    println!("  Tau base:     {}ms", cli.tau_base * 1000.0);
    println!("  Utterances:   {}", utterance_ids.len());
    println!();

    // Training loop
    let start = Instant::now();
    let mut best_accuracy = 0.0f32;

    for epoch in 0..cli.epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;
        let mut processed = 0usize;

        let pb = ProgressBar::new(utterance_ids.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [epoch {msg}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap(),
        );
        pb.set_message(format!("{}/{}", epoch + 1, cli.epochs));

        for utt_id in &utterance_ids {
            let alignment = &alignments[utt_id];

            // Load audio (FLAC or WAV)
            let audio_path = match id_to_audio_path(utt_id, &cli.audio_dir) {
                Some(p) => p,
                None => {
                    pb.inc(1);
                    continue;
                }
            };
            let (audio, _sample_rate) = match AudioFrontend::load_audio(&audio_path) {
                Ok(a) => a,
                Err(_) => {
                    pb.inc(1);
                    continue;
                }
            };

            // Extract mel frames
            let mel_frames = frontend.extract_features(&audio);
            if mel_frames.is_empty() {
                pb.inc(1);
                continue;
            }

            // Convert alignment to per-frame labels
            let n_frames = mel_frames.len();
            let mut frame_labels = vec!["SIL".to_string(); n_frames];
            for seg in &alignment.phonemes {
                let start_frame = (seg.start / cli.frame_hop as f64) as usize;
                let end_frame = (seg.end / cli.frame_hop as f64) as usize;
                for f in start_frame..end_frame.min(n_frames) {
                    frame_labels[f] = seg.phoneme.clone();
                }
            }

            // Reset neuron state for each utterance
            phoneme_ltc.reset();

            // Train on each frame
            let mut utt_sim = 0.0f32;
            for (frame_idx, mel) in mel_frames.iter().take(n_frames).enumerate() {
                // Encode mel → 16,384D HV
                let input_hv = encoder.encode_frame(mel);

                // Train step: evolve + contrastive update
                let sim = phoneme_ltc.train_step(
                    &input_hv,
                    &frame_labels[frame_idx],
                    cli.frame_hop,
                    cli.lr,
                );
                utt_sim += sim;

                // Check accuracy
                let (pred, _) = phoneme_ltc.decode();
                if pred == frame_labels[frame_idx] {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }

            epoch_loss += utt_sim / n_frames.max(1) as f32;
            processed += 1;
            pb.inc(1);
        }

        pb.finish_and_clear();

        let accuracy = if epoch_total > 0 {
            epoch_correct as f32 / epoch_total as f32
        } else {
            0.0
        };
        let avg_loss = if processed > 0 {
            epoch_loss / processed as f32
        } else {
            0.0
        };

        println!(
            "  [epoch {}/{}] loss={:.6}  accuracy={:.1}% ({}/{})  elapsed={:.0}s",
            epoch + 1,
            cli.epochs,
            avg_loss,
            accuracy * 100.0,
            epoch_correct,
            epoch_total,
            start.elapsed().as_secs_f32()
        );

        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            // Save model (genesis phrase is sufficient to reconstruct network)
            let model = UnifiedLtcModel {
                phoneme_labels: phoneme_labels.clone(),
                best_accuracy,
                epoch: epoch + 1,
                genesis_phrase: cli.genesis.clone(),
                num_neurons: phoneme_ltc.num_neurons(),
            };
            if let Ok(data) = bincode::serialize(&model) {
                if let Err(e) = std::fs::write(&cli.output, &data) {
                    eprintln!("  Warning: failed to save model: {e}");
                } else {
                    println!("    ✓ Best model saved ({:.1}% accuracy)", accuracy * 100.0);
                }
            }
        }
    }

    println!();
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").green()
    );
    println!(
        "{}",
        style("          UNIFIED HDC-LTC TRAINING COMPLETE                ").green()
    );
    println!(
        "{}",
        style("═══════════════════════════════════════════════════════════").green()
    );
    println!("  Best accuracy: {:.1}%", best_accuracy * 100.0);
    println!("  Model saved:   {:?}", cli.output);
    println!("  Total time:    {:.0}s", start.elapsed().as_secs_f32());
}

#[derive(serde::Serialize, serde::Deserialize)]
struct UnifiedLtcModel {
    phoneme_labels: Vec<String>,
    best_accuracy: f32,
    epoch: usize,
    genesis_phrase: String,
    num_neurons: usize,
}
