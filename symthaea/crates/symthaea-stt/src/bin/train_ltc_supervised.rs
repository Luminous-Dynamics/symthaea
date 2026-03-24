// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supervised LTC Training — BPTT with Contrastive Phoneme Loss
//!
//! Trains the LTC cell weights to produce phoneme-discriminative hidden states,
//! fixing the core STT bottleneck (BLAKE3 random projection loses phoneme structure).
//!
//! Usage:
//!   cargo run --release -p symthaea-stt --bin train-ltc-supervised -- \
//!     --alignments data/alignments/data/dev_clean-00000-of-00001.parquet \
//!     --audio-dir data/librispeech/LibriSpeech/dev-clean \
//!     --output data/models/v2/ltc_supervised.bin \
//!     --epochs 10 --lr 0.001 --hidden-size 128

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_stt::{
    id_to_audio_path, load_alignments, AudioConfig, AudioFrontend, LtcConfig,
    ltc::LtcCell,
    ltc_training::{PhonemeCentroids, LtcTrainingConfig, train_utterance},
};

#[derive(Parser)]
#[command(
    name = "train-ltc-supervised",
    about = "Train LTC weights via BPTT with contrastive phoneme loss",
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
    #[arg(short, long, default_value = "data/models/v2/ltc_supervised.bin")]
    output: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// LTC hidden size
    #[arg(long, default_value = "128")]
    hidden_size: usize,

    /// BPTT truncation length (frames)
    #[arg(long, default_value = "50")]
    bptt_length: usize,

    /// Contrastive loss temperature
    #[arg(long, default_value = "0.1")]
    temperature: f32,

    /// Gradient clipping norm
    #[arg(long, default_value = "5.0")]
    grad_clip: f32,

    /// Maximum utterances (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Frame hop in seconds
    #[arg(long, default_value = "0.010")]
    frame_hop: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    println!("{}", style("╔══════════════════════════════════════════════════╗").cyan());
    println!("{}", style("║   SUPERVISED LTC TRAINING — BPTT + CONTRASTIVE  ║").cyan());
    println!("{}", style("╚══════════════════════════════════════════════════╝").cyan());
    println!();

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

    // Create LTC cell
    let ltc_config = LtcConfig {
        hidden_size: cli.hidden_size,
        tau_init: 0.020,
        adaptive_tau: false, // We'll train tau via BPTT instead
        ..LtcConfig::default()
    };
    let mel_dim = 40; // Standard mel channels
    let mut ltc = LtcCell::new_reservoir(mel_dim, ltc_config, 0.9, 1.0);

    // Create phoneme centroids
    let centroids = PhonemeCentroids::new(&phoneme_labels, cli.hidden_size, cli.temperature);

    // Training config
    let train_config = LtcTrainingConfig {
        learning_rate: cli.lr,
        grad_clip: cli.grad_clip,
        bptt_length: cli.bptt_length,
        temperature: cli.temperature,
        epochs: cli.epochs,
        report_interval: 100,
        frame_hop: cli.frame_hop,
    };

    // Audio frontend
    let mut frontend = AudioFrontend::new(AudioConfig::default());

    // Prepare utterance IDs
    let mut utterance_ids: Vec<String> = alignments.keys().cloned().collect();
    utterance_ids.sort();
    if cli.max_utterances > 0 && utterance_ids.len() > cli.max_utterances {
        utterance_ids.truncate(cli.max_utterances);
    }

    println!();
    println!("  Training config:");
    println!("    Hidden size:  {}", cli.hidden_size);
    println!("    Epochs:       {}", cli.epochs);
    println!("    LR:           {}", cli.lr);
    println!("    BPTT length:  {}", cli.bptt_length);
    println!("    Temperature:  {}", cli.temperature);
    println!("    Utterances:   {}", utterance_ids.len());
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
        pb.set_style(ProgressStyle::default_bar()
            .template("  [epoch {msg}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .unwrap());
        pb.set_message(format!("{}/{}", epoch + 1, cli.epochs));

        for utt_id in &utterance_ids {
            let alignment = &alignments[utt_id];

            // Load audio
            let audio_path = match id_to_audio_path(utt_id, &cli.audio_dir) {
                Some(p) => p,
                None => { pb.inc(1); continue; }
            };
            let (audio, _sample_rate) = match AudioFrontend::load_wav(&audio_path) {
                Ok(a) => a,
                Err(_) => { pb.inc(1); continue; }
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

            // Train on this utterance
            let (loss, correct, total) = train_utterance(
                &mut ltc,
                &centroids,
                &mel_frames,
                &frame_labels,
                &train_config,
            );

            epoch_loss += loss;
            epoch_correct += correct;
            epoch_total += total;
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
            "  [epoch {}/{}] loss={:.4}  accuracy={:.1}% ({}/{})  elapsed={:.0}s",
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
            // Save best model
            // For now, save the LTC state + centroids
            let model = TrainedLtcModel {
                ltc_state: bincode::serialize(&ltc).unwrap_or_default(),
                centroid_labels: phoneme_labels.clone(),
                hidden_size: cli.hidden_size,
                best_accuracy,
                epoch: epoch + 1,
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
    println!("{}", style("═══════════════════════════════════════════════════").green());
    println!("{}", style("           SUPERVISED LTC TRAINING COMPLETE        ").green());
    println!("{}", style("═══════════════════════════════════════════════════").green());
    println!("  Best accuracy: {:.1}%", best_accuracy * 100.0);
    println!("  Model saved:   {:?}", cli.output);
    println!("  Total time:    {:.0}s", start.elapsed().as_secs_f32());
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TrainedLtcModel {
    ltc_state: Vec<u8>,
    centroid_labels: Vec<String>,
    hidden_size: usize,
    best_accuracy: f32,
    epoch: usize,
}
