// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mirror Test: Compare Training vs Inference HDC projections
//!
//! This forensic test verifies that training and inference produce
//! consistent HDC vectors from the same audio input.

use clap::Parser;
use std::path::PathBuf;

use symthaea_stt::{AudioFrontend, AudioProjector, PhonemeDecoder, TrainedPrototypes};

#[derive(Parser)]
#[command(name = "symthaea-mirror")]
#[command(about = "Mirror Test: Compare HDC projections between training and inference")]
struct Cli {
    /// Audio file to test
    #[arg(short, long)]
    audio: PathBuf,

    /// Prototypes file (to compare against)
    #[arg(
        short,
        long,
        default_value = "data/models/dev-clean_adaptive_prototypes.bin"
    )]
    prototypes: PathBuf,

    /// Number of frames to compare
    #[arg(short, long, default_value = "10")]
    num_frames: usize,
}

fn main() {
    let cli = Cli::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              SYMTHAEA MIRROR TEST                             ║");
    println!("║         Training vs Inference Pipeline Comparison             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Load audio
    println!("Loading audio: {:?}", cli.audio);
    let (audio, sample_rate) = match AudioFrontend::load_audio(&cli.audio) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("ERROR: Failed to load audio: {}", e);
            std::process::exit(1);
        }
    };
    let duration_sec = audio.len() as f32 / sample_rate as f32;
    println!(
        "  Duration: {:.2}s, Sample rate: {} Hz, {} samples",
        duration_sec,
        sample_rate,
        audio.len()
    );
    println!();

    // Load prototypes to compare against
    println!("Loading prototypes: {:?}", cli.prototypes);
    let prototypes = match TrainedPrototypes::load(&cli.prototypes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Failed to load prototypes: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Loaded {} phoneme prototypes", prototypes.len());
    println!();

    // Project audio through inference pipeline
    println!("═══════════════════════════════════════════════════════════════");
    println!("PROJECTING AUDIO (Inference Pipeline)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut projector = AudioProjector::default_config();
    projector.reset();
    let hvs = projector.project(&audio);

    println!("  Generated {} HV frames ({}ms per frame)", hvs.len(), 10);
    println!();

    // Analyze HV properties (using individual frames like training/inference)
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "FRAME HV ANALYSIS (first {} frames)",
        cli.num_frames.min(hvs.len())
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Build decoder to query prototypes
    let mut decoder = PhonemeDecoder::new();
    decoder.load_prototypes(&prototypes.as_pairs());

    // Analyze first N frames
    let mut total_max_sim = 0.0;
    let mut total_min_sim = 0.0;
    let mut total_range = 0.0;

    for (i, hv) in hvs.iter().take(cli.num_frames).enumerate() {
        let scores = decoder.decode_all_scores(hv);

        let (top_phoneme, top_score) = match scores.first() {
            Some(v) => v,
            None => continue,
        };
        let (_, min_score) = match scores.last() {
            Some(v) => v,
            None => continue,
        };
        let range = top_score - min_score;

        total_max_sim += top_score;
        total_min_sim += min_score;
        total_range += range;

        // Show detailed view for first few
        if i < 5 {
            println!(
                "Frame {:3}: top={}({:.4}) min={:.4} range={:.4}",
                i, top_phoneme, top_score, min_score, range
            );

            // Show top-5 candidates
            print!("           ");
            for (p, s) in scores.iter().take(5) {
                print!(" {}:{:.3}", p, s);
            }
            println!();
        }
    }

    let n = cli.num_frames.min(hvs.len()) as f32;
    println!();
    println!("───────────────────────────────────────────────────────────────");
    println!("AGGREGATE STATISTICS ({} frames)", n as usize);
    println!("───────────────────────────────────────────────────────────────");
    println!("  Avg max similarity:  {:.4}", total_max_sim / n);
    println!("  Avg min similarity:  {:.4}", total_min_sim / n);
    println!("  Avg score range:     {:.4}", total_range / n);
    println!();

    // Compare inference HVs with stored prototypes
    println!("═══════════════════════════════════════════════════════════════");
    println!("PROTOTYPE COMPARISON");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Pick first few prototypes and compare with first few HVs
    let proto_pairs = prototypes.as_pairs();
    let sample_protos: Vec<_> = proto_pairs.iter().take(5).collect();

    println!("Sample prototype self-similarity (should be ~1.0):");
    for (label, proto_hv) in &sample_protos {
        let self_sim = proto_hv.similarity(proto_hv);
        println!("  {}: {:.4}", label, self_sim);
    }
    println!();

    println!("Prototype-to-FramedHV cross-similarity (diagnostic):");
    for (label, proto_hv) in &sample_protos {
        // Compare prototype to first 5 BUNDLED inference HVs
        let sims: Vec<f32> = hvs
            .iter()
            .take(5)
            .map(|hv| proto_hv.similarity(hv))
            .collect();
        let avg_sim: f32 = sims.iter().sum::<f32>() / sims.len() as f32;

        println!(
            "  {} -> Frame[0..5]: avg={:.4}, vals={:?}",
            label,
            avg_sim,
            sims.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>()
        );
    }
    println!();

    // BIT PATTERN ANALYSIS
    println!("═══════════════════════════════════════════════════════════════");
    println!("BIT PATTERN ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Check prototype bit density
    let sample_proto_hv = proto_pairs[0].1;
    let proto_ones = sample_proto_hv
        .words
        .iter()
        .map(|w| w.count_ones())
        .sum::<u32>();
    let proto_density = proto_ones as f32 / 2048.0;

    // Check bundled HV bit density (should be ~50% like prototypes)
    let sample_bundle = &hvs[0];
    let bundle_ones = sample_bundle
        .words
        .iter()
        .map(|w| w.count_ones())
        .sum::<u32>();
    let bundle_density = bundle_ones as f32 / 2048.0;

    println!(
        "Prototype bit density: {:.1}% ({}/2048 bits set)",
        proto_density * 100.0,
        proto_ones
    );
    println!(
        "Framed HV bit density: {:.1}% ({}/2048 bits set)",
        bundle_density * 100.0,
        bundle_ones
    );
    println!();

    // Check if they're using the same hash space
    let common_ones = sample_proto_hv
        .words
        .iter()
        .zip(sample_bundle.words.iter())
        .map(|(a, b)| (a & b).count_ones())
        .sum::<u32>();
    let common_zeros = sample_proto_hv
        .words
        .iter()
        .zip(sample_bundle.words.iter())
        .map(|(a, b)| (!a & !b).count_ones())
        .sum::<u32>();

    println!(
        "Bit agreement: {} common 1s + {} common 0s = {} matching bits",
        common_ones,
        common_zeros,
        common_ones + common_zeros
    );
    println!("Expected for orthogonal vectors: ~1024 (50%)");
    println!("Expected for same-source vectors: >1200 (>60%)");
    println!();

    // DIAGNOSIS
    println!("═══════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let avg_max = total_max_sim / n;
    if avg_max < 0.1 {
        println!("⚠️  CRITICAL: Max similarity {:.4} << 0.3", avg_max);
        println!("   This indicates the inference HVs are ORTHOGONAL to prototypes!");
        println!();
        println!("   Likely cause: NORMALIZATION MISMATCH between training and inference.");
        println!("   - Training may have used different mel normalization");
        println!("   - LTC states may be saturating differently");
        println!();
        println!("   Next step: Check AudioFrontend.compute_mel_frame() normalization");
        println!("              and compare against how prototypes were trained.");
    } else if avg_max < 0.3 {
        println!(
            "⚠️  WARNING: Max similarity {:.4} is low (expect >0.3)",
            avg_max
        );
        println!("   Prototypes have weak discrimination.");
    } else {
        println!("✓  Max similarity {:.4} is reasonable (>0.3)", avg_max);
    }
    println!();
}
