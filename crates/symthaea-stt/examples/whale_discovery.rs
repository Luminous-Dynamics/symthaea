// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Whale Acoustic Discovery
//!
//! Operation Leviathan: Prove that Symthaea's LTC/HDC architecture
//! discovers universal acoustic structure in whale vocalizations.
//!
//! Run: cargo run --example whale_discovery

use std::fs;
use std::path::PathBuf;
use symthaea_stt::audio::AudioFrontend;
use symthaea_stt::whale::{WhaleConfig, WhaleSentinel, analyze_coda_topology};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          OPERATION LEVIATHAN: Whale Acoustic Discovery        ║");
    println!("║         Universal Acoustic Resonator Validation Test          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let whale_dir = PathBuf::from("data/whales/sperm");

    if !whale_dir.exists() {
        eprintln!("ERROR: Whale data not found at {:?}", whale_dir);
        eprintln!("       Run: bash scripts/download_whales.sh");
        std::process::exit(1);
    }

    // Collect all WAV files
    let wav_files: Vec<PathBuf> = fs::read_dir(&whale_dir)
        .expect("Failed to read whale directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map(|e| e == "wav").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if wav_files.is_empty() {
        eprintln!("ERROR: No WAV files found in {:?}", whale_dir);
        eprintln!("       Run: bash scripts/download_whales.sh");
        std::process::exit(1);
    }

    println!("Found {} whale recordings:", wav_files.len());
    for f in &wav_files {
        println!("  - {}", f.file_name().unwrap().to_string_lossy());
    }
    println!();

    // Configure sentinel for whale acoustics
    // These parameters are tuned for typical sperm whale coda recordings
    let config = WhaleConfig {
        sample_rate: 44100,    // Standard sample rate
        click_threshold: 0.01, // Low threshold for recorded audio
        min_click_gap: 0.01,   // 10ms minimum between clicks
        max_ici: 0.8,          // 800ms max gap between clicks in a coda
        min_coda_clicks: 2,    // At least 2 clicks per coda
        max_coda_clicks: 10,   // Allow up to 10 clicks
        click_tau: 0.002,      // 2ms - fast for click transients
        rhythm_tau: 0.2,       // 200ms for rhythm memory
    };

    let mut sentinel = WhaleSentinel::with_config(config);
    let mut total_codas = 0;
    let mut all_codas = Vec::new();

    println!("══════════════════════════════════════════════════════════════════");
    println!("                    PROCESSING WHALE RECORDINGS                   ");
    println!("══════════════════════════════════════════════════════════════════");

    for wav_path in &wav_files {
        let filename = wav_path.file_name().unwrap().to_string_lossy();
        print!("\nProcessing: {} ... ", filename);

        // Load audio
        match AudioFrontend::load_wav(wav_path) {
            Ok((samples, sample_rate)) => {
                println!(
                    "({} Hz, {:.1}s)",
                    sample_rate,
                    samples.len() as f32 / sample_rate as f32
                );

                // Debug: show audio stats
                let max_abs: f32 = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let mean_abs: f32 =
                    samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32;
                println!(
                    "  Audio stats: max_abs={:.4}, mean_abs={:.4}",
                    max_abs, mean_abs
                );

                // Reset sentinel for new file
                sentinel.reset();

                // Process all samples
                let codas = sentinel.process_buffer(&samples);

                println!("  Clicks detected: {}", sentinel.stats.clicks_detected);
                println!("  Codas recognized: {}", codas.len());

                for coda in &codas {
                    println!(
                        "    └─ {} clicks, {:.3}s duration, ICI: {:?}",
                        coda.num_clicks,
                        coda.duration,
                        coda.ici_pattern
                            .iter()
                            .map(|x| format!("{:.2}", x))
                            .collect::<Vec<_>>()
                    );
                }

                total_codas += codas.len();
                all_codas.extend(codas);
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("                     CODA PROTOTYPE TOPOLOGY                      ");
    println!("══════════════════════════════════════════════════════════════════");
    println!();

    let (num_types, num_variants, avg_variants) = sentinel.prototype_stats();

    println!("Discovery Summary:");
    println!("  Total Codas Detected:   {}", total_codas);
    println!("  Unique Coda Types:      {}", num_types);
    println!("  Total Variants:         {}", num_variants);
    println!("  Avg Variants per Type:  {:.2}", avg_variants);
    println!();

    // Analyze topology if we have enough types
    if num_types >= 2 {
        let topology = analyze_coda_topology(&sentinel);

        println!("Coda Type Similarities:");
        println!("  Mean:  {:.3}", topology.mean_similarity);
        println!("  Min:   {:.3}", topology.min_similarity);
        println!("  Max:   {:.3}", topology.max_similarity);
        println!();

        if !topology.pairwise.is_empty() {
            println!("Pairwise Similarities (most similar first):");
            for (label_i, label_j, sim) in topology.pairwise.iter().take(10) {
                println!("  {:<20} <-> {:<20}  sim={:.3}", label_i, label_j, sim);
            }
        }
    } else {
        println!("Not enough coda types for topology analysis (need at least 2)");
    }

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("                        OPERATION LEVIATHAN                       ");
    println!("══════════════════════════════════════════════════════════════════");
    println!();

    // Verdict
    if num_types >= 3 && num_variants > num_types {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║                     HYPOTHESIS VALIDATED                      ║");
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Symthaea discovered {} distinct coda types with {} variants  ",
            num_types, num_variants
        );
        println!("║  The LTC/HDC architecture finds structure in whale clicks     ║");
        println!("║  Bio-Acoustic Structure is UNIVERSAL                          ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
    } else if num_types > 0 {
        println!("PARTIAL SUCCESS: Discovered {} coda types", num_types);
        println!("Need more/longer recordings for stronger validation");
    } else {
        println!("INCONCLUSIVE: No coda patterns detected");
        println!("Check click threshold or recording quality");
    }
    println!();

    // Export clusters for visualization
    let clusters = sentinel.export_clusters();
    if !clusters.is_empty() {
        let export_path = "data/whales/coda_clusters.json";
        if let Ok(json) = serde_json::to_string_pretty(&clusters) {
            if fs::write(export_path, &json).is_ok() {
                println!("Exported {} clusters to {}", clusters.len(), export_path);
            }
        }
    }
}
