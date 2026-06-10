// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live Audio Sentinel Demo
//!
//! Real-time temporal pattern recognition from microphone input.
//! Uses the AudioPump for lock-free audio capture and the AudioSentinel
//! for hierarchical HDC/LTC processing.
//!
//! Usage:
//!   cargo run --release --features live-audio --bin live-audio
//!
//! Requirements:
//!   - Working microphone
//!   - ALSA dev libraries on Linux (libasound2-dev)

#[cfg(feature = "live-audio")]
mod live_impl {
    use anyhow::Result;
    use std::io::{self, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use symthaea_sentinel::{
        AudioCategory, AudioConfig, AudioFeatures, AudioPump, AudioSentinel, FREQ_BINS, MEL_BANDS,
        NUM_MFCC, compute_mfcc, compute_onset_strength, compute_spectral_centroid,
        compute_spectral_flatness, compute_temporal_regularity, spectrum_to_mel_bands,
    };

    pub fn run() -> Result<()> {
        println!("\n╔═══════════════════════════════════════════════════════════════════╗");
        println!("║         LIVE AUDIO SENTINEL - Real-Time Pattern Recognition       ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  \"Transformers spatialize time. Symthaea LIVES in time.\"          ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  This demo processes live audio from your microphone.             ║");
        println!("║  Play different music or sounds to learn temporal patterns.       ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  Commands:                                                        ║");
        println!("║    [1-9] Learn pattern (play sound while learning)                ║");
        println!("║    [D]   Detect mode (continuous classification)                  ║");
        println!("║    [M]   Monitor mode (show live audio features)                  ║");
        println!("║    [S]   Summary of learned patterns                              ║");
        println!("║    [Q]   Quit                                                     ║");
        println!("╚═══════════════════════════════════════════════════════════════════╝\n");

        // Initialize audio pump
        let config = AudioConfig::default();
        let mut pump = AudioPump::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;

        println!("  Control Rate: {:.1} Hz", pump.control_rate());
        println!("  Sample Rate:  {:.0} Hz\n", pump.sample_rate());

        // Initialize sentinel
        let mut sentinel = AudioSentinel::new();

        // For onset detection and temporal regularity
        let mut prev_spectrum: Vec<f32> = Vec::new();
        let mut onset_history: Vec<f32> = Vec::new();

        // Ctrl+C handler
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        ctrlc::set_handler(move || {
            r.store(false, Ordering::Relaxed);
        })?;

        loop {
            print!("Command> ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let cmd = input.trim().to_uppercase();

            match cmd.chars().next() {
                Some('1'..='9') => {
                    let pattern_num = cmd.chars().next().unwrap();
                    let name = format!("Pattern{}", pattern_num);

                    println!("\n  Learning '{}' - Play your sound now!", name);
                    println!("  (Recording for 5 seconds...)\n");

                    sentinel.start_learning(&name, AudioCategory::Music);

                    let start = Instant::now();
                    let duration = Duration::from_secs(5);
                    let mut frame_count = 0;

                    running.store(true, Ordering::Relaxed);

                    while start.elapsed() < duration && running.load(Ordering::Relaxed) {
                        if let Some(spectrum) = pump.next_power_spectrum() {
                            // Compute features
                            let mel_bands =
                                spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
                            let mfcc = compute_mfcc(&mel_bands);

                            let spectral_centroid = compute_spectral_centroid(
                                &spectrum,
                                pump.sample_rate(),
                                pump.window_size(),
                            );
                            let spectral_flatness = compute_spectral_flatness(&spectrum);

                            let onset_strength = if prev_spectrum.is_empty() {
                                0.0
                            } else {
                                compute_onset_strength(&prev_spectrum, &spectrum)
                            };
                            prev_spectrum = spectrum.clone();

                            // Track onset history for regularity
                            onset_history.push(onset_strength);
                            if onset_history.len() > 100 {
                                onset_history.remove(0);
                            }
                            let temporal_regularity = compute_temporal_regularity(&onset_history);

                            // Estimate RMS from spectrum (approximation)
                            let rms_energy =
                                (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

                            // Spectral rolloff (85% energy point)
                            let total: f32 = spectrum.iter().sum();
                            let mut cumsum = 0.0;
                            let mut rolloff_bin = 0;
                            for (i, &p) in spectrum.iter().enumerate() {
                                cumsum += p;
                                if cumsum >= total * 0.85 {
                                    rolloff_bin = i;
                                    break;
                                }
                            }
                            let spectral_rolloff = rolloff_bin as f32
                                * (pump.sample_rate() / pump.window_size() as f32);

                            // Zero crossing rate approximation from high-frequency content
                            let high_freq_energy: f32 =
                                spectrum.iter().skip(spectrum.len() / 2).sum();
                            let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);

                            let features = AudioFeatures {
                                mel_bands,
                                mfcc,
                                spectral_centroid,
                                spectral_rolloff,
                                onset_strength,
                                rms_energy,
                                zero_crossing_rate,
                                spectral_flatness,
                                temporal_regularity,
                            };

                            let result = sentinel.process(&features);
                            frame_count += 1;

                            // Progress display
                            let progress = start.elapsed().as_secs_f32() / duration.as_secs_f32();
                            let bar = (progress * 30.0) as usize;

                            // Show live audio level
                            let level = (rms_energy * 50.0).min(20.0) as usize;
                            let level_bar = "█".repeat(level) + &"░".repeat(20 - level);

                            print!(
                                "\r  [{:30}] {:.0}%  Level: [{}] Φr={:.4}",
                                "█".repeat(bar) + &"░".repeat(30 - bar),
                                progress * 100.0,
                                level_bar,
                                result.phi_rhythm
                            );
                            io::stdout().flush()?;
                        }

                        std::thread::sleep(Duration::from_millis(10));
                    }

                    sentinel.stop_learning();
                    println!(
                        "\n  ✓ Pattern '{}' learned! ({} frames)\n",
                        name, frame_count
                    );
                }

                Some('M') => {
                    println!("\n  Monitor mode - showing live audio features");
                    println!("  (Press Ctrl+C to stop)\n");

                    running.store(true, Ordering::Relaxed);

                    while running.load(Ordering::Relaxed) {
                        if let Some(spectrum) = pump.next_power_spectrum() {
                            let mel_bands =
                                spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
                            let centroid = compute_spectral_centroid(
                                &spectrum,
                                pump.sample_rate(),
                                pump.window_size(),
                            );
                            let onset = if prev_spectrum.is_empty() {
                                0.0
                            } else {
                                compute_onset_strength(&prev_spectrum, &spectrum)
                            };
                            prev_spectrum = spectrum.clone();

                            let rms = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

                            // Visualize mel bands (first 20)
                            let mut mel_viz = String::new();
                            for &band in mel_bands.iter().take(20) {
                                let normalized = ((band + 10.0) / 10.0).clamp(0.0, 1.0);
                                let char = match (normalized * 8.0) as u8 {
                                    0 => ' ',
                                    1 => '▁',
                                    2 => '▂',
                                    3 => '▃',
                                    4 => '▄',
                                    5 => '▅',
                                    6 => '▆',
                                    7 => '▇',
                                    _ => '█',
                                };
                                mel_viz.push(char);
                            }

                            print!(
                                "\r  Mel:[{}] Cent:{:5.0}Hz Onset:{:.3} RMS:{:.3}   ",
                                mel_viz, centroid, onset, rms
                            );
                            io::stdout().flush()?;
                        }

                        std::thread::sleep(Duration::from_millis(20));
                    }
                    println!("\n");
                }

                Some('D') => {
                    if sentinel.patterns.len() < 2 {
                        println!("\n  Learn at least 2 patterns first!\n");
                        continue;
                    }

                    println!("\n  Detection mode - classifying live audio");
                    println!("  (Press Ctrl+C to stop)\n");

                    running.store(true, Ordering::Relaxed);

                    while running.load(Ordering::Relaxed) {
                        if let Some(spectrum) = pump.next_power_spectrum() {
                            let mel_bands =
                                spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
                            let mfcc = compute_mfcc(&mel_bands);

                            let spectral_centroid = compute_spectral_centroid(
                                &spectrum,
                                pump.sample_rate(),
                                pump.window_size(),
                            );
                            let spectral_flatness = compute_spectral_flatness(&spectrum);

                            let onset_strength = if prev_spectrum.is_empty() {
                                0.0
                            } else {
                                compute_onset_strength(&prev_spectrum, &spectrum)
                            };
                            prev_spectrum = spectrum.clone();

                            // Track onset history for regularity
                            onset_history.push(onset_strength);
                            if onset_history.len() > 100 {
                                onset_history.remove(0);
                            }
                            let temporal_regularity = compute_temporal_regularity(&onset_history);

                            let rms_energy =
                                (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

                            let total: f32 = spectrum.iter().sum();
                            let mut cumsum = 0.0;
                            let mut rolloff_bin = 0;
                            for (i, &p) in spectrum.iter().enumerate() {
                                cumsum += p;
                                if cumsum >= total * 0.85 {
                                    rolloff_bin = i;
                                    break;
                                }
                            }
                            let spectral_rolloff = rolloff_bin as f32
                                * (pump.sample_rate() / pump.window_size() as f32);
                            let high_freq_energy: f32 =
                                spectrum.iter().skip(spectrum.len() / 2).sum();
                            let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);

                            let features = AudioFeatures {
                                mel_bands,
                                mfcc,
                                spectral_centroid,
                                spectral_rolloff,
                                onset_strength,
                                rms_energy,
                                zero_crossing_rate,
                                spectral_flatness,
                                temporal_regularity,
                            };

                            let result = sentinel.process(&features);

                            // Build similarity bars
                            let mut sim_parts: Vec<String> = Vec::new();
                            let mut sorted: Vec<_> = result.similarities.iter().collect();
                            sorted.sort_by(|a, b| a.0.cmp(b.0));

                            for (name, sim) in sorted {
                                let bar_len = (sim.combined * 10.0).min(10.0) as usize;
                                let is_best = result.detected_pattern == *name;
                                let color = if is_best { "\x1b[32m" } else { "\x1b[90m" };
                                sim_parts.push(format!(
                                    "{}:{}{:.2}\x1b[0m",
                                    &name[..name.len().min(8)],
                                    color,
                                    sim.combined
                                ));
                            }

                            // Level meter
                            let level = (rms_energy * 50.0).min(10.0) as usize;
                            let level_bar = "█".repeat(level) + &"░".repeat(10 - level);

                            print!(
                                "\r  [{}] {} │ \x1b[1m{:10}\x1b[0m ({:.0}%)   ",
                                level_bar,
                                sim_parts.join(" "),
                                result.detected_pattern,
                                result.confidence * 100.0
                            );
                            io::stdout().flush()?;
                        }

                        std::thread::sleep(Duration::from_millis(20));
                    }
                    println!("\n");
                }

                Some('S') => {
                    println!("\n  === Live Audio Sentinel Summary ===");
                    println!("  Sample Rate:   {} Hz", pump.sample_rate());
                    println!("  Control Rate:  {:.1} Hz", pump.control_rate());
                    println!("  Freq Bins:     {:?}", FREQ_BINS);
                    println!("\n  Learned Patterns:");

                    for (name, pattern) in &sentinel.patterns {
                        if pattern.frame_count == 0 {
                            continue;
                        }
                        println!(
                            "    {} ({:?}, {} frames)",
                            name, pattern.category, pattern.frame_count
                        );

                        // Find dominant rhythm frequency
                        let rhythm_max_idx = pattern
                            .rhythm_freq_signature
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(i, _)| i)
                            .unwrap_or(0);

                        let estimated_bpm = FREQ_BINS[rhythm_max_idx] * 60.0;

                        print!("      Rhythm:  [");
                        for (i, &f) in pattern.rhythm_freq_signature.iter().enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            if i == rhythm_max_idx && f > 0.01 {
                                print!("\x1b[33m{:.3}\x1b[0m", f);
                            } else {
                                print!("{:.3}", f);
                            }
                        }
                        println!("]  ~{:.0} BPM", estimated_bpm);

                        print!("      Timbre:  [");
                        for (i, &f) in pattern.timbre_freq_signature.iter().enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            if f > 0.05 {
                                print!("\x1b[36m{:.3}\x1b[0m", f);
                            } else {
                                print!("{:.3}", f);
                            }
                        }
                        println!("]");
                    }
                    println!();
                }

                Some('Q') => {
                    println!("\nGoodbye!");
                    break;
                }

                _ => {
                    println!(
                        "  Commands: 1-9 (learn), D (detect), M (monitor), S (summary), Q (quit)"
                    );
                }
            }
        }

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "live-audio")]
    {
        live_impl::run()
    }

    #[cfg(not(feature = "live-audio"))]
    {
        eprintln!("Error: live-audio feature not enabled!");
        eprintln!("Run with: cargo run --release --features live-audio --bin live-audio");
        std::process::exit(1);
    }
}
