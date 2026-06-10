// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio Sentinel Demo
//!
//! Demonstrates temporal pattern recognition from audio using:
//! - Hierarchical HDC encoding (Timbre + Rhythm → Context)
//! - Dual LTC networks for parallel timbre/rhythm processing
//! - 8-bin frequency signature for fine tempo discrimination
//!
//! Uses synthetic audio patterns since we don't have real-time audio capture yet.

use anyhow::Result;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use symthaea_sentinel::*;

// =============================================================================
// Synthetic Audio Pattern Generation
// =============================================================================

#[derive(Clone, Copy)]
enum SyntheticPattern {
    /// Steady beat like a metronome or kick drum
    SteadyBeat { bpm: u32 },
    /// Music with melody (varying timbre) and beat
    MusicLike { bpm: u32, melody_speed: f32 },
    /// Speech-like: irregular rhythm, formant-like timbre
    SpeechLike { words_per_min: u32 },
    /// Traffic: continuous noise with occasional peaks
    Traffic,
    /// Nature: irregular bird-like chirps, wind
    Nature,
    /// Office: low level, keyboard clicks, HVAC
    Office,
    /// Silence with occasional small sounds
    Quiet,
}

struct SyntheticAudioGenerator {
    pattern: SyntheticPattern,
    frame: u64,
    control_rate: f32,
}

impl SyntheticAudioGenerator {
    fn new(pattern: SyntheticPattern) -> Self {
        Self {
            pattern,
            frame: 0,
            control_rate: CONTROL_RATE,
        }
    }

    fn next_features(&mut self) -> AudioFeatures {
        let t = self.frame as f32 / self.control_rate;
        let frame_index = self.frame as usize;
        self.frame += 1;

        match self.pattern {
            SyntheticPattern::SteadyBeat { bpm } => {
                let beat_freq = bpm as f32 / 60.0;
                let phase = (t * beat_freq * 2.0 * PI).sin();
                let on_beat = phase > 0.7;

                let onset_strength = if on_beat { 0.8 } else { 0.05 };
                let rms_energy = if on_beat { 0.6 } else { 0.1 };

                // Steady timbre (kick drum-like)
                let mut mel_bands = vec![0.0; MEL_BANDS];
                for i in 0..10 {
                    // Low frequency emphasis
                    mel_bands[i] = if on_beat { 0.8 - i as f32 * 0.05 } else { 0.1 };
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC], // Synthetic - no real MFCC
                    spectral_centroid: if on_beat { 200.0 } else { 100.0 },
                    spectral_rolloff: if on_beat { 500.0 } else { 200.0 },
                    onset_strength,
                    rms_energy,
                    zero_crossing_rate: 0.05,
                    spectral_flatness: 0.2,   // Low (tonal beat)
                    temporal_regularity: 0.9, // High (steady beat)
                    envelope_delta: if on_beat { 0.5 } else { 0.02 }, // Sharp transients on beat
                    envelope_variance: 0.15,  // High modulation (rhythmic)
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.0,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::MusicLike { bpm, melody_speed } => {
                let beat_freq = bpm as f32 / 60.0;
                let beat_phase = (t * beat_freq * 2.0 * PI).sin();
                let on_beat = beat_phase > 0.7;

                // Melody varies the spectral centroid
                let melody = (t * melody_speed * 2.0 * PI).sin() * 0.5 + 0.5;

                let onset_strength = if on_beat { 0.6 } else { 0.1 + melody * 0.1 };
                let rms_energy = 0.3 + melody * 0.2;

                // Timbre varies with melody
                let mut mel_bands = vec![0.0; MEL_BANDS];
                let center_band = ((melody * 20.0) as usize).min(MEL_BANDS - 5);
                for i in 0..MEL_BANDS {
                    let dist = (i as i32 - center_band as i32).abs() as f32;
                    mel_bands[i] = (0.8 - dist * 0.1).max(0.0);
                }
                // Add bass on beat
                if on_beat {
                    for i in 0..5 {
                        mel_bands[i] = (mel_bands[i] + 0.5).min(1.0);
                    }
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: 500.0 + melody * 2000.0,
                    spectral_rolloff: 2000.0 + melody * 4000.0,
                    onset_strength,
                    rms_energy,
                    zero_crossing_rate: 0.1 + melody * 0.1,
                    spectral_flatness: 0.3,   // Some noise in music
                    temporal_regularity: 0.7, // Fairly regular rhythm
                    envelope_delta: if on_beat { 0.3 } else { 0.05 },
                    envelope_variance: 0.12, // Moderate modulation
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::SpeechLike { words_per_min } => {
                // Speech: ~150 wpm = 2.5 words/sec, syllables ~5/sec
                let syllable_rate = words_per_min as f32 / 60.0 * 2.0;

                // Irregular rhythm (simulated with multiple frequencies)
                let rhythm1 = (t * syllable_rate * 2.0 * PI).sin();
                let rhythm2 = (t * syllable_rate * 0.7 * 2.0 * PI + 1.0).sin();
                let combined = (rhythm1 + rhythm2 * 0.5) / 1.5;
                let speaking = combined > 0.3;

                // Formant-like timbre (peaks at certain bands)
                let mut mel_bands = vec![0.1; MEL_BANDS];
                if speaking {
                    // Simulate formants F1 (~500Hz), F2 (~1500Hz), F3 (~2500Hz)
                    let formant_bands = [8, 15, 22]; // Approximate mel band indices
                    for &fb in &formant_bands {
                        if fb < MEL_BANDS {
                            mel_bands[fb] = 0.7;
                            if fb > 0 {
                                mel_bands[fb - 1] = 0.4;
                            }
                            if fb < MEL_BANDS - 1 {
                                mel_bands[fb + 1] = 0.4;
                            }
                        }
                    }
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: if speaking { 1200.0 } else { 200.0 },
                    spectral_rolloff: if speaking { 4000.0 } else { 500.0 },
                    onset_strength: if speaking {
                        0.3 + combined.abs() * 0.3
                    } else {
                        0.02
                    },
                    rms_energy: if speaking { 0.4 } else { 0.05 },
                    zero_crossing_rate: if speaking { 0.15 } else { 0.02 },
                    spectral_flatness: 0.4,   // Mixed tonal/noise
                    temporal_regularity: 0.3, // Irregular (speech)
                    envelope_delta: if speaking { 0.15 } else { 0.02 }, // Variable speech envelope
                    envelope_variance: 0.08,  // Irregular modulation
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::Traffic => {
                // Continuous noise with occasional car passes (Doppler-like)
                let car_pass = (t * 0.2 * 2.0 * PI).sin().abs(); // ~5 sec period
                let honk = if (t * 0.05).fract() < 0.02 { 0.8 } else { 0.0 };

                let mut mel_bands = vec![0.3; MEL_BANDS];
                // Broadband noise
                for i in 0..MEL_BANDS {
                    mel_bands[i] = 0.2 + car_pass * 0.3 + (i as f32 / MEL_BANDS as f32) * 0.2;
                }
                // Honk adds mid-frequency peak
                if honk > 0.0 {
                    for i in 10..20 {
                        mel_bands[i] = (mel_bands[i] + honk).min(1.0);
                    }
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: 800.0 + car_pass * 500.0,
                    spectral_rolloff: 3000.0 + car_pass * 2000.0,
                    onset_strength: 0.1 + car_pass * 0.2 + honk * 0.5,
                    rms_energy: 0.3 + car_pass * 0.3,
                    zero_crossing_rate: 0.2,
                    spectral_flatness: 0.7,            // High (noise-like)
                    temporal_regularity: 0.2,          // Low (aperiodic)
                    envelope_delta: 0.03 + honk * 0.4, // Low except for honk
                    envelope_variance: 0.05,           // Low modulation
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::Nature => {
                // Bird chirps (high, brief), wind (low, continuous)
                let wind = (t * 0.1 * 2.0 * PI).sin() * 0.3 + 0.3;
                let bird = if (t * 1.5).fract() < 0.1 {
                    (t * 50.0).sin() * 0.5 + 0.5 // Fast chirp
                } else {
                    0.0
                };

                let mut mel_bands = vec![0.1; MEL_BANDS];
                // Wind in low bands
                for i in 0..10 {
                    mel_bands[i] = wind * 0.5;
                }
                // Bird in high bands
                if bird > 0.0 {
                    for i in 25..35 {
                        mel_bands[i.min(MEL_BANDS - 1)] = bird;
                    }
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: 500.0 + bird * 3000.0,
                    spectral_rolloff: 2000.0 + bird * 6000.0,
                    onset_strength: 0.05 + bird * 0.6,
                    rms_energy: 0.15 + wind * 0.1 + bird * 0.3,
                    zero_crossing_rate: 0.05 + bird * 0.2,
                    spectral_flatness: 0.5,     // Mixed
                    temporal_regularity: 0.2,   // Low (aperiodic nature sounds)
                    envelope_delta: bird * 0.4, // Sharp bird chirps
                    envelope_variance: 0.06,    // Irregular
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::Office => {
                // HVAC hum + keyboard clicks
                let hvac = 0.2;
                let keyboard = if (t * 3.0).fract() < 0.05 { 0.4 } else { 0.0 };

                let mut mel_bands = vec![0.1; MEL_BANDS];
                // HVAC: low frequency hum
                for i in 0..5 {
                    mel_bands[i] = hvac;
                }
                // Keyboard: brief broadband click
                if keyboard > 0.0 {
                    for i in 10..30 {
                        mel_bands[i.min(MEL_BANDS - 1)] = keyboard * 0.3;
                    }
                }

                AudioFeatures {
                    mel_bands: mel_bands.clone(),
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: 300.0 + keyboard * 1000.0,
                    spectral_rolloff: 1000.0 + keyboard * 2000.0,
                    onset_strength: 0.02 + keyboard * 0.3,
                    rms_energy: 0.1 + keyboard * 0.2,
                    zero_crossing_rate: 0.03,
                    spectral_flatness: 0.3,         // Some noise (HVAC)
                    temporal_regularity: 0.4,       // Somewhat regular (typing rhythm)
                    envelope_delta: keyboard * 0.3, // Clicks have sharp delta
                    envelope_variance: 0.04,        // Low modulation
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }

            SyntheticPattern::Quiet => {
                // Very low level ambient
                let ambient = 0.05;
                let occasional = if (t * 0.1).fract() < 0.02 { 0.2 } else { 0.0 };

                AudioFeatures {
                    mel_bands: vec![ambient + occasional * 0.1; MEL_BANDS],
                    mfcc: vec![0.0; NUM_MFCC],
                    spectral_centroid: 200.0,
                    spectral_rolloff: 500.0,
                    onset_strength: occasional * 0.1,
                    rms_energy: ambient + occasional * 0.1,
                    zero_crossing_rate: 0.01,
                    spectral_flatness: 0.5,           // Neutral
                    temporal_regularity: 0.1,         // Very low (silence)
                    envelope_delta: occasional * 0.1, // Very low delta
                    envelope_variance: 0.01,          // Minimal modulation
                    frame_index,
                    spectral_flux: 0.1,
                    harmonic_ratio: 0.5,
                    cfc_theta_gamma: 0.5,
                    cfc_delta_beta: 0.5,
                    attack_sharpness: 0.5,
                    decay_roughness: 0.5,
                    silence_ratio: 0.5,
                    burst_density: 0.5,
                }
            }
        }
    }
}

// =============================================================================
// Demo
// =============================================================================

fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║        AUDIO SENTINEL - LTC Temporal Audio Recognition            ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Hierarchical HDC: Timbre + Rhythm → Context                      ║");
    println!("║  Dual LTC Networks: Parallel timbre/rhythm processing             ║");
    println!("║  8-bin Frequency Signature: 0.25Hz - 6Hz (15-360 BPM)             ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Patterns:                                                        ║");
    println!("║    [1] 60 BPM Beat      [2] 120 BPM Beat    [3] 90 BPM Music      ║");
    println!("║    [4] Speech           [5] Traffic         [6] Nature            ║");
    println!("║    [7] Office           [8] Quiet                                 ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Commands:                                                        ║");
    println!("║    [D] Detect mode      [S] Summary         [Q] Quit              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let patterns: Vec<(&str, SyntheticPattern, AudioCategory)> = vec![
        (
            "60BPM",
            SyntheticPattern::SteadyBeat { bpm: 60 },
            AudioCategory::Music,
        ),
        (
            "120BPM",
            SyntheticPattern::SteadyBeat { bpm: 120 },
            AudioCategory::Music,
        ),
        (
            "Music90",
            SyntheticPattern::MusicLike {
                bpm: 90,
                melody_speed: 0.5,
            },
            AudioCategory::Music,
        ),
        (
            "Speech",
            SyntheticPattern::SpeechLike { words_per_min: 150 },
            AudioCategory::Speech,
        ),
        ("Traffic", SyntheticPattern::Traffic, AudioCategory::Urban),
        ("Nature", SyntheticPattern::Nature, AudioCategory::Nature),
        ("Office", SyntheticPattern::Office, AudioCategory::Urban),
        ("Quiet", SyntheticPattern::Quiet, AudioCategory::Silence),
    ];

    let mut sentinel = AudioSentinel::new();

    loop {
        print!("Command> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let cmd = input.trim().to_uppercase();

        match cmd.as_str() {
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" => {
                let idx: usize = cmd.parse::<usize>().unwrap() - 1;
                let (name, pattern, category) = patterns[idx];

                println!("\n  Learning '{}' ({:?})...", name, category);
                sentinel.start_learning(name, category);

                let mut audio_gen = SyntheticAudioGenerator::new(pattern);
                let start = Instant::now();
                let duration = Duration::from_secs(4);

                while start.elapsed() < duration {
                    let features = audio_gen.next_features();
                    let result = sentinel.process(&features);

                    let progress = start.elapsed().as_secs_f32() / duration.as_secs_f32();
                    let bar = (progress * 30.0) as usize;
                    print!(
                        "\r  [{}{}] {:.0}%  Φr={:.4} Φt={:.4}",
                        "█".repeat(bar),
                        "░".repeat(30 - bar),
                        progress * 100.0,
                        result.phi_rhythm,
                        result.phi_timbre
                    );
                    io::stdout().flush()?;

                    // ~43Hz control rate
                    std::thread::sleep(Duration::from_micros(23000));
                }

                sentinel.stop_learning();
                println!("\n  ✓ Pattern '{}' learned!\n", name);
            }

            "D" => {
                if sentinel.patterns.len() < 2 {
                    println!("\n  Learn at least 2 patterns first!\n");
                    continue;
                }

                println!("\n  Detection mode (Ctrl+C to stop)...\n");
                println!("  Cycling through learned patterns every 5 seconds.\n");

                let learned: Vec<_> = sentinel.patterns.keys().cloned().collect();
                let pattern_map: HashMap<String, SyntheticPattern> = patterns
                    .iter()
                    .map(|(name, pat, _)| (name.to_string(), *pat))
                    .collect();

                let mut idx = 0;
                let mut pattern_start = Instant::now();
                let mut current_pattern = &learned[0];

                loop {
                    // Switch pattern every 5 seconds
                    if pattern_start.elapsed() > Duration::from_secs(5) {
                        idx = (idx + 1) % learned.len();
                        current_pattern = &learned[idx];
                        pattern_start = Instant::now();
                        println!("\n  [Switching to {}]", current_pattern);
                    }

                    let syn_pattern = pattern_map
                        .get(current_pattern)
                        .copied()
                        .unwrap_or(SyntheticPattern::Quiet);

                    let mut audio_gen = SyntheticAudioGenerator::new(syn_pattern);

                    // Process a few frames
                    for _ in 0..5 {
                        let features = audio_gen.next_features();
                        let result = sentinel.process(&features);

                        // Build similarity display
                        let mut sim_parts: Vec<String> = Vec::new();
                        let mut sorted: Vec<_> = result.similarities.iter().collect();
                        sorted.sort_by(|a, b| a.0.cmp(b.0));

                        for (name, sim) in sorted {
                            let _bar_len = (sim.combined * 10.0) as usize;
                            let color = if result.detected_pattern == *name {
                                "\x1b[32m"
                            } else {
                                "\x1b[90m"
                            };
                            sim_parts.push(format!("{}:{}{:.2}\x1b[0m", name, color, sim.combined));
                        }

                        print!(
                            "\r  Playing: {:8} │ {} │ Detected: {:8} ({:.0}%)    ",
                            current_pattern,
                            sim_parts.join(" "),
                            result.detected_pattern,
                            result.confidence * 100.0
                        );
                        io::stdout().flush()?;
                    }

                    std::thread::sleep(Duration::from_millis(100));

                    // Check for input (non-blocking would be better, but this works)
                    if pattern_start.elapsed() > Duration::from_secs(30) {
                        break;
                    }
                }
                println!("\n");
            }

            "S" => {
                println!("\n  === Audio Sentinel Summary ===");
                println!("  Control Rate: {:.1} Hz", CONTROL_RATE);
                println!("  HDC Dimension: {}", HDC_DIM);
                println!("  Mel Bands: {}", MEL_BANDS);
                println!("  Freq Bins: {:?}", FREQ_BINS);
                println!("\n  Learned Patterns:");

                for (name, pattern) in &sentinel.patterns {
                    if pattern.frame_count == 0 {
                        continue;
                    }
                    println!(
                        "    {} ({:?}, {} frames)",
                        name, pattern.category, pattern.frame_count
                    );
                    print!("      Rhythm Freq Sig: [");
                    for (i, &f) in pattern.rhythm_freq_signature.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        // Highlight dominant frequency
                        if f > 0.1 {
                            print!("\x1b[33m{:.3}\x1b[0m", f);
                        } else {
                            print!("{:.3}", f);
                        }
                    }
                    println!("]");
                    print!("      Timbre Freq Sig: [");
                    for (i, &f) in pattern.timbre_freq_signature.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        if f > 0.1 {
                            print!("\x1b[36m{:.3}\x1b[0m", f);
                        } else {
                            print!("{:.3}", f);
                        }
                    }
                    println!("]");
                }
                println!();
            }

            "Q" => {
                println!("\nGoodbye!");
                break;
            }

            _ => {
                println!(
                    "  Unknown command. Use 1-8 to learn, D to detect, S for summary, Q to quit."
                );
            }
        }
    }

    Ok(())
}
