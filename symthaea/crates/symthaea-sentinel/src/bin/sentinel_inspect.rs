// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inspect the frequency signatures learned from audio files

use symthaea_sentinel::{
    AudioCategory, AudioFeatures, AudioSentinel, CONTROL_RATE, FREQ_BINS, FileAudioConfig,
    FileAudioPump, MEL_BANDS, compute_ioi_variance, compute_mfcc, compute_onset_strength,
    compute_spectral_centroid, compute_spectral_flatness, compute_temporal_regularity,
    spectrum_to_mel_bands,
};

fn process_and_learn(path: &str, name: &str, sentinel: &mut AudioSentinel) {
    let config = FileAudioConfig::default();
    let mut pump = match FileAudioPump::new(path, config) {
        Ok(p) => p,
        Err(e) => {
            println!("  Failed to load {}: {}", path, e);
            return;
        }
    };

    sentinel.start_learning(name, AudioCategory::Unknown);

    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut onset_history: Vec<f32> = Vec::new();
    let mut rms_history: Vec<f32> = Vec::new();
    let mut prev_rms: f32 = 0.0;
    let mut frame_count = 0;

    while let Some(spectrum) = pump.next_power_spectrum() {
        let mel_bands = spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
        let mfcc = compute_mfcc(&mel_bands);
        let spectral_centroid =
            compute_spectral_centroid(&spectrum, pump.sample_rate(), pump.window_size());
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

        let rms_energy = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

        // Envelope features
        rms_history.push(rms_energy);
        if rms_history.len() > 32 {
            rms_history.remove(0);
        }
        let envelope_delta = (rms_energy - prev_rms).abs();
        prev_rms = rms_energy;
        let envelope_variance = if rms_history.len() > 2 {
            let mean: f32 = rms_history.iter().sum::<f32>() / rms_history.len() as f32;
            rms_history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / rms_history.len() as f32
        } else {
            0.0
        };

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
        let spectral_rolloff =
            rolloff_bin as f32 * (pump.sample_rate() / pump.window_size() as f32);
        let high_freq_energy: f32 = spectrum.iter().skip(spectrum.len() / 2).sum();
        let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);

        // Compute actual silence_ratio and burst_density
        let silence_ratio = if rms_energy < 0.02 { 1.0 } else { 0.0 };
        let burst_density = compute_ioi_variance(&onset_history, CONTROL_RATE);

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
            envelope_delta,
            envelope_variance,
            frame_index: frame_count,
            spectral_flux: 0.1,
            harmonic_ratio: 0.5,
            cfc_theta_gamma: 0.5,
            cfc_delta_beta: 0.5,
            attack_sharpness: 0.5,
            decay_roughness: 0.5,
            silence_ratio,
            burst_density,
        };

        sentinel.process(&features);
        frame_count += 1;
    }

    sentinel.stop_learning();
    println!("  {} frames processed", frame_count);
}

fn main() {
    let files = [
        ("test-data/clock/clock1.wav", "Clock"),
        ("test-data/dog/dog1.wav", "Dog"),
        ("test-data/birds/crow/crow1.wav", "Bird-Crow"),
        ("test-data/birds/robin/robin1.wav", "Bird-Robin"),
        ("test-data/rain/rain1.wav", "Rain"),
        ("test-data/static/static1.wav", "Static"),
    ];

    let mut sentinel = AudioSentinel::new();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         SIGNATURE INSPECTION - What is being learned?       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for (path, name) in &files {
        println!("Learning {}...", name);
        process_and_learn(path, name, &mut sentinel);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    LEARNED SIGNATURES");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Frequency bins: {:?} Hz\n", FREQ_BINS);

    for (name, pattern) in &sentinel.patterns {
        println!("{}:", name);

        // Find dominant rhythm frequency
        let (rhythm_max_idx, _rhythm_max) = pattern
            .rhythm_freq_signature
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        let (timbre_max_idx, _timbre_max) = pattern
            .timbre_freq_signature
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        print!("  Rhythm:  [");
        for (i, &f) in pattern.rhythm_freq_signature.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            if i == rhythm_max_idx && f > 0.01 {
                print!("\x1b[33m{:.4}\x1b[0m", f);
            } else {
                print!("{:.4}", f);
            }
        }
        println!("]");
        println!(
            "           Peak at {:.2} Hz ({:.0} BPM equivalent)",
            FREQ_BINS[rhythm_max_idx],
            FREQ_BINS[rhythm_max_idx] * 60.0
        );

        print!("  Timbre:  [");
        for (i, &f) in pattern.timbre_freq_signature.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            if i == timbre_max_idx && f > 0.01 {
                print!("\x1b[36m{:.4}\x1b[0m", f);
            } else {
                print!("{:.4}", f);
            }
        }
        println!("]");
        println!("           Peak at {:.2} Hz", FREQ_BINS[timbre_max_idx]);

        // Diagnostic: Show IOI variance (rhythm regularity), silence_ratio, and temporal_regularity
        // IOI Variance: 0 = very regular (clock), 1 = very irregular (dog/birds)
        println!(
            "  IOI Variance:  {:.4}  (0=regular, 1=irregular)",
            pattern.mean_burst_density
        );
        println!("  SilenceRatio:  {:.4}", pattern.mean_silence_ratio);
        println!("  Regularity:    {:.4}", pattern.mean_temporal_regularity);
        println!();
    }
}
