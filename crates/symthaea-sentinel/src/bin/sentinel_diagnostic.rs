// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Detailed similarity diagnostic - shows all components

use std::collections::HashMap;
use symthaea_sentinel::{
    AudioCategory, AudioFeatures, AudioSentinel, FileAudioConfig, FileAudioPump, MEL_BANDS,
    compute_mfcc, compute_onset_strength, compute_spectral_centroid, compute_spectral_flatness,
    compute_temporal_regularity, spectrum_to_mel_bands,
};

/// State for feature extraction with envelope tracking
struct FeatureState {
    prev_spectrum: Vec<f32>,
    onset_history: Vec<f32>,
    rms_history: Vec<f32>,
    prev_rms: f32,
    frame_counter: usize,
}

impl FeatureState {
    fn new() -> Self {
        Self {
            prev_spectrum: Vec::new(),
            onset_history: Vec::new(),
            rms_history: Vec::new(),
            prev_rms: 0.0,
            frame_counter: 0,
        }
    }
}

fn learn_file(path: &str, name: &str, sentinel: &mut AudioSentinel) {
    let config = FileAudioConfig::default();
    let mut pump = match FileAudioPump::new(path, config) {
        Ok(p) => p,
        Err(e) => {
            println!("  Failed: {}", e);
            return;
        }
    };

    sentinel.start_learning(name, AudioCategory::Unknown);
    let mut state = FeatureState::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let features = extract_features(&spectrum, &mut state, &pump);
        sentinel.process(&features);
    }
    sentinel.stop_learning();
}

fn test_file(
    path: &str,
    sentinel: &mut AudioSentinel,
) -> HashMap<String, (f32, f32, f32, f32, f32)> {
    let config = FileAudioConfig::default();
    let mut pump = match FileAudioPump::new(path, config) {
        Ok(p) => p,
        Err(e) => {
            println!("  Failed: {}", e);
            return HashMap::new();
        }
    };

    let mut state = FeatureState::new();
    let mut accum: HashMap<String, (f32, f32, f32, f32, f32, usize)> = HashMap::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let features = extract_features(&spectrum, &mut state, &pump);
        let result = sentinel.process(&features);

        for (name, sim) in &result.similarities {
            let entry = accum
                .entry(name.clone())
                .or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0));
            entry.0 += sim.rhythm;
            entry.1 += sim.timbre;
            entry.2 += sim.rhythm_freq;
            entry.3 += sim.timbre_freq;
            entry.4 += sim.combined;
            entry.5 += 1;
        }
    }

    // Average
    accum
        .into_iter()
        .map(|(k, v)| {
            let n = v.5 as f32;
            (k, (v.0 / n, v.1 / n, v.2 / n, v.3 / n, v.4 / n))
        })
        .collect()
}

fn extract_features(
    spectrum: &[f32],
    state: &mut FeatureState,
    pump: &FileAudioPump,
) -> AudioFeatures {
    let mel_bands = spectrum_to_mel_bands(spectrum, MEL_BANDS, pump.sample_rate());
    let mfcc = compute_mfcc(&mel_bands);
    let spectral_centroid =
        compute_spectral_centroid(spectrum, pump.sample_rate(), pump.window_size());
    let spectral_flatness = compute_spectral_flatness(spectrum);
    let onset_strength = if state.prev_spectrum.is_empty() {
        0.0
    } else {
        compute_onset_strength(&state.prev_spectrum, spectrum)
    };
    state.prev_spectrum = spectrum.to_vec();

    // Track onset history for regularity
    state.onset_history.push(onset_strength);
    if state.onset_history.len() > 100 {
        state.onset_history.remove(0);
    }
    let temporal_regularity = compute_temporal_regularity(&state.onset_history);

    let rms_energy = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

    // Envelope features
    state.rms_history.push(rms_energy);
    if state.rms_history.len() > 32 {
        state.rms_history.remove(0);
    }
    let envelope_delta = (rms_energy - state.prev_rms).abs();
    state.prev_rms = rms_energy;
    let envelope_variance = if state.rms_history.len() > 2 {
        let mean: f32 = state.rms_history.iter().sum::<f32>() / state.rms_history.len() as f32;
        state
            .rms_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / state.rms_history.len() as f32
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
    let spectral_rolloff = rolloff_bin as f32 * (pump.sample_rate() / pump.window_size() as f32);
    let high_freq_energy: f32 = spectrum.iter().skip(spectrum.len() / 2).sum();
    let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);

    let frame_index = state.frame_counter;
    state.frame_counter += 1;

    AudioFeatures {
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

fn main() {
    let mut sentinel = AudioSentinel::new();

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              SIMILARITY COMPONENT DIAGNOSTIC                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Learn Dog
    println!("Learning 'Dog' from dog1.wav...");
    learn_file("test-data/dog/dog1.wav", "Dog", &mut sentinel);

    println!(
        "\nTesting similarity components (Rhythm HDC, Timbre HDC, RhyFreq, TmbFreq, Combined):"
    );
    println!(
        "─────────────────────────────────────────────────────────────────────────────────────"
    );
    println!(
        "{:14} {:>9} {:>9} {:>9} {:>9} │ {:>9} │ {:>8}",
        "Sound", "Rhy HDC", "Tmb HDC", "RhyFreq", "TmbFreq", "Combined", "Expect"
    );
    println!(
        "─────────────────────────────────────────────────────────────────────────────────────"
    );

    let tests = [
        ("test-data/dog/dog2.wav", "Dog2", "MATCH"),
        ("test-data/dog/dog3.wav", "Dog3", "MATCH"),
        ("test-data/glass/glass1.wav", "Glass1", "REJECT"),
        ("test-data/glass/glass2.wav", "Glass2", "REJECT"),
        ("test-data/rain/rain1.wav", "Rain", "REJECT"),
        ("test-data/clock/clock1.wav", "Clock", "REJECT"),
    ];

    for (path, label, expect) in tests {
        let sims = test_file(path, &mut sentinel);
        if let Some((rhythm, timbre, rhy_freq, tmb_freq, combined)) = sims.get("Dog") {
            let color = if *combined > 0.5 {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "{:14} {:9.3} {:9.3} {:9.3} {:9.3} │ {}{:9.3}\x1b[0m │ {}",
                label, rhythm, timbre, rhy_freq, tmb_freq, color, combined, expect
            );
        }
    }

    println!(
        "─────────────────────────────────────────────────────────────────────────────────────"
    );
    println!("\nWeights: Combined = 0.20*RhyHDC + 0.15*TmbHDC + 0.40*RhyFreq + 0.25*TmbFreq");
    println!("\nObservation: Which component has the LEAST variance between MATCH and REJECT?");
}
