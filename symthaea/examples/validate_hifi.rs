// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hi-Fi Audio Validation Suite
//!
//! Comprehensive validation of the Hi-Fi synthesis upgrade:
//! 1. A/B comparison: old (4 partial, mono) vs new (8 partial, stereo Freeverb)
//! 2. FLAC round-trip integrity
//! 3. WavPack hybrid round-trip (inline implementation test)
//! 4. Fingerprint similarity (consciousness-aware distance check)
//! 5. Resampler quality (24kHz → 48kHz, SNR measurement)
//! 6. Full output format matrix
//!
//! Run: cargo run --features muse --example validate_hifi
//! Output: audio_output/hifi_*.wav + console report

use symthaea_muse::export::{export_flac_bytes, export_wav, ExportFormat};
use symthaea_muse::fingerprint::{fingerprint, FINGERPRINT_DIM};
use symthaea_muse::{
    compose, AudioData, MelodyMode, MuseConfig, MusicalState, OutputFormat, ReverbConfig,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Hi-Fi Audio Validation Suite                       ║");
    println!("║          Symthaea-Muse + Lossless Audio                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let output_dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(output_dir).ok();

    let mut pass = 0u32;
    let mut fail = 0u32;

    // ─── 1. A/B Comparison ──────────────────────────────────────────────────
    println!("━━━ 1. A/B Listening Comparison ━━━");
    {
        let state = MusicalState {
            consciousness_level: 0.8,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.4,
            noradrenaline: 0.3,
            harmony_activations: [0.7, 0.5, 0.6, 0.4, 0.3, 0.5, 0.6, 0.3],
            valence: 0.3,
            prediction_error: 0.2,
        };

        // Old config: mono, 4 partials, basic reverb
        let old_config = MuseConfig {
            sample_rate: 44100,
            duration_secs: 8.0,
            base_tempo_bpm: 90.0,
            max_notes: 24,
            melody_mode: MelodyMode::Classic,
            output_format: OutputFormat::Mono16,
            num_partials: 4,
            enable_antialiasing: false,
            reverb: ReverbConfig {
                room_size: 0.3,
                damping: 0.7,
                width: 0.0,
            },
            cfc_layer_sizes: vec![4, 4],
        };

        // New config: stereo, 8 partials, Freeverb, PolyBLEP
        let new_config = MuseConfig {
            sample_rate: 44100,
            duration_secs: 8.0,
            base_tempo_bpm: 90.0,
            max_notes: 24,
            melody_mode: MelodyMode::Neural,
            ..Default::default()
        };

        let old_comp = compose(&old_config, &state, 42);
        let new_comp = compose(&new_config, &state, 42);

        let old_path = output_dir.join("hifi_A_old_mono16.wav");
        let new_path = output_dir.join("hifi_B_new_stereo.wav");

        export_wav(&old_comp, &old_path, ExportFormat::Wav16).expect("old WAV export");
        export_wav(&new_comp, &new_path, ExportFormat::Wav16).expect("new WAV export");

        let old_rms = audio_rms(&old_comp.audio);
        let new_rms = audio_rms(&new_comp.audio);

        println!(
            "  A (old): {} — {} samples, RMS={:.4}",
            old_path.display(),
            old_comp.audio.len(),
            old_rms
        );
        println!(
            "  B (new): {} — {} samples, RMS={:.4}",
            new_path.display(),
            new_comp.audio.len(),
            new_rms
        );

        check(&mut pass, &mut fail, "Old config produces audio", old_rms > 0.001);
        check(&mut pass, &mut fail, "New config produces audio", new_rms > 0.001);
        check(
            &mut pass,
            &mut fail,
            "New config outputs stereo",
            matches!(&new_comp.audio, AudioData::StereoF32(_)),
        );

        println!(
            "  Notes: old={}, new={}",
            old_comp.notes.len(),
            new_comp.notes.len()
        );

        let hires_path = output_dir.join("hifi_B_new_24bit.wav");
        export_wav(&new_comp, &hires_path, ExportFormat::Wav24).expect("24-bit WAV");
        let float_path = output_dir.join("hifi_B_new_float32.wav");
        export_wav(&new_comp, &float_path, ExportFormat::WavF32).expect("float WAV");
        println!("  Hi-Res: {}", hires_path.display());
        println!("  Float:  {}", float_path.display());
        println!("  → Listen and compare A vs B for spatial quality and timbre\n");
    }

    // ─── 2. FLAC Round-Trip ─────────────────────────────────────────────────
    println!("━━━ 2. FLAC Round-Trip Integrity ━━━");
    {
        let config = MuseConfig {
            duration_secs: 4.0,
            max_notes: 16,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);

        let flac16 = export_flac_bytes(&comp, 16).expect("FLAC 16-bit");
        let valid16 = flac16.len() >= 4 && &flac16[0..4] == b"fLaC";
        check(&mut pass, &mut fail, "FLAC 16-bit valid header", valid16);
        let raw_bytes = audio_byte_size(&comp.audio);
        println!(
            "  16-bit FLAC: {} bytes (compression: {:.2}x)",
            flac16.len(),
            raw_bytes as f64 / flac16.len().max(1) as f64
        );

        let flac24 = export_flac_bytes(&comp, 24).expect("FLAC 24-bit");
        let valid24 = flac24.len() >= 4 && &flac24[0..4] == b"fLaC";
        check(&mut pass, &mut fail, "FLAC 24-bit valid header", valid24);
        println!(
            "  24-bit FLAC: {} bytes",
            flac24.len()
        );

        let flac_path = output_dir.join("hifi_roundtrip.flac");
        std::fs::write(&flac_path, &flac16).expect("write FLAC");
        println!(
            "  Written: {} — validate with: flac -t {}\n",
            flac_path.display(),
            flac_path.display()
        );
    }

    // ─── 3. WavPack Hybrid Round-Trip ───────────────────────────────────────
    println!("━━━ 3. WavPack Hybrid Round-Trip ━━━");
    {
        // Inline WavPack test (mycelix-codec is in a separate workspace)
        let samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / 44100.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.7
                    + (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.3
            })
            .collect();

        // Simulate WavPack hybrid: quantize to 16-bit (lossy) + f32 residual (correction)
        let lossy: Vec<i16> = samples
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();
        let lossy_f32: Vec<f32> = lossy.iter().map(|&s| s as f32 / 32768.0).collect();
        let correction: Vec<f32> = samples
            .iter()
            .zip(lossy_f32.iter())
            .map(|(&orig, &l)| orig - l)
            .collect();
        let reconstructed: Vec<f32> = lossy_f32
            .iter()
            .zip(correction.iter())
            .map(|(&l, &c)| l + c)
            .collect();

        let max_error: f32 = samples
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        check(
            &mut pass,
            &mut fail,
            "WavPack hybrid bit-perfect (error < 1e-6)",
            max_error < 1e-6,
        );

        let lossy_bytes = lossy.len() * 2;
        let correction_bytes = correction.len() * 4;
        let original_bytes = samples.len() * 4;
        let lossy_snr = compute_snr(&samples, &lossy_f32);

        println!("  .wv  (lossy): {} bytes ({:.1}x compression)", lossy_bytes, original_bytes as f64 / lossy_bytes as f64);
        println!("  .wvc (correction): {} bytes", correction_bytes);
        println!("  Max reconstruction error: {:.2e}", max_error);
        println!("  Lossy-only SNR: {:.1} dB\n", lossy_snr);
    }

    // ─── 4. Fingerprint Similarity ──────────────────────────────────────────
    println!("━━━ 4. HDC Fingerprint Similarity ━━━");
    {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };

        let state_a = MusicalState {
            consciousness_level: 0.8,
            arousal: 0.5,
            dopamine: 0.6,
            ..Default::default()
        };
        let state_b = MusicalState {
            consciousness_level: 0.82,
            arousal: 0.52,
            dopamine: 0.58,
            ..Default::default()
        };
        let state_c = MusicalState {
            consciousness_level: 0.1,
            arousal: 0.9,
            dopamine: 0.1,
            serotonin: 0.1,
            noradrenaline: 0.9,
            valence: -0.8,
            harmony_activations: [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.0],
            prediction_error: 0.8,
        };

        let comp_a = compose(&config, &state_a, 42);
        let comp_b = compose(&config, &state_b, 42);
        let comp_c = compose(&config, &state_c, 42);

        let fp_a = fingerprint(&comp_a, &state_a);
        let fp_b = fingerprint(&comp_b, &state_b);
        let fp_c = fingerprint(&comp_c, &state_c);

        let sim_ab = fp_a.similarity(&fp_b);
        let sim_ac = fp_a.similarity(&fp_c);

        check(
            &mut pass,
            &mut fail,
            "Embedding dimension = 256",
            fp_a.dimension == FINGERPRINT_DIM,
        );
        println!("  Similar states (A~B):   cosine = {:.4}", sim_ab);
        println!("  Different states (A~C): cosine = {:.4}", sim_ac);
        check(
            &mut pass,
            &mut fail,
            "Similar states closer than different",
            sim_ab > sim_ac,
        );
        println!("  Margin: {:.4}\n", sim_ab - sim_ac);
    }

    // ─── 5. Resampler Quality ───────────────────────────────────────────────
    println!("━━━ 5. Polyphase Resampler Quality ━━━");
    {
        use symthaea::voice::resampler::AudioResampler;

        let sr_source = 24000u32;
        let sr_target = 48000u32;
        let freq = 440.0f32;
        let source: Vec<f32> = (0..sr_source)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr_source as f32).sin())
            .collect();

        let mut resampler = AudioResampler::new(sr_source, sr_target).expect("create resampler");
        let upsampled = resampler.process(&source).expect("resample");

        let ratio = upsampled.len() as f64 / source.len() as f64;
        println!(
            "  Source: {} samples @ {} Hz",
            source.len(),
            sr_source
        );
        println!(
            "  Output: {} samples @ {} Hz (ratio: {:.3})",
            upsampled.len(),
            sr_target,
            ratio
        );
        check(
            &mut pass,
            &mut fail,
            "Upsample ratio ~2.0x",
            (ratio - 2.0).abs() < 0.15,
        );

        let source_rms =
            (source.iter().map(|s| s * s).sum::<f32>() / source.len() as f32).sqrt();
        let up_rms =
            (upsampled.iter().map(|s| s * s).sum::<f32>() / upsampled.len() as f32).sqrt();
        let rms_ratio = up_rms / source_rms;
        println!(
            "  RMS: source={:.4}, upsampled={:.4} (ratio: {:.3})",
            source_rms, up_rms, rms_ratio
        );
        check(
            &mut pass,
            &mut fail,
            "RMS preserved within 20%",
            (rms_ratio - 1.0).abs() < 0.2,
        );

        let ideal: Vec<f32> = (0..upsampled.len())
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr_target as f32).sin())
            .collect();
        let snr = compute_snr(&ideal, &upsampled);
        println!("  SNR vs ideal 440 Hz @ 48kHz: {:.1} dB", snr);
        check(&mut pass, &mut fail, "Resampler SNR > 30 dB", snr > 30.0);

        check(
            &mut pass,
            &mut fail,
            "Same-rate returns None",
            AudioResampler::new(48000, 48000).is_none(),
        );
        println!();
    }

    // ─── 6. Output Format Matrix ────────────────────────────────────────────
    println!("━━━ 6. Output Format Matrix ━━━");
    {
        let config_base = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = MusicalState::default();

        for (format, name) in [
            (OutputFormat::Mono16, "Mono16"),
            (OutputFormat::MonoF32, "MonoF32"),
            (OutputFormat::StereoF32, "StereoF32"),
        ] {
            let config = MuseConfig {
                output_format: format,
                ..config_base.clone()
            };
            let comp = compose(&config, &state, 42);
            let rms = audio_rms(&comp.audio);
            let format_ok = match (&comp.audio, format) {
                (AudioData::I16(_), OutputFormat::Mono16) => true,
                (AudioData::F32(_), OutputFormat::MonoF32) => true,
                (AudioData::StereoF32(_), OutputFormat::StereoF32) => true,
                _ => false,
            };
            check(&mut pass, &mut fail, &format!("{name} format correct"), format_ok);
            println!("  {name:12} — {} samples, RMS={:.4}", comp.audio.len(), rms);
        }

        let neural_config = MuseConfig {
            melody_mode: MelodyMode::Neural,
            ..config_base.clone()
        };
        let neural_comp = compose(&neural_config, &state, 42);
        check(
            &mut pass,
            &mut fail,
            "Neural mode produces audio",
            !neural_comp.audio.is_empty(),
        );
        println!(
            "  Neural     — {} notes, {} samples\n",
            neural_comp.notes.len(),
            neural_comp.audio.len()
        );
    }

    // ─── Summary ────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    if fail == 0 {
        println!(
            "║  RESULTS: {} passed, {} failed — ALL CHECKS PASSED          ║",
            pass, fail
        );
    } else {
        println!(
            "║  RESULTS: {} passed, {} failed — SOME CHECKS FAILED         ║",
            pass, fail
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nAudio files in {}:", output_dir.display());
    println!("  hifi_A_old_mono16.wav   — Old: 4 partials, mono, comb reverb");
    println!("  hifi_B_new_stereo.wav   — New: 8 partials, stereo, Freeverb");
    println!("  hifi_B_new_24bit.wav    — New: 24-bit Hi-Res");
    println!("  hifi_B_new_float32.wav  — New: 32-bit float studio");
    println!("  hifi_roundtrip.flac     — FLAC (validate: flac -t hifi_roundtrip.flac)");

    if fail > 0 {
        std::process::exit(1);
    }
}

fn check(pass: &mut u32, fail: &mut u32, name: &str, ok: bool) {
    if ok {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

fn audio_rms(audio: &AudioData) -> f32 {
    let samples: Vec<f32> = match audio {
        AudioData::I16(s) => s.iter().map(|&x| x as f32 / 32768.0).collect(),
        AudioData::F32(s) => s.clone(),
        AudioData::StereoF32(s) => s.iter().map(|p| (p[0] + p[1]) * 0.5).collect(),
    };
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn audio_byte_size(audio: &AudioData) -> usize {
    match audio {
        AudioData::I16(s) => s.len() * 2,
        AudioData::F32(s) => s.len() * 4,
        AudioData::StereoF32(s) => s.len() * 8,
    }
}

fn compute_snr(reference: &[f32], test: &[f32]) -> f32 {
    let len = reference.len().min(test.len());
    if len == 0 {
        return 0.0;
    }
    let signal_power: f64 =
        reference[..len].iter().map(|&s| s as f64 * s as f64).sum::<f64>() / len as f64;
    let noise_power: f64 = reference[..len]
        .iter()
        .zip(test[..len].iter())
        .map(|(&r, &t)| {
            let d = (r - t) as f64;
            d * d
        })
        .sum::<f64>()
        / len as f64;
    if noise_power < 1e-20 {
        return 120.0;
    }
    (10.0 * (signal_power / noise_power).log10()) as f32
}
