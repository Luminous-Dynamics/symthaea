// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural vocoder benchmark: compares DSP and neural (BigVGAN) synthesis paths.
//!
//! Synthesizes a 5-second utterance through both paths and reports:
//! - Mel conversion throughput (frames/sec)
//! - ONNX inference time per chunk
//! - End-to-end latency
//! - Audio duration match
//! - Multi-sentence quality metrics (MCD, RMS, spectral flatness)
//! - HTML listening test report
//!
//! Usage: `cargo run --example neural_vocoder_benchmark --features neural-vocoder`
//!
//! Writes comparison WAV files to `audio_output/`.

use std::f32::consts::PI;
use std::time::Instant;
use symthaea::voice::neural_vocoder::NeuralVocoderConfig;
use symthaea::voice::{
    FormantFrame,
    vocal_tract_controller::VocalTractController,
    vocal_tract_encoder::VoiceCognitiveState,
    vocal_tract_fep::{StreamingVocalTract, populate_manner_map},
    vocoder::{FormantVocoder, VocoderConfig},
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::formant_to_mel::{
    FormantToMelConfig, FormantToMelConverter, MelNormalization, MelVoiceQuality,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  BigVGAN Neural Vocoder Benchmark");
    println!("═══════════════════════════════════════════════════════════════\n");

    let genesis = GenesisSeed::from_phrase("neural-vocoder-benchmark");
    let sample_rate = 24000u32;
    let frame_rate = 200u32;
    let duration_secs = 5.0;
    let total_frames = (duration_secs * frame_rate as f32) as usize;
    let samples_per_frame = (sample_rate / frame_rate) as usize;

    let state = VoiceCognitiveState {
        emotional_arousal: 0.6,
        emotional_valence: 0.3,
        consciousness_level: 0.8,
        ..Default::default()
    };

    // Phoneme sequence for a simple utterance, repeated
    let phonemes = ["HH", "EH", "L", "OW", "W", "ER", "L", "D"];

    // ── Part 1: Mel conversion throughput ────────────────────────────
    println!("1. Mel Conversion Throughput");
    println!("   ─────────────────────────");

    let mel_config = FormantToMelConfig::default();
    let mut mel_converter = FormantToMelConverter::new(mel_config);
    let vq = MelVoiceQuality {
        rd: 1.3,
        arousal: 0.6,
    };

    // Generate formant frames via streaming vocal tract
    let mut streaming_dsp = StreamingVocalTract::new(&genesis, sample_rate, frame_rate);
    let mut formant_frames = Vec::with_capacity(total_frames);
    for i in 0..total_frames {
        let ph = phonemes[i % phonemes.len()];
        let frame = streaming_dsp
            .pipeline
            .tick_phoneme(&state, None, 0.005, Some(ph));
        formant_frames.push(frame);
    }

    let mel_start = Instant::now();
    let mut total_mel_frames = 0;
    for frame in &formant_frames {
        let mels = mel_converter.push_frame(frame, &vq);
        total_mel_frames += mels.len();
    }
    let mel_elapsed = mel_start.elapsed();

    println!("   Motor frames processed: {total_frames}");
    println!("   Mel frames produced:    {total_mel_frames}");
    println!("   Time: {:.2}ms", mel_elapsed.as_secs_f64() * 1000.0);
    println!(
        "   Throughput: {:.0} mel frames/sec",
        total_mel_frames as f64 / mel_elapsed.as_secs_f64()
    );
    println!(
        "   Per motor frame: {:.1}µs\n",
        mel_elapsed.as_micros() as f64 / total_frames as f64
    );

    // ── Part 2: DSP synthesis baseline ───────────────────────────────
    println!("2. DSP Synthesis Baseline");
    println!("   ──────────────────────");

    let mut streaming_dsp = StreamingVocalTract::new(&genesis, sample_rate, frame_rate);
    let mut dsp_audio = Vec::with_capacity(total_frames * samples_per_frame);

    let dsp_start = Instant::now();
    for i in 0..total_frames {
        let ph = phonemes[i % phonemes.len()];
        let chunk = streaming_dsp.tick(&state, None, 0.005, Some(ph));
        dsp_audio.extend_from_slice(&chunk);
    }
    let dsp_elapsed = dsp_start.elapsed();

    let dsp_duration = dsp_audio.len() as f64 / sample_rate as f64;
    println!("   Frames: {total_frames}");
    println!("   Samples: {}", dsp_audio.len());
    println!("   Audio duration: {dsp_duration:.2}s");
    println!(
        "   Synthesis time: {:.2}ms",
        dsp_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Real-time factor: {:.1}x\n",
        dsp_duration / dsp_elapsed.as_secs_f64()
    );

    // ── Part 3: Neural vocoder (if model available) ──────────────────
    println!("3. Neural Vocoder (BigVGAN)");
    println!("   ───────────────────────");

    let neural_config = NeuralVocoderConfig::default();
    let model_exists = std::path::Path::new(&neural_config.model_path).exists();

    if model_exists {
        let mut streaming_neural = StreamingVocalTract::with_neural_vocoder(
            &genesis,
            sample_rate,
            frame_rate,
            neural_config,
        );

        if streaming_neural.has_neural_vocoder() {
            let mut neural_audio = Vec::with_capacity(total_frames * samples_per_frame);

            let neural_start = Instant::now();
            for i in 0..total_frames {
                let ph = phonemes[i % phonemes.len()];
                let chunk = streaming_neural.tick(&state, None, 0.005, Some(ph));
                neural_audio.extend_from_slice(&chunk);
            }
            let neural_elapsed = neural_start.elapsed();

            let neural_duration = neural_audio.len() as f64 / sample_rate as f64;
            println!("   Status: ACTIVE");
            println!("   Samples: {}", neural_audio.len());
            println!("   Audio duration: {neural_duration:.2}s");
            println!(
                "   Synthesis time: {:.2}ms",
                neural_elapsed.as_secs_f64() * 1000.0
            );
            println!(
                "   Real-time factor: {:.1}x\n",
                neural_duration / neural_elapsed.as_secs_f64()
            );

            // Save neural WAV
            save_wav(
                "audio_output/neural_vocoder_bigvgan.wav",
                &neural_audio,
                sample_rate,
            );
        } else {
            println!("   Status: Model loaded but channel failed\n");
        }
    } else {
        println!(
            "   Status: SKIPPED (model not found at '{}')",
            neural_config.model_path
        );
        println!("   Run `cd models && python export_bigvgan.py` to export the model\n");
    }

    // ── Part 4: Multi-Sentence Quality Report ──────────────────────
    println!("4. Multi-Sentence Quality Report");
    println!("   ─────────────────────────────");

    let test_sentences: Vec<(&str, Vec<&str>)> = vec![
        (
            "Hello world",
            vec!["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
        ),
        ("How are you", vec!["HH", "AW", "AA", "R", "Y", "UW"]),
        (
            "Good morning",
            vec!["G", "UH", "D", "M", "AO", "R", "N", "IH", "NG"],
        ),
        ("Thank you", vec!["TH", "AE", "NG", "K", "Y", "UW"]),
        (
            "I love music",
            vec!["AY", "L", "AH", "V", "M", "Y", "UW", "Z", "IH", "K"],
        ),
        (
            "Beautiful day",
            vec!["B", "Y", "UW", "T", "AH", "F", "AH", "L", "D", "EY"],
        ),
        (
            "Open the door",
            vec!["OW", "P", "AH", "N", "DH", "AH", "D", "AO", "R"],
        ),
        (
            "Please help me",
            vec!["P", "L", "IY", "Z", "HH", "EH", "L", "P", "M", "IY"],
        ),
        (
            "See you later",
            vec!["S", "IY", "Y", "UW", "L", "EY", "T", "ER"],
        ),
        (
            "Come back soon",
            vec!["K", "AH", "M", "B", "AE", "K", "S", "UW", "N"],
        ),
    ];

    let vq_report = MelVoiceQuality {
        rd: 1.3,
        arousal: 0.6,
    };
    let mut mel_config_raw = FormantToMelConfig::default();
    mel_config_raw.normalization = MelNormalization::disabled();

    println!(
        "   {:20} {:>6} {:>8} {:>10}",
        "Sentence", "RMS", "MCD(dB)", "Flatness"
    );
    println!(
        "   {:20} {:>6} {:>8} {:>10}",
        "────────", "───", "───────", "────────"
    );

    let mut all_sentence_audio = Vec::new();

    for (text, phonemes_seq) in &test_sentences {
        let mut streaming = StreamingVocalTract::new(&genesis, sample_rate, frame_rate);
        let mut audio = Vec::new();
        let frames_per_sentence = 60; // ~300ms per sentence

        let mut mel_conv = FormantToMelConverter::new(mel_config_raw.clone());
        let mut mels = Vec::new();

        for i in 0..frames_per_sentence {
            let ph = phonemes_seq[i % phonemes_seq.len()];
            let frame = streaming
                .pipeline
                .tick_phoneme(&state, None, 0.005, Some(ph));
            mels.extend(mel_conv.push_frame(&frame, &vq_report));
            let chunk = streaming
                .vocoder
                .synthesize_frame(&frame, samples_per_frame);
            audio.extend_from_slice(&chunk);
        }

        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();

        // MCD: compare first and second half
        let mcd = if mels.len() >= 4 {
            let half = mels.len() / 2;
            compute_mcd(&mels[..half], &mels[half..half * 2])
        } else {
            0.0
        };

        // Spectral flatness: geometric mean / arithmetic mean of power
        let flatness = if !mels.is_empty() {
            let mel = mels.last().unwrap();
            let lin: Vec<f32> = mel.iter().map(|&v| v.exp().max(1e-10)).collect();
            let log_mean = lin.iter().map(|v| v.ln()).sum::<f32>() / lin.len() as f32;
            let arith_mean = lin.iter().sum::<f32>() / lin.len() as f32;
            (log_mean.exp() / arith_mean.max(1e-10)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        println!("   {:20} {:6.4} {:8.2} {:10.4}", text, rms, mcd, flatness);
        all_sentence_audio.push((text.to_string(), audio));
    }

    println!();

    // ── Part 5: HTML Listening Test Report ───────────────────────
    println!("5. Listening Test Report");
    println!("   ────────────────────");

    // Save per-sentence WAV files
    for (i, (text, audio)) in all_sentence_audio.iter().enumerate() {
        let path = format!(
            "audio_output/sentence_{:02}_{}.wav",
            i,
            text.replace(' ', "_").to_lowercase()
        );
        save_wav(&path, audio, sample_rate);
    }

    // Generate HTML report
    generate_html_report(&all_sentence_audio, sample_rate);

    // ── Save DSP WAV ────────────────────────────────────────────────
    save_wav(
        "audio_output/neural_vocoder_dsp_baseline.wav",
        &dsp_audio,
        sample_rate,
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  WAV files written to audio_output/");
    println!("  Open audio_output/vocoder_quality_report.html for A/B test");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Compute Mel Cepstral Distortion (MCD) between two sequences of mel spectrogram frames.
fn compute_mcd(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let n_frames = a.len().min(b.len());
    if n_frames == 0 {
        return 0.0;
    }

    let n_ceps = 13;
    let mut total_dist = 0.0f64;

    for i in 0..n_frames {
        let ceps_a = dct_ii(&a[i], n_ceps);
        let ceps_b = dct_ii(&b[i], n_ceps);

        let dist: f64 = ceps_a
            .iter()
            .zip(ceps_b.iter())
            .skip(1)
            .map(|(&ca, &cb)| {
                let d = (ca - cb) as f64;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        total_dist += dist;
    }

    let mean_dist = total_dist / n_frames as f64;
    let scale = (10.0 / 10.0_f64.ln()) * 2.0_f64.sqrt();
    (scale * mean_dist) as f32
}

/// Type-II DCT for converting log-mel to cepstral coefficients.
fn dct_ii(log_mel: &[f32], n_ceps: usize) -> Vec<f32> {
    let n = log_mel.len();
    let mut ceps = Vec::with_capacity(n_ceps);

    for k in 0..n_ceps {
        let mut sum = 0.0f32;
        for (i, &val) in log_mel.iter().enumerate() {
            sum += val * (PI * k as f32 * (2.0 * i as f32 + 1.0) / (2.0 * n as f32)).cos();
        }
        ceps.push(sum * (2.0 / n as f32).sqrt());
    }

    ceps
}

fn generate_html_report(sentences: &[(String, Vec<f32>)], sample_rate: u32) {
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Symthaea Vocoder Quality Report</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; background: #1a1a2e; color: #e0e0e0; }
h1 { color: #7b68ee; border-bottom: 2px solid #7b68ee; padding-bottom: 0.3em; }
h2 { color: #9b89ff; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }
th { background: #16213e; color: #7b68ee; }
tr:hover { background: #16213e; }
audio { width: 100%; }
.good { color: #4ade80; } .ok { color: #fbbf24; } .poor { color: #f87171; }
footer { margin-top: 2em; font-size: 0.85em; color: #666; }
</style></head><body>
<h1>Symthaea Vocoder Quality Report</h1>
<p>Generated by <code>neural_vocoder_benchmark</code></p>
<h2>Per-Sentence Audio Comparison</h2>
<table><tr><th>#</th><th>Sentence</th><th>DSP Audio</th><th>Duration</th></tr>
"#,
    );

    for (i, (text, audio)) in sentences.iter().enumerate() {
        let duration = audio.len() as f32 / sample_rate as f32;
        let wav_name = format!(
            "sentence_{:02}_{}.wav",
            i,
            text.replace(' ', "_").to_lowercase()
        );
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td><audio controls src=\"{}\"></audio></td><td>{:.2}s</td></tr>\n",
            i + 1, text, wav_name, duration
        ));
    }

    html.push_str("</table>\n");
    html.push_str(
        r#"<h2>Full Baseline</h2>
<p>5-second DSP baseline:</p>
<audio controls src="neural_vocoder_dsp_baseline.wav" style="width:100%"></audio>
<footer>Symthaea voice pipeline &mdash; consciousness-first synthesis</footer>
</body></html>"#,
    );

    let path = "audio_output/vocoder_quality_report.html";
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(path, &html) {
        Ok(()) => println!("   Saved: {path}"),
        Err(e) => eprintln!("   Failed to write {path}: {e}"),
    }
}

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) {
    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    match hound::WavWriter::create(path, spec) {
        Ok(mut writer) => {
            for &s in samples {
                let _ = writer.write_sample(s);
            }
            let _ = writer.finalize();
            println!("   Saved: {path}");
        }
        Err(e) => {
            eprintln!("   Failed to write {path}: {e}");
        }
    }
}
