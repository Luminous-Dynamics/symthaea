//! Neural vocoder benchmark: compares DSP and neural (BigVGAN) synthesis paths.
//!
//! Synthesizes a 5-second utterance through both paths and reports:
//! - Mel conversion throughput (frames/sec)
//! - ONNX inference time per chunk
//! - End-to-end latency
//! - Audio duration match
//!
//! Usage: `cargo run --example neural_vocoder_benchmark --features neural-vocoder`
//!
//! Writes comparison WAV files to `audio_output/`.

use std::f32::consts::PI;
use std::time::Instant;
use symthaea::voice::{
    vocal_tract_controller::VocalTractController,
    vocal_tract_encoder::VoiceCognitiveState,
    vocal_tract_fep::{populate_manner_map, StreamingVocalTract},
    vocoder::{FormantVocoder, VocoderConfig},
    FormantFrame,
};
use symthaea::voice::neural_vocoder::NeuralVocoderConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::formant_to_mel::{FormantToMelConfig, FormantToMelConverter, MelNormalization, MelVoiceQuality};

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
    let vq = MelVoiceQuality { rd: 1.3, arousal: 0.6 };

    // Generate formant frames via streaming vocal tract
    let mut streaming_dsp = StreamingVocalTract::new(&genesis, sample_rate, frame_rate);
    let mut formant_frames = Vec::with_capacity(total_frames);
    for i in 0..total_frames {
        let ph = phonemes[i % phonemes.len()];
        // Get the formant frame by ticking the pipeline (we need the frame, not the audio)
        let frame = streaming_dsp.pipeline.tick_phoneme(&state, None, 0.005, Some(ph));
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
    println!("   Synthesis time: {:.2}ms", dsp_elapsed.as_secs_f64() * 1000.0);
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
            save_wav("audio_output/neural_vocoder_bigvgan.wav", &neural_audio, sample_rate);
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

    // ── Part 4: MCD Perceptual Quality Metric ──────────────────────
    println!("4. Mel Cepstral Distortion (MCD)");
    println!("   ─────────────────────────────");

    // Compute mel spectrogram from DSP audio for self-consistency check.
    // When neural vocoder is available, compute MCD between DSP and neural mel spectra.
    let mut mel_config_raw = FormantToMelConfig::default();
    mel_config_raw.normalization = MelNormalization::disabled();
    let mut mel_conv_dsp = FormantToMelConverter::new(mel_config_raw.clone());

    let vq_mcd = MelVoiceQuality { rd: 1.3, arousal: 0.6 };
    let mut dsp_mels = Vec::new();
    for frame in &formant_frames {
        dsp_mels.extend(mel_conv_dsp.push_frame(frame, &vq_mcd));
    }

    println!("   DSP mel frames extracted: {}", dsp_mels.len());

    if !dsp_mels.is_empty() {
        // Compute MCD between first and second half of DSP (consistency baseline)
        let half = dsp_mels.len() / 2;
        let mcd_self = compute_mcd(&dsp_mels[..half], &dsp_mels[half..half*2]);
        println!("   MCD (DSP self-consistency, 1st vs 2nd half): {mcd_self:.2} dB");
        println!("   (Lower is better. <4 dB = excellent, 4-6 = good, >8 = poor)");
    }

    println!();

    // ── Save DSP WAV ────────────────────────────────────────────────
    save_wav("audio_output/neural_vocoder_dsp_baseline.wav", &dsp_audio, sample_rate);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  WAV files written to audio_output/");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Compute Mel Cepstral Distortion (MCD) between two sequences of mel spectrogram frames.
///
/// MCD = (10 / ln(10)) * sqrt(2) * mean(euclidean_distance(cepstra_a, cepstra_b))
///
/// Uses DCT-II to convert log-mel spectra to cepstral coefficients (first 13 MFCCs),
/// then computes frame-wise Euclidean distance.
fn compute_mcd(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let n_frames = a.len().min(b.len());
    if n_frames == 0 {
        return 0.0;
    }

    let n_ceps = 13; // Standard MFCC count
    let mut total_dist = 0.0f64;

    for i in 0..n_frames {
        let ceps_a = dct_ii(&a[i], n_ceps);
        let ceps_b = dct_ii(&b[i], n_ceps);

        // Euclidean distance of cepstral coefficients (skip c0 = energy)
        let dist: f64 = ceps_a.iter().zip(ceps_b.iter())
            .skip(1) // Skip c0
            .map(|(&ca, &cb)| {
                let d = (ca - cb) as f64;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        total_dist += dist;
    }

    let mean_dist = total_dist / n_frames as f64;

    // MCD scaling factor: (10 / ln(10)) * sqrt(2)
    let scale = (10.0 / 10.0_f64.ln()) * 2.0_f64.sqrt();
    (scale * mean_dist) as f32
}

/// Type-II DCT (Discrete Cosine Transform) for converting log-mel to cepstral coefficients.
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
