// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Round-trip intelligibility eval: the mouth and the semantic ear grade each other.
//!
//! Speak each test sentence with the native HDC/CfC formant voice, transcribe
//! the produced audio with the Whisper JSONL worker, and score word error rate
//! against the original text (normalized scoring from `symthaea-communication`
//! — the single scoring authority, per voice plan P4/WER-dedup).
//!
//! This turns "voice quality" from vibes into a number: a committable baseline
//! and a regression gate for any change to G2P, formant tables, or the
//! vocal-tract controller. It is also the future source for the
//! `VoiceOutputMetrics.listener_prediction` field (hardcoded 0.5 until now —
//! this eval is what "a listener could predict what was said" actually means).
//!
//! Usage (worker needs the provisioned Whisper model):
//! ```bash
//! SYMTHAEA_WHISPER_MODEL_PATH=/srv/luminous-dynamics/models/communication/whisper-large-v3 \
//! cargo run --example voice_roundtrip_wer --features voice-tts [--release] -- \
//!     [--worker communication/worker/run_whisper_nixos.sh] \
//!     [--report /path/to/report.json] [--save-wavs /tmp/dir]
//! ```
//!
//! Interpretation note: Whisper large-v3 scores ~4.8% WER (normalized) on real
//! English speech, so round-trip WER here overwhelmingly measures the *mouth*,
//! not the ear. Expect a high baseline — the formant voice was never
//! intelligibility-optimized. The point is the number and its trend, not
//! flattery.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use symthaea::voice::repl_voice::{ReplVoiceConfig, ReplVoiceOutput};
use symthaea::voice::transcribe_samples;
use symthaea_communication::human::{LocalJsonlProvider, WorkerPolicy};
use symthaea_communication::metrics::{normalize_for_scoring, word_error_rate};

const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Test sentences, easiest→hardest for the current G2P:
/// the first few are fully in the hand dictionary; later ones exercise the
/// letter-fallback path and longer utterances.
const SENTENCES: &[&str] = &[
    "hello world",
    "I feel awareness",
    "I can see the light",
    "this is calm and still",
    "I know something is rising",
    "the world feels right now",
    "peace and harmony",
    "I think I can hear you",
    "consciousness is expanding here",
    "the quick brown fox jumps over the lazy dog",
    "please open the pod bay doors",
    "speech synthesis quality measurement",
];

fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..output_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            match (input.get(idx), input.get(idx + 1)) {
                (Some(&a), Some(&b)) => a * (1.0 - frac) + b * frac,
                (Some(&a), None) => a,
                _ => 0.0,
            }
        })
        .collect()
}

fn save_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Load a distillation corpus: `manifest.tsv` lines of `NN.wav<TAB>text`,
/// each WAV run through the formant extractor to produce teacher targets.
#[cfg(feature = "vocal-tract")]
fn load_distill_corpus(
    dir: &std::path::Path,
) -> anyhow::Result<Vec<(String, Vec<symthaea::voice::FormantFrame>)>> {
    use symthaea_vocal_tract::formant_extraction::{ExtractionConfig, extract_formant_frames};

    let manifest = std::fs::read_to_string(dir.join("manifest.tsv"))?;
    let extraction = ExtractionConfig::default();
    let mut corpus = Vec::new();
    for line in manifest.lines() {
        let Some((wav, text)) = line.split_once('\t') else {
            continue;
        };
        let mut reader = hound::WavReader::open(dir.join(wav))?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<Result<_, _>>()?
            }
        };
        let frames = extract_formant_frames(&samples, spec.sample_rate, &extraction);
        if !frames.is_empty() {
            corpus.push((text.to_string(), frames));
        }
    }
    Ok(corpus)
}

/// Contiguous runs of confidently-voiced vowel frames ("voiced islands").
/// `min_len` filters single-frame LPC blips (3 frames = 15ms at 200Hz).
#[cfg(feature = "vocal-tract")]
fn voiced_islands(frames: &[symthaea::voice::FormantFrame], min_len: usize) -> Vec<(usize, usize)> {
    use symthaea_vocal_tract::types::SourceType;
    let mut islands = Vec::new();
    let mut start: Option<usize> = None;
    for (i, f) in frames.iter().enumerate() {
        let voiced = f.source_type == SourceType::Vowel && f.voicing > 0.5;
        match (voiced, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if i - s >= min_len {
                    islands.push((s, i));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start
        && frames.len() - s >= min_len
    {
        islands.push((s, frames.len()));
    }
    islands
}

/// Distillation v2/v3: per-phoneme MEDIAN formant targets from the teacher
/// corpus.
///
/// v2 (duration-proportional core window) gated 100.0% vs 98.6% control —
/// diagnosed via `symthaea-vocal-tract::f1_probe` as an alignment failure,
/// not an extractor or teacher-voice-f0 issue (whole-utterance F1
/// distributions are healthy; only the per-phoneme attribution was wrong).
/// v3 fixes this the way the v1 diagnosis called out: anchor each vowel
/// phoneme to the nearest actual **voiced island** in the extracted track
/// (a contiguous confidently-voiced run) instead of a proportional-timing
/// guess. Vowels are matched to islands nearest their duration-proportional
/// midpoint, greedily, without reuse; a vowel with no island left falls back
/// to the old proportional core-window (rare — only when islands run out).
#[cfg(feature = "vocal-tract")]
fn median_vowel_targets(
    corpus: &[(String, Vec<symthaea::voice::FormantFrame>)],
    base_duration: f32,
    min_samples: usize,
) -> Vec<(String, symthaea::voice::FormantTarget)> {
    use std::collections::HashMap;
    use symthaea::voice::SimpleG2P;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea_vocal_tract::types::SourceType;

    let g2p = SimpleG2P::new();
    let db = FormantDatabase::new();
    let mut samples: HashMap<String, Vec<(f32, f32, f32)>> = HashMap::new();

    for (text, ref_frames) in corpus {
        if ref_frames.is_empty() {
            continue;
        }
        let phonemes = g2p.text_to_phonemes(text, base_duration);
        let total_dur: f32 = phonemes.iter().map(|p| p.duration).sum();
        if total_dur <= 0.0 {
            continue;
        }
        let n = ref_frames.len() as f32;
        let islands = voiced_islands(ref_frames, 3);
        let mut island_used = vec![false; islands.len()];

        // Pass 1: duration-proportional spans + vowel midpoints.
        let mut t_cursor = 0.0f32;
        let mut spans = Vec::with_capacity(phonemes.len());
        for tp in &phonemes {
            let span_start = (t_cursor / total_dur) * n;
            t_cursor += tp.duration;
            let span_end = ((t_cursor / total_dur) * n).min(n);
            spans.push((span_start, span_end));
        }

        // Pass 2: greedy nearest-island assignment for vowels, in
        // proportional-time order (so earlier vowels don't steal a later
        // vowel's closer island).
        let mut vowel_order: Vec<usize> = (0..phonemes.len())
            .filter(|&i| {
                phonemes[i].phoneme != "SIL"
                    && phonemes[i].phoneme != "SP"
                    && db.lookup(&phonemes[i].phoneme).is_some_and(|t| t.is_vowel)
            })
            .collect();
        vowel_order.sort_by(|&a, &b| spans[a].0.total_cmp(&spans[b].0));

        for &pi in &vowel_order {
            let (span_start, span_end) = spans[pi];
            let midpoint = (span_start + span_end) * 0.5;

            let nearest = islands
                .iter()
                .enumerate()
                .filter(|(idx, _)| !island_used[*idx])
                .map(|(idx, &(s, e))| {
                    let center = (s + e) as f32 * 0.5;
                    (idx, (center - midpoint).abs())
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));

            let frame_range: Option<(usize, usize)> = if let Some((idx, _)) = nearest {
                island_used[idx] = true;
                let (s, e) = islands[idx];
                // Trim onset/offset transition frames when the island is
                // long enough to spare them.
                if e - s >= 5 {
                    Some((s + 1, e - 1))
                } else {
                    Some((s, e))
                }
            } else {
                // No island left: fall back to the old proportional core
                // window (middle 50% of the span).
                let width = span_end - span_start;
                let core_start = (span_start + width * 0.25) as usize;
                let core_end = ((span_start + width * 0.75) as usize).min(ref_frames.len());
                (core_start < core_end).then_some((core_start, core_end))
            };

            let Some((fs, fe)) = frame_range else {
                continue;
            };
            for frame in &ref_frames[fs..fe.min(ref_frames.len())] {
                if frame.source_type == SourceType::Vowel && frame.voicing > 0.5 {
                    samples
                        .entry(phonemes[pi].phoneme.clone())
                        .or_default()
                        .push((frame.f1, frame.f2, frame.f3));
                }
            }
        }
    }

    finalize_median_targets(samples, &db, min_samples)
}

/// Shared tail for all median-target distillation variants (v2 duration-
/// proportional, v3 voiced-island, v4 DTW): given per-phoneme (F1,F2,F3)
/// sample triples, take medians and keep only phonemes the table already
/// considers vowels.
#[cfg(feature = "vocal-tract")]
fn finalize_median_targets(
    samples: std::collections::HashMap<String, Vec<(f32, f32, f32)>>,
    db: &symthaea::voice::formant_targets::FormantDatabase,
    min_samples: usize,
) -> Vec<(String, symthaea::voice::FormantTarget)> {
    fn median(values: &mut [f32]) -> f32 {
        values.sort_by(|a, b| a.total_cmp(b));
        values[values.len() / 2]
    }

    let mut targets = Vec::new();
    for (phoneme, mut triples) in samples {
        if triples.len() < min_samples {
            continue;
        }
        // Only update phonemes the table considers vowels.
        let Some(existing) = db.lookup(&phoneme) else {
            continue;
        };
        if !existing.is_vowel {
            continue;
        }
        let mut f1s: Vec<f32> = triples.iter().map(|t| t.0).collect();
        let mut f2s: Vec<f32> = triples.iter().map(|t| t.1).collect();
        let mut f3s: Vec<f32> = triples.iter().map(|t| t.2).collect();
        let mut target = existing.clone();
        target.f1 = median(&mut f1s);
        target.f2 = median(&mut f2s);
        target.f3 = median(&mut f3s);
        triples.clear();
        targets.push((phoneme, target));
    }
    targets.sort_by(|a, b| a.0.cmp(&b.0));
    targets
}

/// Distillation v4: real forced alignment via DTW over trained HDC
/// prototypes (`symthaea-stt::{DtwAligner, DtwTrainer}`), replacing v2/v3's
/// heuristic segmentation (duration-proportional windows, voiced islands —
/// both diagnosed negative: extraction is fine, but neither heuristic
/// recovers true per-phoneme frame boundaries against Kokoro's real
/// prosody). Each `FormantFrame` is encoded into a 2048-bit `HV16` via a
/// fixed random sign-projection (record-based HDC encoding: output bit i is
/// the sign of a random ±1-weighted combination of F1/F2/F3/f0/energy/
/// voicing). Two-pass EM: pass 1 bootstraps phoneme prototypes via uniform
/// alignment (`DtwTrainer` with no prototypes yet); pass 2 re-aligns with
/// those prototypes via real dynamic time warping and refines them. The
/// final aligner then produces genuine DTW segment boundaries per
/// utterance, and F1/F2/F3 medians are extracted from those segments
/// (same trim + `finalize_median_targets` tail as v2/v3).
#[cfg(all(feature = "vocal-tract", feature = "voice-stt"))]
fn encode_frame_hv(frame: &symthaea::voice::FormantFrame) -> symthaea_stt::HV16 {
    // Deterministic sign-random-projection: no stored weight matrix, no PRNG
    // state (sidesteps the LCG low-order-bit pitfall entirely) — each
    // (output_bit, input_feature) pair gets a fixed ±1 sign from a
    // splitmix64-style hash of its own indices.
    fn hash_sign(i: usize, j: usize) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        x ^= x >> 33;
        if (x >> 63) & 1 == 1 { 1.0 } else { -1.0 }
    }

    let features = [
        frame.f1 / 1000.0,
        frame.f2 / 2000.0,
        frame.f3 / 3000.0,
        frame.f0 / 300.0,
        frame.energy,
        frame.voicing,
    ];
    let mut values = vec![0.0f32; symthaea_stt::HDC_DIM];
    for (i, v) in values.iter_mut().enumerate() {
        *v = features
            .iter()
            .enumerate()
            .map(|(j, &f)| hash_sign(i, j) * f)
            .sum();
    }
    symthaea_stt::HV16::from_continuous(&values)
}

#[cfg(all(feature = "vocal-tract", feature = "voice-stt"))]
fn dtw_vowel_targets(
    corpus: &[(String, Vec<symthaea::voice::FormantFrame>)],
    base_duration: f32,
    min_samples: usize,
) -> Vec<(String, symthaea::voice::FormantTarget)> {
    use std::collections::HashMap;
    use symthaea::voice::SimpleG2P;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea_stt::{DtwAligner, DtwTrainer};
    use symthaea_vocal_tract::types::SourceType;

    let g2p = SimpleG2P::new();
    let db = FormantDatabase::new();

    let prepared: Vec<(
        Vec<String>,
        Vec<symthaea_stt::HV16>,
        &Vec<symthaea::voice::FormantFrame>,
    )> = corpus
        .iter()
        .filter_map(|(text, frames)| {
            if frames.is_empty() {
                return None;
            }
            let phonemes: Vec<String> = g2p
                .text_to_phonemes(text, base_duration)
                .into_iter()
                .filter(|tp| tp.phoneme != "SIL" && tp.phoneme != "SP")
                .map(|tp| tp.phoneme)
                .collect();
            if phonemes.is_empty() {
                return None;
            }
            let hvs: Vec<symthaea_stt::HV16> = frames.iter().map(encode_frame_hv).collect();
            Some((phonemes, hvs, frames))
        })
        .collect();

    // Pass 1: bootstrap prototypes via uniform alignment (no prototypes yet).
    let mut trainer = DtwTrainer::new();
    for (phonemes, hvs, _) in &prepared {
        trainer.train_utterance(hvs, phonemes);
    }
    let bootstrap = trainer.finalize(1);
    println!(
        "  DTW pass 1 (uniform bootstrap): {} phoneme prototypes",
        bootstrap.len()
    );

    // Pass 2: refine via real DTW against the bootstrapped prototypes.
    let mut trainer2 = DtwTrainer::with_prototypes(bootstrap);
    for (phonemes, hvs, _) in &prepared {
        trainer2.train_utterance(hvs, phonemes);
    }
    let refined = trainer2.finalize(1);
    println!(
        "  DTW pass 2 (refined via real alignment): {} phoneme prototypes",
        refined.len()
    );

    // Final: align every utterance against the refined prototypes and pull
    // F1/F2/F3 samples from the real DTW segment boundaries.
    let aligner = DtwAligner::with_prototypes(refined);
    let mut samples: HashMap<String, Vec<(f32, f32, f32)>> = HashMap::new();
    for (phonemes, hvs, frames) in &prepared {
        let alignment = aligner.align(hvs, phonemes);
        for seg in &alignment.segments {
            let start = seg.start_frame;
            let end = seg.end_frame.min(frames.len());
            if start >= end {
                continue;
            }
            // Trim onset/offset transition frames when the segment is long
            // enough to spare them (mirrors v3's island trimming).
            let (fs, fe) = if end - start >= 5 {
                (start + 1, end - 1)
            } else {
                (start, end)
            };
            for frame in &frames[fs..fe.min(frames.len())] {
                if frame.source_type == SourceType::Vowel && frame.voicing > 0.5 {
                    samples
                        .entry(seg.phoneme.clone())
                        .or_default()
                        .push((frame.f1, frame.f2, frame.f3));
                }
            }
        }
    }

    finalize_median_targets(samples, &db, min_samples)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let worker = PathBuf::from(
        arg_value(&args, "--worker")
            .unwrap_or_else(|| "communication/worker/run_whisper_nixos.sh".to_string()),
    );
    let report_path = arg_value(&args, "--report");
    let save_wavs = arg_value(&args, "--save-wavs").map(PathBuf::from);
    if let Some(ref dir) = save_wavs {
        std::fs::create_dir_all(dir)?;
    }

    // Diagnostic knobs (voice plan live-fire follow-up): isolate WHERE
    // intelligibility is lost by varying the synthesis path and pacing.
    // --mode ltc|articulatory   (ltc requires the vocal-tract feature; the
    //                            2026-07-15 121.5% baseline was articulatory —
    //                            the harness was built without vocal-tract)
    // --rate <f32>              base speech-rate multiplier (default 1.0)
    // --f0 <f32>                base fundamental (Hz, default 150)
    let mode = arg_value(&args, "--mode").unwrap_or_else(|| "default".into());
    let rate: f32 = arg_value(&args, "--rate")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1.0);
    let f0: f32 = arg_value(&args, "--f0")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(150.0);

    let mut config = ReplVoiceConfig {
        base_rate: rate,
        base_f0: f0,
        ..ReplVoiceConfig::default()
    };
    match mode.as_str() {
        "ltc" => {
            config.use_ltc_pipeline = true;
            config.use_articulatory = true;
        }
        "articulatory" => {
            config.use_ltc_pipeline = false;
            config.use_articulatory = true;
        }
        // "kokoro": neural teacher engine — handled below.
        _ => {}
    }
    println!(
        "voice config: mode={mode} (ltc_pipeline={}, articulatory={}), rate={rate}, f0={f0}",
        config.use_ltc_pipeline, config.use_articulatory
    );

    let voice_sr;
    // Either the native formant/vocal-tract voice or the Kokoro neural engine.
    enum Synth {
        Repl(Box<ReplVoiceOutput>),
        Kokoro(Box<symthaea::voice::KokoroEngine>),
    }
    let mut synth = if mode == "kokoro" {
        // --kokoro-model overrides the ONNX filename within the HF cache
        // snapshot (e.g. a locally-quantized "onnx/model_int8.onnx" placed
        // there for an fp32-vs-int8 A/B, TTS-BETTER-1, 2026-07-17). Defaults
        // to the verified fp32 model.
        let kokoro_config = || {
            let mut c = symthaea::voice::KokoroConfig::default();
            if let Some(m) = arg_value(&args, "--kokoro-model") {
                c.model_filename = m;
            }
            c
        };
        let engine = symthaea::voice::KokoroEngine::load(kokoro_config())
            .ok_or_else(|| anyhow::anyhow!("Kokoro engine failed to load (ORT_DYLIB_PATH set?)"))?;
        voice_sr = engine.sample_rate();

        // Liquid speed-knob sanity check: the `speed` input must actually
        // change output duration (0.75x should be ~1.5x longer than 1.25x).
        let mut probe = symthaea::voice::KokoroEngine::load(kokoro_config())
            .ok_or_else(|| anyhow::anyhow!("Kokoro probe engine failed to load"))?;
        probe.speed = Some(0.75);
        let slow = probe.synthesize("testing the speed control", None);
        probe.speed = Some(1.25);
        let fast = probe.synthesize("testing the speed control", None);
        if let (Some(s), Some(f)) = (slow, fast) {
            println!(
                "  speed-knob check: 0.75x -> {:.2}s, 1.25x -> {:.2}s (ratio {:.2}, expect ~1.67)",
                s.len() as f32 / voice_sr as f32,
                f.len() as f32 / voice_sr as f32,
                s.len() as f32 / f.len().max(1) as f32
            );
        } else {
            println!("  speed-knob check: probe synthesis FAILED");
        }
        drop(probe);
        Synth::Kokoro(Box::new(engine))
    } else {
        voice_sr = config.sample_rate;
        Synth::Repl(Box::new(ReplVoiceOutput::new(config)?))
    };

    // Whisper worker: generous timeout — first request pays model load.
    let mut policy =
        WorkerPolicy::allow_one(&worker).map_err(|e| anyhow::anyhow!("worker policy: {e}"))?;
    policy.response_timeout = Duration::from_secs(900);
    // The provider_id must match what the worker self-reports in its response
    // envelope (whisper_worker.py's SYMTHAEA_PROVIDER_ID default), or every
    // response is rejected as an identity mismatch.
    let provider_id =
        std::env::var("SYMTHAEA_PROVIDER_ID").unwrap_or_else(|_| "whisper-large-v3-local".into());
    let mut provider = LocalJsonlProvider::spawn(provider_id, &worker, &[], policy)
        .map_err(|e| anyhow::anyhow!("spawn worker: {e}"))?;

    // Distillation v2: --distill-medians <corpus_dir> [--min-samples N]
    // computes per-phoneme median vowel targets from the teacher corpus and
    // re-pretrains the controller on the updated FormantDatabase. Seconds
    // instead of ~40 min, and alignment-error-robust (v1's frame-gradient
    // approach was an honest negative: 100.0% vs 98.6% control).
    #[cfg(feature = "vocal-tract")]
    if let Some(dir) = arg_value(&args, "--distill-medians") {
        let min_samples: usize = arg_value(&args, "--min-samples")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(30);
        match &mut synth {
            Synth::Repl(voice) => {
                let corpus = load_distill_corpus(std::path::Path::new(&dir))?;
                let base_duration = 0.08 / rate; // matches ReplVoiceConfig default
                let targets = median_vowel_targets(&corpus, base_duration, min_samples);
                println!("  median targets ({} vowels updated):", targets.len());
                for (ph, t) in &targets {
                    println!(
                        "    {ph:>4}: F1 {:>4.0} F2 {:>4.0} F3 {:>4.0}",
                        t.f1, t.f2, t.f3
                    );
                }
                let n = voice.apply_formant_targets(&targets)?;
                println!("  applied {n} targets + re-pretrained controller\n");
            }
            Synth::Kokoro(_) => {
                anyhow::bail!("--distill-medians applies to the native voice, not kokoro mode")
            }
        }
    }

    // Distillation v4: --distill-dtw <corpus_dir> [--min-samples N] — real
    // forced alignment (DTW over trained HDC prototypes) instead of v2/v3's
    // heuristic segmentation. See dtw_vowel_targets doc comment.
    #[cfg(all(feature = "vocal-tract", feature = "voice-stt"))]
    if let Some(dir) = arg_value(&args, "--distill-dtw") {
        let min_samples: usize = arg_value(&args, "--min-samples")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(30);
        match &mut synth {
            Synth::Repl(voice) => {
                let corpus = load_distill_corpus(std::path::Path::new(&dir))?;
                let base_duration = 0.08 / rate;
                let targets = dtw_vowel_targets(&corpus, base_duration, min_samples);
                println!(
                    "  DTW-aligned median targets ({} vowels updated):",
                    targets.len()
                );
                for (ph, t) in &targets {
                    println!(
                        "    {ph:>4}: F1 {:>4.0} F2 {:>4.0} F3 {:>4.0}",
                        t.f1, t.f2, t.f3
                    );
                }
                let n = voice.apply_formant_targets(&targets)?;
                println!("  applied {n} targets + re-pretrained controller\n");
            }
            Synth::Kokoro(_) => {
                anyhow::bail!("--distill-dtw applies to the native voice, not kokoro mode")
            }
        }
    }

    // Kokoro-teacher distillation: --distill <corpus_dir> [--epochs N] [--lr X]
    // trains the vocal-tract controller against teacher trajectories BEFORE
    // the gate runs. Corpus sentences are disjoint from the gate (no leakage).
    #[cfg(feature = "vocal-tract")]
    if let Some(dir) = arg_value(&args, "--distill") {
        let epochs: usize = arg_value(&args, "--epochs")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(3);
        let lr: f32 = arg_value(&args, "--lr")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(1e-4);
        match &mut synth {
            Synth::Repl(voice) => {
                let corpus = load_distill_corpus(std::path::Path::new(&dir))?;
                let n_utts = corpus.len();
                let t = Instant::now();
                let (pairs, skipped) = voice.distill_from_references(&corpus, epochs, lr)?;
                println!(
                    "  distilled: {n_utts} utterances, {pairs} frame pairs trained \
                     ({skipped} silent skipped), {epochs} epochs, lr {lr}, {:.1}s\n",
                    t.elapsed().as_secs_f32()
                );
            }
            Synth::Kokoro(_) => {
                anyhow::bail!("--distill applies to the native voice, not kokoro mode")
            }
        }
    }

    println!("Round-trip intelligibility: {mode} voice → Whisper large-v3\n");

    // Low-prior control sentences (kokoro mode only): "Hello, this is a
    // test." is exactly the kind of high-prior phrase Whisper's language
    // model can reconstruct from degraded audio. These are grammatical but
    // improbable — a correct transcription here means the AUDIO carries the
    // phonetics, not the listener's prior.
    const LOW_PRIOR: &[&str] = &[
        "the purple architect measured seven quiet plums",
        "my walrus prefers grapefruit opera on Tuesdays",
        "eleven cautious spoons orbit the marble lighthouse",
    ];
    let sentences: Vec<&str> = if mode == "kokoro" {
        SENTENCES.iter().chain(LOW_PRIOR.iter()).copied().collect()
    } else {
        SENTENCES.to_vec()
    };

    let mut rows = Vec::new();
    let mut total_wer = 0.0f64;
    for (i, sentence) in sentences.iter().enumerate() {
        let synth_start = Instant::now();
        let samples = match &mut synth {
            Synth::Repl(voice) => voice.synthesize(sentence)?,
            Synth::Kokoro(engine) => engine.synthesize(sentence, None).unwrap_or_default(),
        };
        let synth_ms = synth_start.elapsed().as_millis();
        if samples.is_empty() {
            println!("  [{i:2}] SYNTHESIS EMPTY: {sentence:?}");
            continue;
        }
        let audio_secs = samples.len() as f32 / voice_sr as f32;
        let resampled = linear_resample(&samples, voice_sr, WHISPER_SAMPLE_RATE);

        if let Some(ref dir) = save_wavs {
            save_wav(
                &dir.join(format!("roundtrip_{i:02}.wav")),
                &samples,
                voice_sr,
            )?;
        }

        let asr_start = Instant::now();
        let preserved =
            transcribe_samples(&mut provider, resampled, WHISPER_SAMPLE_RATE, Some("en"))?;
        let asr_ms = asr_start.elapsed().as_millis();

        let reference = normalize_for_scoring(sentence);
        let hypothesis = normalize_for_scoring(&preserved.original);
        let wer = word_error_rate(&reference, &hypothesis);
        total_wer += wer;

        println!(
            "  [{i:2}] WER {:>5.1}%  ({audio_secs:.1}s audio, synth {synth_ms}ms, asr {asr_ms}ms)",
            wer * 100.0
        );
        println!("        said:  {sentence:?}");
        println!("        heard: {:?}", preserved.original.trim());

        rows.push(serde_json::json!({
            "sentence": sentence,
            "heard": preserved.original.trim(),
            "wer": wer,
            "audio_seconds": audio_secs,
            "synth_ms": synth_ms,
            "asr_ms": asr_ms,
        }));
    }

    let mean_wer = total_wer / rows.len().max(1) as f64;
    println!(
        "\n  Mean round-trip WER: {:.1}% over {} sentences",
        mean_wer * 100.0,
        rows.len()
    );
    println!(
        "  (normalized scoring; measures mouth intelligibility — Whisper's own WER on real speech is ~4.8%)"
    );

    if let Some(path) = report_path {
        let report = serde_json::json!({
            "eval": "voice_roundtrip_wer",
            "voice": if mode == "kokoro" { "KokoroEngine (neural, onnx-community)" } else { "ReplVoiceOutput (formant/vocal-tract)" },
            "mode": mode,
            "rate": rate,
            "f0": f0,
            "vocal_tract_feature": cfg!(feature = "vocal-tract"),
            "asr": "whisper-large-v3-local (JSONL worker)",
            "scoring": "symthaea-communication normalize_for_scoring + word_error_rate",
            "mean_wer": mean_wer,
            "sentences": rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("  Report written to {path}");
    }

    Ok(())
}
