// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate Broca training data from LibriSpeech using the HDC-LTC speech encoder.
//!
//! Three-phase pipeline:
//!   Phase A: Load LibriSpeech audio + alignments → compute heuristic ThoughtChannels
//!   Phase B: Train SpeechThoughtEncoder's readout probes on heuristic targets
//!   Phase C: Re-encode all utterances with LEARNED encoder → output Broca JSONL
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --features speech-data --bin r#gen-speech-training
//!
//! Outputs:
//!   data/training/speech-v1-train.jsonl  (80%)
//!   data/training/speech-v1-eval.jsonl   (20%)
//!   data/models/speech_thought_encoder.bin

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::speech_encoder::SpeechThoughtEncoder;
use symthaea_broca::speech_heuristics::{self, PhonemeInfo, UtteranceAcoustics};
use symthaea_core::genesis::GenesisSeed;
use symthaea_stt::alignment_loader;
use symthaea_stt::audio::{AudioConfig, AudioFrontend};

// ============================================================================
// Configuration
// ============================================================================

const LIBRISPEECH_DIR: &str = "data/librispeech/LibriSpeech/dev-clean";
const ALIGNMENT_PATH: &str = "data/alignments/data/dev_clean-00000-of-00001.parquet";
const TRAIN_OUTPUT: &str = "data/training/speech-v1-train.jsonl";
const EVAL_OUTPUT: &str = "data/training/speech-v1-eval.jsonl";
const ENCODER_OUTPUT: &str = "data/models/speech_thought_encoder.bin";
const ENCODER_TRAIN_EPOCHS: usize = 5;
const ENCODER_LR: f32 = 0.001;
const FRAME_DT: f32 = 0.01; // 10ms mel frame hop

// ============================================================================
// Deterministic RNG (same as gen_moral_training)
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len();
        for i in (1..n).rev() {
            let j = self.next_u64() as usize % (i + 1);
            items.swap(i, j);
        }
    }
}

// ============================================================================
// Training pair (matches Broca JSONL format)
// ============================================================================

#[derive(serde::Serialize)]
struct TrainingPair {
    channels: Vec<f32>,
    target_text: String,
    #[serde(default)]
    target_ids: Vec<u32>,
}

// ============================================================================
// Data loading
// ============================================================================

/// An utterance with audio, alignment, and transcript.
struct UtteranceData {
    id: String,
    transcript: String,
    mel_frames: Vec<Vec<f32>>,
    phonemes: Vec<PhonemeInfo>,
    duration: f32,
}

/// Load LibriSpeech transcripts from .trans.txt files.
fn load_transcripts(librispeech_dir: &Path) -> HashMap<String, String> {
    let mut transcripts = HashMap::new();

    for entry in walkdir(librispeech_dir) {
        if entry.ends_with(".trans.txt") {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for line in content.lines() {
                    if let Some(space_pos) = line.find(' ') {
                        let id = &line[..space_pos];
                        let text = &line[space_pos + 1..];
                        transcripts.insert(id.to_string(), text.to_string());
                    }
                }
            }
        }
    }

    transcripts
}

/// Simple recursive directory walker.
fn walkdir(dir: &Path) -> Vec<String> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(walkdir(&path));
            } else if let Some(s) = path.to_str() {
                results.push(s.to_string());
            }
        }
    }
    results
}

/// Map utterance ID to audio file path.
/// Format: "SPEAKER-CHAPTER-UTTERANCE" → "SPEAKER/CHAPTER/SPEAKER-CHAPTER-UTTERANCE.flac"
fn id_to_audio_path(id: &str, base_dir: &Path) -> Option<std::path::PathBuf> {
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() >= 3 {
        let speaker = parts[0];
        let chapter = parts[1];
        let path = base_dir
            .join(speaker)
            .join(chapter)
            .join(format!("{id}.flac"));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

// ============================================================================
// Main pipeline
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Speech-to-Thought Training Data Generator                  ║");
    println!("║  HDC-LTC Unified Neuron + LibriSpeech                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_start = Instant::now();
    let genesis = GenesisSeed::from_phrase("speech-thought-v1");
    let base_dir = Path::new(LIBRISPEECH_DIR);

    // ── Phase A: Load data and compute heuristic targets ────────────────

    println!("Phase A: Loading LibriSpeech data...");
    let phase_a_start = Instant::now();

    // Load transcripts
    let transcripts = load_transcripts(base_dir);
    println!("  Loaded {} transcripts", transcripts.len());

    // Load phoneme alignments
    let alignment_path = Path::new(ALIGNMENT_PATH);
    let alignments = if alignment_path.exists() {
        match alignment_loader::load_alignments(alignment_path) {
            Ok(a) => {
                println!("  Loaded {} alignments", a.len());
                a
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load alignments: {e}");
                HashMap::new()
            }
        }
    } else {
        eprintln!(
            "  Warning: Alignment file not found: {}",
            alignment_path.display()
        );
        HashMap::new()
    };

    // Load audio + compute mel frames for matched utterances
    let mut audio_frontend = AudioFrontend::new(AudioConfig::default());
    let mut utterances = Vec::new();
    let mut skipped = 0;

    for (id, transcript) in &transcripts {
        // Find alignment
        let alignment = match alignments.get(id) {
            Some(a) => a,
            None => {
                skipped += 1;
                continue;
            }
        };

        // Find audio file
        let audio_path = match id_to_audio_path(id, base_dir) {
            Some(p) => p,
            None => {
                skipped += 1;
                continue;
            }
        };

        // Load audio and extract mel frames
        let (samples, _sample_rate) = match AudioFrontend::load_flac(&audio_path) {
            Ok(s) => s,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let mel_frames = audio_frontend.extract_features(&samples);
        if mel_frames.is_empty() {
            skipped += 1;
            continue;
        }

        let phonemes: Vec<PhonemeInfo> = alignment
            .phonemes
            .iter()
            .map(|p| PhonemeInfo {
                phoneme: p.phoneme.clone(),
                start: p.start,
                end: p.end,
            })
            .collect();

        let duration = alignment
            .phonemes
            .last()
            .map(|p| p.end as f32)
            .unwrap_or(0.0);

        utterances.push(UtteranceData {
            id: id.clone(),
            transcript: transcript.clone(),
            mel_frames,
            phonemes,
            duration,
        });

        if utterances.len() % 100 == 0 {
            print!(
                "\r  Loaded {} utterances ({} skipped)...",
                utterances.len(),
                skipped
            );
            let _ = std::io::stdout().flush();
        }
    }

    println!(
        "\r  Phase A complete: {} utterances loaded, {} skipped ({:.1}s)",
        utterances.len(),
        skipped,
        phase_a_start.elapsed().as_secs_f32()
    );

    if utterances.is_empty() {
        eprintln!("Error: No utterances loaded. Check paths:");
        eprintln!("  Audio: {}", LIBRISPEECH_DIR);
        eprintln!("  Alignments: {}", ALIGNMENT_PATH);
        return;
    }

    // Compute heuristic targets
    println!("\n  Computing heuristic ThoughtChannel targets...");
    let heuristic_targets: Vec<ThoughtChannels> = utterances
        .iter()
        .map(|utt| {
            let centroid = speech_heuristics::spectral_centroid(&utt.mel_frames);
            let rate = speech_heuristics::speaking_rate(&utt.phonemes);
            let voicing = speech_heuristics::voicing_ratio(&utt.phonemes);
            let pauses = speech_heuristics::pause_ratio(&utt.phonemes, utt.duration);
            let consistency = speech_heuristics::articulation_consistency(&utt.phonemes);
            let (energy_mean, energy_var) = speech_heuristics::energy_stats(&utt.mel_frames);

            let acoustics = UtteranceAcoustics {
                spectral_centroid: centroid,
                speaking_rate: rate,
                voicing_ratio: voicing,
                pause_ratio: pauses,
                articulation_consistency: consistency,
                energy_mean,
                energy_variance: energy_var,
                duration: utt.duration,
            };

            speech_heuristics::heuristic_channels(&acoustics, &utt.transcript)
        })
        .collect();

    println!("  {} heuristic targets computed", heuristic_targets.len());

    // ── Phase B: Train SpeechThoughtEncoder ─────────────────────────────

    println!(
        "\nPhase B: Training SpeechThoughtEncoder ({} epochs)...",
        ENCODER_TRAIN_EPOCHS
    );
    let phase_b_start = Instant::now();

    let mut encoder = SpeechThoughtEncoder::new(&genesis);

    for epoch in 0..ENCODER_TRAIN_EPOCHS {
        let mut total_loss = 0.0f32;
        let mut count = 0;

        for (utt, target) in utterances.iter().zip(heuristic_targets.iter()) {
            // Process utterance through LTC neuron
            encoder.encode_utterance(&utt.mel_frames, FRAME_DT);

            // Train readout probes
            let loss = encoder.loss(target);
            encoder.train_step(target, ENCODER_LR);

            total_loss += loss;
            count += 1;
        }

        let avg_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };
        println!(
            "  Epoch {}/{}: avg_loss={:.4} ({:.1}s)",
            epoch + 1,
            ENCODER_TRAIN_EPOCHS,
            avg_loss,
            phase_b_start.elapsed().as_secs_f32()
        );
    }

    // Save encoder
    std::fs::create_dir_all(Path::new(ENCODER_OUTPUT).parent().unwrap()).ok();
    match encoder.save(ENCODER_OUTPUT) {
        Ok(()) => println!("  Encoder saved to {ENCODER_OUTPUT}"),
        Err(e) => eprintln!("  Warning: Failed to save encoder: {e}"),
    }

    println!(
        "  Phase B complete ({:.1}s)",
        phase_b_start.elapsed().as_secs_f32()
    );

    // ── Phase C: Generate Broca JSONL with learned channels ─────────────

    println!("\nPhase C: Generating Broca training data with learned encoder...");
    let phase_c_start = Instant::now();

    let mut pairs: Vec<TrainingPair> = Vec::new();

    for utt in &utterances {
        let channels = encoder.encode_utterance(&utt.mel_frames, FRAME_DT);

        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: utt.transcript.clone(),
            target_ids: Vec::new(), // Tokenized by broca-train at load time
        });
    }

    // Shuffle deterministically
    let mut rng = Rng::new(42);
    rng.shuffle(&mut pairs);

    // 80/20 split
    let split = (pairs.len() as f32 * 0.8) as usize;
    let (train_pairs, eval_pairs) = pairs.split_at(split);

    // Write JSONL
    std::fs::create_dir_all("data/training").ok();

    let mut train_file = std::fs::File::create(TRAIN_OUTPUT).expect("Failed to create train file");
    for pair in train_pairs {
        serde_json::to_writer(&mut train_file, pair).ok();
        train_file.write_all(b"\n").ok();
    }

    let mut eval_file = std::fs::File::create(EVAL_OUTPUT).expect("Failed to create eval file");
    for pair in eval_pairs {
        serde_json::to_writer(&mut eval_file, pair).ok();
        eval_file.write_all(b"\n").ok();
    }

    println!(
        "  Phase C complete ({:.1}s)",
        phase_c_start.elapsed().as_secs_f32()
    );

    // ── Summary ─────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Utterances processed: {}", utterances.len());
    println!("  Training pairs: {}", train_pairs.len());
    println!("  Eval pairs: {}", eval_pairs.len());
    println!("  Encoder saved: {ENCODER_OUTPUT}");
    println!("  Train JSONL: {TRAIN_OUTPUT}");
    println!("  Eval JSONL: {EVAL_OUTPUT}");
    println!("  Total time: {:.1}s", total_start.elapsed().as_secs_f32());
    println!("═══════════════════════════════════════════════════════════════");
}
