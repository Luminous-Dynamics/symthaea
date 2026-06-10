// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Speech-to-Text Pipeline
//!
//! Complete pipeline: Audio → Mel → HDC → Phonemes → Grammar Rescoring → Words
//!
//! This demonstrates integrating all Symthaea components into a production-ready
//! speech recognition pipeline with unified grammar rescoring.

use std::path::Path;
use symthaea_stt::audio::{AudioConfig, AudioFrontend, AudioProjector};
use symthaea_stt::hdc::HV16;
use symthaea_stt::ltc::LtcConfig;
use symthaea_stt::phoneme::{PhonemeDecoder, PhonemeResonator, TemporalConfig, TemporalDecoder};
use symthaea_stt::temporal_grammar::TemporalEvent;
use symthaea_stt::unified_grammar::UnifiedGrammar;

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn header(title: &str) {
    println!();
    separator('=', 70);
    println!("  {}", title);
    separator('=', 70);
    println!();
}

fn subheader(title: &str) {
    println!();
    separator('-', 70);
    println!("  {}", title);
    separator('-', 70);
    println!();
}

/// Simple word dictionary mapping phoneme sequences to words
struct WordDictionary {
    entries: Vec<(Vec<String>, String)>,
}

impl WordDictionary {
    fn new() -> Self {
        let entries = vec![
            // Common words with their phoneme sequences
            (vec!["DH", "AH"], "the"),
            (vec!["K", "AE", "T"], "cat"),
            (vec!["S", "AE", "T"], "sat"),
            (vec!["D", "AO", "G"], "dog"),
            (vec!["R", "AE", "N"], "ran"),
            (vec!["B", "AE", "T"], "bat"),
            (vec!["HH", "AE", "T"], "hat"),
            (vec!["M", "AE", "T"], "mat"),
            (vec!["P", "AE", "T"], "pat"),
            (vec!["R", "AE", "T"], "rat"),
            (vec!["B", "IH", "G"], "big"),
            (vec!["P", "IH", "G"], "pig"),
            (vec!["D", "IH", "G"], "dig"),
            (vec!["F", "IH", "SH"], "fish"),
            (vec!["D", "IH", "SH"], "dish"),
            (vec!["W", "IH", "SH"], "wish"),
            (vec!["S", "T", "R", "AO", "NG"], "strong"),
            (vec!["S", "T", "R", "IH", "NG"], "string"),
            (vec!["S", "P", "R", "IH", "NG"], "spring"),
            (vec!["S", "T", "R", "IY", "T"], "street"),
            (vec!["AH"], "a"),
            (vec!["AE", "N", "D"], "and"),
            (vec!["IH", "Z"], "is"),
            (vec!["IH", "T"], "it"),
            (vec!["AO", "N"], "on"),
            (vec!["AE", "T"], "at"),
        ]
        .into_iter()
        .map(|(p, w)| (p.into_iter().map(String::from).collect(), w.to_string()))
        .collect();

        Self { entries }
    }

    /// Find the best matching word for a phoneme sequence
    fn find_word(&self, phonemes: &[String]) -> Option<(&str, usize)> {
        let mut best_match: Option<(&str, usize)> = None;

        for (entry_phones, word) in &self.entries {
            if phonemes.starts_with(entry_phones) {
                if best_match.map_or(true, |(_, len)| entry_phones.len() > len) {
                    best_match = Some((word.as_str(), entry_phones.len()));
                }
            }
        }

        best_match
    }

    /// Convert phoneme sequence to words (greedy matching)
    fn phonemes_to_words(&self, phonemes: &[String]) -> Vec<String> {
        let mut words = Vec::new();
        let mut i = 0;

        while i < phonemes.len() {
            if let Some((word, consumed)) = self.find_word(&phonemes[i..]) {
                words.push(word.to_string());
                i += consumed;
            } else {
                // Unknown phoneme - output as-is or skip
                i += 1;
            }
        }

        words
    }
}

/// Complete STT Pipeline integrating all components
pub struct SttPipeline {
    /// Audio projector (Mel + LTC → HDC)
    projector: AudioProjector,
    /// Phoneme decoder (HDC → Phonemes)
    decoder: PhonemeDecoder,
    /// Temporal decoder (smoothing + hysteresis)
    temporal: TemporalDecoder,
    /// Unified grammar for sequence scoring
    grammar: UnifiedGrammar,
    /// Word dictionary
    dictionary: WordDictionary,
}

impl SttPipeline {
    /// Create a new STT pipeline
    pub fn new() -> Self {
        let audio_config = AudioConfig::default();
        let ltc_config = LtcConfig::default();

        Self {
            projector: AudioProjector::new(audio_config, ltc_config),
            decoder: PhonemeDecoder::new(),
            temporal: TemporalDecoder::with_config(TemporalConfig {
                window_size: 5,
                confidence_threshold: 0.25,
                min_duration_frames: 2,
                stability_frames: 2,
            }),
            grammar: UnifiedGrammar::speech(),
            dictionary: WordDictionary::new(),
        }
    }

    /// Load phoneme prototypes from pre-trained data
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.decoder.load_prototypes(prototypes);
    }

    /// Load grammar model from file
    pub fn load_grammar<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        self.grammar = UnifiedGrammar::load(path)?;
        Ok(())
    }

    /// Save grammar model to file
    pub fn save_grammar<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.grammar.save(path)
    }

    /// Train grammar on phoneme sequences
    pub fn train_grammar(&mut self, phoneme_sequences: &[Vec<String>], epochs: usize) {
        for _ in 0..epochs {
            for seq in phoneme_sequences {
                let events = self.phonemes_to_events(seq);
                if events.len() >= 2 {
                    self.grammar.train_sequence(&events);
                }
            }
        }
    }

    /// Convert phoneme list to temporal events
    fn phonemes_to_events(&self, phonemes: &[String]) -> Vec<TemporalEvent> {
        let mut events = Vec::new();
        let mut time = 0.0f32;

        for phoneme in phonemes {
            if let Some(cid) = self.grammar.category_id(phoneme) {
                let duration = estimate_duration(phoneme);
                let intensity = estimate_intensity(phoneme);
                events.push(TemporalEvent::new(phoneme, cid, time, duration, intensity));
                time += duration;
            }
        }

        events
    }

    /// Transcribe audio file to text (without grammar rescoring)
    pub fn transcribe_raw(&mut self, audio: &[f32]) -> TranscriptionResult {
        // Step 1: Project audio to HDC vectors
        let hvs = self.projector.project(audio);

        // Step 2: Decode each frame to phoneme candidates
        self.temporal.reset();

        let mut phoneme_scores_all = Vec::new();

        for hv in &hvs {
            let scores = self.decoder.decode_all_scores(hv);
            phoneme_scores_all.push(scores.clone());
            self.temporal.process_frame(&scores);
        }

        // Step 3: Flush and get smoothed phoneme sequence
        self.temporal.flush();
        let detailed = self.temporal.get_phonemes_detailed();
        let phonemes: Vec<String> = detailed.iter().map(|(p, _, _)| p.clone()).collect();

        // Step 4: Convert to words
        let words = self.dictionary.phonemes_to_words(&phonemes);

        let confidence = if detailed.is_empty() {
            0.0
        } else {
            detailed.iter().map(|(_, c, _)| *c).sum::<f32>() / detailed.len() as f32
        };

        TranscriptionResult {
            phonemes,
            words,
            grammar_score: 0.0,
            confidence,
        }
    }

    /// Transcribe audio with grammar rescoring
    pub fn transcribe(&mut self, audio: &[f32]) -> TranscriptionResult {
        // Get raw transcription
        let mut result = self.transcribe_raw(audio);

        // Score with grammar
        if result.phonemes.len() >= 2 {
            let events = self.phonemes_to_events(&result.phonemes);
            result.grammar_score = self.grammar.score_sequence(&events);
        }

        result
    }

    /// Get grammar statistics
    pub fn grammar_stats(&self) -> symthaea_stt::unified_grammar::UnifiedStats {
        self.grammar.stats()
    }
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Decoded phoneme sequence
    pub phonemes: Vec<String>,
    /// Recognized words
    pub words: Vec<String>,
    /// Grammar score (higher = more likely valid sequence)
    pub grammar_score: f32,
    /// Average phoneme confidence
    pub confidence: f32,
}

impl TranscriptionResult {
    /// Get text output
    pub fn text(&self) -> String {
        self.words.join(" ")
    }
}

// Helper functions

fn estimate_duration(phoneme: &str) -> f32 {
    match phoneme {
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.10
        }
        "S" | "Z" | "SH" | "ZH" | "F" | "V" | "TH" | "DH" => 0.08,
        "P" | "B" | "T" | "D" | "K" | "G" => 0.05,
        "M" | "N" | "NG" => 0.06,
        _ => 0.06,
    }
}

fn estimate_intensity(phoneme: &str) -> f32 {
    match phoneme {
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.8
        }
        "B" | "D" | "G" | "V" | "Z" | "DH" | "ZH" | "M" | "N" | "NG" => 0.6,
        "P" | "T" | "K" | "F" | "S" | "TH" | "SH" | "HH" => 0.5,
        _ => 0.55,
    }
}

/// Generate synthetic phoneme prototypes for testing
fn generate_synthetic_prototypes() -> Vec<(String, HV16)> {
    use symthaea_stt::hdc::BundleAccumulator;

    let phonemes = vec![
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        "B", "D", "G", "K", "P", "T", "CH", "DH", "F", "HH", "JH", "S", "SH", "TH", "V", "Z", "ZH",
        "M", "N", "NG", "L", "R", "W", "Y",
    ];

    let mut prototypes = Vec::new();

    for (i, phoneme) in phonemes.iter().enumerate() {
        // Generate deterministic pseudo-random prototype based on phoneme index
        let mut hv = HV16::zero();

        // Set ~50% of bits based on hash of phoneme
        let hash_base = (i * 12345 + 67890) as u128;
        for w in 0..16 {
            let word_hash = hash_base
                .wrapping_mul(w as u128 + 1)
                .wrapping_add(i as u128);
            hv.words[w] = word_hash;
        }

        prototypes.push((phoneme.to_string(), hv));
    }

    prototypes
}

/// Simulate audio from phoneme sequence (for testing without real audio)
fn simulate_audio_from_phonemes(phonemes: &[&str]) -> Vec<f32> {
    // Generate ~100ms of silence + "audio" per phoneme
    let sample_rate = 16000;
    let mut audio = Vec::new();

    // Silence at start
    audio.extend(vec![0.0f32; sample_rate / 20]); // 50ms

    for (i, _phoneme) in phonemes.iter().enumerate() {
        // Generate ~80ms of synthetic signal per phoneme
        let duration_samples = sample_rate / 12; // ~83ms

        for j in 0..duration_samples {
            let t = j as f32 / sample_rate as f32;
            // Different frequency for each phoneme position
            let freq = 200.0 + (i as f32 * 100.0);
            let sample = 0.3 * (2.0 * std::f32::consts::PI * freq * t).sin();
            audio.push(sample);
        }

        // Small silence between phonemes
        audio.extend(vec![0.0f32; sample_rate / 100]); // 10ms
    }

    // Silence at end
    audio.extend(vec![0.0f32; sample_rate / 20]); // 50ms

    audio
}

fn main() {
    header("SYMTHAEA END-TO-END STT PIPELINE");

    println!("  Complete pipeline demonstration:");
    println!("    Audio → Mel → HDC → Phonemes → Grammar → Words");
    println!();

    // Create pipeline
    let mut pipeline = SttPipeline::new();

    // Load synthetic prototypes (in real use, load from trained model)
    subheader("Phase 1: Initialize Pipeline");

    let prototypes = generate_synthetic_prototypes();
    pipeline.load_prototypes(&prototypes);
    println!("    Loaded {} phoneme prototypes", prototypes.len());

    // Train grammar on example sentences
    let training_sequences: Vec<Vec<String>> = vec![
        vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"],
        vec!["DH", "AH", "D", "AO", "G", "R", "AE", "N"],
        vec!["DH", "AH", "B", "AE", "T", "HH", "AE", "T"],
        vec!["B", "IH", "G", "P", "IH", "G", "D", "IH", "G"],
        vec!["F", "IH", "SH", "D", "IH", "SH", "W", "IH", "SH"],
        vec!["S", "T", "R", "AO", "NG", "S", "T", "R", "IH", "NG"],
    ]
    .into_iter()
    .map(|seq| seq.into_iter().map(String::from).collect())
    .collect();

    pipeline.train_grammar(&training_sequences, 30);
    let stats = pipeline.grammar_stats();
    println!(
        "    Trained grammar: {} clusters, {:.1}% density",
        stats.num_clusters,
        stats.avg_cluster_density * 100.0
    );

    // Simulate transcription
    subheader("Phase 2: Simulated Transcription");

    let test_cases = vec![
        (vec!["DH", "AH", "K", "AE", "T"], "the cat"),
        (vec!["DH", "AH", "D", "AO", "G"], "the dog"),
        (vec!["B", "IH", "G", "F", "IH", "SH"], "big fish"),
    ];

    for (phonemes, expected) in &test_cases {
        // Generate synthetic audio
        let audio = simulate_audio_from_phonemes(phonemes);

        // Transcribe
        let result = pipeline.transcribe(&audio);

        println!("    Expected: \"{}\"", expected);
        println!("    Phonemes: {}", phonemes.join(" "));
        println!(
            "    Decoded:  {} (confidence: {:.2})",
            result.phonemes.join(" "),
            result.confidence
        );
        println!("    Grammar:  {:+.4}", result.grammar_score);
        println!();
    }

    // Test grammar discrimination
    subheader("Phase 3: Grammar Discrimination");

    // Score different phoneme sequences
    let score_tests = vec![
        (vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"], "trained"),
        (vec!["DH", "AH", "P", "AE", "T", "M", "AE", "T"], "similar"),
        (vec!["S", "T", "R", "IY", "T"], "valid cluster"),
        (vec!["T", "L", "IY", "T"], "invalid cluster"),
    ];

    println!("    Phonotactic scoring:");
    for (phonemes, label) in score_tests {
        let events: Vec<TemporalEvent> = phonemes
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                pipeline
                    .grammar
                    .category_id(*p)
                    .map(|cid| TemporalEvent::new(*p, cid, i as f32 * 0.06, 0.06, 0.6))
            })
            .collect();

        let score = if events.len() >= 2 {
            pipeline.grammar.score_sequence(&events)
        } else {
            0.0
        };

        println!("      {:+.4}  {}  [{}]", score, phonemes.join(" "), label);
    }

    // Real audio file processing (if available)
    subheader("Phase 4: Real Audio Processing");

    let test_audio_path = "data/test_audio.wav";
    if Path::new(test_audio_path).exists() {
        match AudioFrontend::load_wav(test_audio_path) {
            Ok((audio, sample_rate)) => {
                println!("    Loaded: {}", test_audio_path);
                println!(
                    "    Duration: {:.2}s, Sample rate: {}Hz",
                    audio.len() as f32 / sample_rate as f32,
                    sample_rate
                );

                let result = pipeline.transcribe(&audio);
                println!("    Transcription: \"{}\"", result.text());
                println!("    Phonemes: {}", result.phonemes.join(" "));
                println!("    Grammar score: {:+.4}", result.grammar_score);
                println!("    Confidence: {:.2}", result.confidence);
            }
            Err(e) => {
                println!("    Could not load {}: {}", test_audio_path, e);
            }
        }
    } else {
        println!("    No test audio file found at {}", test_audio_path);
        println!("    To test with real audio, place a WAV file at that path.");
        println!();
        println!("    The pipeline supports:");
        println!("      - WAV files (16-bit, mono or stereo)");
        println!("      - FLAC files");
        println!("      - Any sample rate (will resample internally)");
    }

    // Summary
    header("PIPELINE SUMMARY");

    let stats = pipeline.grammar_stats();
    println!("    Pipeline Components:");
    println!("      Audio Frontend:    Mel spectrogram (40 bands)");
    println!("      LTC Projector:     Temporal dynamics encoding");
    println!("      HDC Encoder:       Hyperdimensional vectors");
    println!("      Phoneme Decoder:   {} prototypes", prototypes.len());
    println!("      Temporal Decoder:  Schmitt trigger + run-length");
    println!(
        "      Unified Grammar:   {} clusters, {:.1}% density",
        stats.num_clusters,
        stats.avg_cluster_density * 100.0
    );
    println!("      Word Dictionary:   Simple greedy matching");
    println!();
    println!("    Key Features:");
    println!("      - End-to-end neuromorphic pipeline");
    println!("      - Grammar-based phonotactic rescoring");
    println!("      - Hierarchical clustering prevents saturation");
    println!("      - CfC adaptive time constants");
    println!("      - Sparse HDC for efficiency");

    separator('=', 70);
}
