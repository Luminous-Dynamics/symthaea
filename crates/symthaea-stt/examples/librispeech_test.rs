// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LibriSpeech Real Audio Test
//!
//! Tests the STT pipeline on real speech from LibriSpeech corpus.
//! Requires LibriSpeech data at data/librispeech/LibriSpeech/
//!
//! Usage:
//!   cargo run --release --example librispeech_test
//!   cargo run --release --example librispeech_test -- --limit 10
//!   cargo run --release --example librispeech_test -- --subset test-clean

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use symthaea_stt::audio::{AudioConfig, AudioFrontend, AudioProjector};
use symthaea_stt::eval::word_error_rate;
use symthaea_stt::hdc::HV16;
use symthaea_stt::ltc::LtcConfig;
use symthaea_stt::phoneme::{PhonemeDecoder, TemporalConfig, TemporalDecoder};
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

/// Load transcriptions from LibriSpeech .trans.txt files
fn load_transcriptions(librispeech_path: &Path, subset: &str) -> HashMap<String, String> {
    let mut transcriptions = HashMap::new();
    let subset_path = librispeech_path.join(subset);

    if !subset_path.exists() {
        eprintln!("Subset path does not exist: {}", subset_path.display());
        return transcriptions;
    }

    // Walk through speaker/chapter directories
    if let Ok(speakers) = fs::read_dir(&subset_path) {
        for speaker_entry in speakers.flatten() {
            if let Ok(chapters) = fs::read_dir(speaker_entry.path()) {
                for chapter_entry in chapters.flatten() {
                    let chapter_path = chapter_entry.path();

                    // Find .trans.txt file
                    if let Ok(files) = fs::read_dir(&chapter_path) {
                        for file_entry in files.flatten() {
                            let file_path = file_entry.path();
                            if file_path.extension().map_or(false, |e| e == "txt") {
                                if let Ok(file) = File::open(&file_path) {
                                    let reader = BufReader::new(file);
                                    for line in reader.lines().flatten() {
                                        let parts: Vec<&str> = line.splitn(2, ' ').collect();
                                        if parts.len() == 2 {
                                            let utterance_id = parts[0].to_string();
                                            let text = parts[1].to_uppercase();
                                            transcriptions.insert(utterance_id, text);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    transcriptions
}

/// Find all FLAC files in a LibriSpeech subset
fn find_audio_files(librispeech_path: &Path, subset: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let subset_path = librispeech_path.join(subset);

    fn walk_dir(path: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    walk_dir(&entry_path, files);
                } else if entry_path.extension().map_or(false, |e| e == "flac") {
                    files.push(entry_path);
                }
            }
        }
    }

    walk_dir(&subset_path, &mut files);
    files.sort();
    files
}

/// Simple word dictionary for phoneme-to-word conversion
struct WordDictionary {
    entries: Vec<(Vec<String>, String)>,
}

impl WordDictionary {
    fn new() -> Self {
        // Common words - in production, use full CMU dictionary
        let entries = vec![
            (vec!["DH", "AH"], "THE"),
            (vec!["DH", "IY"], "THE"),
            (vec!["AH"], "A"),
            (vec!["EY"], "A"),
            (vec!["AE", "N", "D"], "AND"),
            (vec!["T", "UW"], "TO"),
            (vec!["T", "AH"], "TO"),
            (vec!["AH", "V"], "OF"),
            (vec!["IH", "N"], "IN"),
            (vec!["IH", "T"], "IT"),
            (vec!["IH", "Z"], "IS"),
            (vec!["W", "AA", "Z"], "WAS"),
            (vec!["HH", "IY"], "HE"),
            (vec!["SH", "IY"], "SHE"),
            (vec!["DH", "EY"], "THEY"),
            (vec!["W", "IY"], "WE"),
            (vec!["Y", "UW"], "YOU"),
            (vec!["AY"], "I"),
            (vec!["M", "AY"], "MY"),
            (vec!["B", "IY"], "BE"),
            (vec!["HH", "AE", "D"], "HAD"),
            (vec!["HH", "AE", "V"], "HAVE"),
            (vec!["W", "IH", "TH"], "WITH"),
            (vec!["F", "AO", "R"], "FOR"),
            (vec!["AA", "N"], "ON"),
            (vec!["AE", "T"], "AT"),
            (vec!["B", "AH", "T"], "BUT"),
            (vec!["N", "AA", "T"], "NOT"),
            (vec!["DH", "IH", "S"], "THIS"),
            (vec!["DH", "AE", "T"], "THAT"),
        ]
        .into_iter()
        .map(|(p, w)| (p.into_iter().map(String::from).collect(), w.to_string()))
        .collect();

        Self { entries }
    }

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

    fn phonemes_to_words(&self, phonemes: &[String]) -> Vec<String> {
        let mut words = Vec::new();
        let mut i = 0;

        while i < phonemes.len() {
            if let Some((word, consumed)) = self.find_word(&phonemes[i..]) {
                words.push(word.to_string());
                i += consumed;
            } else {
                i += 1;
            }
        }

        words
    }
}

/// Generate synthetic phoneme prototypes
fn generate_prototypes() -> Vec<(String, HV16)> {
    let phonemes = vec![
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        "B", "D", "G", "K", "P", "T", "CH", "DH", "F", "HH", "JH", "S", "SH", "TH", "V", "Z", "ZH",
        "M", "N", "NG", "L", "R", "W", "Y",
    ];

    phonemes
        .iter()
        .enumerate()
        .map(|(i, phoneme)| {
            let mut hv = HV16::zero();
            let hash_base = (i * 12345 + 67890) as u128;
            for w in 0..16 {
                let word_hash = hash_base
                    .wrapping_mul(w as u128 + 1)
                    .wrapping_add(i as u128);
                hv.words[w] = word_hash;
            }
            (phoneme.to_string(), hv)
        })
        .collect()
}

fn main() {
    header("LIBRISPEECH REAL AUDIO TEST");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut limit = 50; // Default: process 50 utterances
    let mut subset = "dev-clean".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" | "-l" => {
                i += 1;
                if i < args.len() {
                    limit = args[i].parse().unwrap_or(50);
                }
            }
            "--subset" | "-s" => {
                i += 1;
                if i < args.len() {
                    subset = args[i].clone();
                }
            }
            "--help" | "-h" => {
                println!("Usage: librispeech_test [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --limit, -l <N>      Process N utterances (default: 50)");
                println!("  --subset, -s <NAME>  Use subset (dev-clean, test-clean)");
                println!("  --help, -h           Show this help");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // Find LibriSpeech data
    let librispeech_path = std::env::var("LIBRISPEECH_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/librispeech/LibriSpeech"));

    if !librispeech_path.exists() {
        eprintln!(
            "LibriSpeech data not found at: {}",
            librispeech_path.display()
        );
        eprintln!("Set LIBRISPEECH_PATH environment variable or download data to:");
        eprintln!("  data/librispeech/LibriSpeech/");
        return;
    }

    println!("  LibriSpeech path: {}", librispeech_path.display());
    println!("  Subset: {}", subset);
    println!("  Limit: {} utterances", limit);

    // Load transcriptions
    subheader("Phase 1: Loading Data");

    let transcriptions = load_transcriptions(&librispeech_path, &subset);
    println!("  Loaded {} transcriptions", transcriptions.len());

    let audio_files = find_audio_files(&librispeech_path, &subset);
    println!("  Found {} audio files", audio_files.len());

    if audio_files.is_empty() {
        eprintln!(
            "No audio files found in {}/{}",
            librispeech_path.display(),
            subset
        );
        return;
    }

    // Initialize pipeline components
    subheader("Phase 2: Initializing Pipeline");

    let audio_config = AudioConfig::default();
    let ltc_config = LtcConfig::default();
    let mut projector = AudioProjector::new(audio_config, ltc_config);
    let mut decoder = PhonemeDecoder::new();
    let mut temporal = TemporalDecoder::with_config(TemporalConfig {
        window_size: 5,
        confidence_threshold: 0.2,
        min_duration_frames: 2,
        stability_frames: 2,
    });

    // Load prototypes
    let prototypes = generate_prototypes();
    decoder.load_prototypes(&prototypes);
    println!("  Loaded {} phoneme prototypes", prototypes.len());

    // Train grammar on common patterns
    let mut grammar = UnifiedGrammar::speech();
    let training_patterns: Vec<Vec<&str>> = vec![
        vec!["DH", "AH"],
        vec!["DH", "IY"],
        vec!["AE", "N", "D"],
        vec!["T", "UW"],
        vec!["AH", "V"],
        vec!["IH", "N"],
        vec!["IH", "T"],
        vec!["IH", "Z"],
        vec!["W", "AA", "Z"],
        vec!["HH", "IY"],
        vec!["DH", "EY"],
        vec!["W", "IY"],
    ];

    for pattern in &training_patterns {
        let events: Vec<TemporalEvent> = pattern
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                grammar
                    .category_id(*p)
                    .map(|cid| TemporalEvent::new(*p, cid, i as f32 * 0.06, 0.06, 0.6))
            })
            .collect();
        for _ in 0..20 {
            grammar.train_sequence(&events);
        }
    }

    let stats = grammar.stats();
    println!(
        "  Trained grammar: {} clusters, {:.1}% density",
        stats.num_clusters,
        stats.avg_cluster_density * 100.0
    );

    let dictionary = WordDictionary::new();
    println!("  Initialized word dictionary");

    // Process utterances
    subheader("Phase 3: Processing Utterances");

    let mut results = Vec::new();
    let mut total_ref_words = 0;
    let mut total_hyp_words = 0;

    for (idx, audio_path) in audio_files.iter().take(limit).enumerate() {
        // Get utterance ID from filename
        let utterance_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        // Get reference transcription
        let reference = transcriptions
            .get(utterance_id)
            .map(|s| s.as_str())
            .unwrap_or("");

        if reference.is_empty() {
            continue;
        }

        // Load and process audio
        let (audio, _sample_rate) = match AudioFrontend::load_audio(audio_path) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("  Error loading {}: {}", audio_path.display(), e);
                continue;
            }
        };

        // Project to HDC
        projector.reset();
        let hvs = projector.project(&audio);

        // Decode phonemes
        temporal.reset();
        for hv in &hvs {
            let scores = decoder.decode_all_scores(hv);
            temporal.process_frame(&scores);
        }
        temporal.flush();
        let phonemes = temporal.get_phonemes();

        // Convert to words
        let hypothesis_words = dictionary.phonemes_to_words(&phonemes);
        let hypothesis = hypothesis_words.join(" ");

        // Calculate WER for this utterance
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let hyp_words: Vec<&str> = hypothesis_words.iter().map(|s| s.as_str()).collect();

        total_ref_words += ref_words.len();
        total_hyp_words += hyp_words.len();

        // Score with grammar
        let events: Vec<TemporalEvent> = phonemes
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                grammar
                    .category_id(p)
                    .map(|cid| TemporalEvent::new(p, cid, i as f32 * 0.06, 0.06, 0.6))
            })
            .collect();

        let grammar_score = if events.len() >= 2 {
            grammar.score_sequence(&events)
        } else {
            0.0
        };

        results.push((
            utterance_id.to_string(),
            reference.to_string(),
            hypothesis.clone(),
            grammar_score,
        ));

        // Print progress
        if (idx + 1) % 10 == 0 || idx == 0 {
            println!(
                "  [{}/{}] {} -> {} words, {} phonemes, grammar: {:+.3}",
                idx + 1,
                limit.min(audio_files.len()),
                utterance_id,
                hypothesis_words.len(),
                phonemes.len(),
                grammar_score
            );
        }
    }

    // Summary
    header("RESULTS SUMMARY");

    println!("  Processed: {} utterances", results.len());
    println!("  Total reference words: {}", total_ref_words);
    println!("  Total hypothesis words: {}", total_hyp_words);
    println!();

    // Show some examples
    subheader("Sample Results");

    for (i, (id, reference, hypothesis, grammar_score)) in results.iter().take(5).enumerate() {
        println!("  Example {}:", i + 1);
        println!("    ID:  {}", id);
        println!("    REF: {}", reference);
        println!("    HYP: {}", hypothesis);
        println!("    Grammar: {:+.4}", grammar_score);
        println!();
    }

    // Grammar score distribution
    subheader("Grammar Score Analysis");

    let scores: Vec<f32> = results.iter().map(|(_, _, _, s)| *s).collect();
    if !scores.is_empty() {
        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("    Average grammar score: {:+.4}", avg_score);
        println!("    Min: {:+.4}, Max: {:+.4}", min_score, max_score);
        println!();

        // Score histogram
        println!("    Score distribution:");
        let bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        for i in 0..bins.len() - 1 {
            let count = scores
                .iter()
                .filter(|&&s| s >= bins[i] && s < bins[i + 1])
                .count();
            let bar = "#".repeat(count.min(40));
            println!(
                "      [{:.1}-{:.1}): {:3} {}",
                bins[i],
                bins[i + 1],
                count,
                bar
            );
        }
    }

    println!();
    println!("  Note: Actual WER calculation requires alignment and proper");
    println!("  phoneme-to-word conversion with full dictionary coverage.");
    println!();
    println!("  The grammar scores indicate phonotactic plausibility of the");
    println!("  decoded phoneme sequences. Higher scores suggest more natural");
    println!("  English phoneme patterns.");

    separator('=', 70);
}
