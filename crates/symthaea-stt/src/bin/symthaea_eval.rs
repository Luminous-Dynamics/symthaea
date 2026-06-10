// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Evaluation CLI
//!
//! Run the "Bar Exam" - evaluate WER on LibriSpeech test-clean
//!
//! Usage:
//! ```bash
//! symthaea-eval --prototypes models/prototypes.bin --test-dir data/librispeech/LibriSpeech/test-clean
//! ```

use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea_stt::{
    AudioFrontend, AudioProjector, CmuDictionary, EvalResult, NgramLM, PhonemeDecoder,
    PhonemeToWordDecoder, TemporalConfig, TemporalDecoder, TextToPhonemes, TrainedPrototypes,
};

#[derive(Parser)]
#[command(
    name = "symthaea-eval",
    about = "Symthaea Speech Recognition Evaluation",
    version,
    author
)]
struct Cli {
    /// Path to adaptive prototypes file
    #[arg(
        short,
        long,
        default_value = "data/models/dev-clean_adaptive_prototypes.bin"
    )]
    prototypes: PathBuf,

    /// Path to test dataset directory (LibriSpeech format)
    #[arg(short, long, default_value = "data/librispeech/LibriSpeech/test-clean")]
    test_dir: PathBuf,

    /// Path to CMU dictionary
    #[arg(short, long, default_value = "data/dict/cmudict.dict")]
    dictionary: PathBuf,

    /// Path to training transcripts for LM (optional)
    #[arg(long)]
    lm_train: Option<PathBuf>,

    /// Beam width for decoding
    #[arg(long, default_value = "10")]
    beam_width: usize,

    /// Language model weight (0.0 to 1.0)
    #[arg(long, default_value = "0.3")]
    lm_weight: f32,

    /// Maximum number of utterances to evaluate (0 = all)
    #[arg(long, default_value = "0")]
    max_utterances: usize,

    /// Output report path
    #[arg(short, long, default_value = "eval_report.json")]
    output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// LibriSpeech utterance
struct Utterance {
    audio_path: PathBuf,
    transcript: String,
    speaker_id: String,
    #[allow(dead_code)]
    chapter_id: String,
    utterance_id: String,
}

/// Evaluation results
#[derive(Debug, serde::Serialize)]
struct EvalReport {
    num_utterances: usize,
    total_audio_seconds: f32,
    total_inference_seconds: f32,
    rtf: f32, // Real-time factor
    wer: f32,
    per: f32,
    word_correct: usize,
    word_substitutions: usize,
    word_insertions: usize,
    word_deletions: usize,
    phoneme_correct: usize,
    phoneme_substitutions: usize,
    phoneme_insertions: usize,
    phoneme_deletions: usize,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("              SYMTHAEA EVALUATION: THE BAR EXAM                   ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    // Check required files
    if !cli.prototypes.exists() {
        eprintln!(
            "{} Prototypes not found: {:?}",
            style("ERROR:").red().bold(),
            cli.prototypes
        );
        eprintln!("       Run training first: symthaea-librispeech train --adaptive");
        std::process::exit(1);
    }

    if !cli.test_dir.exists() {
        eprintln!(
            "{} Test directory not found: {:?}",
            style("ERROR:").red().bold(),
            cli.test_dir
        );
        eprintln!("       Run: bash scripts/download_librispeech.sh --test-clean");
        std::process::exit(1);
    }

    if !cli.dictionary.exists() {
        eprintln!(
            "{} Dictionary not found: {:?}",
            style("WARNING:").yellow().bold(),
            cli.dictionary
        );
        eprintln!("       Will use phoneme-level evaluation only");
    }

    // Load prototypes
    println!("Loading prototypes from {:?}...", cli.prototypes);
    let prototypes = match TrainedPrototypes::load(&cli.prototypes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "{} Failed to load prototypes: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    println!("  Loaded {} phonemes", prototypes.len());

    // Load dictionary
    let dictionary = if cli.dictionary.exists() {
        println!("Loading dictionary from {:?}...", cli.dictionary);
        match CmuDictionary::load(&cli.dictionary) {
            Ok(d) => {
                println!("  Loaded {} words", d.len());
                Some(d)
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load dictionary: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Build phoneme decoder from prototypes
    println!("Building phoneme decoder...");
    let mut decoder = PhonemeDecoder::new();
    let phoneme_pairs = prototypes.as_pairs();
    decoder.load_prototypes(&phoneme_pairs);
    println!("  Decoder ready with {} phonemes", phoneme_pairs.len());

    // Build phoneme-to-word decoder
    let _p2w_decoder = PhonemeToWordDecoder::new();
    if let Some(ref dict) = dictionary {
        // Build reverse dictionary manually
        for word in dict.words() {
            if let Some(_phonemes) = dict.get(word) {
                // This is a workaround - we'd need to add the phonemes
            }
        }
    }

    // Load test utterances
    println!("Scanning test directory {:?}...", cli.test_dir);
    let utterances = match scan_librispeech_dir(&cli.test_dir) {
        Ok(u) => u,
        Err(e) => {
            eprintln!(
                "{} Failed to scan test directory: {}",
                style("ERROR:").red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    let total_utterances = if cli.max_utterances > 0 {
        utterances.len().min(cli.max_utterances)
    } else {
        utterances.len()
    };

    println!(
        "  Found {} utterances, evaluating {}",
        utterances.len(),
        total_utterances
    );
    println!();

    // Create audio projector
    let mut projector = AudioProjector::default_config();

    // Create temporal decoder with hysteresis
    // Note: HDC similarity scores are low (~0.06) but relative ranking works
    // Try: COMPLETELY DISABLED filtering to establish baseline
    let window_size = 1; // NO smoothing - take frame directly
    let confidence_threshold = 0.0; // DISABLED
    let min_duration_frames = 1; // No duration filter (emit even 1-frame phonemes)
    let stability_frames = 1; // Immediate transitions

    let temporal_config = TemporalConfig {
        window_size,
        confidence_threshold,
        min_duration_frames,
        stability_frames,
    };
    let _temporal_decoder = TemporalDecoder::with_config(temporal_config);
    println!("  Temporal decoder configured:");
    println!(
        "    - Window size: {} frames ({}ms)",
        window_size,
        window_size * 10
    );
    println!("    - Confidence threshold: {}", confidence_threshold);
    println!(
        "    - Min duration: {} frames ({}ms)",
        min_duration_frames,
        min_duration_frames * 10
    );
    println!("    - Stability frames: {}", stability_frames);

    // Train simple LM from training transcripts if available
    let _lm = if let Some(ref lm_path) = cli.lm_train {
        println!("Training language model from {:?}...", lm_path);
        match train_lm_from_transcripts(lm_path) {
            Ok(lm) => {
                println!("  LM trained with {} unigrams", lm.vocab_size);
                Some(lm)
            }
            Err(e) => {
                eprintln!("  Warning: Failed to train LM: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Run evaluation
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                     RUNNING EVALUATION                           ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    let pb = ProgressBar::new(total_utterances as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut total_word_result = EvalResult::default();
    let mut total_phoneme_result = EvalResult::default();
    let mut total_audio_seconds = 0.0f32;
    let mut total_inference_seconds = 0.0f32;

    let text_to_phonemes = dictionary.as_ref().map(|d| TextToPhonemes::new(d.clone()));

    for (i, utterance) in utterances.iter().take(total_utterances).enumerate() {
        pb.set_message(format!(
            "{}_{}",
            utterance.speaker_id, utterance.utterance_id
        ));

        // Load audio (supports WAV and FLAC)
        let (audio, sample_rate) = match AudioFrontend::load_audio(&utterance.audio_path) {
            Ok(a) => a,
            Err(e) => {
                if cli.verbose {
                    eprintln!("Warning: Failed to load {:?}: {}", utterance.audio_path, e);
                }
                continue;
            }
        };

        let audio_duration = audio.len() as f32 / sample_rate as f32;
        total_audio_seconds += audio_duration;

        // Time the inference
        let start = Instant::now();

        // Project audio
        projector.reset();
        // Use raw projection (no temporal windowing) to match training
        let hvs = projector.project_raw(&audio);
        let _saliences = vec![1.0f32; hvs.len()]; // Placeholder saliences

        // Decode from individual frames using the decode() method with:
        // - Minimum confidence threshold (low similarity = SILENCE)
        // - Repetition penalty (prevents mode collapse like "W AY1 W AY1...")
        decoder.reset(); // Reset history for each utterance
        let frame_phonemes: Vec<String> = hvs
            .iter()
            .filter_map(|hv| decoder.decode(hv).map(|(p, _)| p))
            .collect();

        // Window voting: take mode over sliding window
        let vote_window = 8; // ~80ms
        let mut smoothed: Vec<String> = Vec::new();
        for chunk in frame_phonemes.chunks(vote_window) {
            let mut votes: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for p in chunk {
                *votes.entry(p.as_str()).or_insert(0) += 1;
            }
            if let Some((winner, _)) = votes.iter().max_by_key(|(_, count)| *count) {
                smoothed.push(winner.to_string());
            }
        }

        // RLE to collapse consecutive same predictions
        let mut hyp_phonemes: Vec<String> = Vec::new();
        let mut prev = String::new();
        for phoneme in smoothed {
            if phoneme != prev {
                hyp_phonemes.push(phoneme.clone());
                prev = phoneme;
            }
        }

        // Debug: show info for first utterance
        if i == 0 && cli.verbose {
            let first_hv = &hvs[0];
            let scores = decoder.decode_all_scores(first_hv);
            if let Some((top_phoneme, top_score)) = scores.first() {
                let min_score = scores.last().map(|(_, s)| *s).unwrap_or(0.0);
                println!(
                    "\n  DEBUG: Frame similarities - top: {} ({:.3}), min: {:.3}",
                    top_phoneme, top_score, min_score
                );
            }
            // Show salience stats
            let min_sal = _saliences.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_sal = _saliences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let avg_sal = _saliences.iter().sum::<f32>() / _saliences.len() as f32;
            println!(
                "  DEBUG: Salience range: {:.3} to {:.3}, avg: {:.3}",
                min_sal, max_sal, avg_sal
            );
            println!(
                "  DEBUG: {} frames -> {} phonemes (after RLE)",
                hvs.len(),
                hyp_phonemes.len()
            );
        }

        let inference_time = start.elapsed().as_secs_f32();
        total_inference_seconds += inference_time;

        // Get reference phonemes
        let ref_phonemes = if let Some(ref t2p) = text_to_phonemes {
            t2p.convert(&utterance.transcript)
        } else {
            // Without dictionary, skip phoneme evaluation
            Vec::new()
        };

        // Phoneme error rate
        if !ref_phonemes.is_empty() {
            let ref_strs: Vec<&str> = ref_phonemes.iter().map(|s| s.as_str()).collect();
            let hyp_strs: Vec<&str> = hyp_phonemes.iter().map(|s| s.as_str()).collect();

            let alignment = symthaea_stt::eval::levenshtein_align(&ref_strs, &hyp_strs);
            let result = symthaea_stt::eval::eval_from_alignment(&alignment);
            total_phoneme_result.merge(&result);
        }

        // Word error rate (simple greedy phoneme-to-word)
        let hyp_text = phonemes_to_text(&hyp_phonemes, dictionary.as_ref());
        let ref_words: Vec<&str> = utterance.transcript.split_whitespace().collect();
        let hyp_words: Vec<&str> = hyp_text.split_whitespace().collect();

        let alignment = symthaea_stt::eval::levenshtein_align(&ref_words, &hyp_words);
        let result = symthaea_stt::eval::eval_from_alignment(&alignment);
        total_word_result.merge(&result);

        if cli.verbose && i < 5 {
            println!("\n  [{}] REF: {}", i, utterance.transcript);
            println!("       HYP: {}", hyp_text);
            println!("       PH:  {}", hyp_phonemes.join(" "));
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate metrics
    let wer = total_word_result.error_rate() * 100.0;
    let per = total_phoneme_result.error_rate() * 100.0;
    let rtf = total_inference_seconds / total_audio_seconds;

    // Print results
    println!();
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!(
        "{}",
        style("                     EVALUATION RESULTS                           ")
            .bold()
            .cyan()
    );
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    println!("{}:", style("Summary").bold());
    println!("  Utterances evaluated:  {}", total_utterances);
    println!("  Total audio:           {:.1}s", total_audio_seconds);
    println!("  Total inference:       {:.1}s", total_inference_seconds);
    println!("  Real-time factor:      {:.2}x", rtf);
    println!();

    println!("{}:", style("Word Error Rate (WER)").bold());
    println!(
        "  Reference words:    {}",
        total_word_result.reference_length
    );
    println!(
        "  Hypothesis words:   {}",
        total_word_result.hypothesis_length
    );
    println!("  Correct:            {}", total_word_result.correct);
    println!("  Substitutions:      {}", total_word_result.substitutions);
    println!("  Insertions:         {}", total_word_result.insertions);
    println!("  Deletions:          {}", total_word_result.deletions);
    println!("  {}:               {:.1}%", style("WER").bold(), wer);
    println!();

    if total_phoneme_result.reference_length > 0 {
        println!("{}:", style("Phoneme Error Rate (PER)").bold());
        println!(
            "  Reference phonemes: {}",
            total_phoneme_result.reference_length
        );
        println!(
            "  Hypothesis phonemes:{}",
            total_phoneme_result.hypothesis_length
        );
        println!("  Correct:            {}", total_phoneme_result.correct);
        println!(
            "  Substitutions:      {}",
            total_phoneme_result.substitutions
        );
        println!("  Insertions:         {}", total_phoneme_result.insertions);
        println!("  Deletions:          {}", total_phoneme_result.deletions);
        println!("  {}:               {:.1}%", style("PER").bold(), per);
        println!();
    }

    // Verdict
    println!(
        "{}",
        style("══════════════════════════════════════════════════════════════════").cyan()
    );
    println!();

    if wer < 20.0 {
        println!(
            "{}",
            style("╔═══════════════════════════════════════════════════════════════╗").green()
        );
        println!(
            "{}",
            style("║                    BAR EXAM: PASSED                           ║")
                .bold()
                .green()
        );
        println!(
            "{}",
            style("╠═══════════════════════════════════════════════════════════════╣").green()
        );
        println!("{}  WER {:.1}% < 20% target", style("║").green(), wer);
        println!(
            "{}  RTF {:.2}x (real-time capable)",
            style("║").green(),
            rtf
        );
        println!(
            "{}  Symthaea is production-ready for speech recognition",
            style("║").green()
        );
        println!(
            "{}",
            style("╚═══════════════════════════════════════════════════════════════╝").green()
        );
    } else if wer < 50.0 {
        println!(
            "{}",
            style("╔═══════════════════════════════════════════════════════════════╗").yellow()
        );
        println!(
            "{}",
            style("║                 BAR EXAM: PARTIAL PASS                        ║")
                .bold()
                .yellow()
        );
        println!(
            "{}",
            style("╠═══════════════════════════════════════════════════════════════╣").yellow()
        );
        println!("{}  WER {:.1}% (target: <20%)", style("║").yellow(), wer);
        println!(
            "{}  System recognizes speech but needs tuning",
            style("║").yellow()
        );
        println!(
            "{}  Consider: more training data, LM integration, beam search",
            style("║").yellow()
        );
        println!(
            "{}",
            style("╚═══════════════════════════════════════════════════════════════╝").yellow()
        );
    } else {
        println!(
            "{}",
            style("╔═══════════════════════════════════════════════════════════════╗").red()
        );
        println!(
            "{}",
            style("║                    BAR EXAM: FAILED                           ║")
                .bold()
                .red()
        );
        println!(
            "{}",
            style("╠═══════════════════════════════════════════════════════════════╣").red()
        );
        println!("{}  WER {:.1}% (target: <20%)", style("║").red(), wer);
        println!(
            "{}  Decoder is guessing - phoneme mapping may be wrong",
            style("║").red()
        );
        println!(
            "{}  Check: prototype quality, dictionary alignment, audio processing",
            style("║").red()
        );
        println!(
            "{}",
            style("╚═══════════════════════════════════════════════════════════════╝").red()
        );
    }
    println!();

    // Save report
    let report = EvalReport {
        num_utterances: total_utterances,
        total_audio_seconds,
        total_inference_seconds,
        rtf,
        wer,
        per,
        word_correct: total_word_result.correct,
        word_substitutions: total_word_result.substitutions,
        word_insertions: total_word_result.insertions,
        word_deletions: total_word_result.deletions,
        phoneme_correct: total_phoneme_result.correct,
        phoneme_substitutions: total_phoneme_result.substitutions,
        phoneme_insertions: total_phoneme_result.insertions,
        phoneme_deletions: total_phoneme_result.deletions,
    };

    if let Ok(json) = serde_json::to_string_pretty(&report) {
        if let Err(e) = fs::write(&cli.output, &json) {
            eprintln!("Warning: Failed to save report: {}", e);
        } else {
            println!("Report saved to {:?}", cli.output);
        }
    }
}

fn scan_librispeech_dir(dir: &Path) -> std::io::Result<Vec<Utterance>> {
    let mut utterances = Vec::new();

    // LibriSpeech structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
    for speaker_entry in fs::read_dir(dir)? {
        let speaker_entry = speaker_entry?;
        let speaker_path = speaker_entry.path();

        if !speaker_path.is_dir() {
            continue;
        }

        let speaker_id = speaker_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        for chapter_entry in fs::read_dir(&speaker_path)? {
            let chapter_entry = chapter_entry?;
            let chapter_path = chapter_entry.path();

            if !chapter_path.is_dir() {
                continue;
            }

            let chapter_id = chapter_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Find transcript file
            let trans_file = chapter_path.join(format!("{}-{}.trans.txt", speaker_id, chapter_id));

            if !trans_file.exists() {
                continue;
            }

            // Parse transcripts
            let file = File::open(&trans_file)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                let parts: Vec<&str> = line.splitn(2, ' ').collect();

                if parts.len() != 2 {
                    continue;
                }

                let utterance_id = parts[0];
                let transcript = parts[1].to_lowercase();

                // Try FLAC first, then WAV
                let flac_path = chapter_path.join(format!("{}.flac", utterance_id));
                let wav_path = chapter_path.join(format!("{}.wav", utterance_id));

                let audio_path = if flac_path.exists() {
                    flac_path
                } else if wav_path.exists() {
                    wav_path
                } else {
                    continue;
                };

                utterances.push(Utterance {
                    audio_path,
                    transcript,
                    speaker_id: speaker_id.clone(),
                    chapter_id: chapter_id.clone(),
                    utterance_id: utterance_id.to_string(),
                });
            }
        }
    }

    // Sort for reproducibility
    utterances.sort_by(|a, b| a.utterance_id.cmp(&b.utterance_id));

    Ok(utterances)
}

fn train_lm_from_transcripts(dir: &Path) -> std::io::Result<NgramLM> {
    let mut corpus = Vec::new();

    // Scan for transcript files
    fn scan_transcripts(dir: &Path, corpus: &mut Vec<Vec<String>>) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                scan_transcripts(&path, corpus)?;
            } else if path.extension().map(|e| e == "txt").unwrap_or(false)
                && path.to_string_lossy().contains(".trans.")
            {
                let file = File::open(&path)?;
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line?;
                    let parts: Vec<&str> = line.splitn(2, ' ').collect();

                    if parts.len() == 2 {
                        let words: Vec<String> = parts[1]
                            .to_lowercase()
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect();

                        if !words.is_empty() {
                            corpus.push(words);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    scan_transcripts(dir, &mut corpus)?;

    let mut lm = NgramLM::new(2); // Bigram
    lm.train(&corpus);

    Ok(lm)
}

fn phonemes_to_text(phonemes: &[String], dictionary: Option<&CmuDictionary>) -> String {
    // Without dictionary, just return phoneme string
    if dictionary.is_none() || phonemes.is_empty() {
        return format!("/{}/", phonemes.join(" "));
    }

    // Simple approach: map common phoneme sequences to words
    // This is a placeholder - proper implementation needs reverse dictionary lookup
    // For now, we just output phonemes which will give high WER but valid PER

    phonemes.join(" ")
}
