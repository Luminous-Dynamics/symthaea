// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-World Whale Validation
//!
//! Validates the Universal Temporal Grammar on actual whale recordings.
//! This is the "Whale Bridge" - proving the architecture works on real data
//! before scaling to speech.

use std::collections::HashMap;
use std::path::Path;
use symthaea_stt::audio::AudioFrontend;
use symthaea_stt::liquid::{LiquidBank, LiquidFeatures};
use symthaea_stt::temporal_grammar::{DomainConfig, TemporalEvent, TemporalGrammar};

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

/// Extract events from audio using Liquid feature bank
fn extract_events(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    let mut bank = LiquidBank::new(sample_rate, frame_size);
    let mut events: Vec<TemporalEvent> = Vec::new();

    let mut current_event: Option<(String, usize, f32, f32)> = None; // (type, class, start, peak_intensity)
    let mut time = 0.0f32;
    let frame_duration = frame_size as f32 / sample_rate;

    // Thresholds for event detection
    let click_thresh = 0.4;
    let whistle_thresh = 0.35;
    let burst_thresh = 0.3;
    let silence_thresh = 0.02;
    let min_event_duration = 0.02; // 20ms minimum

    for chunk in audio.chunks(frame_size) {
        if chunk.len() < frame_size {
            break;
        }

        let features = bank.process_frame(chunk);

        // Classify frame
        let (event_type, class_id) = if features.energy < silence_thresh {
            ("silence", 8)
        } else if features.click > click_thresh
            && features.click > features.whistle
            && features.click > features.burst
        {
            ("click", 0)
        } else if features.whistle > whistle_thresh && features.whistle > features.burst {
            // Determine whistle direction based on spectral tilt change
            if features.burst > 0.2 {
                ("whistle_up", 1)
            } else {
                ("whistle_flat", 3)
            }
        } else if features.burst > burst_thresh {
            ("burst", 4)
        } else {
            ("silence", 8)
        };

        // State machine for event segmentation
        match &current_event {
            None => {
                if event_type != "silence" {
                    current_event = Some((event_type.to_string(), class_id, time, features.energy));
                }
            }
            Some((curr_type, curr_class, start_time, peak_intensity)) => {
                if event_type != curr_type {
                    // Event transition
                    let duration = time - start_time;
                    if duration >= min_event_duration && curr_type != "silence" {
                        events.push(TemporalEvent::new(
                            curr_type,
                            *curr_class,
                            *start_time,
                            duration,
                            *peak_intensity,
                        ));
                    }

                    if event_type != "silence" {
                        current_event =
                            Some((event_type.to_string(), class_id, time, features.energy));
                    } else {
                        current_event = None;
                    }
                } else {
                    // Same event, update peak intensity
                    let new_peak = peak_intensity.max(features.energy);
                    current_event = Some((curr_type.clone(), *curr_class, *start_time, new_peak));
                }
            }
        }

        time += frame_duration;
    }

    // Flush final event
    if let Some((curr_type, curr_class, start_time, peak_intensity)) = current_event {
        let duration = time - start_time;
        if duration >= min_event_duration && curr_type != "silence" {
            events.push(TemporalEvent::new(
                &curr_type,
                curr_class,
                start_time,
                duration,
                peak_intensity,
            ));
        }
    }

    events
}

/// Analyze a whale recording
fn analyze_recording(
    path: &Path,
    grammar: &mut TemporalGrammar,
) -> Option<(String, Vec<TemporalEvent>, f32)> {
    let filename = path.file_name()?.to_string_lossy().to_string();

    // Load audio
    let (audio, sample_rate) = match AudioFrontend::load_wav(path) {
        Ok(result) => result,
        Err(e) => {
            println!("    Error loading {}: {}", filename, e);
            return None;
        }
    };

    println!(
        "    {} ({:.1}s @ {}Hz)",
        filename,
        audio.len() as f32 / sample_rate as f32,
        sample_rate
    );

    // Extract events
    let frame_size = 2048;
    let events = extract_events(&audio, sample_rate as f32, frame_size);

    // Count event types
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for event in &events {
        *type_counts.entry(event.category.clone()).or_insert(0) += 1;
    }

    print!("      Events: ");
    for (t, c) in &type_counts {
        print!("{}:{} ", t, c);
    }
    println!();

    // Score against grammar
    let score = if !events.is_empty() {
        grammar.score_sequence(&events)
    } else {
        0.0
    };

    Some((filename, events, score))
}

fn main() -> std::io::Result<()> {
    header("WHALE BRIDGE: REAL-WORLD VALIDATION");

    println!("  Testing Universal Temporal Grammar on actual whale recordings.");
    println!("  This validates the architecture on real, messy bioacoustic data.");

    // Find whale audio files
    let whale_dir = Path::new("data/whales/sperm");
    let test_dir = Path::new("data/test_animals");

    let mut whale_files: Vec<_> = std::fs::read_dir(whale_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "wav")
                .unwrap_or(false)
        })
        .collect();

    if whale_files.is_empty() {
        println!("\n  No whale audio files found in {}", whale_dir.display());
        println!("  Run: bash scripts/download_whales.sh");
        return Ok(());
    }

    println!("\n  Found {} whale recordings", whale_files.len());

    // Create cetacean temporal grammar
    let mut grammar = TemporalGrammar::cetacean();
    let stats = grammar.stats();

    subheader("Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Categories:  {}", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!("    Hierarchy:   {} levels", stats.hierarchy_depth);
    println!(
        "    Prediction:  {}",
        if stats.predictive_feedback {
            "ON"
        } else {
            "OFF"
        }
    );

    // Phase 1: Analyze all recordings
    subheader("Phase 1: Analyzing Recordings");

    let mut all_results: Vec<(String, Vec<TemporalEvent>, f32)> = Vec::new();

    for entry in &whale_files {
        if let Some(result) = analyze_recording(&entry.path(), &mut grammar) {
            all_results.push(result);
        }
    }

    // Phase 2: Train on first half, test on second half
    subheader("Phase 2: Train/Test Split");

    let mid = all_results.len() / 2;
    let (train_set, test_set) = all_results.split_at(mid);

    println!("    Training set: {} recordings", train_set.len());
    println!("    Test set:     {} recordings", test_set.len());

    // Train on first half
    println!("\n    Training on:");
    for (filename, events, _) in train_set {
        if !events.is_empty() {
            for _ in 0..10 {
                grammar.train_sequence(events);
            }
            println!("      - {} ({} events)", filename, events.len());
        }
    }

    let stats = grammar.stats();
    println!(
        "\n    Grammar density after training: {:.3}",
        stats.grammar_density
    );

    // Test on second half
    subheader("Phase 3: Discrimination Testing");

    println!("    Scoring test recordings against trained grammar:\n");

    let mut train_scores: Vec<f32> = Vec::new();
    let mut test_scores: Vec<f32> = Vec::new();

    // Re-score training set
    for (filename, events, _) in train_set {
        if !events.is_empty() {
            let score = grammar.score_sequence(events);
            train_scores.push(score);
            println!("      [TRAIN] {}: {:.4}", filename, score);
        }
    }

    println!();

    // Score test set
    for (filename, events, _) in test_set {
        if !events.is_empty() {
            let score = grammar.score_sequence(events);
            test_scores.push(score);
            println!("      [TEST]  {}: {:.4}", filename, score);
        }
    }

    // Calculate statistics
    let train_avg = if !train_scores.is_empty() {
        train_scores.iter().sum::<f32>() / train_scores.len() as f32
    } else {
        0.0
    };

    let test_avg = if !test_scores.is_empty() {
        test_scores.iter().sum::<f32>() / test_scores.len() as f32
    } else {
        0.0
    };

    subheader("Results");

    println!("    Training set average: {:.4}", train_avg);
    println!("    Test set average:     {:.4}", test_avg);
    println!("    Generalization gap:   {:+.4}", train_avg - test_avg);

    // Phase 4: Cross-species test (if test_animals exists)
    if test_dir.exists() {
        subheader("Phase 4: Cross-Species Discrimination");

        println!("    Testing against other animal recordings:\n");

        let test_files: Vec<_> = std::fs::read_dir(test_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "wav")
                    .unwrap_or(false)
            })
            .collect();

        for entry in &test_files {
            if let Some((filename, events, _)) = analyze_recording(&entry.path(), &mut grammar) {
                if !events.is_empty() {
                    let score = grammar.score_sequence(&events);
                    let species = if filename.contains("whale") {
                        "[WHALE]"
                    } else if filename.contains("dolphin") {
                        "[DOLPHIN]"
                    } else if filename.contains("bird") {
                        "[BIRD]"
                    } else {
                        "[OTHER]"
                    };
                    println!("      {} {}: {:.4}", species, filename, score);
                }
            }
        }

        println!("\n    If whale scores > non-whale scores, species discrimination works!");
    }

    // Summary
    header("VALIDATION SUMMARY");

    println!("  Universal Temporal Grammar on Real Whale Audio:");
    println!();
    println!("    Recordings processed:  {}", all_results.len());
    println!(
        "    Total events detected: {}",
        all_results.iter().map(|(_, e, _)| e.len()).sum::<usize>()
    );
    println!("    Training patterns:     {}", stats.training_count);
    println!("    Grammar density:       {:.3}", stats.grammar_density);
    println!();

    if train_avg > 0.3 {
        println!("  [SUCCESS] Grammar learned patterns from real whale recordings!");
    } else if train_avg > 0.1 {
        println!("  [PARTIAL] Grammar shows some learning, may need more data.");
    } else {
        println!("  [INFO] Low scores may indicate threshold tuning needed.");
    }

    if (train_avg - test_avg).abs() < 0.2 {
        println!("  [SUCCESS] Good generalization to unseen recordings!");
    }

    separator('=', 70);
    println!("  WHALE BRIDGE VALIDATION COMPLETE");
    separator('=', 70);

    Ok(())
}
