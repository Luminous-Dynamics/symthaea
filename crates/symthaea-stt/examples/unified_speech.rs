// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Speech Grammar Test
//!
//! Compare unified vs baseline temporal grammar on speech phoneme patterns.

use symthaea_stt::temporal_grammar::{DomainConfig, Sparsity, TemporalEvent, TemporalGrammar};
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

/// Built-in phoneme sequences for common words
fn word_to_events(word: &str, start: &mut f32) -> Vec<TemporalEvent> {
    let phonemes: Option<Vec<(&str, f32, f32)>> = match word.to_uppercase().as_str() {
        "THE" => Some(vec![("DH", 0.04, 0.5), ("AH", 0.06, 0.6)]),
        "CAT" => Some(vec![("K", 0.06, 0.6), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "SAT" => Some(vec![("S", 0.08, 0.5), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "DOG" => Some(vec![
            ("D", 0.05, 0.6),
            ("AO", 0.12, 0.75),
            ("G", 0.06, 0.55),
        ]),
        "RAN" => Some(vec![("R", 0.06, 0.5), ("AE", 0.1, 0.8), ("N", 0.05, 0.5)]),
        "MAT" => Some(vec![("M", 0.06, 0.5), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "BAT" => Some(vec![("B", 0.05, 0.6), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "HAT" => Some(vec![("HH", 0.05, 0.4), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "RAT" => Some(vec![("R", 0.06, 0.5), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "PAT" => Some(vec![("P", 0.05, 0.6), ("AE", 0.1, 0.8), ("T", 0.05, 0.5)]),
        "BIG" => Some(vec![
            ("B", 0.05, 0.6),
            ("IH", 0.08, 0.75),
            ("G", 0.06, 0.55),
        ]),
        "PIG" => Some(vec![
            ("P", 0.05, 0.6),
            ("IH", 0.08, 0.75),
            ("G", 0.06, 0.55),
        ]),
        "DIG" => Some(vec![
            ("D", 0.05, 0.6),
            ("IH", 0.08, 0.75),
            ("G", 0.06, 0.55),
        ]),
        "FISH" => Some(vec![("F", 0.08, 0.5), ("IH", 0.08, 0.75), ("SH", 0.1, 0.5)]),
        "DISH" => Some(vec![("D", 0.05, 0.6), ("IH", 0.08, 0.75), ("SH", 0.1, 0.5)]),
        "WISH" => Some(vec![("W", 0.06, 0.5), ("IH", 0.08, 0.75), ("SH", 0.1, 0.5)]),
        "STRONG" => Some(vec![
            ("S", 0.08, 0.5),
            ("T", 0.04, 0.5),
            ("R", 0.05, 0.5),
            ("AO", 0.1, 0.7),
            ("NG", 0.06, 0.5),
        ]),
        "STRING" => Some(vec![
            ("S", 0.08, 0.5),
            ("T", 0.04, 0.5),
            ("R", 0.05, 0.5),
            ("IH", 0.08, 0.7),
            ("NG", 0.06, 0.5),
        ]),
        "SPRING" => Some(vec![
            ("S", 0.08, 0.5),
            ("P", 0.04, 0.5),
            ("R", 0.05, 0.5),
            ("IH", 0.08, 0.7),
            ("NG", 0.06, 0.5),
        ]),
        "STREET" => Some(vec![
            ("S", 0.08, 0.5),
            ("T", 0.04, 0.5),
            ("R", 0.05, 0.5),
            ("IY", 0.1, 0.75),
            ("T", 0.05, 0.5),
        ]),
        _ => None,
    };

    let mut events = Vec::new();
    if let Some(phones) = phonemes {
        for (phoneme, dur, intensity) in phones {
            events.push(TemporalEvent::new(phoneme, 0, *start, dur, intensity));
            *start += dur;
        }
        *start += 0.05; // Word gap
    }
    events
}

fn sentence_to_events(words: &[&str]) -> Vec<TemporalEvent> {
    let mut events = Vec::new();
    let mut time = 0.0f32;
    for word in words {
        events.extend(word_to_events(word, &mut time));
    }
    events
}

fn remap_events(events: &[TemporalEvent], grammar: &UnifiedGrammar) -> Vec<TemporalEvent> {
    events
        .iter()
        .filter_map(|e| {
            grammar.category_id(&e.category).map(|cid| {
                TemporalEvent::new(&e.category, cid, e.start_time, e.duration, e.intensity)
            })
        })
        .collect()
}

fn remap_events_baseline(
    events: &[TemporalEvent],
    grammar: &TemporalGrammar,
) -> Vec<TemporalEvent> {
    events
        .iter()
        .filter_map(|e| {
            grammar.category_id(&e.category).map(|cid| {
                TemporalEvent::new(&e.category, cid, e.start_time, e.duration, e.intensity)
            })
        })
        .collect()
}

fn baseline_speech_config() -> DomainConfig {
    let phonemes = vec![
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        "B", "D", "G", "K", "P", "T", "CH", "DH", "F", "JH", "S", "SH", "TH", "V", "Z", "ZH", "M",
        "N", "NG", "L", "R", "W", "Y", "HH", "SIL",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    DomainConfig {
        name: "speech_baseline".to_string(),
        categories: phonemes,
        sample_rate: 16000.0,
        frame_size: 400,
        sparsity: Sparsity::Sparse5,
        duration_bins: 8,
        intensity_bins: 4,
        predictive_feedback: true,
        prediction_boost: 0.3,
        hierarchy_depth: 2,
    }
}

fn main() {
    header("UNIFIED SPEECH GRAMMAR COMPARISON");

    println!("  Comparing unified vs baseline on English phonotactics.");

    // Create grammars
    let mut unified = UnifiedGrammar::speech();
    let mut baseline = TemporalGrammar::new(baseline_speech_config());

    // Training patterns - CVC words with similar structure
    let train_words = vec![
        vec!["THE", "CAT", "SAT"],
        vec!["THE", "DOG", "RAN"],
        vec!["THE", "BAT", "HAT"],
        vec!["THE", "RAT", "MAT"],
        vec!["BIG", "PIG", "DIG"],
        vec!["FISH", "DISH", "WISH"],
        vec!["STRONG", "STRING", "SPRING"],
    ];

    subheader("Phase 1: Training");

    for words in &train_words {
        let events = sentence_to_events(words);
        let u_events = remap_events(&events, &unified);
        let b_events = remap_events_baseline(&events, &baseline);

        if u_events.len() >= 3 && b_events.len() >= 3 {
            for _ in 0..30 {
                unified.train_sequence(&u_events);
                baseline.train_sequence(&b_events);
            }
            println!("    Trained: {:?} ({} phonemes)", words, u_events.len());
        }
    }

    let u_stats = unified.stats();
    let b_stats = baseline.stats();

    println!(
        "\n    Unified:  {} clusters, {:.1}% cluster density",
        u_stats.num_clusters,
        u_stats.avg_cluster_density * 100.0
    );
    println!(
        "    Baseline: {:.1}% density",
        b_stats.grammar_density * 100.0
    );

    // Test patterns
    subheader("Phase 2: Scoring");

    let test_patterns = vec![
        (vec!["THE", "CAT", "SAT"], "trained"),          // Exact match
        (vec!["THE", "PAT", "MAT"], "similar"),          // Similar CVC
        (vec!["STREET", "STRONG"], "consonant_cluster"), // Different structure
    ];

    let mut u_scores = Vec::new();
    let mut b_scores = Vec::new();

    for (words, label) in &test_patterns {
        let events = sentence_to_events(words);
        let u_events = remap_events(&events, &unified);
        let b_events = remap_events_baseline(&events, &baseline);

        if u_events.len() >= 2 && b_events.len() >= 2 {
            let u_score = unified.score_sequence(&u_events);
            let b_score = baseline.score_sequence(&b_events);

            u_scores.push((label.to_string(), u_score));
            b_scores.push((label.to_string(), b_score));

            println!("    [{}] {:?}", label.to_uppercase(), words);
            println!("      Unified:  {:+.4}", u_score);
            println!("      Baseline: {:+.4}", b_score);
            println!("      Delta:    {:+.4}", u_score - b_score);
            println!();
        }
    }

    // Phonotactic test: valid vs invalid onset clusters
    subheader("Phase 3: Phonotactic Discrimination");

    // Valid: str-, spr-, consonant clusters that exist in English
    // Invalid: tl-, sr-, clusters that don't exist

    let valid_onset: Vec<TemporalEvent> = vec![
        TemporalEvent::new("S", 0, 0.0, 0.08, 0.5),
        TemporalEvent::new("T", 0, 0.08, 0.04, 0.5),
        TemporalEvent::new("R", 0, 0.12, 0.05, 0.5),
        TemporalEvent::new("IY", 0, 0.17, 0.1, 0.75),
        TemporalEvent::new("T", 0, 0.27, 0.05, 0.5),
    ];

    let invalid_onset: Vec<TemporalEvent> = vec![
        TemporalEvent::new("T", 0, 0.0, 0.04, 0.5),
        TemporalEvent::new("L", 0, 0.04, 0.05, 0.5), // TL- invalid in English
        TemporalEvent::new("IY", 0, 0.09, 0.1, 0.75),
        TemporalEvent::new("T", 0, 0.19, 0.05, 0.5),
    ];

    let u_valid = unified.score_sequence(&remap_events(&valid_onset, &unified));
    let u_invalid = unified.score_sequence(&remap_events(&invalid_onset, &unified));
    let b_valid = baseline.score_sequence(&remap_events_baseline(&valid_onset, &baseline));
    let b_invalid = baseline.score_sequence(&remap_events_baseline(&invalid_onset, &baseline));

    println!("    Valid onset (STR-):");
    println!("      Unified:  {:+.4}", u_valid);
    println!("      Baseline: {:+.4}", b_valid);
    println!();
    println!("    Invalid onset (TL-):");
    println!("      Unified:  {:+.4}", u_invalid);
    println!("      Baseline: {:+.4}", b_invalid);
    println!();
    println!("    Phonotactic discrimination (valid - invalid):");
    println!("      Unified:  {:+.4}", u_valid - u_invalid);
    println!("      Baseline: {:+.4}", b_valid - b_invalid);

    // Results
    header("SPEECH COMPARISON RESULTS");

    let u_trained = u_scores
        .iter()
        .find(|(l, _)| l == "trained")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let u_similar = u_scores
        .iter()
        .find(|(l, _)| l == "similar")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let b_trained = b_scores
        .iter()
        .find(|(l, _)| l == "trained")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    let b_similar = b_scores
        .iter()
        .find(|(l, _)| l == "similar")
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let u_pattern_disc = u_trained - u_similar;
    let b_pattern_disc = b_trained - b_similar;
    let u_phonotactic_disc = u_valid - u_invalid;
    let b_phonotactic_disc = b_valid - b_invalid;

    println!("    Pattern discrimination (trained - similar):");
    println!("      Unified:  {:+.4}", u_pattern_disc);
    println!("      Baseline: {:+.4}", b_pattern_disc);
    println!();
    println!("    Phonotactic discrimination (valid - invalid onset):");
    println!("      Unified:  {:+.4}", u_phonotactic_disc);
    println!("      Baseline: {:+.4}", b_phonotactic_disc);
    println!();

    let u_total = u_pattern_disc + u_phonotactic_disc;
    let b_total = b_pattern_disc + b_phonotactic_disc;

    println!("    Combined discrimination:");
    println!("      Unified:  {:+.4}", u_total);
    println!("      Baseline: {:+.4}", b_total);
    println!("      Improvement: {:+.4}", u_total - b_total);
    println!();

    if u_total > b_total {
        println!("  [SUCCESS] Unified architecture outperforms baseline!");
    } else if (u_total - b_total).abs() < 0.001 {
        println!("  [INFO] Both architectures perform similarly.");
    } else {
        println!("  [INFO] Baseline performed better on speech.");
    }

    separator('=', 70);
}
