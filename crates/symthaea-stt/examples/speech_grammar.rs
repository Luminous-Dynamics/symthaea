// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Speech Temporal Grammar
//!
//! Apply Universal Temporal Grammar to phoneme sequences for improved
//! speech recognition. Learns phonotactic patterns from training data.

use std::collections::HashMap;
use symthaea_stt::temporal_grammar::{DomainConfig, Sparsity, TemporalEvent, TemporalGrammar};

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

/// Create speech grammar config
fn speech_grammar_config() -> DomainConfig {
    // ARPAbet phonemes (39 + silence)
    let phonemes = vec![
        // Vowels
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        // Consonants - Stops
        "B", "D", "G", "K", "P", "T", // Consonants - Fricatives
        "CH", "DH", "F", "JH", "S", "SH", "TH", "V", "Z", "ZH", // Consonants - Nasals
        "M", "N", "NG", // Consonants - Liquids/Glides
        "L", "R", "W", "Y", "HH", // Silence
        "SIL",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    DomainConfig {
        name: "speech_phoneme".to_string(),
        categories: phonemes,
        sample_rate: 16000.0,
        frame_size: 400, // 25ms at 16kHz
        sparsity: Sparsity::Sparse5,
        duration_bins: 8,
        intensity_bins: 4,
        predictive_feedback: true,
        prediction_boost: 0.35,
        hierarchy_depth: 3, // phoneme → syllable → word
    }
}

/// Built-in word to phoneme dictionary (common words)
fn get_word_phonemes(word: &str) -> Option<Vec<&'static str>> {
    match word.to_uppercase().as_str() {
        // Common words
        "THE" => Some(vec!["DH", "AH"]),
        "A" => Some(vec!["AH"]),
        "AN" => Some(vec!["AH", "N"]),
        "CAT" => Some(vec!["K", "AE", "T"]),
        "SAT" => Some(vec!["S", "AE", "T"]),
        "ON" => Some(vec!["AA", "N"]),
        "MAT" => Some(vec!["M", "AE", "T"]),
        "DOG" => Some(vec!["D", "AO", "G"]),
        "RAN" => Some(vec!["R", "AE", "N"]),
        "TO" => Some(vec!["T", "UW"]),
        "PARK" => Some(vec!["P", "AA", "R", "K"]),
        "SHE" => Some(vec!["SH", "IY"]),
        "WILL" => Some(vec!["W", "IH", "L"]),
        "GO" => Some(vec!["G", "OW"]),
        "STORE" => Some(vec!["S", "T", "AO", "R"]),
        "HOW" => Some(vec!["HH", "AW"]),
        "ARE" => Some(vec!["AA", "R"]),
        "YOU" => Some(vec!["Y", "UW"]),
        "DOING" => Some(vec!["D", "UW", "IH", "NG"]),
        "TODAY" => Some(vec!["T", "AH", "D", "EY"]),
        "WHAT" => Some(vec!["W", "AH", "T"]),
        "IS" => Some(vec!["IH", "Z"]),
        "YOUR" => Some(vec!["Y", "AO", "R"]),
        "NAME" => Some(vec!["N", "EY", "M"]),
        "NICE" => Some(vec!["N", "AY", "S"]),
        "MEET" => Some(vec!["M", "IY", "T"]),
        "THANK" => Some(vec!["TH", "AE", "NG", "K"]),
        "VERY" => Some(vec!["V", "EH", "R", "IY"]),
        "MUCH" => Some(vec!["M", "AH", "CH"]),
        "QUICK" => Some(vec!["K", "W", "IH", "K"]),
        "BROWN" => Some(vec!["B", "R", "AW", "N"]),
        "FOX" => Some(vec!["F", "AA", "K", "S"]),
        "JUMPS" => Some(vec!["JH", "AH", "M", "P", "S"]),
        "PLEASE" => Some(vec!["P", "L", "IY", "Z"]),
        "SPEAK" => Some(vec!["S", "P", "IY", "K"]),
        "CLEARLY" => Some(vec!["K", "L", "IH", "R", "L", "IY"]),
        "NOW" => Some(vec!["N", "AW"]),
        "STRONG" => Some(vec!["S", "T", "R", "AO", "NG"]),
        "STRANGE" => Some(vec!["S", "T", "R", "EY", "N", "JH"]),
        "STRING" => Some(vec!["S", "T", "R", "IH", "NG"]),
        "STRUCK" => Some(vec!["S", "T", "R", "AH", "K"]),
        "SO" => Some(vec!["S", "OW"]),
        "DAY" => Some(vec!["D", "EY"]),
        "GOING" => Some(vec!["G", "OW", "IH", "NG"]),
        "HELLO" => Some(vec!["HH", "AH", "L", "OW"]),
        "WORLD" => Some(vec!["W", "ER", "L", "D"]),
        "XYLOPHONE" => Some(vec!["Z", "AY", "L", "AH", "F", "OW", "N"]),
        "RHYTHM" => Some(vec!["R", "IH", "DH", "AH", "M"]),
        "SPHINX" => Some(vec!["S", "F", "IH", "NG", "K", "S"]),
        "OF" => Some(vec!["AH", "V"]),
        "QUARTZ" => Some(vec!["K", "W", "AO", "R", "T", "S"]),
        "JAZZ" => Some(vec!["JH", "AE", "Z"]),
        "AND" => Some(vec!["AE", "N", "D"]),
        "BLUES" => Some(vec!["B", "L", "UW", "Z"]),
        "MUSIC" => Some(vec!["M", "Y", "UW", "Z", "IH", "K"]),
        _ => None,
    }
}

/// Convert word to phoneme events
fn word_to_events(word: &str, start_time: &mut f32) -> Vec<TemporalEvent> {
    let mut events = Vec::new();

    if let Some(phonemes) = get_word_phonemes(word) {
        for (i, phoneme) in phonemes.iter().enumerate() {
            // Estimate duration based on phoneme type
            let duration = if "AEIOU".chars().any(|v| phoneme.starts_with(v)) {
                0.08 // Vowels longer
            } else {
                0.05 // Consonants shorter
            };

            // Estimate intensity (vowels louder)
            let intensity = if "AEIOU".chars().any(|v| phoneme.starts_with(v)) {
                0.8
            } else {
                0.5
            };

            events.push(TemporalEvent::new(
                phoneme,
                i, // Will be remapped by grammar
                *start_time,
                duration,
                intensity,
            ));

            *start_time += duration;
        }
    }

    events
}

/// Convert sentence to phoneme events
fn sentence_to_events(sentence: &str) -> Vec<TemporalEvent> {
    let mut events = Vec::new();
    let mut time = 0.0f32;

    for word in sentence.split_whitespace() {
        let clean_word: String = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>()
            .to_uppercase();

        if !clean_word.is_empty() {
            let word_events = word_to_events(&clean_word, &mut time);
            events.extend(word_events);

            // Add small pause between words
            time += 0.05;
        }
    }

    events
}

/// Remap event class IDs to grammar categories
fn remap_events(events: &[TemporalEvent], grammar: &TemporalGrammar) -> Vec<TemporalEvent> {
    events
        .iter()
        .filter_map(|e| {
            grammar.category_id(&e.category).map(|cid| {
                TemporalEvent::new(&e.category, cid, e.start_time, e.duration, e.intensity)
            })
        })
        .collect()
}

fn main() {
    header("SPEECH TEMPORAL GRAMMAR");

    println!("  Applying Universal Temporal Grammar to phoneme sequences.");
    println!("  Uses built-in dictionary for word→phoneme conversion.");

    // Create speech grammar
    let mut grammar = TemporalGrammar::new(speech_grammar_config());
    let stats = grammar.stats();

    subheader("Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Phonemes:    {}", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!("    Hierarchy:   {} levels", stats.hierarchy_depth);

    // Training sentences - common English patterns
    let training_sentences = vec![
        // Simple CVC words
        "the cat sat on the mat",
        "a dog ran to the park",
        "she will go to the store",
        // Common phrases
        "how are you doing today",
        "what is your name",
        "nice to meet you",
        "thank you very much",
        // Phonotactically diverse
        "the quick brown fox jumps",
        "please speak clearly now",
        "strong strange string struck",
    ];

    subheader("Phase 1: Training on Common Phrases");

    let mut training_events: Vec<Vec<TemporalEvent>> = Vec::new();

    for sentence in &training_sentences {
        let events = sentence_to_events(sentence);
        let remapped = remap_events(&events, &grammar);

        if remapped.len() >= 3 {
            println!("    \"{}\" → {} phonemes", sentence, remapped.len());
            training_events.push(remapped);
        }
    }

    // Train grammar
    for events in &training_events {
        for _ in 0..30 {
            grammar.train_sequence(events);
        }
    }

    let stats = grammar.stats();
    println!(
        "\n    Grammar density after training: {:.3}",
        stats.grammar_density
    );

    // Test sentences
    subheader("Phase 2: Scoring Test Sentences");

    let test_sentences = vec![
        // Similar to training (should score high)
        ("the dog sat on the mat", "similar"),
        ("how is your day going", "similar"),
        ("thank you so much", "similar"),
        // Different patterns (should score lower)
        ("xylophone rhythm", "unusual"),
        ("sphinx of quartz", "unusual"),
        ("jazz and blues music", "unusual"),
        // Nonsense (should score lowest)
        ("blorft snizzle prak", "nonsense"),
    ];

    let mut similar_scores = Vec::new();
    let mut unusual_scores = Vec::new();

    for (sentence, category) in &test_sentences {
        let events = sentence_to_events(sentence);
        let remapped = remap_events(&events, &grammar);

        if remapped.len() >= 2 {
            let score = grammar.score_sequence(&remapped);

            match *category {
                "similar" => similar_scores.push(score),
                "unusual" => unusual_scores.push(score),
                _ => {}
            }

            println!(
                "    [{}] \"{}\": {:+.4} ({} phonemes)",
                category.to_uppercase(),
                sentence,
                score,
                remapped.len()
            );
        } else {
            println!(
                "    [{}] \"{}\" - not in dictionary",
                category.to_uppercase(),
                sentence
            );
        }
    }

    // Phonotactic analysis (rule-based)
    subheader("Phase 3: Phonotactic Pattern Analysis");

    // Valid English onsets (simplified)
    let valid_onsets = vec![
        vec!["S", "T", "R"], // "str" - strong, string
        vec!["S", "P", "L"], // "spl" - split
        vec!["S", "T"],      // "st" - stop, store
        vec!["S", "P"],      // "sp" - speak
        vec!["S", "K"],      // "sk" - skip
        vec!["P", "R"],      // "pr" - pretty
        vec!["B", "R"],      // "br" - brown
        vec!["T", "R"],      // "tr" - tree
        vec!["D", "R"],      // "dr" - drive
        vec!["K", "R"],      // "cr" - cry
        vec!["G", "R"],      // "gr" - green
    ];

    // Invalid English onsets
    let invalid_onsets = vec![
        vec!["T", "L"], // No "tl-" words
        vec!["S", "R"], // No "sr-" words
        vec!["N", "G"], // "ng" only as coda
        vec!["Z", "T"], // No "zt-" words
    ];

    println!("    Valid English syllable onsets:");
    for onset in &valid_onsets {
        println!("      ✓ {:?}", onset);
    }

    println!("\n    Invalid English syllable onsets:");
    for onset in &invalid_onsets {
        println!("      ✗ {:?}", onset);
    }

    println!("\n    The grammar implicitly learns these constraints from training data.");

    // Summary
    header("RESULTS");

    let similar_avg = similar_scores.iter().sum::<f32>() / similar_scores.len().max(1) as f32;
    let unusual_avg = unusual_scores.iter().sum::<f32>() / unusual_scores.len().max(1) as f32;
    let discrimination = similar_avg - unusual_avg;

    println!("    Similar sentences avg:  {:+.4}", similar_avg);
    println!("    Unusual sentences avg:  {:+.4}", unusual_avg);
    println!("    Discrimination:         {:+.4}", discrimination);
    println!();

    if discrimination > 0.0 {
        println!("  [SUCCESS] Grammar discriminates common from unusual patterns!");
    } else {
        println!("  [INFO] Need more training data for better discrimination.");
    }

    // Show bigram patterns learned
    subheader("Phase 4: Common Phoneme Bigrams");

    let mut bigram_counts: HashMap<(String, String), usize> = HashMap::new();

    for events in &training_events {
        for window in events.windows(2) {
            let key = (window[0].category.clone(), window[1].category.clone());
            *bigram_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut bigrams: Vec<_> = bigram_counts.iter().collect();
    bigrams.sort_by(|a, b| b.1.cmp(a.1));

    println!("    Top 15 phoneme bigrams in training:\n");
    for ((p1, p2), count) in bigrams.iter().take(15) {
        let bar_len = **count;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("      {}-{}: {} {}", p1, p2, count, bar);
    }

    separator('=', 70);
    println!("  SPEECH GRAMMAR COMPLETE");
    separator('=', 70);
}
