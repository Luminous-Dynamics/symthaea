// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grammar CLI Tool
//!
//! Command-line interface for training, saving, loading, and using
//! the UnifiedGrammar for speech recognition.
//!
//! Usage:
//!   grammar_cli train --input <phoneme_file> --output <model_file>
//!   grammar_cli score --model <model_file> --input <phoneme_file>
//!   grammar_cli info --model <model_file>

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use symthaea_stt::temporal_grammar::TemporalEvent;
use symthaea_stt::unified_grammar::UnifiedGrammar;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "score" => cmd_score(&args[2..]),
        "info" => cmd_info(&args[2..]),
        "demo" => cmd_demo(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("Symthaea Grammar CLI - Unified Temporal Grammar for Speech");
    println!();
    println!("COMMANDS:");
    println!("  train   Train grammar on phoneme sequences");
    println!("  score   Score phoneme sequences with trained model");
    println!("  info    Show model statistics");
    println!("  demo    Run interactive demo");
    println!();
    println!("EXAMPLES:");
    println!("  grammar_cli train --input train.txt --output model.json --epochs 30");
    println!("  grammar_cli score --model model.json --input test.txt");
    println!("  grammar_cli info --model model.json");
    println!("  grammar_cli demo");
    println!();
    println!("FILE FORMAT (phoneme sequences):");
    println!("  Each line contains space-separated phonemes:");
    println!("    DH AH K AE T S AE T");
    println!("    DH AH D AO G R AE N");
}

fn cmd_train(args: &[String]) {
    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut epochs = 30;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                if i < args.len() {
                    input_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--epochs" | "-e" => {
                i += 1;
                if i < args.len() {
                    epochs = args[i].parse().unwrap_or(30);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let input = match input_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --input required");
            return;
        }
    };

    let output = output_path.unwrap_or_else(|| PathBuf::from("grammar_model.json"));

    println!("Training UnifiedGrammar");
    println!("  Input:  {}", input.display());
    println!("  Output: {}", output.display());
    println!("  Epochs: {}", epochs);
    println!();

    // Create grammar
    let mut grammar = UnifiedGrammar::speech();

    // Load training sequences
    let sequences = load_phoneme_sequences(&input);
    if sequences.is_empty() {
        eprintln!("Error: No training sequences found in {}", input.display());
        return;
    }

    println!("Loaded {} training sequences", sequences.len());

    // Train
    for epoch in 0..epochs {
        for seq in &sequences {
            let events = phonemes_to_events(seq, &grammar);
            if events.len() >= 2 {
                grammar.train_sequence(&events);
            }
        }

        if (epoch + 1) % 10 == 0 {
            let stats = grammar.stats();
            println!(
                "  Epoch {}: {} clusters, {:.1}% density",
                epoch + 1,
                stats.num_clusters,
                stats.avg_cluster_density * 100.0
            );
        }
    }

    // Save model
    match grammar.save(&output) {
        Ok(_) => println!("\nModel saved to {}", output.display()),
        Err(e) => eprintln!("\nError saving model: {}", e),
    }

    // Final stats
    let stats = grammar.stats();
    println!("\nFinal Statistics:");
    println!("  Categories: {}", stats.num_categories);
    println!("  Clusters:   {}", stats.num_clusters);
    println!("  Density:    {:.2}%", stats.avg_cluster_density * 100.0);
}

fn cmd_score(args: &[String]) {
    let mut model_path: Option<PathBuf> = None;
    let mut input_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--input" | "-i" => {
                i += 1;
                if i < args.len() {
                    input_path = Some(PathBuf::from(&args[i]));
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --model required");
            return;
        }
    };

    let input = match input_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --input required");
            return;
        }
    };

    // Load model
    let mut grammar = match UnifiedGrammar::load(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            return;
        }
    };

    println!("Loaded model from {}", model.display());
    println!();

    // Score sequences
    let sequences = load_phoneme_sequences(&input);
    if sequences.is_empty() {
        eprintln!("Error: No sequences found in {}", input.display());
        return;
    }

    println!("Scoring {} sequences:", sequences.len());
    println!();

    let mut total_score = 0.0;
    for seq in &sequences {
        let events = phonemes_to_events(seq, &grammar);
        if events.len() >= 2 {
            let score = grammar.score_sequence(&events);
            total_score += score;
            let phoneme_str: Vec<&str> = seq.iter().map(|s| s.as_str()).collect();
            println!("  {:+.4}  {}", score, phoneme_str.join(" "));
        }
    }

    println!();
    println!(
        "Average score: {:+.4}",
        total_score / sequences.len() as f32
    );
}

fn cmd_info(args: &[String]) {
    let mut model_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --model required");
            return;
        }
    };

    // Load model
    let grammar = match UnifiedGrammar::load(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            return;
        }
    };

    let stats = grammar.stats();

    println!("Model Information: {}", model.display());
    println!("==========================================");
    println!("  Domain:           {}", stats.domain);
    println!("  Categories:       {}", stats.num_categories);
    println!("  Clusters:         {}", stats.num_clusters);
    println!("  Active Clusters:  {}", stats.active_clusters);
    println!("  Global Density:   {:.2}%", stats.global_density * 100.0);
    println!(
        "  Cluster Density:  {:.2}%",
        stats.avg_cluster_density * 100.0
    );
    println!("  Training Count:   {}", stats.training_count);
    println!("  Sparsity:         {:.0}%", stats.sparsity * 100.0);
}

fn cmd_demo(_args: &[String]) {
    println!("Unified Grammar Interactive Demo");
    println!("================================");
    println!();
    println!("Creating speech grammar and training on example sentences...");
    println!();

    let mut grammar = UnifiedGrammar::speech();

    // Built-in training examples
    let training_sentences = vec![
        vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"], // the cat sat
        vec!["DH", "AH", "D", "AO", "G", "R", "AE", "N"], // the dog ran
        vec!["DH", "AH", "B", "AE", "T", "HH", "AE", "T"], // the bat hat
        vec!["B", "IH", "G", "P", "IH", "G", "D", "IH", "G"], // big pig dig
        vec!["F", "IH", "SH", "D", "IH", "SH", "W", "IH", "SH"], // fish dish wish
    ];

    // Train
    for sentence in &training_sentences {
        let events = phoneme_list_to_events(sentence, &grammar);
        for _ in 0..30 {
            grammar.train_sequence(&events);
        }
        println!("  Trained: {}", sentence.join(" "));
    }

    let stats = grammar.stats();
    println!();
    println!(
        "Training complete: {} clusters, {:.1}% density",
        stats.num_clusters,
        stats.avg_cluster_density * 100.0
    );
    println!();

    // Test examples
    let test_cases = vec![
        (
            vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"],
            "trained (the cat sat)",
        ),
        (
            vec!["DH", "AH", "P", "AE", "T", "M", "AE", "T"],
            "similar (the pat mat)",
        ),
        (vec!["S", "T", "R", "IY", "T"], "consonant cluster (street)"),
        (vec!["T", "L", "IY", "T"], "invalid onset (tleet)"),
    ];

    println!("Testing phoneme sequences:");
    println!();

    for (phonemes, description) in test_cases {
        let events = phoneme_list_to_events(&phonemes, &grammar);
        let score = grammar.score_sequence(&events);
        println!("  {:+.4}  {}  [{}]", score, phonemes.join(" "), description);
    }

    println!();
    println!("Positive scores = likely English phonotactics");
    println!("Negative scores = unlikely sequences");
}

// Helper functions

fn load_phoneme_sequences(path: &PathBuf) -> Vec<Vec<String>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = BufReader::new(file);
    let mut sequences = Vec::new();

    for line in reader.lines().flatten() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let phonemes: Vec<String> = line.split_whitespace().map(|s| s.to_uppercase()).collect();

        if phonemes.len() >= 2 {
            sequences.push(phonemes);
        }
    }

    sequences
}

fn phonemes_to_events(phonemes: &[String], grammar: &UnifiedGrammar) -> Vec<TemporalEvent> {
    let mut events = Vec::new();
    let mut time = 0.0f32;

    for phoneme in phonemes {
        if let Some(cid) = grammar.category_id(phoneme) {
            // Estimate duration based on phoneme type
            let duration = estimate_phoneme_duration(phoneme);
            let intensity = estimate_phoneme_intensity(phoneme);

            events.push(TemporalEvent::new(phoneme, cid, time, duration, intensity));
            time += duration;
        }
    }

    events
}

fn phoneme_list_to_events(phonemes: &[&str], grammar: &UnifiedGrammar) -> Vec<TemporalEvent> {
    let owned: Vec<String> = phonemes.iter().map(|s| s.to_string()).collect();
    phonemes_to_events(&owned, grammar)
}

fn estimate_phoneme_duration(phoneme: &str) -> f32 {
    // Rough estimates based on phoneme class
    match phoneme {
        // Vowels - longer
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.10
        }
        // Fricatives - medium
        "S" | "Z" | "SH" | "ZH" | "F" | "V" | "TH" | "DH" => 0.08,
        // Stops - short
        "P" | "B" | "T" | "D" | "K" | "G" => 0.05,
        // Nasals - medium
        "M" | "N" | "NG" => 0.06,
        // Others
        _ => 0.06,
    }
}

fn estimate_phoneme_intensity(phoneme: &str) -> f32 {
    // Rough intensity estimates
    match phoneme {
        // Vowels - high energy
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.8
        }
        // Voiced consonants - medium
        "B" | "D" | "G" | "V" | "Z" | "DH" | "ZH" | "M" | "N" | "NG" => 0.6,
        // Voiceless consonants - lower
        "P" | "T" | "K" | "F" | "S" | "TH" | "SH" | "HH" => 0.5,
        _ => 0.55,
    }
}
