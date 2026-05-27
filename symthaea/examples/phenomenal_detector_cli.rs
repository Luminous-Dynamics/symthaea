// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phenomenal Content Detector CLI
//!
//! A command-line tool for detecting phenomenal content in text.
//!
//! ## Usage
//!
//! ```bash
//! # Score a single text
//! cargo run --example phenomenal_detector_cli --features neural-bridge --release -- \
//!     "The vivid redness of the sunset filled my awareness"
//!
//! # Compare two texts
//! cargo run --example phenomenal_detector_cli --features neural-bridge --release -- \
//!     --compare "The experience of seeing red" "The algorithm processes input"
//!
//! # Analyze a file
//! cargo run --example phenomenal_detector_cli --features neural-bridge --release -- \
//!     --file input.txt
//!
//! # Interactive mode
//! cargo run --example phenomenal_detector_cli --features neural-bridge --release -- \
//!     --interactive
//! ```

use anyhow::Result;
use std::io::{self, BufRead, Write};

#[cfg(feature = "neural-bridge")]
use symthaea::perception::phenomenal_detector::{
    ClassLabel, DetectionMethod, DetectorConfig, PhenomenalDetector,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This tool requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example phenomenal_detector_cli --features neural-bridge --release"
        );
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_cli()
}

#[cfg(feature = "neural-bridge")]
fn run_cli() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    // Parse arguments
    let mut interactive = false;
    let mut compare_mode = false;
    let mut file_mode = false;
    let mut verbose = false;
    let mut texts: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--interactive" | "-i" => interactive = true,
            "--compare" | "-c" => compare_mode = true,
            "--file" | "-f" => file_mode = true,
            "--verbose" | "-v" => verbose = true,
            arg if !arg.starts_with('-') => texts.push(arg.to_string()),
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                return Ok(());
            }
        }
        i += 1;
    }

    // Load detector
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   PHENOMENAL CONTENT DETECTOR                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Loading model...");
    let mut detector = PhenomenalDetector::new()?;
    println!("Calibrating...");
    let cal = detector.auto_calibrate()?;
    println!(
        "  Φ baselines: phen={:.2}, func={:.2}",
        cal.phen_phi_mean, cal.func_phi_mean
    );
    println!("Ready!\n");

    if interactive {
        run_interactive(&detector)?;
    } else if compare_mode && texts.len() >= 2 {
        run_compare(&detector, &texts[0], &texts[1], verbose)?;
    } else if file_mode && !texts.is_empty() {
        run_file(&detector, &texts[0], verbose)?;
    } else if !texts.is_empty() {
        for text in &texts {
            run_single(&detector, text, verbose)?;
        }
    } else {
        print_usage();
    }

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn run_single(detector: &PhenomenalDetector, text: &str, verbose: bool) -> Result<()> {
    if verbose {
        let analysis = detector.analyze(text)?;
        println!("Text: \"{}\"", truncate(text, 60));
        println!("────────────────────────────────────────────────────────────────");
        println!("  Classification: {}", analysis.classification.label);
        println!("  Combined Score: {:.3}", analysis.combined_score);
        println!("  Unity Score:    {:.3}", analysis.unity_score);
        println!("  Φ Score:        {:.3}", analysis.phi_score);
        println!(
            "  Confidence:     {:.1}%",
            analysis.classification.confidence * 100.0
        );
        println!("  Raw Unity:      {:.4}", analysis.unity);
        println!("  Raw Φ Loading:  {:.4}", analysis.phi_loading);
        println!();
    } else {
        let classification = detector.classify(text)?;
        let icon = match classification.label {
            ClassLabel::Phenomenal => "●",
            ClassLabel::Functional => "○",
            ClassLabel::Ambiguous => "◐",
        };
        println!(
            "{} [{:.2}] {} - \"{}\"",
            icon,
            classification.score,
            classification.label,
            truncate(text, 50)
        );
    }
    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn run_compare(
    detector: &PhenomenalDetector,
    text_a: &str,
    text_b: &str,
    verbose: bool,
) -> Result<()> {
    let result = detector.compare(text_a, text_b)?;

    println!("COMPARISON");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("A: \"{}\"", truncate(text_a, 50));
    println!("   Score: {:.3}", result.score_a);
    println!();
    println!("B: \"{}\"", truncate(text_b, 50));
    println!("   Score: {:.3}", result.score_b);
    println!();
    println!("────────────────────────────────────────────────────────────────");

    match result.more_phenomenal {
        Some(true) => println!("Result: A is MORE PHENOMENAL (+{:.3})", result.difference),
        Some(false) => println!("Result: B is MORE PHENOMENAL (+{:.3})", -result.difference),
        None => println!(
            "Result: APPROXIMATELY EQUAL (diff={:.3})",
            result.difference.abs()
        ),
    }

    if verbose {
        println!();
        let analysis_a = detector.analyze(text_a)?;
        let analysis_b = detector.analyze(text_b)?;
        println!("Detailed Analysis:");
        println!(
            "  A - Unity: {:.3}, Φ: {:.3}",
            analysis_a.unity_score, analysis_a.phi_score
        );
        println!(
            "  B - Unity: {:.3}, Φ: {:.3}",
            analysis_b.unity_score, analysis_b.phi_score
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn run_file(detector: &PhenomenalDetector, path: &str, verbose: bool) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let analysis = detector.analyze_document(&content)?;

    println!("FILE ANALYSIS: {}", path);
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Overall Score: {:.3}", analysis.overall_score);
    println!(
        "Phenomenal Ratio: {:.1}%",
        analysis.phenomenal_ratio * 100.0
    );
    println!();

    println!("Top Phenomenal Passages:");
    for (i, (passage, score)) in analysis.top_phenomenal_passages.iter().enumerate() {
        println!("  {}. [{:.2}] \"{}\"", i + 1, score, truncate(passage, 50));
    }

    if verbose {
        println!();
        println!("All Passages:");
        for (passage, score) in &analysis.passage_scores {
            let icon = if *score > 0.65 {
                "●"
            } else if *score < 0.35 {
                "○"
            } else {
                "◐"
            };
            println!("  {} [{:.2}] \"{}\"", icon, score, truncate(passage, 40));
        }
    }

    println!();
    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn run_interactive(detector: &PhenomenalDetector) -> Result<()> {
    println!("INTERACTIVE MODE");
    println!("════════════════════════════════════════════════════════════════");
    println!("Enter text to analyze. Commands:");
    println!("  :compare <text1> | <text2>  - Compare two texts");
    println!("  :verbose                    - Toggle verbose mode");
    println!("  :quit or :q                 - Exit");
    println!("════════════════════════════════════════════════════════════════\n");

    let stdin = io::stdin();
    let mut verbose = false;

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line == ":quit" || line == ":q" {
            println!("Goodbye!");
            break;
        }

        if line == ":verbose" {
            verbose = !verbose;
            println!("Verbose mode: {}", if verbose { "ON" } else { "OFF" });
            continue;
        }

        if line.starts_with(":compare ") {
            let rest = &line[9..];
            if let Some(idx) = rest.find(" | ") {
                let text_a = &rest[..idx];
                let text_b = &rest[idx + 3..];
                run_compare(detector, text_a, text_b, verbose)?;
            } else {
                println!("Usage: :compare <text1> | <text2>");
            }
            continue;
        }

        if line.starts_with(':') {
            println!("Unknown command. Type :quit to exit.");
            continue;
        }

        run_single(detector, line, verbose)?;
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

fn print_usage() {
    println!("Phenomenal Content Detector CLI");
    println!();
    println!("USAGE:");
    println!("  phenomenal_detector_cli [OPTIONS] [TEXT...]");
    println!();
    println!("OPTIONS:");
    println!("  -h, --help         Show this help message");
    println!("  -i, --interactive  Interactive mode");
    println!("  -c, --compare      Compare two texts");
    println!("  -f, --file         Analyze a file");
    println!("  -v, --verbose      Show detailed analysis");
    println!();
    println!("EXAMPLES:");
    println!("  # Score a text");
    println!("  phenomenal_detector_cli \"The vivid experience of seeing red\"");
    println!();
    println!("  # Compare two texts");
    println!(
        "  phenomenal_detector_cli --compare \"Subjective awareness\" \"Algorithm execution\""
    );
    println!();
    println!("  # Analyze a file");
    println!("  phenomenal_detector_cli --file essay.txt --verbose");
    println!();
    println!("  # Interactive mode");
    println!("  phenomenal_detector_cli --interactive");
}
