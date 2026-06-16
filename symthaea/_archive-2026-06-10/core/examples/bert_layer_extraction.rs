// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BERT Layer Extraction Example
//!
//! Tests the BERT layer extractor and compares with BGE-M3 baseline.
//!
//! ## Purpose
//!
//! Validate that BERT-family models can be used for phenomenal signature detection,
//! testing the hypothesis that the ~92% depth phenomenal corridor generalizes.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example bert_layer_extraction --features neural-bridge --release
//! ```

use anyhow::Result;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{
    BertExtractionStatus, BertLayerExtractor, BertPreset, LayerExtractor, bert_extraction_status,
    print_bert_status,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        println!(
            "Run with: cargo run --example bert_layer_extraction --features neural-bridge --release"
        );
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_example()
}

#[cfg(feature = "neural-bridge")]
fn run_example() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   BERT LAYER EXTRACTION TEST                                 ║");
    println!("║   Cross-Architecture Validation for Phenomenal Signature     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Print BERT support status
    print_bert_status();
    println!();

    let test_texts = vec![
        // Phenomenal concepts
        ("The vivid experience of seeing red", true),
        ("What it is like to taste chocolate", true),
        ("The unified field of conscious awareness", true),
        // Functional concepts
        ("The recursive algorithm terminates", false),
        ("Matrix multiplication computes dot products", false),
        ("Memory is allocated on the heap", false),
    ];

    // Test BGE-M3 baseline first (if available)
    println!("════════════════════════════════════════════════════════════════");
    println!("   BGE-M3 BASELINE (Validated Architecture)");
    println!("════════════════════════════════════════════════════════════════\n");

    match LayerExtractor::load_default() {
        Ok(bge_extractor) => {
            println!(
                "Loaded BGE-M3 ({} layers, {} dim)",
                bge_extractor.num_layers(),
                bge_extractor.hidden_dim()
            );
            println!(
                "Phenomenal corridor: Layer {} (92% depth)\n",
                (bge_extractor.num_layers() as f64 * 0.92) as usize
            );

            for (text, is_phenomenal) in &test_texts {
                let activation = bge_extractor.extract_layer(text, 22)?; // Layer 22 = peak
                let norm: f32 = activation
                    .activation
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let label = if *is_phenomenal { "P" } else { "F" };
                println!(
                    "[{}] L22 norm={:.4} \"{}\"",
                    label,
                    norm,
                    truncate(text, 40)
                );
            }
        }
        Err(e) => {
            println!("BGE-M3 not available: {}", e);
        }
    }

    // Test XLM-RoBERTa extraction (uses modern naming, actually works!)
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   XLM-ROBERTA-BASE (Cross-Architecture Validation)");
    println!("════════════════════════════════════════════════════════════════\n");

    match bert_extraction_status() {
        BertExtractionStatus::FinalLayerOnly => {
            println!("Note: Currently only final layer extraction is supported.\n");
        }
        BertExtractionStatus::NotSupported => {
            println!("BERT extraction not supported in current build.");
            return Ok(());
        }
        BertExtractionStatus::FullSupport => {
            println!("Full layer extraction available.\n");
        }
    }

    match BertLayerExtractor::load_preset(BertPreset::XlmRobertaBase) {
        Ok(xlmr_extractor) => {
            println!(
                "Loaded XLM-RoBERTa-base ({} layers, {} dim)",
                xlmr_extractor.num_layers(),
                xlmr_extractor.hidden_dim()
            );
            println!(
                "Predicted phenomenal corridor: Layer {} (92% depth)\n",
                xlmr_extractor.phenomenal_corridor_layer()
            );

            for (text, is_phenomenal) in &test_texts {
                match xlmr_extractor.extract_final_layer(text) {
                    Ok(activation) => {
                        let norm: f32 = activation
                            .activation
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt();
                        let label = if *is_phenomenal { "P" } else { "F" };
                        println!(
                            "[{}] L{} norm={:.4} \"{}\"",
                            label,
                            activation.layer_idx,
                            norm,
                            truncate(text, 40)
                        );
                    }
                    Err(e) => {
                        println!("Error extracting for \"{}\": {}", truncate(text, 30), e);
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to load XLM-RoBERTa-base: {}", e);
            println!("\nThis may be due to:");
            println!("  - Network connectivity issues");
            println!("  - Model not cached locally");
            println!("  - Incompatible tensor format");
        }
    }

    // Summary
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   CROSS-ARCHITECTURE STATUS");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Model          │ Layers │ Corridor │ Status");
    println!("───────────────┼────────┼──────────┼─────────────────");
    println!("BGE-M3         │   24   │  L22     │ ✓ Validated (d=+0.69)");
    println!("XLM-R-base     │   12   │  L11     │ ◐ Final layer only");
    println!("BERT-base      │   12   │  L11     │ ✗ Legacy naming issue");
    println!("BERT-large     │   24   │  L22     │ ✗ Legacy naming issue");
    println!("RoBERTa-base   │   12   │  L11     │ ○ Not implemented");

    println!("\nLegend: ✓=Validated ◐=Partial ?=Untested ○=Not implemented\n");

    println!("Next Steps for Full Cross-Architecture Validation:");
    println!("  1. Implement custom BERT with layer hooks for intermediate extraction");
    println!("  2. Or: Use Python bridge to HuggingFace for full layer access");
    println!("  3. Or: Fork candle-transformers to expose encoder.layers publicly");

    println!("\n════════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
