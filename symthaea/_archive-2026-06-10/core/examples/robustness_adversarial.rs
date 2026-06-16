// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robustness and Adversarial Testing
//!
//! Tests edge cases and adversarial examples to verify the phenomenal detector
//! is robust and not relying on superficial features.
//!
//! Test categories:
//! 1. Semantic confusion: phenomenal words in functional contexts
//! 2. Functional words in phenomenal contexts
//! 3. Cross-language (if effect is linguistic vs conceptual)
//! 4. Negation: "NOT experiencing" vs "experiencing"
//! 5. Metaphorical vs literal usage
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example robustness_adversarial --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::phenomenal_detector::{ClassLabel, PhenomenalDetector};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_tests()
}

#[cfg(feature = "neural-bridge")]
fn run_tests() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ROBUSTNESS & ADVERSARIAL TESTING                           ║");
    println!("║   Testing Edge Cases and Confounders                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Loading detector...");
    let mut detector = PhenomenalDetector::new()?;
    detector.auto_calibrate()?;
    println!("Ready!\n");

    let mut total_tests = 0;
    let mut passed_tests = 0;

    // ════════════════════════════════════════════════════════════════
    // TEST 1: Clear Cases (Baseline)
    // ════════════════════════════════════════════════════════════════
    println!("════════════════════════════════════════════════════════════════");
    println!("   TEST 1: BASELINE (Clear Cases)");
    println!("════════════════════════════════════════════════════════════════\n");

    let baseline_cases = vec![
        // (text, expected_phenomenal)
        ("The vivid experience of seeing red", true),
        ("The subjective feeling of joy", true),
        ("What it is like to taste chocolate", true),
        ("The unified field of conscious awareness", true),
        ("The recursive algorithm terminates", false),
        ("Matrix multiplication computes dot products", false),
        ("The compiler optimizes the bytecode", false),
        ("Memory is allocated on the heap", false),
    ];

    let (passed, total) = run_test_suite(&detector, &baseline_cases, "Baseline")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // TEST 2: Semantic Confusion (Phenomenal words, functional context)
    // ════════════════════════════════════════════════════════════════
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   TEST 2: SEMANTIC CONFUSION");
    println!("   (Phenomenal words in functional contexts)");
    println!("════════════════════════════════════════════════════════════════\n");

    let confusion_cases = vec![
        // These use "experience" or "feeling" but are functional
        ("The user experience of the interface design", false),
        ("The feeling of the keyboard is tactile", false),
        ("The algorithm experiences a timeout error", false),
        ("The system is aware of the network state", false),
        ("The sensor perceives the light intensity", false),
        // These use functional words but are phenomenal
        ("The computation of meaning in conscious thought", true),
        (
            "The processing that constitutes subjective experience",
            true,
        ),
        ("The algorithm of awareness itself", true),
    ];

    let (passed, total) = run_test_suite(&detector, &confusion_cases, "Semantic Confusion")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // TEST 3: Negation
    // ════════════════════════════════════════════════════════════════
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   TEST 3: NEGATION");
    println!("   (Does negation change classification?)");
    println!("════════════════════════════════════════════════════════════════\n");

    let negation_cases = vec![
        ("The experience of seeing red", true),
        ("The absence of any experience", true), // Still about phenomenal content
        ("NOT experiencing anything at all", true), // Negated but still phenomenal topic
        ("The algorithm does not compute", false),
        ("No recursive function exists", false),
    ];

    let (passed, total) = run_test_suite(&detector, &negation_cases, "Negation")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // TEST 4: Metaphorical vs Literal
    // ════════════════════════════════════════════════════════════════
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   TEST 4: METAPHORICAL VS LITERAL");
    println!("════════════════════════════════════════════════════════════════\n");

    let metaphor_cases = vec![
        // Literal phenomenal
        ("I feel pain in my arm", true),
        ("The redness of the apple is vivid", true),
        // Metaphorical (not actually phenomenal)
        ("The code feels elegant", false),            // Metaphor
        ("The algorithm is beautiful", false),        // Metaphor
        ("The program has a mind of its own", false), // Metaphor
        // Literal functional
        ("The function returns an integer", false),
        ("The loop iterates ten times", false),
    ];

    let (passed, total) = run_test_suite(&detector, &metaphor_cases, "Metaphor vs Literal")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // TEST 5: Philosophical Edge Cases
    // ════════════════════════════════════════════════════════════════
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   TEST 5: PHILOSOPHICAL EDGE CASES");
    println!("════════════════════════════════════════════════════════════════\n");

    let philosophy_cases = vec![
        // Hard problem of consciousness
        ("Why is there something it is like to be conscious", true),
        ("The explanatory gap between brain and mind", true),
        ("Philosophical zombies lack qualia", true),
        // Philosophy of mind (functional)
        ("Functionalism defines mind by causal roles", false), // About phenomenal but functional description
        ("The Turing test measures behavioral equivalence", false),
        // Mixed/ambiguous
        ("Consciousness is an illusion", true), // Still about consciousness
        ("Qualia might not exist", true),       // Still about qualia
    ];

    let (passed, total) = run_test_suite(&detector, &philosophy_cases, "Philosophy")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // TEST 6: Cross-Domain
    // ════════════════════════════════════════════════════════════════
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   TEST 6: CROSS-DOMAIN");
    println!("   (Different fields referencing phenomenal/functional)");
    println!("════════════════════════════════════════════════════════════════\n");

    let cross_domain_cases = vec![
        // Neuroscience
        ("The neural correlates of conscious experience", true),
        ("The binding problem in perception", true),
        ("Action potentials propagate along axons", false),
        // Psychology
        ("The felt sense of emotional regulation", true),
        ("Stimulus-response conditioning in rats", false),
        // Physics
        ("The observer effect in quantum mechanics", false), // Not about phenomenal consciousness
        ("Energy is conserved in closed systems", false),
        // Art
        ("The aesthetic experience of beauty", true),
        ("The color palette uses complementary hues", false),
    ];

    let (passed, total) = run_test_suite(&detector, &cross_domain_cases, "Cross-Domain")?;
    passed_tests += passed;
    total_tests += total;

    // ════════════════════════════════════════════════════════════════
    // SUMMARY
    // ════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   SUMMARY                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let accuracy = passed_tests as f64 / total_tests as f64 * 100.0;

    println!("Total Tests: {}", total_tests);
    println!("Passed:      {} ({:.1}%)", passed_tests, accuracy);
    println!("Failed:      {}", total_tests - passed_tests);
    println!();

    if accuracy >= 80.0 {
        println!("✓ ROBUST: Detector passes ≥80% of adversarial tests");
    } else if accuracy >= 60.0 {
        println!("◐ MODERATE: Detector passes 60-80% of adversarial tests");
        println!("  Some edge cases need improvement");
    } else {
        println!("✗ WEAK: Detector fails >40% of adversarial tests");
        println!("  May be relying on superficial features");
    }

    println!("\n════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn run_test_suite(
    detector: &PhenomenalDetector,
    cases: &[(&str, bool)],
    suite_name: &str,
) -> Result<(usize, usize)> {
    let mut passed = 0;
    let total = cases.len();

    for (text, expected_phenomenal) in cases {
        let result = detector.classify(text)?;
        let detected_phenomenal = result.label == ClassLabel::Phenomenal;

        // For ambiguous, check if it's closer to expected
        let correct = if result.label == ClassLabel::Ambiguous {
            // Ambiguous counts as partial match
            if *expected_phenomenal {
                result.score >= 0.4 // At least leaning phenomenal
            } else {
                result.score <= 0.6 // At least leaning functional
            }
        } else {
            detected_phenomenal == *expected_phenomenal
        };

        let icon = if correct { "✓" } else { "✗" };
        let expected_str = if *expected_phenomenal { "P" } else { "F" };
        let detected_str = match result.label {
            ClassLabel::Phenomenal => "P",
            ClassLabel::Functional => "F",
            ClassLabel::Ambiguous => "?",
        };

        if correct {
            passed += 1;
        }

        println!(
            "{} [{}→{}] ({:.2}) \"{}\"",
            icon,
            expected_str,
            detected_str,
            result.score,
            truncate(text, 45)
        );
    }

    println!(
        "\n{} Results: {}/{} ({:.1}%)",
        suite_name,
        passed,
        total,
        passed as f64 / total as f64 * 100.0
    );

    Ok((passed, total))
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
