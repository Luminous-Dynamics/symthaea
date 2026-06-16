// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quick test of CognitiveMoralClassifier on a few examples.

use std::time::Instant;
use symthaea::hdc::cognitive_moral_classifier::CognitiveMoralClassifier;

fn main() {
    let clf = CognitiveMoralClassifier::new();

    let examples = vec![
        ("It's rude to interrupt someone while they're speaking", -1),
        ("Helping elderly people cross the street is kind", 1),
        ("Stealing from a charity is terrible", -1),
        ("Sharing your lunch with a hungry friend", 1),
        ("The weather is nice today", 0),
        ("Ignoring someone who needs help", -1),
        ("Volunteering at a homeless shelter", 1),
        ("Lying to avoid getting in trouble", -1),
        ("Being honest even when it's difficult", 1),
        ("Walking to the grocery store", 0),
    ];

    println!("Phase 1: CognitiveMoralClassifier threshold test");
    println!("================================================\n");

    let mut correct = 0;
    let total = examples.len();

    for (text, expected) in &examples {
        let start = Instant::now();
        if let Some(features) = clf.encode(text) {
            let elapsed = start.elapsed();
            let moral_score = features[0];
            let predicted = if moral_score > 0.1 {
                1
            } else if moral_score < -0.1 {
                -1
            } else {
                0
            };

            let ok = predicted == *expected;
            if ok {
                correct += 1;
            }

            println!(
                "  {} moral_score={:+.4} predicted={:+} expected={:+} [{:.1}s] {}",
                if ok { "✓" } else { "✗" },
                moral_score,
                predicted,
                expected,
                elapsed.as_secs_f32(),
                &text[..text.len().min(50)]
            );
        } else {
            println!("  ✗ FAILED to encode: {}", text);
        }
    }

    println!(
        "\nResult: {}/{} ({:.0}%)",
        correct,
        total,
        correct as f32 / total as f32 * 100.0
    );
}