// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for luminous-nix

use luminous_nix::{IntentClassifier, Intent};

#[test]
fn test_full_classification_pipeline() {
    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    // Test search
    let result = classifier.classify("search firefox");
    assert!(matches!(result.intent, Intent::Search { .. }));
    assert!(result.confidence > 0.9);

    // Test install
    let result = classifier.classify("install vim");
    assert!(matches!(result.intent, Intent::Install { .. }));
    assert!(result.confidence > 0.9);

    // Test remove
    let result = classifier.classify("remove htop");
    assert!(matches!(result.intent, Intent::Remove { .. }));
    assert!(result.confidence > 0.9);

    // Test generations
    let result = classifier.classify("list generations");
    assert!(matches!(result.intent, Intent::ListGenerations));
    assert!(result.confidence > 0.9);

    // Test gc
    let result = classifier.classify("garbage collect");
    assert!(matches!(result.intent, Intent::GarbageCollect));
    assert!(result.confidence > 0.9);
}

#[test]
fn test_pattern_variations() {
    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    // Different search variations
    for query in &["search vim", "find vim", "look for vim", "locate vim"] {
        let result = classifier.classify(query);
        assert!(
            matches!(result.intent, Intent::Search { .. }),
            "Failed for: {}",
            query
        );
    }

    // Different install variations
    for query in &["install git", "add git", "get git"] {
        let result = classifier.classify(query);
        assert!(
            matches!(result.intent, Intent::Install { .. }),
            "Failed for: {}",
            query
        );
    }
}

#[test]
fn test_argument_extraction() {
    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    let result = classifier.classify("search terminal emulator");
    if let Intent::Search { query } = result.intent {
        assert!(query.contains("terminal") || query.contains("emulator"));
    } else {
        panic!("Expected Search intent");
    }

    let result = classifier.classify("install neovim");
    if let Intent::Install { package } = result.intent {
        assert!(package.contains("neovim"));
    } else {
        panic!("Expected Install intent");
    }
}

#[test]
fn test_online_learning() {
    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    // Initial classification
    let _result = classifier.classify("search firefox");

    // Provide positive feedback
    classifier.learn_from_outcome(
        "search firefox",
        &Intent::Search { query: "firefox".to_string() },
        1.0,
    );

    // Classification should still work
    let result = classifier.classify("search firefox");
    assert!(matches!(result.intent, Intent::Search { .. }));
}

#[test]
fn test_performance() {
    use std::time::Instant;

    let mut classifier = IntentClassifier::new();

    // Measure bootstrap time
    let start = Instant::now();
    classifier.bootstrap();
    let bootstrap_time = start.elapsed();

    // Bootstrap should complete in reasonable time
    // Note: 2^14 (16384) dimensions require more processing than 2^12 (4096)
    // ~60 bootstrap examples × 16K dimensions = ~1-2s is acceptable
    assert!(
        bootstrap_time.as_millis() < 3000,
        "Bootstrap took too long: {:?}",
        bootstrap_time
    );

    // Measure classification time (average over 100 queries)
    let queries = vec!["search vim", "install git", "remove htop", "generations", "gc"];
    let start = Instant::now();
    for _ in 0..100 {
        for query in &queries {
            let _ = classifier.classify(query);
        }
    }
    let total_time = start.elapsed();
    let avg_time = total_time.as_micros() / 500; // 100 iterations * 5 queries

    // Each classification should be < 1ms on average
    assert!(
        avg_time < 1000,
        "Classification too slow: {} microseconds average",
        avg_time
    );
}
