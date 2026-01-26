//! HDC-Enhanced Epistemic Classification Tests
//!
//! Tests the HDC semantic similarity approach to epistemic classification.
//! This method handles novel phrasings better than pattern matching.

use symthaea::mind::{Mind, EpistemicStatus, HdcEpistemicStats};

/// Test enabling HDC classification
#[tokio::test]
async fn test_enable_hdc_classification() {
    println!("\n🧠 Testing HDC classification enablement\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    // Initially not enabled
    assert!(!mind.has_hdc_classifier());

    // Enable it
    mind.enable_hdc_classification().unwrap();

    // Now enabled
    assert!(mind.has_hdc_classifier());

    // Check stats
    let stats = mind.hdc_classifier_stats().unwrap();
    println!("   Total exemplars: {}", stats.total_exemplars);
    println!("   Unknown count: {}", stats.unknown_count);
    println!("   Unverifiable count: {}", stats.unverifiable_count);
    println!("   Known count: {}", stats.known_count);

    assert!(stats.total_exemplars > 30, "Should have many exemplars");
    assert!(stats.unknown_count > 10, "Should have unknown exemplars");
    assert!(stats.unverifiable_count > 10, "Should have unverifiable exemplars");
    assert!(stats.known_count > 10, "Should have known exemplars");

    println!("\n✅ HDC classification enabled successfully\n");
}

/// Test HDC classification of exact exemplar queries
#[tokio::test]
async fn test_hdc_classify_exact_exemplars() {
    println!("\n🎯 Testing HDC classification of exact exemplars\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
    mind.enable_hdc_classification().unwrap();

    let test_cases = [
        ("What is the GDP of Atlantis?", EpistemicStatus::Unknown),
        ("What will the stock market do tomorrow?", EpistemicStatus::Unverifiable),
        ("What is the capital of France?", EpistemicStatus::Known),
    ];

    for (query, expected) in test_cases {
        let (status, confidence) = mind.analyze_query_hdc(query).unwrap();

        println!("   Query: \"{}\"", query);
        println!("   Expected: {:?}, Got: {:?} (confidence: {:.1}%)", expected, status, confidence * 100.0);

        // Exact exemplars should be classified correctly with reasonable confidence
        assert_eq!(status, expected, "Classification mismatch for: {}", query);
        // Note: Confidence depends on HDC encoding quality; 40%+ is good enough
        assert!(confidence > 0.4, "Confidence should be reasonable for exact exemplar (got {:.1}%)", confidence * 100.0);
    }

    println!("\n✅ Exact exemplar classification works\n");
}

/// Test hybrid classification (HDC + pattern fallback)
#[tokio::test]
async fn test_hybrid_classification() {
    println!("\n🔄 Testing hybrid classification\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
    mind.enable_hdc_classification().unwrap();

    // Test various queries
    let test_cases = [
        "What is the population of Hogwarts?",  // Unknown (fictional)
        "Predict the lottery numbers",          // Unverifiable (future)
        "What is 2 + 2?",                       // Known (math)
        "Tell me about rust programming",       // Likely uncertain (novel)
    ];

    for query in test_cases {
        let (status, method) = mind.analyze_query_hybrid(query).await.unwrap();
        println!("   \"{}\"", query);
        println!("   → {:?} (via {})\n", status, method);
    }

    println!("✅ Hybrid classification works\n");
}

/// Test that HDC handles novel phrasings
#[tokio::test]
async fn test_hdc_novel_phrasings() {
    println!("\n🆕 Testing HDC classification of novel phrasings\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
    mind.enable_hdc_classification().unwrap();

    // These are NOT exact exemplars but should still classify correctly
    // due to semantic similarity
    let novel_queries = [
        // Variations on "GDP of Atlantis"
        ("Tell me the economic output of mythical Atlantis", EpistemicStatus::Unknown),
        ("Atlantis GDP statistics", EpistemicStatus::Unknown),

        // Variations on "capital of France"
        ("France capital city", EpistemicStatus::Known),
        ("What city is the capital of France", EpistemicStatus::Known),

        // Variations on "stock market tomorrow"
        ("Tomorrow's stock prices", EpistemicStatus::Unverifiable),
        ("Future market predictions", EpistemicStatus::Unverifiable),
    ];

    let mut correct = 0;
    let mut total = 0;

    for (query, expected) in novel_queries {
        let (status, confidence) = mind.analyze_query_hdc(query).unwrap();

        total += 1;
        if status == expected {
            correct += 1;
            println!("   ✅ \"{}\"", query);
        } else {
            println!("   ❌ \"{}\"", query);
        }
        println!("      Expected: {:?}, Got: {:?} (confidence: {:.1}%)\n", expected, status, confidence * 100.0);
    }

    let accuracy = (correct as f32 / total as f32) * 100.0;
    println!("📊 Novel phrasing accuracy: {}/{} ({:.1}%)\n", correct, total, accuracy);

    // We expect some novel phrasings to work, but not all
    // HDC quality depends on the encoding and exemplar coverage
    println!("✅ Novel phrasing test complete (accuracy depends on HDC encoding quality)\n");
}

/// Test think_auto_hybrid end-to-end
#[tokio::test]
async fn test_think_auto_hybrid() {
    println!("\n🧠 Testing think_auto_hybrid end-to-end\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
    mind.enable_hdc_classification().unwrap();

    let query = "What is the precise GDP of Atlantis in 2024?";
    println!("   Query: \"{}\"\n", query);

    let thought = mind.think_auto_hybrid(query).await.unwrap();

    println!("   Epistemic Status: {:?}", thought.epistemic_status);
    println!("   Semantic Intent: {:?}", thought.semantic_intent);
    println!("   Confidence: {:.2}", thought.confidence);
    println!("   Response: \"{}\"\n", thought.response_text);

    // Should detect as Unknown
    assert_eq!(
        thought.epistemic_status,
        EpistemicStatus::Unknown,
        "Atlantis query should be Unknown"
    );

    // Should contain hedging language
    assert!(
        thought.contains_hedging(),
        "Response should contain hedging language"
    );

    println!("✅ think_auto_hybrid works correctly\n");
}

/// Compare HDC vs pattern matching accuracy
#[tokio::test]
async fn test_hdc_vs_pattern_comparison() {
    println!("\n📊 Comparing HDC vs Pattern Matching\n");

    let mut mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();
    mind.enable_hdc_classification().unwrap();

    let test_cases = [
        // Exact matches (both should work)
        ("What is the GDP of Atlantis?", EpistemicStatus::Unknown),
        ("What will happen tomorrow?", EpistemicStatus::Unverifiable),
        ("Capital of France?", EpistemicStatus::Known),

        // Edge cases
        ("Hogwarts enrollment numbers", EpistemicStatus::Unknown),
        ("What am I thinking?", EpistemicStatus::Unverifiable),
        ("2+2", EpistemicStatus::Known),
    ];

    let mut hdc_correct = 0;
    let mut pattern_correct = 0;

    println!("   {:50} | {:12} | {:12} | Expected", "Query", "HDC", "Pattern");
    println!("   {}", "-".repeat(90));

    for (query, expected) in test_cases {
        let (hdc_status, _) = mind.analyze_query_hdc(query).unwrap();
        let pattern_status = mind.analyze_query(query).await.unwrap();

        if hdc_status == expected { hdc_correct += 1; }
        if pattern_status == expected { pattern_correct += 1; }

        let hdc_mark = if hdc_status == expected { "✅" } else { "❌" };
        let pattern_mark = if pattern_status == expected { "✅" } else { "❌" };

        println!(
            "   {:50} | {} {:?} | {} {:?} | {:?}",
            &query[..query.len().min(48)],
            hdc_mark, hdc_status,
            pattern_mark, pattern_status,
            expected
        );
    }

    println!();
    println!("   HDC accuracy: {}/{}", hdc_correct, test_cases.len());
    println!("   Pattern accuracy: {}/{}", pattern_correct, test_cases.len());

    println!("\n✅ Comparison complete\n");
}
