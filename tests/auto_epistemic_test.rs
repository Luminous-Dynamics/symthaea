//! Automatic Epistemic Detection Tests
//!
//! Tests the Mind's ability to automatically classify queries
//! into the correct epistemic category without manual intervention.

use symthaea::mind::{Mind, EpistemicStatus};

/// Test that Unknown entities are correctly detected
#[tokio::test]
async fn test_auto_detect_unknown_fictional() {
    println!("\n🔍 Testing automatic detection of UNKNOWN (fictional entities)\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let unknown_queries = [
        ("What is the GDP of Atlantis?", "Atlantis"),
        ("Who is the king of Narnia?", "Narnia"),
        ("What is the population of Hogwarts?", "Hogwarts"),
        ("What's the weather in Mordor?", "Mordor"),
        ("How do I get to El Dorado?", "El Dorado"),
    ];

    for (query, entity) in unknown_queries {
        let status = mind.analyze_query(query).await.unwrap();
        println!("   Query: \"{}\"", query);
        println!("   Entity: {} → Status: {:?}", entity, status);

        assert_eq!(
            status,
            EpistemicStatus::Unknown,
            "Query about {} should be Unknown, got {:?}",
            entity,
            status
        );
    }

    println!("\n✅ All fictional entities correctly detected as Unknown\n");
}

/// Test that Unverifiable queries are correctly detected
#[tokio::test]
async fn test_auto_detect_unverifiable() {
    println!("\n🔮 Testing automatic detection of UNVERIFIABLE\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let unverifiable_queries = [
        ("What will the stock market do tomorrow?", "future prediction"),
        ("Will I be successful next year?", "personal future"),
        ("What am I thinking right now?", "subjective/mind reading"),
        ("What if Napoleon had won at Waterloo?", "counterfactual"),
        ("Predict the lottery numbers", "impossible prediction"),
    ];

    for (query, reason) in unverifiable_queries {
        let status = mind.analyze_query(query).await.unwrap();
        println!("   Query: \"{}\"", query);
        println!("   Type: {} → Status: {:?}", reason, status);

        assert_eq!(
            status,
            EpistemicStatus::Unverifiable,
            "Query '{}' ({}) should be Unverifiable, got {:?}",
            query,
            reason,
            status
        );
    }

    println!("\n✅ All unverifiable queries correctly detected\n");
}

/// Test that Known facts are correctly detected
#[tokio::test]
async fn test_auto_detect_known() {
    println!("\n📚 Testing automatic detection of KNOWN\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let known_queries = [
        ("What is the capital of France?", "geography"),
        ("What is 2 + 2?", "math"),
        ("What is 15 * 3?", "math"),
        ("Who wrote Hamlet?", "literature"),
        ("What is the boiling point of water?", "science"),
    ];

    for (query, category) in known_queries {
        let status = mind.analyze_query(query).await.unwrap();
        println!("   Query: \"{}\"", query);
        println!("   Category: {} → Status: {:?}", category, status);

        assert_eq!(
            status,
            EpistemicStatus::Known,
            "Query '{}' ({}) should be Known, got {:?}",
            query,
            category,
            status
        );
    }

    println!("\n✅ All known facts correctly detected\n");
}

/// Test the think_auto method end-to-end
#[tokio::test]
async fn test_think_auto_atlantis() {
    println!("\n🧠 Testing think_auto with Atlantis query\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let query = "What is the precise GDP of Atlantis in 2024?";
    println!("   Query: \"{}\"\n", query);

    // Use automatic detection
    let thought = mind.think_auto(query).await.unwrap();

    println!("   Auto-detected Status: {:?}", thought.epistemic_status);
    println!("   Semantic Intent: {:?}", thought.semantic_intent);
    println!("   Confidence: {:.2}", thought.confidence);
    println!("   Response: \"{}\"\n", thought.response_text);

    // Verify
    assert_eq!(
        thought.epistemic_status,
        EpistemicStatus::Unknown,
        "Atlantis query should auto-detect as Unknown"
    );

    assert!(
        thought.contains_hedging(),
        "Response should contain hedging language"
    );

    println!("✅ think_auto correctly detected and handled Atlantis query\n");
}

/// Test the think_auto method with a known query
#[tokio::test]
async fn test_think_auto_known() {
    println!("\n🧠 Testing think_auto with known query\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let query = "What is 2 + 2?";
    println!("   Query: \"{}\"\n", query);

    let thought = mind.think_auto(query).await.unwrap();

    println!("   Auto-detected Status: {:?}", thought.epistemic_status);
    println!("   Semantic Intent: {:?}", thought.semantic_intent);
    println!("   Confidence: {:.2}", thought.confidence);
    println!("   Response: \"{}\"\n", thought.response_text);

    assert_eq!(
        thought.epistemic_status,
        EpistemicStatus::Known,
        "Math query should auto-detect as Known"
    );

    println!("✅ think_auto correctly detected known query\n");
}

/// Test the think_auto method with an unverifiable query
#[tokio::test]
async fn test_think_auto_future() {
    println!("\n🔮 Testing think_auto with future prediction\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let query = "What will the stock market do tomorrow?";
    println!("   Query: \"{}\"\n", query);

    let thought = mind.think_auto(query).await.unwrap();

    println!("   Auto-detected Status: {:?}", thought.epistemic_status);
    println!("   Semantic Intent: {:?}", thought.semantic_intent);
    println!("   Response: \"{}\"\n", thought.response_text);

    assert_eq!(
        thought.epistemic_status,
        EpistemicStatus::Unverifiable,
        "Future prediction should auto-detect as Unverifiable"
    );

    println!("✅ think_auto correctly detected unverifiable query\n");
}

/// Comprehensive test of all epistemic categories
#[tokio::test]
async fn test_comprehensive_epistemic_detection() {
    println!("\n🎯 COMPREHENSIVE EPISTEMIC DETECTION TEST\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mind = Mind::new_with_simulated_llm(512, 32).await.unwrap();

    let test_cases: Vec<(&str, EpistemicStatus)> = vec![
        // Unknown - Fictional
        ("What is the GDP of Atlantis?", EpistemicStatus::Unknown),
        ("Population of Hogwarts?", EpistemicStatus::Unknown),
        ("Capital of Mordor?", EpistemicStatus::Unknown),

        // Unknown - Nonsensical
        ("What is the color of happiness?", EpistemicStatus::Unknown),
        ("What does Tuesday smell like?", EpistemicStatus::Unknown),

        // Unverifiable - Future
        ("What will happen next year?", EpistemicStatus::Unverifiable),
        ("Stock market tomorrow?", EpistemicStatus::Unverifiable),

        // Unverifiable - Subjective
        ("What am I thinking?", EpistemicStatus::Unverifiable),
        ("Read my mind", EpistemicStatus::Unverifiable),

        // Known - Facts
        ("Capital of France?", EpistemicStatus::Known),
        ("2 + 2?", EpistemicStatus::Known),
        ("Who wrote Hamlet?", EpistemicStatus::Known),

        // Uncertain - Novel/unclear
        ("Tell me about quantum computing", EpistemicStatus::Uncertain),
        ("What is consciousness?", EpistemicStatus::Uncertain),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (query, expected) in &test_cases {
        let actual = mind.analyze_query(query).await.unwrap();
        let ok = actual == *expected;

        if ok {
            passed += 1;
            println!("   ✅ \"{}\"", query);
            println!("      Expected: {:?}, Got: {:?}\n", expected, actual);
        } else {
            failed += 1;
            println!("   ❌ \"{}\"", query);
            println!("      Expected: {:?}, Got: {:?}\n", expected, actual);
        }
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📊 Results: {}/{} passed ({:.1}%)\n",
             passed,
             passed + failed,
             100.0 * passed as f64 / (passed + failed) as f64);

    assert_eq!(failed, 0, "{} tests failed", failed);
    println!("🏆 ALL EPISTEMIC DETECTION TESTS PASSED!\n");
}
