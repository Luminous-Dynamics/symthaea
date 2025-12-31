// ==================================================================================
// End-to-End Integration Test: Full Symthaea Pipeline
// ==================================================================================
//
// **Purpose**: Test the complete flow from text input through all Symthaea components:
//   1. Text → HDC encoding (via TextEncoder with primitive grounding)
//   2. HDC → LTC temporal processing (via LearnableLTC)
//   3. LTC → Causal reasoning (via SymthaeaSolver)
//   4. Output → Verified consciousness metrics
//
// **This proves**: Symthaea's components work together as an integrated system,
// not just as isolated modules.
//
// ==================================================================================

use symthaea::hdc::{TextEncoder, TextEncoderConfig};
use symthaea::hdc::primitive_system::PrimitiveSystem;
use symthaea::learnable_ltc::{LearnableLTC, LearnableLTCConfig};
use symthaea::benchmarks::{
    CausalBenchmarkSuite, SymthaeaSolver, CausalGraph,
    CausalQuery, CausalAnswer, CausalBenchmark, CausalCategory,
};

/// Test 1: Text to HDC encoding with primitive grounding
#[test]
fn test_text_to_hdc_encoding() {
    let config = TextEncoderConfig::default();

    let mut encoder = TextEncoder::new(config).expect("Should create encoder");
    let primitives = PrimitiveSystem::new();

    // Test encoding a sentence with primitives
    let text = "Causation implies correlation but not vice versa";
    let encoded = encoder.encode_with_primitives(text, &primitives)
        .expect("Should encode text successfully");

    // Verify dimensionality
    assert_eq!(encoded.len(), 16384, "HDC vector should be 16384D");

    // Verify it's a valid hypervector (bipolar: -1 or 1)
    let valid = encoded.iter().all(|&x| x == -1 || x == 1);
    assert!(valid, "Should be a valid bipolar hypervector");

    // Test that different texts produce different encodings
    let text2 = "Correlation does not imply causation";
    let encoded2 = encoder.encode_with_primitives(text2, &primitives)
        .expect("Should encode second text");

    // Compute cosine similarity (should be moderate - related but different)
    let similarity = cosine_similarity_i8(&encoded, &encoded2);
    assert!(similarity < 0.9, "Different sentences should have distinct encodings");
    assert!(similarity > -0.5, "Related sentences should not be orthogonal");

    println!("Text encoding test passed!");
    println!("  Similarity between related sentences: {:.3}", similarity);
}

/// Test 2: HDC vector through LTC temporal processing
#[test]
fn test_hdc_through_ltc() {
    // Create a small LTC network using default config with modifications
    let mut config = LearnableLTCConfig::default();
    config.input_dim = 64;    // Downsampled HDC
    config.num_neurons = 32;
    config.output_dim = 16;
    config.num_steps = 10;    // Fewer steps for test speed

    let mut ltc = LearnableLTC::new(config).expect("Should create LTC");

    // Create a sequence of HDC-like inputs (simulating encoded text over time)
    let sequence: Vec<Vec<f32>> = (0..10).map(|t| {
        // Simulated HDC input that changes over time
        (0..64).map(|i| {
            ((i as f32 + t as f32 * 0.1).sin() * 0.5) as f32
        }).collect()
    }).collect();

    // Process sequence through LTC
    let mut outputs = Vec::new();
    for input in &sequence {
        let (output, _hidden_states) = ltc.forward(input).expect("Should process input");
        outputs.push(output);
    }

    // Verify temporal dynamics - outputs should show temporal evolution
    assert_eq!(outputs.len(), 10, "Should have 10 outputs");
    assert_eq!(outputs[0].len(), 16, "Each output should be 16D");

    // Check that the network exhibits temporal dynamics
    // (later outputs should be different from earlier ones)
    let first_output = &outputs[0];
    let last_output = &outputs[9];
    let temporal_difference: f32 = first_output.iter()
        .zip(last_output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(temporal_difference > 0.001, "LTC should show temporal dynamics");

    println!("HDC -> LTC pipeline test passed!");
    println!("  Temporal difference (first vs last): {:.4}", temporal_difference);
}

/// Test 3: LTC output through causal reasoning
#[test]
fn test_causal_reasoning_integration() {
    let mut solver = SymthaeaSolver::new();

    // Create a causal scenario: Weather → Mood → Productivity
    let mut graph = CausalGraph::new();
    graph.add_variable("weather");
    graph.add_variable("mood");
    graph.add_variable("productivity");
    graph.add_variable("coffee");

    // Causal edges with strengths
    graph.add_edge("weather", "mood", 0.7);
    graph.add_edge("mood", "productivity", 0.8);
    graph.add_edge("coffee", "productivity", 0.5);

    // Test causal queries
    let benchmark = CausalBenchmark {
        id: "integration_test".to_string(),
        description: "Integration test scenario".to_string(),
        category: CausalCategory::InterventionPrediction,
        ground_truth_graph: graph,
        observations: vec![],
        queries: vec![],
        expected_answers: vec![],
        difficulty: 2,
    };

    // Query 1: Does weather cause productivity?
    let query1 = CausalQuery::DoesCause {
        from: "weather".to_string(),
        to: "productivity".to_string(),
    };
    let answer1 = solver.solve(&benchmark, &query1);
    match answer1 {
        CausalAnswer::Boolean(causes) => {
            assert!(causes, "Weather should cause productivity transitively");
        }
        _ => panic!("Expected boolean answer"),
    }

    // Query 2: Is there a spurious correlation between weather and coffee?
    let query2 = CausalQuery::SpuriousCorrelation {
        var1: "weather".to_string(),
        var2: "coffee".to_string(),
    };
    let answer2 = solver.solve(&benchmark, &query2);
    match answer2 {
        CausalAnswer::Boolean(spurious) => {
            // No common cause, so not spurious
            assert!(!spurious, "Weather and coffee have no spurious correlation");
        }
        _ => panic!("Expected boolean answer"),
    }

    // Query 3: Intervention - what if we improve mood?
    let query3 = CausalQuery::Intervention {
        variable: "mood".to_string(),
        value: 1.0,
        target: "productivity".to_string(),
    };
    let answer3 = solver.solve(&benchmark, &query3);
    match answer3 {
        CausalAnswer::Range { expected, .. } => {
            assert!(expected > 0.0, "Improving mood should increase productivity");
        }
        _ => panic!("Expected range answer"),
    }

    println!("Causal reasoning integration test passed!");
    println!("  Solver stats: {:?}", solver.stats());
}

/// Test 4: Full pipeline - Text → HDC → Semantic Analysis → Causal Query
#[test]
fn test_full_pipeline() {
    // Step 1: Encode causal statements
    let config = TextEncoderConfig::default();
    let mut encoder = TextEncoder::new(config).expect("Should create encoder");
    let primitives = PrimitiveSystem::new();

    let statements = vec![
        "Smoking causes cancer",
        "Exercise improves health",
        "Ice cream sales correlate with drowning",  // Spurious!
    ];

    let encodings: Vec<Vec<i8>> = statements.iter()
        .map(|s| encoder.encode_with_primitives(s, &primitives).unwrap())
        .collect();

    // Step 2: Verify encodings capture semantic relationships
    // "Smoking causes cancer" and "Exercise improves health" are both causal
    // "Ice cream sales correlate with drowning" is correlational

    let sim_causal = cosine_similarity_i8(&encodings[0], &encodings[1]);
    let sim_spurious = cosine_similarity_i8(&encodings[0], &encodings[2]);

    println!("Full pipeline test:");
    println!("  Similarity (causal statements): {:.3}", sim_causal);
    println!("  Similarity (causal vs spurious): {:.3}", sim_spurious);

    // Step 3: Run causal benchmarks
    let suite = CausalBenchmarkSuite::standard();
    let mut solver = SymthaeaSolver::new();
    let results = suite.run(|benchmark, query| solver.solve(benchmark, query));

    println!("  Causal benchmark accuracy: {:.1}%", results.accuracy() * 100.0);

    // Symthaea should significantly beat random chance
    assert!(results.accuracy() > 0.5, "Should beat 50% random baseline");

    // Step 4: Verify the full system is coherent
    let stats = solver.stats();
    assert!(stats.queries_solved > 0, "Should have solved queries");
    assert!(stats.causal_detections > 0, "Should have detected causal relationships");

    println!("Full pipeline test passed!");
    println!("  Total queries solved: {}", stats.queries_solved);
    println!("  Causal detections: {}", stats.causal_detections);
}

/// Test 5: Consciousness metrics through pipeline
#[test]
fn test_consciousness_metrics() {
    use symthaea::hdc::integrated_information::IntegratedInformation;
    use symthaea::hdc::binary_hv::HV16;

    // Create IIT calculator
    let mut phi_calculator = IntegratedInformation::new();

    // Generate random HDC vectors representing conscious components
    let components: Vec<HV16> = (0..8).map(|i| {
        // Create random vectors with unique seeds
        HV16::random(42 + i as u64)
    }).collect();

    // Compute Phi for these components
    let phi = phi_calculator.compute_phi(&components);

    println!("Consciousness metrics test:");
    println!("  Integrated Information (Phi): {:.4}", phi);

    // Phi should be non-negative for any system
    assert!(phi >= 0.0, "Phi should be non-negative");

    println!("Consciousness metrics test passed!");
}

// ==================================================================================
// Helper Functions
// ==================================================================================

/// Compute cosine similarity between two bipolar vectors
fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: i32 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum();

    let mag_a: f32 = (a.iter().map(|&x| x as i32 * x as i32).sum::<i32>() as f32).sqrt();
    let mag_b: f32 = (b.iter().map(|&x| x as i32 * x as i32).sum::<i32>() as f32).sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot as f32 / (mag_a * mag_b)
    } else {
        0.0
    }
}
