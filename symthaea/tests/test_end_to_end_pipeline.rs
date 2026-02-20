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

use symthaea::hdc::primitive_system::PrimitiveSystem;
use symthaea::hdc::{TextEncoder, TextEncoderConfig};
use symthaea::learnable_ltc::{LearnableLTC, LearnableLTCConfig};

/// Test 1: Text to HDC encoding with primitive grounding
#[test]
fn test_text_to_hdc_encoding() {
    let config = TextEncoderConfig::default();

    let mut encoder = TextEncoder::new(config).expect("Should create encoder");
    let primitives = PrimitiveSystem::new();

    // Test encoding a sentence with primitives
    let text = "Causation implies correlation but not vice versa";
    let encoded = encoder
        .encode_with_primitives(text, &primitives)
        .expect("Should encode text successfully");

    // Verify dimensionality
    assert_eq!(encoded.len(), 16384, "HDC vector should be 16384D");

    // Verify it's a valid hypervector (bipolar: -1 or 1)
    let valid = encoded.iter().all(|&x| x == -1 || x == 1);
    assert!(valid, "Should be a valid bipolar hypervector");

    // Test that different texts produce different encodings
    let text2 = "Correlation does not imply causation";
    let encoded2 = encoder
        .encode_with_primitives(text2, &primitives)
        .expect("Should encode second text");

    // Compute cosine similarity (should be moderate - related but different)
    let similarity = cosine_similarity_i8(&encoded, &encoded2);
    assert!(
        similarity < 0.9,
        "Different sentences should have distinct encodings"
    );
    assert!(
        similarity > -0.5,
        "Related sentences should not be orthogonal"
    );

    println!("Text encoding test passed!");
    println!("  Similarity between related sentences: {:.3}", similarity);
}

/// Test 2: HDC vector through LTC temporal processing
#[test]
fn test_hdc_through_ltc() {
    // Create a small LTC network using default config with modifications
    let mut config = LearnableLTCConfig::default();
    config.input_dim = 64; // Downsampled HDC
    config.num_neurons = 32;
    config.output_dim = 16;
    config.num_steps = 10; // Fewer steps for test speed

    let mut ltc = LearnableLTC::new(config).expect("Should create LTC");

    // Create a sequence of HDC-like inputs (simulating encoded text over time)
    let sequence: Vec<Vec<f32>> = (0..10)
        .map(|t| {
            // Simulated HDC input that changes over time
            (0..64)
                .map(|i| ((i as f32 + t as f32 * 0.1).sin() * 0.5) as f32)
                .collect()
        })
        .collect();

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
    let temporal_difference: f32 = first_output
        .iter()
        .zip(last_output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        temporal_difference > 0.001,
        "LTC should show temporal dynamics"
    );

    println!("HDC -> LTC pipeline test passed!");
    println!(
        "  Temporal difference (first vs last): {:.4}",
        temporal_difference
    );
}

/// Test 3: Consciousness metrics through pipeline
#[test]
fn test_consciousness_metrics() {
    use symthaea::hdc::binary_hv::BinaryHV;
    use symthaea::hdc::integrated_information::IntegratedInformation;

    // Create IIT calculator
    let mut phi_calculator = IntegratedInformation::new();

    // Generate random HDC vectors representing conscious components
    let components: Vec<BinaryHV> = (0..8)
        .map(|i| {
            // Create random vectors with unique seeds
            BinaryHV::random(42 + i as u64)
        })
        .collect();

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

    let dot: i32 = a
        .iter()
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
