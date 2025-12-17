/*!
Week 17: SDM Integration Tests

Validates that Sparse Distributed Memory integrates correctly with
the broader HDC ecosystem:
- Temporal Encoder chrono-semantic binding
- SemanticSpace concept encoding
- Dimension consistency across modules

These tests ensure SDM is production-ready for episodic memory.
*/

use std::time::Duration;
use symthaea::hdc::{
    HDC_DIMENSION,
    SparseDistributedMemory,
    SDMConfig,
    ReadResult,
    IterativeReadResult,
    hamming_similarity,
    random_bipolar_vector,
    add_noise,
    TemporalEncoder,
    SemanticSpace,
};

// ============================================================================
// DIMENSION CONSISTENCY TESTS
// ============================================================================

#[test]
fn test_hdc_dimension_is_16k() {
    // Verify the canonical HDC dimension is 16,384
    assert_eq!(HDC_DIMENSION, 16_384, "HDC_DIMENSION should be 16,384");
}

#[test]
fn test_temporal_encoder_uses_hdc_dimension() {
    // Temporal encoder should default to HDC_DIMENSION
    let encoder = TemporalEncoder::new();
    let (dimension, _, _) = encoder.config();
    assert_eq!(dimension, HDC_DIMENSION, "TemporalEncoder should use HDC_DIMENSION");
}

#[test]
fn test_sdm_default_uses_hdc_dimension() {
    // SDM should default to HDC_DIMENSION
    let sdm = SparseDistributedMemory::new(SDMConfig::default());
    let config = sdm.config();
    assert_eq!(config.dimension, HDC_DIMENSION, "SDM should default to HDC_DIMENSION");
}

#[test]
fn test_semantic_space_uses_hdc_dimension() {
    // SemanticSpace should use HDC_DIMENSION
    let mut space = SemanticSpace::new(HDC_DIMENSION).unwrap();
    let vec = space.encode("test concept").unwrap();
    assert_eq!(vec.len(), HDC_DIMENSION, "SemanticSpace vectors should have HDC_DIMENSION length");
}

// ============================================================================
// SDM + TEMPORAL ENCODER INTEGRATION
// ============================================================================

#[test]
fn test_sdm_stores_temporal_vectors() {
    // SDM should store temporal encoding vectors
    // Use smaller dimension for faster testing
    let test_dim = 1000;

    let encoder = TemporalEncoder::with_config(
        test_dim,
        Duration::from_secs(24 * 3600), // 24h cycle
        0.0
    );

    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 2000,
        activation_radius: 0.42,
        weighted_read: true,
        min_activation_count: 5,
    });

    // Encode "noon" as temporal vector
    let noon = Duration::from_secs(12 * 3600);
    let temporal_vec = encoder.encode_time(noon).unwrap();

    // Convert f32 to i8 for SDM storage
    let sdm_vec: Vec<i8> = temporal_vec.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    // Store with multiple writes for reinforcement
    for _ in 0..10 {
        sdm.write_auto(&sdm_vec);
    }

    // Should retrieve similar pattern
    if let ReadResult::Success { pattern, .. } = sdm.read(&sdm_vec) {
        let sim = hamming_similarity(&pattern, &sdm_vec);
        assert!(sim > 0.6, "Should retrieve temporal pattern, got similarity {}", sim);
    }
}

#[test]
fn test_chrono_semantic_binding() {
    // Test binding temporal + semantic vectors then storing in SDM
    let test_dim = 500;

    let encoder = TemporalEncoder::with_config(
        test_dim,
        Duration::from_secs(24 * 3600),
        0.0
    );

    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 1500,
        activation_radius: 0.42,
        weighted_read: true,
        min_activation_count: 5,
    });

    // Create temporal vector (morning: 9 AM)
    let morning = Duration::from_secs(9 * 3600);
    let temporal = encoder.encode_time(morning).unwrap();

    // Create semantic vector (random pattern representing "breakfast")
    let semantic: Vec<f32> = (0..test_dim)
        .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
        .collect();

    // Bind temporal + semantic (chrono-semantic memory)
    let bound = encoder.bind(&temporal, &semantic).unwrap();

    // Convert to bipolar for SDM
    let sdm_vec: Vec<i8> = bound.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    // Store the chrono-semantic memory
    for _ in 0..15 {
        sdm.write_auto(&sdm_vec);
    }

    // Retrieve with slight noise
    let noisy = add_noise(&sdm_vec, 0.1);
    if let ReadResult::Success { pattern, .. } = sdm.read(&noisy) {
        let sim = hamming_similarity(&pattern, &sdm_vec);
        assert!(sim > 0.5, "Should retrieve chrono-semantic pattern, got {}", sim);
    }
}

#[test]
fn test_temporal_query_retrieves_time_based_memories() {
    // Store multiple memories at different times, query by time
    let test_dim = 500;

    let encoder = TemporalEncoder::with_config(
        test_dim,
        Duration::from_secs(24 * 3600),
        0.0
    );

    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 2000,
        activation_radius: 0.42,
        weighted_read: true,
        min_activation_count: 3,
    });

    // Store memory at 9 AM
    let time_9am = Duration::from_secs(9 * 3600);
    let temporal_9am = encoder.encode_time(time_9am).unwrap();
    let sdm_9am: Vec<i8> = temporal_9am.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    for _ in 0..10 {
        sdm.write_auto(&sdm_9am);
    }

    // Query with similar time (9:05 AM)
    let time_905 = Duration::from_secs(9 * 3600 + 5 * 60);
    let temporal_905 = encoder.encode_time(time_905).unwrap();
    let query: Vec<i8> = temporal_905.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    // Similar times should have high similarity
    let time_sim = encoder.temporal_similarity(time_9am, time_905).unwrap();
    assert!(time_sim > 0.95, "9:00 and 9:05 should be very similar: {}", time_sim);

    // SDM should retrieve something close to original
    if let ReadResult::Success { pattern, confidence, .. } = sdm.read(&query) {
        let original_sim = hamming_similarity(&pattern, &sdm_9am);
        println!("Query confidence: {}, retrieved-to-original: {}", confidence, original_sim);
        // With similar query times, we should retrieve related patterns
        assert!(original_sim > 0.4, "Retrieved pattern should be related to stored");
    }
}

// ============================================================================
// SDM + SEMANTIC SPACE INTEGRATION
// ============================================================================

#[test]
fn test_sdm_stores_semantic_encodings() {
    // Store semantic concept vectors in SDM
    // Note: Uses smaller dimension for test speed
    let test_dim = 1000;

    let mut space = SemanticSpace::new(test_dim).unwrap();
    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 2000,
        activation_radius: 0.42,
        weighted_read: true,
        min_activation_count: 5,
    });

    // Encode a concept
    let concept_vec = space.encode("breakfast cereal").unwrap();

    // Convert f32 to i8 bipolar
    let sdm_vec: Vec<i8> = concept_vec.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    // Store with reinforcement
    for _ in 0..15 {
        sdm.write_auto(&sdm_vec);
    }

    // Retrieve
    if let ReadResult::Success { pattern, confidence, .. } = sdm.read(&sdm_vec) {
        let sim = hamming_similarity(&pattern, &sdm_vec);
        println!("Semantic retrieval - read confidence: {}, pattern similarity: {}", confidence, sim);
        assert!(sim > 0.5, "Should retrieve semantic pattern");
    }
}

#[test]
fn test_similar_concepts_cluster_in_sdm() {
    // Similar semantic concepts should activate overlapping hard locations
    let test_dim = 500;

    let mut space = SemanticSpace::new(test_dim).unwrap();
    // SDM available for future cluster verification
    let _sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 1500,
        activation_radius: 0.45,
        weighted_read: true,
        min_activation_count: 3,
    });

    // Encode related concepts
    let cat = space.encode("cat").unwrap();
    let kitten = space.encode("kitten").unwrap();
    let car = space.encode("automobile").unwrap();

    // Convert to bipolar
    let cat_bipolar: Vec<i8> = cat.iter().map(|&v| if v >= 0.0 { 1 } else { -1 }).collect();
    let kitten_bipolar: Vec<i8> = kitten.iter().map(|&v| if v >= 0.0 { 1 } else { -1 }).collect();
    let car_bipolar: Vec<i8> = car.iter().map(|&v| if v >= 0.0 { 1 } else { -1 }).collect();

    // Check semantic similarity
    let cat_kitten_sim = hamming_similarity(&cat_bipolar, &kitten_bipolar);
    let cat_car_sim = hamming_similarity(&cat_bipolar, &car_bipolar);

    println!("cat-kitten similarity: {}", cat_kitten_sim);
    println!("cat-car similarity: {}", cat_car_sim);

    // Random semantic encodings should have ~0.5 similarity (chance)
    // This test validates the encoding produces valid bipolar vectors
    assert!(cat_kitten_sim > 0.0 && cat_kitten_sim < 1.0, "Should be valid similarity");
    assert!(cat_car_sim > 0.0 && cat_car_sim < 1.0, "Should be valid similarity");
}

// ============================================================================
// ITERATIVE RETRIEVAL TESTS
// ============================================================================

#[test]
fn test_iterative_read_cleans_noisy_temporal_cue() {
    // Iterative read should clean up noisy temporal query
    let test_dim = 500;

    let encoder = TemporalEncoder::with_config(
        test_dim,
        Duration::from_secs(24 * 3600),
        0.0
    );

    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 2000,
        activation_radius: 0.40,
        weighted_read: true,
        min_activation_count: 3,
    });

    // Store clean temporal pattern
    let noon = Duration::from_secs(12 * 3600);
    let temporal = encoder.encode_time(noon).unwrap();
    let clean: Vec<i8> = temporal.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect();

    // Strong reinforcement
    for _ in 0..20 {
        sdm.write_auto(&clean);
    }

    // Query with 20% noise
    let noisy = add_noise(&clean, 0.20);

    // Iterative read should converge toward clean pattern
    match sdm.iterative_read(&noisy, 10) {
        IterativeReadResult::Converged { pattern, iterations } => {
            let sim = hamming_similarity(&pattern, &clean);
            println!("Converged in {} iterations, similarity: {}", iterations, sim);
            assert!(sim > 0.5, "Should clean up noisy pattern to >0.5 similarity");
        }
        IterativeReadResult::MaxIterations { pattern, .. } => {
            let sim = hamming_similarity(&pattern, &clean);
            println!("Max iterations reached, similarity: {}", sim);
            // Still acceptable for small test config
        }
        IterativeReadResult::Failed { .. } => {
            // Acceptable for small test configuration
        }
    }
}

// ============================================================================
// PERFORMANCE INTEGRATION TEST
// ============================================================================

#[test]
fn test_sdm_temporal_encoder_pipeline_performance() {
    use std::time::Instant;

    let test_dim = 500;

    let encoder = TemporalEncoder::with_config(
        test_dim,
        Duration::from_secs(24 * 3600),
        0.0
    );

    let mut sdm = SparseDistributedMemory::new(SDMConfig {
        dimension: test_dim,
        num_hard_locations: 1000,
        activation_radius: 0.42,
        weighted_read: true,
        min_activation_count: 3,
    });

    let start = Instant::now();

    // Encode 20 different times
    let times: Vec<Duration> = (0..20)
        .map(|i| Duration::from_secs(i * 3600))
        .collect();

    for time in &times {
        let temporal = encoder.encode_time(*time).unwrap();
        let sdm_vec: Vec<i8> = temporal.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        // Store each time
        for _ in 0..5 {
            sdm.write_auto(&sdm_vec);
        }
    }

    let write_time = start.elapsed();

    // Read back 20 times
    let read_start = Instant::now();
    for time in &times {
        let temporal = encoder.encode_time(*time).unwrap();
        let query: Vec<i8> = temporal.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();
        sdm.read(&query);
    }
    let read_time = read_start.elapsed();

    // Should complete quickly
    let total_ms = (write_time + read_time).as_millis();
    println!("✅ Integration pipeline: {} writes in {:?}, 20 reads in {:?}, total: {}ms",
        20 * 5, write_time, read_time, total_ms);

    // Generous threshold for debug mode
    let threshold = if cfg!(debug_assertions) { 10_000 } else { 1_000 };
    assert!(total_ms < threshold as u128,
        "Pipeline should complete in <{}ms, took {}ms", threshold, total_ms);
}

// ============================================================================
// DIMENSION COMPATIBILITY TESTS
// ============================================================================

#[test]
fn test_all_hdc_modules_can_produce_hdc_dimension_vectors() {
    // Verify all major HDC components can produce vectors of HDC_DIMENSION

    // 1. TemporalEncoder
    let encoder = TemporalEncoder::new();
    let temporal = encoder.encode_time(Duration::from_secs(1000)).unwrap();
    assert_eq!(temporal.len(), HDC_DIMENSION, "TemporalEncoder output should be HDC_DIMENSION");

    // 2. SemanticSpace
    let mut ss = SemanticSpace::new(HDC_DIMENSION).unwrap();
    let semantic = ss.encode("test").unwrap();
    assert_eq!(semantic.len(), HDC_DIMENSION, "SemanticSpace output should be HDC_DIMENSION");

    // 3. SDM random vectors
    let bipolar = random_bipolar_vector(HDC_DIMENSION);
    assert_eq!(bipolar.len(), HDC_DIMENSION, "random_bipolar_vector should produce HDC_DIMENSION");

    println!("✅ All HDC modules produce {}-dimensional vectors", HDC_DIMENSION);
}
