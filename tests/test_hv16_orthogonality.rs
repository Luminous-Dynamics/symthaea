//! Test HV16 Orthogonality at 16,384 Dimensions

use symthaea::hdc::binary_hv::HV16;

#[test]
fn test_random_vector_orthogonality() {
    println!("\n=== HV16 Orthogonality at {} dimensions ===\n", HV16::DIM);

    // Generate 100 random vectors
    let vectors: Vec<HV16> = (0..100).map(|i| HV16::random(i as u64)).collect();

    // Measure pairwise similarities
    let mut similarities = Vec::new();
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let sim = vectors[i].similarity(&vectors[j]);
            similarities.push(sim);
        }
    }

    // Statistics
    let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;
    let variance = similarities
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f32>()
        / similarities.len() as f32;
    let std_dev = variance.sqrt();

    let min_sim = similarities
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_sim = similarities
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Pairwise similarity statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std dev: {:.6}", std_dev);
    println!("  Min: {:.6}", min_sim);
    println!("  Max: {:.6}", max_sim);
    println!("  Total pairs: {}", similarities.len());

    // Expected values for random binary hypervectors
    // At 16,384 dimensions, similarity should be ~0 with very small std
    let expected_mean = 0.0;
    let expected_std = 1.0 / (HV16::DIM as f32).sqrt(); // ≈ 0.0078

    println!("\nExpected values:");
    println!("  Mean: {:.6}", expected_mean);
    println!("  Std dev: {:.6}", expected_std);

    // Validation checks
    assert!(
        mean.abs() < 0.02,
        "Mean similarity should be near 0, got {}",
        mean
    );
    assert!(
        (std_dev - expected_std).abs() < 0.005,
        "Std deviation should be ~{}, got {}",
        expected_std,
        std_dev
    );
    assert!(
        max_sim < 0.1,
        "Max similarity should be < 0.1, got {}",
        max_sim
    );
    assert!(
        min_sim > -0.1,
        "Min similarity should be > -0.1, got {}",
        min_sim
    );

    println!("\n✅ All orthogonality checks passed!");
}

#[test]
fn test_bind_orthogonality() {
    println!("\n=== Bind Operation Orthogonality ===\n");

    let a = HV16::random(1);
    let b = HV16::random(2);
    let c = a.bind(&b);

    // c should be orthogonal to both a and b
    let sim_ca = c.similarity(&a);
    let sim_cb = c.similarity(&b);

    println!("Bind orthogonality:");
    println!("  c = a ⊗ b");
    println!("  sim(c, a) = {:.6}", sim_ca);
    println!("  sim(c, b) = {:.6}", sim_cb);

    assert!(
        sim_ca.abs() < 0.1,
        "Bound vector should be orthogonal to operands"
    );
    assert!(
        sim_cb.abs() < 0.1,
        "Bound vector should be orthogonal to operands"
    );

    println!("✅ Bind preserves orthogonality");
}

#[test]
fn test_bundle_similarity() {
    println!("\n=== Bundle Operation Similarity ===\n");

    let a = HV16::random(10);
    let b = HV16::random(20);
    let c = HV16::random(30);

    let bundle = HV16::bundle(&[a.clone(), b.clone(), c.clone()]);

    // Bundle should be similar to all inputs
    let sim_a = bundle.similarity(&a);
    let sim_b = bundle.similarity(&b);
    let sim_c = bundle.similarity(&c);

    println!("Bundle similarity:");
    println!("  bundle = a + b + c");
    println!("  sim(bundle, a) = {:.6}", sim_a);
    println!("  sim(bundle, b) = {:.6}", sim_b);
    println!("  sim(bundle, c) = {:.6}", sim_c);

    // Bundle should be moderately similar to inputs (typically ~0.3-0.6)
    assert!(
        sim_a > 0.1,
        "Bundle should be similar to component a, got {}",
        sim_a
    );
    assert!(
        sim_b > 0.1,
        "Bundle should be similar to component b, got {}",
        sim_b
    );
    assert!(
        sim_c > 0.1,
        "Bundle should be similar to component c, got {}",
        sim_c
    );

    println!("✅ Bundle preserves similarity to components");
}

#[test]
fn test_memory_and_performance() {
    use std::mem::size_of;
    use std::time::Instant;

    println!("\n=== Memory & Performance ===\n");

    // Memory
    let hv16_size = size_of::<HV16>();
    println!("HV16 size: {} bytes ({} KB)", hv16_size, hv16_size / 1024);
    assert_eq!(hv16_size, 2048, "HV16 should be 2048 bytes");

    // Performance: bind
    let a = HV16::random(100);
    let b = HV16::random(200);

    let start = Instant::now();
    for _ in 0..10000 {
        let _ = a.bind(&b);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed.as_nanos() / 10000;

    println!("\nBind performance:");
    println!("  10,000 operations in {:?}", elapsed);
    println!("  Average: {} ns/op", per_op);

    // Should be fast (<200ns on modern CPU)
    assert!(
        per_op < 500,
        "Bind should be < 500ns, got {}ns",
        per_op
    );

    // Performance: similarity
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = a.similarity(&b);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed.as_nanos() / 10000;

    println!("\nSimilarity performance:");
    println!("  10,000 operations in {:?}", elapsed);
    println!("  Average: {} ns/op", per_op);

    // Should be fast (<400ns on modern CPU)
    assert!(
        per_op < 1000,
        "Similarity should be < 1000ns, got {}ns",
        per_op
    );

    println!("\n✅ Performance within expected ranges");
}

#[test]
fn test_theoretical_properties() {
    println!("\n=== Theoretical Properties ===\n");

    let dim = HV16::DIM as f32;

    // Hamming distance properties
    let expected_hamming = dim / 2.0; // 8192 bits
    let hamming_std = (dim / 4.0).sqrt(); // ≈ 64 bits

    println!("For {}-bit binary hypervectors:", HV16::DIM);
    println!("  Expected Hamming distance: {:.0} bits", expected_hamming);
    println!("  Hamming std deviation: {:.0} bits", hamming_std);

    // Test actual Hamming distances
    let a = HV16::random(1000);
    let b = HV16::random(2000);
    let hamming = a.hamming_distance(&b);

    println!("\nActual Hamming distance: {} bits", hamming);

    // Should be within a few std deviations of expected
    let diff = (hamming as f32 - expected_hamming).abs();
    assert!(
        diff < hamming_std * 5.0,
        "Hamming distance should be within 5 std deviations of expected"
    );

    println!("✅ Theoretical properties validated");
}
