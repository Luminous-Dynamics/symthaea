//! HDC (Hyperdimensional Computing) Integration Tests
//!
//! Tests for the HV16 binary hypervector type and related operations.
//! These tests verify the mathematical properties and practical usage
//! patterns of hyperdimensional computing.

mod common;

use symthaea_core::hdc::binary_hv::HV16;

// ============================================================================
// DETERMINISM TESTS
// ============================================================================

#[test]
fn test_random_is_deterministic() {
    // Same seed should always produce same vector
    let v1 = HV16::random(42);
    let v2 = HV16::random(42);

    assert_eq!(v1, v2, "Same seed should produce identical vectors");
}

#[test]
fn test_different_seeds_produce_different_vectors() {
    let v1 = HV16::random(1);
    let v2 = HV16::random(2);
    let v3 = HV16::random(3);

    assert_ne!(v1, v2, "Different seeds should produce different vectors");
    assert_ne!(v2, v3, "Different seeds should produce different vectors");
    assert_ne!(v1, v3, "Different seeds should produce different vectors");
}

#[test]
fn test_basis_vectors_are_unique() {
    let basis_0 = HV16::basis(0);
    let basis_1 = HV16::basis(1);
    let basis_2 = HV16::basis(2);

    assert_ne!(basis_0, basis_1, "Basis vectors should be unique");
    assert_ne!(basis_1, basis_2, "Basis vectors should be unique");
    assert_ne!(basis_0, basis_2, "Basis vectors should be unique");
}

// ============================================================================
// MATHEMATICAL PROPERTY TESTS
// ============================================================================

#[test]
fn test_bind_is_self_inverse() {
    // A ⊗ A = 0 (all zeros)
    let a = HV16::random(100);
    let result = a.bind(&a);

    assert_eq!(result, HV16::zero(), "A bind A should equal zero vector");
}

#[test]
fn test_bind_is_commutative() {
    // A ⊗ B = B ⊗ A
    let a = HV16::random(200);
    let b = HV16::random(201);

    let ab = a.bind(&b);
    let ba = b.bind(&a);

    assert_eq!(ab, ba, "Bind should be commutative");
}

#[test]
fn test_bind_is_associative() {
    // (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
    let a = HV16::random(300);
    let b = HV16::random(301);
    let c = HV16::random(302);

    let ab_c = a.bind(&b).bind(&c);
    let a_bc = a.bind(&b.bind(&c));

    assert_eq!(ab_c, a_bc, "Bind should be associative");
}

#[test]
fn test_bind_with_zero_is_identity() {
    // A ⊗ 0 = A
    let a = HV16::random(400);
    let zero = HV16::zero();

    let result = a.bind(&zero);

    assert_eq!(result, a, "Bind with zero should be identity");
}

#[test]
fn test_unbind_recovers_original() {
    // Given C = A ⊗ B, then C ⊗ A = B
    let a = HV16::random(500);
    let b = HV16::random(501);

    let c = a.bind(&b);
    let recovered_b = c.bind(&a);

    assert_eq!(recovered_b, b, "Unbinding should recover original vector");
}

// ============================================================================
// SIMILARITY TESTS
// ============================================================================

#[test]
fn test_self_similarity_is_one() {
    let v = HV16::random(600);
    let sim = v.similarity(&v);

    assert!(
        (sim - 1.0).abs() < 0.0001,
        "Self-similarity should be 1.0, got {}",
        sim
    );
}

#[test]
fn test_similarity_with_inverse_is_zero() {
    let v = HV16::random(700);
    let inv = v.invert();
    let sim = v.similarity(&inv);

    assert!(
        sim.abs() < 0.0001,
        "Similarity with inverse should be 0.0, got {}",
        sim
    );
}

#[test]
fn test_random_vectors_have_low_similarity() {
    // Random vectors should be nearly orthogonal (~0.5 similarity)
    let similarities: Vec<f32> = (0..100)
        .map(|i| {
            let a = HV16::random(1000 + i * 2);
            let b = HV16::random(1000 + i * 2 + 1);
            a.similarity(&b)
        })
        .collect();

    let avg_sim: f32 = similarities.iter().sum::<f32>() / similarities.len() as f32;

    // Average similarity of random vectors should be around 0.5 (±0.1)
    assert!(
        (avg_sim - 0.5).abs() < 0.1,
        "Average random similarity should be ~0.5, got {}",
        avg_sim
    );
}

#[test]
fn test_bound_vectors_dissimilar_to_originals() {
    // C = A ⊗ B should be dissimilar to both A and B
    let a = HV16::random(800);
    let b = HV16::random(801);
    let c = a.bind(&b);

    let sim_ca = c.similarity(&a);
    let sim_cb = c.similarity(&b);

    // Bound vectors should be ~0.5 similar to originals (like random)
    assert!(
        sim_ca < 0.6,
        "Bound vector should be dissimilar to A, got {}",
        sim_ca
    );
    assert!(
        sim_cb < 0.6,
        "Bound vector should be dissimilar to B, got {}",
        sim_cb
    );
}

// ============================================================================
// HAMMING DISTANCE TESTS
// ============================================================================

#[test]
fn test_self_hamming_distance_is_zero() {
    let v = HV16::random(900);
    let dist = v.hamming_distance(&v);

    assert_eq!(dist, 0, "Self hamming distance should be 0");
}

#[test]
fn test_hamming_distance_with_inverse_is_max() {
    let v = HV16::random(1000);
    let inv = v.invert();
    let dist = v.hamming_distance(&inv);

    // Maximum hamming distance is DIM (16384)
    assert_eq!(
        dist as usize,
        HV16::DIM,
        "Hamming distance with inverse should be max ({})",
        HV16::DIM
    );
}

#[test]
fn test_hamming_distance_is_symmetric() {
    let a = HV16::random(1100);
    let b = HV16::random(1101);

    let dist_ab = a.hamming_distance(&b);
    let dist_ba = b.hamming_distance(&a);

    assert_eq!(dist_ab, dist_ba, "Hamming distance should be symmetric");
}

// ============================================================================
// BUNDLE TESTS
// ============================================================================

#[test]
fn test_bundle_preserves_similarity() {
    // Bundle of vectors should be similar to all constituents
    let vectors: Vec<HV16> = (0..5).map(|i| HV16::random(1200 + i)).collect();
    let bundled = HV16::bundle(&vectors);

    for (i, v) in vectors.iter().enumerate() {
        let sim = bundled.similarity(v);
        assert!(
            sim > 0.6,
            "Bundle should be similar to constituent {}, got {}",
            i,
            sim
        );
    }
}

#[test]
fn test_bundle_single_is_identity() {
    let v = HV16::random(1300);
    let bundled = HV16::bundle(&[v]);

    assert_eq!(bundled, v, "Bundle of single vector should be identity");
}

#[test]
fn test_bundle_empty_is_zero() {
    let bundled = HV16::bundle(&[]);

    assert_eq!(bundled, HV16::zero(), "Bundle of empty should be zero vector");
}

// ============================================================================
// PERMUTE TESTS
// ============================================================================

#[test]
fn test_permute_zero_is_identity() {
    let v = HV16::random(1400);
    let permuted = v.permute(0);

    assert_eq!(permuted, v, "Permute by 0 should be identity");
}

#[test]
fn test_permute_preserves_density() {
    let v = HV16::random(1500);
    let original_density = v.density();

    // Try various permutation amounts
    for shift in [1, 8, 64, 128, 1024, 8192] {
        let permuted = v.permute(shift);
        let permuted_density = permuted.density();

        assert!(
            (permuted_density - original_density).abs() < 0.01,
            "Permute by {} should preserve density ({} vs {})",
            shift,
            original_density,
            permuted_density
        );
    }
}

#[test]
fn test_permute_changes_vector() {
    let v = HV16::random(1600);
    let permuted = v.permute(1);

    assert_ne!(v, permuted, "Permute should change the vector");
}

// ============================================================================
// DENSITY TESTS
// ============================================================================

#[test]
fn test_zero_density() {
    let zero = HV16::zero();
    let density = zero.density();

    assert!(
        density.abs() < 0.0001,
        "Zero vector should have 0.0 density, got {}",
        density
    );
}

#[test]
fn test_ones_density() {
    let ones = HV16::ones();
    let density = ones.density();

    assert!(
        (density - 1.0).abs() < 0.0001,
        "Ones vector should have 1.0 density, got {}",
        density
    );
}

#[test]
fn test_random_density_is_balanced() {
    // Random vectors should have ~0.5 density
    let densities: Vec<f32> = (0..100)
        .map(|i| HV16::random(1700 + i).density())
        .collect();

    let avg_density: f32 = densities.iter().sum::<f32>() / densities.len() as f32;

    assert!(
        (avg_density - 0.5).abs() < 0.05,
        "Average random density should be ~0.5, got {}",
        avg_density
    );
}

// ============================================================================
// INVERT TESTS
// ============================================================================

#[test]
fn test_double_invert_is_identity() {
    let v = HV16::random(1800);
    let double_inverted = v.invert().invert();

    assert_eq!(double_inverted, v, "Double invert should be identity");
}

#[test]
fn test_invert_flips_density() {
    let v = HV16::random(1900);
    let original_density = v.density();
    let inverted_density = v.invert().density();

    assert!(
        (original_density + inverted_density - 1.0).abs() < 0.0001,
        "Invert should flip density: {} + {} should equal 1.0",
        original_density,
        inverted_density
    );
}

// ============================================================================
// ENCODING ROUNDTRIP TESTS
// ============================================================================

#[test]
fn test_bipolar_conversion_roundtrip() {
    let original = HV16::random(2000);
    let bipolar = original.to_bipolar();
    let recovered = HV16::from_bipolar(&bipolar);

    assert_eq!(
        original, recovered,
        "Bipolar conversion should be lossless roundtrip"
    );
}

#[test]
fn test_from_bits_creates_valid_vector() {
    // Create from all zeros
    let zero_bits = vec![0u64; 256];
    let from_zeros = HV16::from_bits(&zero_bits);
    assert_eq!(from_zeros, HV16::zero(), "From zero bits should equal zero vector");

    // Create from all ones
    let one_bits = vec![u64::MAX; 256];
    let from_ones = HV16::from_bits(&one_bits);
    assert_eq!(from_ones, HV16::ones(), "From one bits should equal ones vector");
}

// ============================================================================
// SEQUENCE ENCODING TESTS
// ============================================================================

#[test]
fn test_sequence_encoding_pattern() {
    // Common HDC pattern: encode sequence using permute + bind
    // Sequence [A, B, C] encoded as: A ⊗ ρ(B) ⊗ ρ²(C)
    let a = HV16::random(2100);
    let b = HV16::random(2101);
    let c = HV16::random(2102);

    let sequence = a.bind(&b.permute(1)).bind(&c.permute(2));

    // The sequence encoding should be a valid vector
    let density = sequence.density();
    assert!(
        density > 0.3 && density < 0.7,
        "Sequence encoding should have balanced density, got {}",
        density
    );

    // It should be dissimilar to original vectors (position matters)
    assert!(
        sequence.similarity(&a) < 0.7,
        "Sequence should encode position information"
    );
}

// ============================================================================
// MEMORY AND CLONE TESTS
// ============================================================================

#[test]
fn test_clone_is_equal() {
    let original = HV16::random(2200);
    let cloned = original.clone();

    assert_eq!(original, cloned, "Clone should equal original");
}

#[test]
fn test_default_is_zero() {
    let default = HV16::default();
    let zero = HV16::zero();

    assert_eq!(default, zero, "Default should be zero vector");
}
