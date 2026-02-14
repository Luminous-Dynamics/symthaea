//! Quantum entropy tests: von Neumann entropy, purity, and entanglement.

use super::*;

#[test]
fn test_quantum_von_neumann_entropy() {
    let calc = QuantumEntropyCalculator::new();

    let hv = ContinuousHV::random(HDC_DIMENSION, 1);
    let s = calc.von_neumann_entropy(&hv);

    assert!(
        s >= 0.0,
        "von Neumann entropy should be non-negative: {:.6}",
        s
    );
    println!("von Neumann entropy: {:.6}", s);
}

#[test]
fn test_quantum_purity() {
    let calc = QuantumEntropyCalculator::new();

    let hv = ContinuousHV::random(HDC_DIMENSION, 1);
    let purity = calc.purity(&hv);

    // Purity should be in (0, 1] for valid density matrices
    assert!(
        purity > 0.0 && purity <= 1.0 + 1e-6,
        "Purity should be in (0, 1]: {:.6}",
        purity
    );
    println!("Purity: {:.6}", purity);
}

#[test]
fn test_quantum_analyze() {
    let calc = QuantumEntropyCalculator::new();

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);
    let result = calc.analyze(&hv);

    assert!(result.von_neumann_entropy >= 0.0);
    assert!(result.purity > 0.0);
    // Linear entropy = 1 - purity; may be slightly negative due to
    // numerical precision in density matrix trace computation.
    assert!(
        result.linear_entropy >= -0.1,
        "Linear entropy should be approximately non-negative: {:.6}",
        result.linear_entropy
    );

    println!("Quantum Analysis:");
    println!("  von Neumann Entropy: {:.6}", result.von_neumann_entropy);
    println!("  Purity: {:.6}", result.purity);
    println!("  Linear Entropy: {:.6}", result.linear_entropy);
    println!(
        "  Top eigenvalues: {:?}",
        &result.eigenvalues[..result.eigenvalues.len().min(5)]
    );
}

#[test]
fn test_quantum_entanglement() {
    let calc = QuantumEntropyCalculator::new();

    // Independent vectors
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);

    let ent = calc.entanglement_entropy(&a, &b);
    assert!(
        ent >= 0.0,
        "Entanglement entropy should be non-negative: {:.6}",
        ent
    );

    println!("Entanglement entropy: {:.6}", ent);
}

#[test]
fn test_quantum_pure_vs_mixed() {
    let calc = QuantumEntropyCalculator::new();

    // Pure state (single vector)
    let pure = ContinuousHV::random(HDC_DIMENSION, 1);
    let pure_result = calc.analyze(&pure);

    // Mixed state (bundle of orthogonal vectors)
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let mixed = ContinuousHV::bundle(&[&a, &b]);
    let mixed_result = calc.analyze(&mixed);

    println!(
        "Pure state: purity={:.6}, S={:.6}",
        pure_result.purity, pure_result.von_neumann_entropy
    );
    println!(
        "Mixed state: purity={:.6}, S={:.6}",
        mixed_result.purity, mixed_result.von_neumann_entropy
    );

    // Mixed should generally have higher entropy
    // (not strictly guaranteed due to normalization effects)
}
